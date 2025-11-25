"""
Migration script to re-embed all chunks with all-mpnet-base-v2 model
--------------------------------------------------------------------
This script will:
1. Read all chunks from Supabase
2. Re-embed them using the mpnet model (768-dim)
3. Update the embeddings in the database

Usage:
    python migrate_embeddings_to_mpnet.py [--user-id USER_ID] [--notebook-id NOTEBOOK_ID] [--batch-size BATCH_SIZE]
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import argparse
from tqdm import tqdm
import psycopg2
from psycopg2.extras import execute_values

# Load environment variables
load_dotenv()

# Initialize the mpnet model (same as rag_pipeline.py)
_sentence_model = None

def _get_sentence_model():
    """Get or initialize the sentence transformer model."""
    global _sentence_model
    if _sentence_model is None:
        print("Loading sentence transformer model: all-mpnet-base-v2...")
        _sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        print("Model loaded successfully!")
    return _sentence_model

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize a numpy array."""
    if x.ndim == 1:
        denom = max(np.linalg.norm(x), eps)
        return (x / denom).astype(np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return (x / denom).astype(np.float32)

def _embed_texts(texts: List[str]) -> np.ndarray:
    """Vectorise a list of texts → (n, 768) numpy array."""
    model = _get_sentence_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    return _l2_normalize(embeddings)

def migrate_chunks(
    user_id: Optional[str] = None,
    notebook_id: Optional[str] = None,
    batch_size: int = 50,
    dry_run: bool = False
):
    """Migrate all chunks to use mpnet embeddings."""
    
    # Initialize Supabase client (for reading)
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    db_url = os.getenv("SUPABASE_DB_URL")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY environment variables are required")
        sys.exit(1)
    
    if not db_url:
        print("⚠️  Warning: SUPABASE_DB_URL not set. Using Supabase REST API (may not work for vectors).")
        print("   For best results, set SUPABASE_DB_URL in your .env file")
        use_db = False
    else:
        use_db = True
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Build query
    query = supabase.table("chunks").select("id, content, embedding, user_id, notebook_id")
    
    if user_id:
        query = query.eq("user_id", user_id)
    
    if notebook_id:
        query = query.eq("notebook_id", notebook_id)
    
    # Get all chunks
    print("Fetching chunks from database...")
    result = query.execute()
    chunks = result.data
    
    if not chunks:
        print("No chunks found to migrate.")
        return
    
    print(f"Found {len(chunks)} chunks to migrate.")
    
    # Check existing embeddings to see how many need migration
    needs_migration = 0
    already_migrated = 0
    no_embedding = 0
    
    for chunk in chunks:
        embedding = chunk.get("embedding")
        if embedding is None:
            no_embedding += 1
        elif isinstance(embedding, list):
            if len(embedding) != 768:
                needs_migration += 1
            else:
                already_migrated += 1
        else:
            needs_migration += 1
    
    print(f"\nMigration status:")
    print(f"  - Needs migration: {needs_migration}")
    print(f"  - Already migrated (768-dim): {already_migrated}")
    print(f"  - No embedding: {no_embedding}")
    
    if needs_migration == 0 and no_embedding == 0:
        print("\nAll chunks are already migrated!")
        return
    
    if dry_run:
        print("\n[DRY RUN] Would migrate chunks, but dry-run mode is enabled.")
        return
    
    # Process in batches
    print(f"\nMigrating chunks in batches of {batch_size}...")
    
    total_updated = 0
    total_errors = 0
    
    # Filter chunks that need migration
    chunks_to_migrate = []
    for chunk in chunks:
        embedding = chunk.get("embedding")
        needs_update = False
        
        if embedding is None:
            needs_update = True
        elif isinstance(embedding, list):
            if len(embedding) != 768:
                needs_update = True
        else:
            needs_update = True
        
        if needs_update:
            chunks_to_migrate.append(chunk)
    
    print(f"Migrating {len(chunks_to_migrate)} chunks...")
    
    for i in tqdm(range(0, len(chunks_to_migrate), batch_size), desc="Processing batches"):
        batch = chunks_to_migrate[i:i + batch_size]
        
        # Extract content
        contents = [chunk["content"] for chunk in batch]
        chunk_ids = [chunk["id"] for chunk in batch]
        
        try:
            # Re-embed batch
            embeddings = _embed_texts(contents)
            
            if use_db:
                # Use direct PostgreSQL connection for vector updates
                try:
                    with psycopg2.connect(db_url) as conn:
                        with conn.cursor() as cur:
                            # Update embeddings using PostgreSQL vector type
                            for j, chunk_id in enumerate(chunk_ids):
                                try:
                                    # Convert numpy array to list and then to PostgreSQL vector format
                                    embedding_list = embeddings[j].tolist()
                                    # PostgreSQL vector type accepts array format: [1,2,3,...]
                                    embedding_str = '[' + ','.join([str(x) for x in embedding_list]) + ']'
                                    
                                    cur.execute(
                                        "UPDATE chunks SET embedding = %s::vector(768) WHERE id = %s",
                                        (embedding_str, chunk_id)
                                    )
                                    total_updated += 1
                                except Exception as e:
                                    print(f"\nError updating chunk {chunk_id}: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    total_errors += 1
                            conn.commit()
                except Exception as e:
                    print(f"\nError connecting to database: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Falling back to Supabase REST API...")
                    use_db = False  # Fall back to REST API
            
            if not use_db:
                # Fallback: Use Supabase REST API (may not work properly for vectors)
                for j, chunk_id in enumerate(chunk_ids):
                    try:
                        update_data = {
                            "embedding": embeddings[j].tolist()
                        }
                        
                        supabase.table("chunks").update(update_data).eq("id", chunk_id).execute()
                        total_updated += 1
                    except Exception as e:
                        print(f"\nError updating chunk {chunk_id}: {e}")
                        total_errors += 1
        
        except Exception as e:
            print(f"\nError processing batch {i//batch_size + 1}: {e}")
            total_errors += len(batch)
    
    print(f"\n✅ Migration complete!")
    print(f"  - Updated: {total_updated} chunks")
    print(f"  - Errors: {total_errors} chunks")
    
    if total_errors > 0:
        print(f"\n⚠️  Some chunks failed to update. Please check the errors above.")
    
    # Verify migration
    print("\nVerifying migration...")
    verify_result = supabase.table("chunks").select("id, embedding").limit(10).execute()
    verified_768 = 0
    verified_wrong = 0
    
    for chunk in verify_result.data:
        embedding = chunk.get("embedding")
        if isinstance(embedding, list) and len(embedding) == 768:
            verified_768 += 1
        else:
            verified_wrong += 1
    
    if verified_768 > 0:
        print(f"✅ Verification: {verified_768} sample chunks have correct 768-dim embeddings")
    if verified_wrong > 0:
        print(f"⚠️  Verification: {verified_wrong} sample chunks still have wrong dimensions")
        print("   You may need to re-run the migration script.")

def main():
    parser = argparse.ArgumentParser(description="Migrate chunk embeddings to mpnet model")
    parser.add_argument("--user-id", help="Filter by user ID")
    parser.add_argument("--notebook-id", help="Filter by notebook ID")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing (default: 50)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode (don't actually update)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Chunk Embedding Migration Script")
    print("Migrating to: all-mpnet-base-v2 (768 dimensions)")
    print("=" * 60)
    print()
    
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No changes will be made")
        print()
    
    migrate_chunks(
        user_id=args.user_id,
        notebook_id=args.notebook_id,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()









