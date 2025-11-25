"""
Verification script to check if embeddings are correctly migrated
----------------------------------------------------------------
This script checks:
1. Embedding dimensions (should be 768)
2. Embedding format (should be lists, not strings)
3. Sample similarity calculations

Usage:
    python verify_embeddings.py [--user-id USER_ID] [--notebook-id NOTEBOOK_ID]
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
import numpy as np
import argparse
from sentence_transformers import SentenceTransformer

load_dotenv()

def _get_sentence_model():
    """Get or initialize the sentence transformer model."""
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _sentence_model

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize a numpy array."""
    if x.ndim == 1:
        denom = max(np.linalg.norm(x), eps)
        return (x / denom).astype(np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return (x / denom).astype(np.float32)

def _embed_text(text: str) -> list:
    """Return a 768-dim embedding for text."""
    model = _get_sentence_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
    embedding = _l2_normalize(embedding)
    return embedding.tolist()

def verify_embeddings(user_id: str = None, notebook_id: str = None):
    """Verify embeddings are correctly migrated."""
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY environment variables are required")
        sys.exit(1)
    
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Build query
    query = supabase.table("chunks").select("id, content, embedding, user_id, notebook_id")
    
    if user_id:
        query = query.eq("user_id", user_id)
    
    if notebook_id:
        query = query.eq("notebook_id", notebook_id)
    
    # Get sample of chunks
    print("Fetching chunks from database...")
    result = query.limit(100).execute()  # Sample first 100
    chunks = result.data
    
    if not chunks:
        print("No chunks found.")
        return
    
    print(f"Checking {len(chunks)} chunks...\n")
    
    # Statistics
    correct_dim = 0
    wrong_dim = 0
    string_format = 0
    list_format = 0
    no_embedding = 0
    dimensions_found = set()
    
    # Sample embeddings for similarity test
    sample_embeddings = []
    sample_texts = []
    
    for chunk in chunks:
        embedding = chunk.get("embedding")
        
        if embedding is None:
            no_embedding += 1
        elif isinstance(embedding, str):
            string_format += 1
            # Try to parse
            try:
                import json
                import ast
                try:
                    parsed = json.loads(embedding)
                except:
                    parsed = ast.literal_eval(embedding)
                
                dim = len(parsed) if isinstance(parsed, list) else 0
                dimensions_found.add(dim)
                if dim == 768:
                    correct_dim += 1
                    sample_embeddings.append(parsed)
                    sample_texts.append(chunk["content"])
                else:
                    wrong_dim += 1
            except:
                wrong_dim += 1
        elif isinstance(embedding, list):
            list_format += 1
            dim = len(embedding)
            dimensions_found.add(dim)
            if dim == 768:
                correct_dim += 1
                sample_embeddings.append(embedding)
                sample_texts.append(chunk["content"])
            else:
                wrong_dim += 1
        else:
            wrong_dim += 1
    
    print("=" * 60)
    print("Embedding Verification Results")
    print("=" * 60)
    print(f"Total chunks checked: {len(chunks)}")
    print(f"  ✓ Correct dimension (768): {correct_dim}")
    print(f"  ✗ Wrong dimension: {wrong_dim}")
    print(f"  ✗ No embedding: {no_embedding}")
    print(f"  - String format: {string_format}")
    print(f"  - List format: {list_format}")
    print(f"  - Dimensions found: {sorted(dimensions_found)}")
    print()
    
    # Test similarity calculation
    if sample_embeddings and len(sample_embeddings) >= 2:
        print("Testing similarity calculation...")
        test_query = "test query"
        q_embed = np.array(_embed_text(test_query), dtype=np.float32)
        q_norm = np.linalg.norm(q_embed)
        if q_norm > 1e-8:
            q_embed = q_embed / q_norm
        
        similarities = []
        for i, doc_embed in enumerate(sample_embeddings[:10]):  # Test first 10
            doc_embed_np = np.array(doc_embed, dtype=np.float32)
            doc_norm = np.linalg.norm(doc_embed_np)
            if doc_norm > 1e-8:
                doc_embed_np = doc_embed_np / doc_norm
            
            similarity = float(np.dot(q_embed, doc_embed_np))
            similarities.append(similarity)
        
        if similarities:
            sim_array = np.array(similarities)
            print(f"  Sample similarities: min={sim_array.min():.3f}, max={sim_array.max():.3f}, mean={sim_array.mean():.3f}")
            print(f"  (Expected range for random text: -0.2 to 0.2)")
            print(f"  (Expected range for relevant content: 0.3 to 0.9)")
    
    print()
    if wrong_dim > 0 or no_embedding > 0:
        print("⚠️  Some chunks need migration!")
        print(f"   Run: python migrate_embeddings_to_mpnet.py")
        if user_id:
            print(f"   With: --user-id {user_id}")
        if notebook_id:
            print(f"   With: --notebook-id {notebook_id}")
    else:
        print("✅ All checked chunks have correct embeddings!")

def main():
    parser = argparse.ArgumentParser(description="Verify chunk embeddings")
    parser.add_argument("--user-id", help="Filter by user ID")
    parser.add_argument("--notebook-id", help="Filter by notebook ID")
    
    args = parser.parse_args()
    
    verify_embeddings(user_id=args.user_id, notebook_id=args.notebook_id)

if __name__ == "__main__":
    main()







