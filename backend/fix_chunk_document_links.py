#!/usr/bin/env python3
"""
Script to fix chunks that have NULL document_id by linking them to documents
"""

import os
import sys
from supabase import create_client, Client

def get_supabase_client():
    """Get Supabase client"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: Supabase credentials not found")
        sys.exit(1)
    
    return create_client(supabase_url, supabase_key)

def fix_chunk_document_links(supabase: Client, notebook_id: str):
    """Fix chunks that have NULL document_id"""
    try:
        print("🔍 Checking chunks with NULL document_id...")
        
        # Get all chunks with NULL document_id in this notebook
        chunks = supabase.table('chunks').select('*').eq('notebook_id', notebook_id).is_('document_id', 'null').execute()
        
        if not chunks.data:
            print("✅ No chunks with NULL document_id found!")
            return
        
        print(f"❌ Found {len(chunks.data)} chunks with NULL document_id")
        
        # Get all documents in this notebook
        documents = supabase.table('documents').select('*').eq('notebook_id', notebook_id).execute()
        
        if not documents.data:
            print("❌ No documents found in this notebook!")
            return
        
        print(f"📚 Found {len(documents.data)} documents in notebook")
        
        # Group chunks by user_id to match with documents
        chunks_by_user = {}
        for chunk in chunks.data:
            user_id = chunk.get('user_id')
            if user_id not in chunks_by_user:
                chunks_by_user[user_id] = []
            chunks_by_user[user_id].append(chunk)
        
        # For each user, try to link their chunks to documents
        total_updated = 0
        
        for user_id, user_chunks in chunks_by_user.items():
            print(f"\n👤 Processing user: {user_id}")
            print(f"   Chunks: {len(user_chunks)}")
            
            # Get documents for this user
            user_documents = [doc for doc in documents.data if doc.get('user_id') == user_id]
            print(f"   Documents: {len(user_documents)}")
            
            if not user_documents:
                print(f"   ⚠️  No documents found for user {user_id}")
                continue
            
            # For now, link all chunks to the most recent document
            # This is a simple heuristic - in practice, you might want more sophisticated matching
            latest_doc = max(user_documents, key=lambda x: x.get('created_at', ''))
            print(f"   📄 Linking to document: {latest_doc['filename']}")
            
            # Update chunks for this user
            chunk_ids = [chunk['id'] for chunk in user_chunks]
            
            try:
                result = supabase.table('chunks').update({
                    'document_id': latest_doc['id']
                }).in_('id', chunk_ids).execute()
                
                updated_count = len(result.data) if result.data else 0
                print(f"   ✅ Updated {updated_count} chunks")
                total_updated += updated_count
                
            except Exception as e:
                print(f"   ❌ Error updating chunks for user {user_id}: {e}")
        
        print(f"\n📊 Results:")
        print(f"✅ Total chunks updated: {total_updated}")
        
        if total_updated > 0:
            print(f"\n🎉 Fixed {total_updated} chunk-document links!")
            print("💡 You can now try generating flashcards again.")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("🔗 Chunk-Document Link Fix Script")
    print("=" * 50)
    
    # Get Supabase client
    supabase = get_supabase_client()
    
    # Get notebook ID from environment or use default
    notebook_id = os.getenv("NOTEBOOK_ID", "e802764b-7692-40a5-a29b-1acc89ec08d8")
    print(f"📚 Fixing notebook: {notebook_id}")
    
    # Fix chunk-document links
    fix_chunk_document_links(supabase, notebook_id)

if __name__ == "__main__":
    main()
