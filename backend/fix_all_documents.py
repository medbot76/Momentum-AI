#!/usr/bin/env python3
"""
Script to reprocess all documents that have no chunks
"""

import os
import sys
from supabase import create_client, Client
import requests
import json

def get_supabase_client():
    """Get Supabase client"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: Supabase credentials not found")
        sys.exit(1)
    
    return create_client(supabase_url, supabase_key)

def get_documents_without_chunks(supabase: Client, notebook_id: str):
    """Get all documents that have no chunks"""
    try:
        # Get all documents in the notebook
        documents = supabase.table('documents').select('*').eq('notebook_id', notebook_id).execute()
        
        documents_without_chunks = []
        
        for doc in documents.data:
            # Check if document has chunks
            chunks = supabase.table('chunks').select('id').eq('document_id', doc['id']).execute()
            
            if not chunks.data:
                documents_without_chunks.append(doc)
                print(f"❌ {doc['filename']} - 0 chunks")
            else:
                print(f"✅ {doc['filename']} - {len(chunks.data)} chunks")
        
        return documents_without_chunks
        
    except Exception as e:
        print(f"Error getting documents: {e}")
        return []

def reprocess_document(document_id: str, backend_url: str = "http://localhost:5000"):
    """Reprocess a single document"""
    try:
        response = requests.post(
            f"{backend_url}/api/reprocess-document",
            json={"document_id": document_id},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return True, "Success"
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    print("🔧 Document Reprocessing Script")
    print("=" * 50)
    
    # Get Supabase client
    supabase = get_supabase_client()
    
    # Get notebook ID from environment or use default
    notebook_id = os.getenv("NOTEBOOK_ID", "e802764b-7692-40a5-a29b-1acc89ec08d8")
    print(f"📚 Checking notebook: {notebook_id}")
    
    # Get documents without chunks
    print("\n🔍 Checking documents for chunks...")
    documents_without_chunks = get_documents_without_chunks(supabase, notebook_id)
    
    if not documents_without_chunks:
        print("\n✅ All documents have chunks! No reprocessing needed.")
        return
    
    print(f"\n❌ Found {len(documents_without_chunks)} documents without chunks")
    print("\n🔄 Starting reprocessing...")
    
    # Reprocess each document
    success_count = 0
    for i, doc in enumerate(documents_without_chunks, 1):
        print(f"\n[{i}/{len(documents_without_chunks)}] Reprocessing: {doc['filename']}")
        
        success, message = reprocess_document(doc['id'])
        
        if success:
            print(f"✅ Success: {doc['filename']}")
            success_count += 1
        else:
            print(f"❌ Failed: {doc['filename']} - {message}")
    
    print(f"\n📊 Results:")
    print(f"✅ Successfully reprocessed: {success_count}")
    print(f"❌ Failed to reprocess: {len(documents_without_chunks) - success_count}")
    
    if success_count > 0:
        print(f"\n🎉 {success_count} documents are now ready for flashcards!")

if __name__ == "__main__":
    main()
