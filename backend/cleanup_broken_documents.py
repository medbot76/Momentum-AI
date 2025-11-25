#!/usr/bin/env python3
"""
Script to clean up broken document records (documents without chunks and missing files)
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

def cleanup_broken_documents(supabase: Client, notebook_id: str):
    """Clean up documents that have no chunks"""
    try:
        # Get all documents in the notebook
        documents = supabase.table('documents').select('*').eq('notebook_id', notebook_id).execute()
        
        broken_documents = []
        
        print("🔍 Checking documents...")
        for doc in documents.data:
            # Check if document has chunks
            chunks = supabase.table('chunks').select('id').eq('document_id', doc['id']).execute()
            
            if not chunks.data:
                broken_documents.append(doc)
                print(f"❌ {doc['filename']} - 0 chunks")
            else:
                print(f"✅ {doc['filename']} - {len(chunks.data)} chunks")
        
        if not broken_documents:
            print("\n✅ No broken documents found!")
            return
        
        print(f"\n🗑️  Found {len(broken_documents)} broken documents")
        
        # Ask for confirmation
        response = input(f"\n⚠️  Delete {len(broken_documents)} broken document records? (y/N): ")
        
        if response.lower() != 'y':
            print("❌ Cancelled")
            return
        
        # Delete broken documents
        deleted_count = 0
        for doc in broken_documents:
            try:
                # Delete the document record
                result = supabase.table('documents').delete().eq('id', doc['id']).execute()
                if result.data:
                    print(f"✅ Deleted: {doc['filename']}")
                    deleted_count += 1
                else:
                    print(f"❌ Failed to delete: {doc['filename']}")
            except Exception as e:
                print(f"❌ Error deleting {doc['filename']}: {e}")
        
        print(f"\n📊 Results:")
        print(f"✅ Successfully deleted: {deleted_count}")
        print(f"❌ Failed to delete: {len(broken_documents) - deleted_count}")
        
        if deleted_count > 0:
            print(f"\n🎉 Cleaned up {deleted_count} broken document records!")
            print("💡 You can now re-upload your files through the frontend.")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("🧹 Broken Documents Cleanup Script")
    print("=" * 50)
    
    # Get Supabase client
    supabase = get_supabase_client()
    
    # Get notebook ID from environment or use default
    notebook_id = os.getenv("NOTEBOOK_ID", "e802764b-7692-40a5-a29b-1acc89ec08d8")
    print(f"📚 Cleaning notebook: {notebook_id}")
    
    # Clean up broken documents
    cleanup_broken_documents(supabase, notebook_id)

if __name__ == "__main__":
    main()
