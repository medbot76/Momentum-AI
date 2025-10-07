#!/usr/bin/env python3
"""
Create chunks table using Supabase client instead of direct DB connection
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client

def create_chunks_table_via_supabase():
    """Create chunks table using Supabase client"""
    load_dotenv()
    
    print("🔧 Creating chunks table via Supabase client...")
    
    try:
        # Use Supabase client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            print("❌ SUPABASE_URL and SUPABASE_KEY must be set in .env file")
            return False
            
        supabase: Client = create_client(url, key)
        
        # Test if chunks table exists by trying to query it
        try:
            result = supabase.table('chunks').select('*').limit(1).execute()
            print("✅ chunks table already exists")
            return True
        except Exception as e:
            if "relation" in str(e) and "does not exist" in str(e):
                print("❌ chunks table does not exist")
                print("\n📝 Please run the following SQL in your Supabase SQL Editor:")
                print("=" * 60)
                print("""
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    notebook_id UUID,
    document_id UUID,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    tokens INTEGER DEFAULT 0,
    chunk_index INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_user_notebook_idx 
ON chunks (user_id, notebook_id);

-- Enable RLS
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

-- Create RLS policy
CREATE POLICY IF NOT EXISTS "Users can manage their chunks"
ON chunks
FOR ALL
USING (auth.uid() = user_id OR user_id IS NULL);
                """)
                print("=" * 60)
                print("\n📋 Instructions:")
                print("1. Go to your Supabase dashboard")
                print("2. Navigate to SQL Editor")
                print("3. Copy and paste the SQL above")
                print("4. Click 'Run' to execute")
                print("5. Come back and test the chatbot")
                return False
            else:
                print(f"❌ Error checking chunks table: {str(e)}")
                return False
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    create_chunks_table_via_supabase()
