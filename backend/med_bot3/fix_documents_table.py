#!/usr/bin/env python3
"""
Fix the documents table for RAG pipeline compatibility
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import psycopg2

def fix_documents_table():
    """Create or fix the documents table for RAG pipeline"""
    load_dotenv()
    
    print("🔧 Fixing documents table for RAG pipeline...")
    
    # Get environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    db_url = os.getenv("SUPABASE_DB_URL")
    
    if not all([supabase_url, supabase_key, db_url]):
        print("❌ Missing required environment variables:")
        print(f"   SUPABASE_URL: {'✓' if supabase_url else '✗'}")
        print(f"   SUPABASE_KEY: {'✓' if supabase_key else '✗'}")
        print(f"   SUPABASE_DB_URL: {'✓' if db_url else '✗'}")
        return False
    
    try:
        # Method 1: Try direct PostgreSQL connection
        if db_url:
            print("   Using direct PostgreSQL connection...")
            with psycopg2.connect(db_url) as conn:
                with conn.cursor() as cur:
                    # Enable pgvector extension
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    print("   ✓ pgvector extension enabled")
                    
                    # Drop existing documents table if it exists (to avoid conflicts)
                    cur.execute("DROP TABLE IF EXISTS documents CASCADE;")
                    print("   ✓ Dropped existing documents table")
                    
                    # Create the correct documents table for RAG pipeline
                    cur.execute("""
                        CREATE TABLE documents (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            user_id UUID,
                            notebook_id TEXT NOT NULL,
                            chunk_id TEXT NOT NULL,
                            content TEXT NOT NULL,
                            embedding VECTOR(768),
                            metadata JSONB DEFAULT '{}',
                            chunk_type TEXT DEFAULT 'text',
                            tokens INTEGER,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            UNIQUE(chunk_id)
                        );
                    """)
                    print("   ✓ Created documents table")
                    
                    # Create indexes for better performance
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                        ON documents USING hnsw (embedding vector_cosine_ops);
                    """)
                    print("   ✓ Created vector similarity index")
                    
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS documents_user_notebook_idx 
                        ON documents (user_id, notebook_id);
                    """)
                    print("   ✓ Created user/notebook index")
                    
                    # Enable RLS (Row Level Security)
                    cur.execute("ALTER TABLE documents ENABLE ROW LEVEL SECURITY;")
                    print("   ✓ Enabled Row Level Security")
                    
                    # Create RLS policy for user isolation
                    cur.execute("""
                        DROP POLICY IF EXISTS "Users can access their own documents" ON documents;
                        CREATE POLICY "Users can access their own documents"
                            ON documents
                            FOR ALL
                            USING (auth.uid() = user_id OR user_id IS NULL);
                    """)
                    print("   ✓ Created RLS policy")
                    
                    conn.commit()
                    print("   ✅ Documents table setup complete!")
                    return True
                    
        else:
            print("   No direct database URL available, trying Supabase client...")
            # Method 2: Use Supabase client (less reliable for schema changes)
            supabase: Client = create_client(supabase_url, supabase_key)
            
            # This is a simplified approach - direct SQL execution via Supabase
            # Note: This might not work for all schema changes
            print("   ⚠️  Direct PostgreSQL connection recommended for schema changes")
            return False
            
    except Exception as e:
        print(f"   ❌ Error setting up documents table: {str(e)}")
        return False

def test_documents_table():
    """Test if the documents table is working"""
    load_dotenv()
    
    print("\n🧪 Testing documents table...")
    
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Try to query the documents table
        result = supabase.table("documents").select("id").limit(1).execute()
        print("   ✅ Documents table is accessible")
        print(f"   📊 Found {len(result.data)} existing documents")
        return True
        
    except Exception as e:
        print(f"   ❌ Documents table test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Fixing Documents Table for RAG Pipeline")
    print("=" * 50)
    
    success = fix_documents_table()
    
    if success:
        test_documents_table()
        print("\n✅ Documents table setup complete!")
        print("   You can now upload files to your chatbot.")
    else:
        print("\n❌ Documents table setup failed!")
        print("   Please check your environment variables and database connection.")
