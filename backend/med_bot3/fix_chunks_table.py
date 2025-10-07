#!/usr/bin/env python3
"""
Fix the chunks table to match the RAG pipeline requirements
"""
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

def fix_chunks_table():
    """Create the chunks table with correct schema for the RAG pipeline"""
    load_dotenv()
    
    print("🔧 Fixing chunks table for RAG pipeline...")
    
    try:
        # Connect to database
        db_url = os.getenv("SUPABASE_DB_URL")
        if not db_url:
            print("❌ SUPABASE_DB_URL not set. Please set it in your .env file")
            return False
            
        # Check if the password is still a placeholder
        if "your-password" in db_url:
            print("❌ SUPABASE_DB_URL contains placeholder password 'your-password'")
            print("Please update the .env file with the actual database password")
            print("You can find it in your Supabase dashboard under Settings > Database")
            return False
            
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                print("✅ pgvector extension enabled")
                
                # Create chunks table if it doesn't exist (matching RAG pipeline schema)
                cur.execute("""
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
                """)
                print("✅ chunks table created/verified")
                
                # Create indexes for better performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
                    ON chunks USING hnsw (embedding vector_cosine_ops);
                """)
                print("✅ embedding index created")
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS chunks_user_notebook_idx 
                    ON chunks (user_id, notebook_id);
                """)
                print("✅ user/notebook index created")
                
                # Enable RLS
                cur.execute("ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;")
                print("✅ RLS enabled")
                
                # Create RLS policy
                cur.execute("""
                    CREATE POLICY IF NOT EXISTS "Users can manage their chunks"
                    ON chunks
                    FOR ALL
                    USING (auth.uid() = user_id OR user_id IS NULL);
                """)
                print("✅ RLS policy created")
                
                conn.commit()
                print("\n🎉 chunks table setup completed successfully!")
                return True
                
    except Exception as e:
        print(f"❌ Error setting up chunks table: {str(e)}")
        return False

if __name__ == "__main__":
    fix_chunks_table()
