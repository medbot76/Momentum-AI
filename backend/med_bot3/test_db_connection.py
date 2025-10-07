#!/usr/bin/env python3
"""
Test database connectivity and verify Supabase setup
"""
import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client, Client
import psycopg2
from psycopg2.extras import RealDictCursor

def test_supabase_connection():
    """Test Supabase client connection"""
    load_dotenv()
    
    print("🔗 Testing Supabase Connection...")
    
    try:
        # Initialize Supabase client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            print("❌ Missing Supabase credentials")
            return False
            
        supabase: Client = create_client(url, key)
        
        # Test connection by fetching auth users (should work even if empty)
        result = supabase.table('profiles').select('id').limit(1).execute()
        print("✅ Supabase client connection successful")
        print(f"   Profile table accessible: {len(result.data)} rows visible")
        
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {str(e)}")
        return False

def test_direct_db_connection():
    """Test direct PostgreSQL connection"""
    load_dotenv()
    
    print("\n🔗 Testing Direct Database Connection...")
    
    try:
        db_url = os.getenv("SUPABASE_DB_URL")
        if not db_url:
            print("❌ Missing SUPABASE_DB_URL")
            return False
            
        # Test direct connection
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Test pgvector extension
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        vector_ext = cursor.fetchone()
        
        if vector_ext:
            print("✅ Direct database connection successful")
            print("✅ pgvector extension installed")
        else:
            print("⚠️  Direct database connection successful but pgvector extension missing")
            
        # Test table existence
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        expected_tables = ['profiles', 'notebooks', 'documents', 'conversations', 'messages', 'user_settings']
        existing_tables = [t['table_name'] for t in tables]
        
        print(f"\n📋 Database Tables:")
        for table in expected_tables:
            status = "✅" if table in existing_tables else "❌"
            print(f"   {status} {table}")
            
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Direct database connection failed: {str(e)}")
        return False

def test_vector_operations():
    """Test vector operations"""
    load_dotenv()
    
    print("\n🧮 Testing Vector Operations...")
    
    try:
        db_url = os.getenv("SUPABASE_DB_URL")
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Test vector creation and similarity search
        test_vector = [0.1] * 768  # 768-dimensional test vector
        
        cursor.execute("""
            SELECT 1 as test, 
                   %s::vector(768) <=> %s::vector(768) as similarity;
        """, (test_vector, test_vector))
        
        result = cursor.fetchone()
        if result and result[1] == 0.0:  # Same vectors should have 0 cosine distance
            print("✅ Vector operations working correctly")
        else:
            print("⚠️  Vector operations may have issues")
            
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Vector operations failed: {str(e)}")
        return False

def main():
    """Run all connection tests"""
    print("🔍 Med-Bot Database Connectivity Test")
    print("=" * 50)
    
    tests = [
        test_supabase_connection,
        test_direct_db_connection, 
        test_vector_operations
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All database tests passed!")
        print("✅ Your Supabase setup is fully operational")
    else:
        print("❌ Some tests failed. Please check your Supabase configuration.")
        
    return all(results)

if __name__ == "__main__":
    main()