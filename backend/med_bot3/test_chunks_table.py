#!/usr/bin/env python3
"""
Test chunks table access and structure
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client

def test_chunks_table():
    """Test if we can access the chunks table"""
    load_dotenv()
    
    print("🔍 Testing chunks table access...")
    
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            print("❌ SUPABASE_URL and SUPABASE_KEY must be set")
            return False
            
        supabase: Client = create_client(url, key)
        
        # Try to query the chunks table
        try:
            result = supabase.table('chunks').select('*').limit(1).execute()
            print("✅ Successfully queried chunks table")
            print(f"   Found {len(result.data)} rows")
            return True
        except Exception as e:
            print(f"❌ Error querying chunks table: {str(e)}")
            
            # Check if it's a permissions issue
            if "permission denied" in str(e).lower():
                print("   This might be a Row Level Security (RLS) issue")
                print("   Try running this SQL in Supabase SQL Editor:")
                print("   ALTER TABLE chunks DISABLE ROW LEVEL SECURITY;")
                print("   Then re-enable it after testing:")
                print("   ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;")
            elif "relation" in str(e) and "does not exist" in str(e):
                print("   The chunks table doesn't exist or isn't accessible")
            else:
                print(f"   Unknown error: {str(e)}")
            return False
                
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False

if __name__ == "__main__":
    test_chunks_table()
