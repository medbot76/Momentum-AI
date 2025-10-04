#!/usr/bin/env python3
"""
Check current storage RLS policies
"""

import os
from supabase import create_client, Client

def check_storage_policies():
    """Check current storage RLS policies"""
    try:
        # Get Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            print("❌ SUPABASE_URL and SUPABASE_KEY must be set")
            return False
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        print("🔍 Checking current storage RLS policies...")
        
        # Query to check current policies
        check_sql = """
        SELECT 
            schemaname,
            tablename,
            policyname,
            permissive,
            roles,
            cmd,
            qual,
            with_check
        FROM pg_policies 
        WHERE tablename = 'objects' AND schemaname = 'storage'
        ORDER BY policyname;
        """
        
        print("📝 Run this SQL in your Supabase SQL editor to see current policies:")
        print("\n" + "="*80)
        print(check_sql)
        print("="*80)
        
        print("\n🔧 If you haven't run the fix yet, here's the SQL to fix the storage policy:")
        print("\n" + "="*80)
        print("DROP POLICY IF EXISTS \"Allow authenticated users to upload documents\" ON storage.objects;")
        print("")
        print("CREATE POLICY \"Allow authenticated users to upload documents\" ON storage.objects")
        print("FOR INSERT WITH CHECK (")
        print("    bucket_id = 'documents' AND ")
        print("    (auth.role() = 'authenticated' OR auth.role() = 'service_role')")
        print(");")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking storage policies: {e}")
        return False

if __name__ == "__main__":
    check_storage_policies()
