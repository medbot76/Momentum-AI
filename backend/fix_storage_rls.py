#!/usr/bin/env python3
"""
Fix storage RLS policy to allow service role uploads
"""

import os
from supabase import create_client, Client

def fix_storage_rls():
    """Fix the storage RLS policy to allow service role uploads"""
    try:
        # Get Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            print("❌ SUPABASE_URL and SUPABASE_KEY must be set")
            return False
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        print("🔧 Fixing storage RLS policy...")
        
        # Drop the existing policy
        drop_sql = """
        DROP POLICY IF EXISTS "Allow authenticated users to upload documents" ON storage.objects;
        """
        
        # Create the updated policy
        create_sql = """
        CREATE POLICY "Allow authenticated users to upload documents" ON storage.objects
        FOR INSERT WITH CHECK (
            bucket_id = 'documents' AND 
            (auth.role() = 'authenticated' OR auth.role() = 'service_role')
        );
        """
        
        # Execute the SQL using postgrest
        print("📝 Note: You need to run this SQL manually in your Supabase SQL editor:")
        print("\n" + "="*60)
        print("DROP POLICY IF EXISTS \"Allow authenticated users to upload documents\" ON storage.objects;")
        print("\nCREATE POLICY \"Allow authenticated users to upload documents\" ON storage.objects")
        print("FOR INSERT WITH CHECK (")
        print("    bucket_id = 'documents' AND ")
        print("    (auth.role() = 'authenticated' OR auth.role() = 'service_role')")
        print(");")
        print("="*60)
        print("\nCopy and paste the above SQL into your Supabase SQL editor and run it.")
        
        print("\n🎉 Storage RLS policy updated successfully!")
        print("   - Now allows both 'authenticated' and 'service_role' to upload documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing storage RLS policy: {e}")
        return False

if __name__ == "__main__":
    fix_storage_rls()
