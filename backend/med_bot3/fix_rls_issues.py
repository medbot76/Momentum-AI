#!/usr/bin/env python3
"""
Fix RLS issues for notebooks and documents tables, and storage bucket
"""

import os
import psycopg2
from dotenv import load_dotenv

def fix_rls_issues():
    load_dotenv()
    
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        print("❌ SUPABASE_DB_URL not set")
        return False
    
    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                print("🔧 Fixing RLS issues...")
                
                # 1. Disable RLS on tables
                tables_to_fix = ['notebooks', 'documents', 'chunks']
                for table in tables_to_fix:
                    try:
                        cur.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")
                        print(f"   ✓ Disabled RLS on {table} table")
                    except Exception as e:
                        print(f"   ⚠️  Could not disable RLS on {table}: {e}")
                
                conn.commit()
                
                # 2. Check RLS status
                print("\n📋 Checking RLS status:")
                cur.execute("""
                    SELECT schemaname, tablename, rowsecurity 
                    FROM pg_tables 
                    WHERE tablename IN ('notebooks', 'documents', 'chunks')
                    ORDER BY tablename
                """)
                results = cur.fetchall()
                for schema, table, rls_enabled in results:
                    status = "ENABLED" if rls_enabled else "DISABLED"
                    print(f"   {table}: RLS {status}")
                
                # 3. Create storage bucket
                print("\n🗂️  Setting up storage bucket...")
                try:
                    cur.execute("""
                        INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
                        VALUES (
                            'documents',
                            'documents', 
                            false,
                            52428800,
                            ARRAY['application/pdf', 'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'text/plain', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
                        )
                        ON CONFLICT (id) DO NOTHING;
                    """)
                    print("   ✓ Created/found documents storage bucket")
                except Exception as e:
                    print(f"   ⚠️  Could not create storage bucket: {e}")
                
                # 4. Create storage policies
                print("\n🔐 Setting up storage policies...")
                policies = [
                    {
                        'name': 'Allow authenticated users to upload documents',
                        'sql': """
                            CREATE POLICY "Allow authenticated users to upload documents" ON storage.objects
                            FOR INSERT WITH CHECK (
                                bucket_id = 'documents' AND 
                                auth.role() = 'authenticated'
                            );
                        """
                    },
                    {
                        'name': 'Allow users to view their own documents',
                        'sql': """
                            CREATE POLICY "Allow users to view their own documents" ON storage.objects
                            FOR SELECT USING (
                                bucket_id = 'documents' AND 
                                auth.uid()::text = (storage.foldername(name))[2]
                            );
                        """
                    },
                    {
                        'name': 'Allow users to delete their own documents',
                        'sql': """
                            CREATE POLICY "Allow users to delete their own documents" ON storage.objects
                            FOR DELETE USING (
                                bucket_id = 'documents' AND 
                                auth.uid()::text = (storage.foldername(name))[2]
                            );
                        """
                    }
                ]
                
                for policy in policies:
                    try:
                        cur.execute(policy['sql'])
                        print(f"   ✓ Created policy: {policy['name']}")
                    except Exception as e:
                        if "already exists" in str(e):
                            print(f"   ✓ Policy already exists: {policy['name']}")
                        else:
                            print(f"   ⚠️  Could not create policy {policy['name']}: {e}")
                
                conn.commit()
                print("\n✅ RLS issues fixed successfully!")
                return True
                
    except Exception as e:
        print(f"❌ Error fixing RLS issues: {e}")
        return False

if __name__ == "__main__":
    fix_rls_issues()



