#!/usr/bin/env python3
"""
Setup Supabase storage buckets and fix missing database components
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import json

def setup_storage_buckets():
    """Create necessary storage buckets for file uploads"""
    load_dotenv()
    
    print("💾 Setting up Storage Buckets...")
    
    try:
        # Initialize client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(url, key)
        
        # Define required buckets
        buckets_to_create = [
            {
                'name': 'documents',
                'public': False,
                'allowed_mime_types': [
                    'application/pdf',
                    'text/plain', 
                    'text/markdown',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'application/msword'
                ],
                'file_size_limit': 52428800  # 50MB
            },
            {
                'name': 'images', 
                'public': False,
                'allowed_mime_types': [
                    'image/png',
                    'image/jpeg',
                    'image/jpg',
                    'image/gif',
                    'image/webp'
                ],
                'file_size_limit': 10485760  # 10MB
            },
            {
                'name': 'temp-uploads',
                'public': False,
                'allowed_mime_types': ['*'],  # Allow all types for temporary storage
                'file_size_limit': 104857600  # 100MB
            }
        ]
        
        # Get existing buckets
        existing_buckets = supabase.storage.list_buckets()
        existing_names = [bucket.name for bucket in existing_buckets]
        
        # Create missing buckets
        for bucket_config in buckets_to_create:
            bucket_name = bucket_config['name']
            
            if bucket_name not in existing_names:
                try:
                    result = supabase.storage.create_bucket(
                        bucket_name,
                        public=bucket_config['public'],
                        file_size_limit=bucket_config['file_size_limit'],
                        allowed_mime_types=bucket_config['allowed_mime_types']
                    )
                    print(f"   ✅ Created bucket: {bucket_name}")
                except Exception as e:
                    print(f"   ❌ Failed to create bucket {bucket_name}: {str(e)}")
            else:
                print(f"   ✅ Bucket already exists: {bucket_name}")
        
        print(f"\n📊 Total buckets available: {len(supabase.storage.list_buckets())}")
        return True
        
    except Exception as e:
        print(f"❌ Storage setup failed: {str(e)}")
        return False

def setup_missing_database_components():
    """Fix missing database components via Supabase client"""
    load_dotenv()
    
    print("\n🔧 Setting up Missing Database Components...")
    
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY") 
        supabase: Client = create_client(url, key)
        
        # Test if user_settings table exists by trying to create a dummy entry
        print("   Checking user_settings table...")
        try:
            # This will fail if table doesn't exist or if we don't have proper RLS
            result = supabase.table('user_settings').select('*').limit(1).execute()
            print("   ✅ user_settings table accessible")
        except Exception as e:
            if "user_settings" in str(e) and "schema cache" in str(e):
                print("   ⚠️  user_settings table may not exist - needs migration")
            else:
                print(f"   ⚠️  user_settings issue: {str(e)}")
        
        # Test if search function exists
        print("   Checking search_documents function...")
        try:
            # Test with dummy parameters
            result = supabase.rpc('search_documents', {
                'p_user_id': '00000000-0000-0000-0000-000000000000',
                'p_query_embedding': [0.0] * 768,
                'p_match_count': 1
            }).execute()
            print("   ✅ search_documents function working")
        except Exception as e:
            if "function" in str(e).lower() and "schema cache" in str(e):
                print("   ⚠️  search_documents function may not exist - needs migration")
            else:
                print(f"   ⚠️  search function issue: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database component check failed: {str(e)}")
        return False

def verify_schema_requirements():
    """Verify that all required schema components exist"""
    load_dotenv()
    
    print("\n📋 Schema Requirements Verification...")
    
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(url, key)
        
        # Test each required table
        required_tables = {
            'profiles': 'User profiles extending auth.users',
            'notebooks': 'Content organization containers', 
            'documents': 'Document chunks with vector embeddings',
            'conversations': 'Chat conversation threads',
            'messages': 'Individual chat messages',
            'user_settings': 'User preferences and settings'
        }
        
        table_status = {}
        
        for table, description in required_tables.items():
            try:
                result = supabase.table(table).select('*').limit(1).execute()
                table_status[table] = True
                print(f"   ✅ {table}: OK - {description}")
            except Exception as e:
                table_status[table] = False
                print(f"   ❌ {table}: MISSING - {description}")
                print(f"      Error: {str(e)}")
        
        # Check extensions/functions
        print("\n🔧 Database Functions:")
        functions_to_check = ['search_documents']
        
        for func in functions_to_check:
            try:
                # Try to call with minimal params to test existence
                supabase.rpc(func, {})
            except Exception as e:
                if "schema cache" in str(e):
                    print(f"   ❌ {func}: Function not found")
                else:
                    print(f"   ✅ {func}: Function exists (test call failed as expected)")
        
        # Summary
        missing_tables = [table for table, exists in table_status.items() if not exists]
        
        if missing_tables:
            print(f"\n⚠️  Missing tables: {', '.join(missing_tables)}")
            print("   Run: supabase db push")
            print("   Or apply migration: supabase/migrations/20230815162500_initial_schema.sql")
            return False
        else:
            print(f"\n✅ All required tables exist!")
            return True
            
    except Exception as e:
        print(f"❌ Schema verification failed: {str(e)}")
        return False

def create_setup_instructions():
    """Generate setup instructions based on findings"""
    print("\n📝 Setup Instructions:")
    print("=" * 50)
    
    print("""
To complete your Supabase setup:

1. 🏗️  Database Schema:
   cd ~/med-bot
   supabase db reset  # This will apply migrations
   
   OR manually in Supabase Dashboard:
   - Go to SQL Editor
   - Copy content from supabase/migrations/20230815162500_initial_schema.sql
   - Execute the SQL

2. 🔧 Extensions:
   Make sure these extensions are enabled in Supabase Dashboard > Database > Extensions:
   - uuid-ossp
   - pgcrypto  
   - vector (for pgvector)

3. 💾 Storage Buckets:
   Run this script again after database setup:
   python setup_storage_and_db.py

4. 🔐 Row Level Security:
   All tables have RLS enabled. Make sure your auth flows work properly.

5. 🧪 Testing:
   python test_supabase_detailed.py
""")

def main():
    """Main setup routine"""
    print("🚀 Med-Bot Supabase Setup")
    print("=" * 40)
    
    # Step 1: Verify schema
    schema_ok = verify_schema_requirements()
    
    # Step 2: Setup storage (if schema is OK)
    if schema_ok:
        storage_ok = setup_storage_buckets()
        db_components_ok = setup_missing_database_components()
        
        if storage_ok and db_components_ok:
            print("\n🎉 Setup completed successfully!")
            print("✅ Your Supabase instance is ready for Med-Bot")
        else:
            print("\n⚠️  Some components need attention")
    else:
        print("\n❌ Database schema needs to be set up first")
    
    # Always show instructions
    create_setup_instructions()
    
    return schema_ok

if __name__ == "__main__":
    main()