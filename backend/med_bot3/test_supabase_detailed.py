#!/usr/bin/env python3
"""
Detailed Supabase testing focusing on REST API operations
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client

def test_detailed_supabase():
    """Comprehensive Supabase testing via REST API"""
    load_dotenv()
    
    print("🔍 Detailed Supabase Testing")
    print("=" * 40)
    
    try:
        # Initialize client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(url, key)
        
        print("✅ Client initialized successfully")
        
        # Test 1: Check if we can access table schemas
        print("\n📊 Testing Table Access...")
        
        tables_to_test = ['profiles', 'notebooks', 'documents', 'conversations', 'messages', 'user_settings']
        
        for table in tables_to_test:
            try:
                result = supabase.table(table).select('*').limit(1).execute()
                print(f"   ✅ {table}: Accessible ({len(result.data)} rows)")
            except Exception as e:
                print(f"   ❌ {table}: {str(e)}")
        
        # Test 2: Test storage bucket access
        print("\n💾 Testing Storage Access...")
        try:
            buckets = supabase.storage.list_buckets()
            print(f"   ✅ Storage accessible, {len(buckets)} buckets found")
            for bucket in buckets:
                print(f"      - {bucket.name} ({'public' if bucket.public else 'private'})")
        except Exception as e:
            print(f"   ⚠️  Storage access: {str(e)}")
        
        # Test 3: Check authentication
        print("\n🔐 Testing Authentication...")
        try:
            # This will fail but tells us if auth endpoint is working
            auth_result = supabase.auth.get_session()
            print(f"   ✅ Auth endpoint accessible")
        except Exception as e:
            if "Invalid JWT" in str(e) or "session" in str(e).lower():
                print(f"   ✅ Auth endpoint working (no active session expected)")
            else:
                print(f"   ❌ Auth error: {str(e)}")
        
        # Test 4: Test database functions
        print("\n🔧 Testing Database Functions...")
        try:
            # Test if we can call the search function (should fail gracefully)
            result = supabase.rpc('search_documents', {
                'p_user_id': '00000000-0000-0000-0000-000000000000',
                'p_query_embedding': [0.1] * 768,
                'p_match_count': 1
            }).execute()
            print(f"   ✅ RPC functions accessible")
        except Exception as e:
            if "function" in str(e).lower():
                print(f"   ⚠️  Function may not exist yet: {str(e)}")
            else:
                print(f"   ❌ RPC error: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Supabase testing failed: {str(e)}")
        return False

def test_environment_setup():
    """Test environment configuration"""
    load_dotenv()
    
    print("\n🔧 Environment Configuration:")
    
    # Check all required environment variables
    required_vars = {
        'SUPABASE_URL': os.getenv('SUPABASE_URL'),
        'SUPABASE_KEY': os.getenv('SUPABASE_KEY'), 
        'SUPABASE_DB_URL': os.getenv('SUPABASE_DB_URL'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY')
    }
    
    for var, value in required_vars.items():
        if value:
            if 'KEY' in var:
                display_value = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else value
            else:
                display_value = value
            print(f"   ✅ {var}: {display_value}")
        else:
            print(f"   ❌ {var}: Not set")
    
    return all(required_vars.values())

if __name__ == "__main__":
    env_ok = test_environment_setup()
    if env_ok:
        supabase_ok = test_detailed_supabase()
        
        print("\n" + "=" * 40)
        if supabase_ok:
            print("🎉 Supabase is operational via REST API!")
            print("📝 Note: Direct database connection may require network configuration")
        else:
            print("❌ Supabase setup needs attention")
    else:
        print("\n❌ Environment configuration incomplete")