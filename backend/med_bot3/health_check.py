#!/usr/bin/env python3
"""
Database health check and maintenance script
"""
import os
import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

def health_check():
    load_dotenv()
    
    print("🏥 Med-Bot Database Health Check")
    print("=" * 35)
    
    try:
        supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
        
        # Check table row counts
        tables = ['profiles', 'notebooks', 'documents', 'conversations', 'messages', 'user_settings']
        total_records = 0
        
        print("📊 Table Statistics:")
        for table in tables:
            try:
                result = supabase.table(table).select('*', count='exact').execute()
                count = result.count if hasattr(result, 'count') else len(result.data)
                total_records += count
                print(f"   {table}: {count:,} records")
            except Exception as e:
                print(f"   {table}: Error - {str(e)}")
        
        print(f"\n📈 Total Records: {total_records:,}")
        
        # Check storage usage
        print("\n💾 Storage Check:")
        try:
            buckets = supabase.storage.list_buckets()
            print(f"   Storage buckets: {len(buckets)}")
            for bucket in buckets:
                print(f"   - {bucket.name} ({'public' if bucket.public else 'private'})")
        except Exception as e:
            print(f"   Storage error: {str(e)}")
        
        print(f"\n✅ Health check completed at {datetime.datetime.now()}")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

if __name__ == "__main__":
    health_check()
