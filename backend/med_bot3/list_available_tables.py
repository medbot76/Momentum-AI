#!/usr/bin/env python3
"""
List all available tables in Supabase
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client

def list_available_tables():
    """List all tables we can access"""
    load_dotenv()
    
    print("🔍 Listing available tables...")
    
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            print("❌ SUPABASE_URL and SUPABASE_KEY must be set")
            return False
            
        supabase: Client = create_client(url, key)
        
        # Try to access each table mentioned in the schema
        tables_to_test = [
            'chunks', 'documents', 'notebooks', 'profiles', 
            'conversations', 'messages', 'flashcards', 'exams',
            'study_plans', 'calendar_events'
        ]
        
        accessible_tables = []
        inaccessible_tables = []
        
        for table in tables_to_test:
            try:
                result = supabase.table(table).select('*').limit(1).execute()
                accessible_tables.append(table)
                print(f"✅ {table}: accessible ({len(result.data)} rows)")
            except Exception as e:
                inaccessible_tables.append((table, str(e)))
                print(f"❌ {table}: {str(e)}")
        
        print(f"\n📊 Summary:")
        print(f"   Accessible tables: {len(accessible_tables)}")
        print(f"   Inaccessible tables: {len(inaccessible_tables)}")
        
        if accessible_tables:
            print(f"\n✅ Accessible: {', '.join(accessible_tables)}")
        
        if inaccessible_tables:
            print(f"\n❌ Inaccessible:")
            for table, error in inaccessible_tables:
                print(f"   {table}: {error}")
        
        return len(accessible_tables) > 0
                
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False

if __name__ == "__main__":
    list_available_tables()
