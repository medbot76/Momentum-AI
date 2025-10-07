#!/usr/bin/env python3
"""
Apply missing schema components manually using SQL execution
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client

def apply_missing_schema_components():
    """Apply missing schema components via SQL execution"""
    load_dotenv()
    
    print("🔧 Applying Missing Schema Components...")
    
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(url, key)
        
        # SQL for missing user_settings table
        user_settings_sql = """
        -- Create user_settings table if not exists
        CREATE TABLE IF NOT EXISTS public.user_settings (
            user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
            theme TEXT DEFAULT 'light',
            default_model TEXT DEFAULT 'gemini-pro',
            default_embedding_model TEXT DEFAULT 'text-embedding-004',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Enable RLS
        ALTER TABLE public.user_settings ENABLE ROW LEVEL SECURITY;
        
        -- Create RLS policy
        DROP POLICY IF EXISTS "Users can manage their settings" ON public.user_settings;
        CREATE POLICY "Users can manage their settings"
            ON public.user_settings
            FOR ALL
            USING (auth.uid() = user_id);
            
        -- Create trigger for updated_at
        DROP TRIGGER IF EXISTS update_user_settings_modtime ON public.user_settings;
        CREATE TRIGGER update_user_settings_modtime
            BEFORE UPDATE ON public.user_settings
            FOR EACH ROW EXECUTE FUNCTION update_modified_column();
        """
        
        print("   Applying user_settings table...")
        try:
            result = supabase.rpc('exec_sql', {'sql': user_settings_sql})
            print("   ✅ user_settings table created successfully")
        except Exception as e:
            # Try alternative approach - individual statements
            print(f"   ⚠️  Direct SQL execution not available: {str(e)}")
            print("   📝 Manual action required - see instructions below")
        
        # SQL for search function if missing
        search_function_sql = """
        -- Create search function if not exists
        CREATE OR REPLACE FUNCTION search_documents(
            p_user_id UUID,
            p_notebook_id UUID DEFAULT NULL,
            p_query_embedding VECTOR(768),
            p_match_count INTEGER DEFAULT 5,
            p_min_similarity FLOAT DEFAULT 0.3
        )
        RETURNS TABLE (
            id BIGINT,
            content TEXT,
            metadata JSONB,
            similarity FLOAT
        )
        LANGUAGE SQL
        STABLE
        AS $$
            SELECT
                d.id,
                d.content,
                d.metadata,
                1 - (d.embedding <=> p_query_embedding) as similarity
            FROM public.documents d
            WHERE d.user_id = p_user_id
                AND (p_notebook_id IS NULL OR d.notebook_id = p_notebook_id)
                AND (d.embedding <=> p_query_embedding) <= (1 - p_min_similarity)
            ORDER BY d.embedding <=> p_query_embedding
            LIMIT LEAST(p_match_count, 100);
        $$;
        """
        
        print("   Applying search_documents function...")
        try:
            result = supabase.rpc('exec_sql', {'sql': search_function_sql})
            print("   ✅ search_documents function created successfully")
        except Exception as e:
            print(f"   ⚠️  Function creation may need manual setup: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema fixing failed: {str(e)}")
        return False

def provide_manual_sql_instructions():
    """Provide SQL that can be run manually in Supabase dashboard"""
    
    print("\n📝 Manual SQL Instructions:")
    print("=" * 50)
    print("""
If automatic schema application failed, run this SQL manually in Supabase Dashboard > SQL Editor:

-- 1. Create user_settings table
CREATE TABLE IF NOT EXISTS public.user_settings (
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    theme TEXT DEFAULT 'light',
    default_model TEXT DEFAULT 'gemini-pro', 
    default_embedding_model TEXT DEFAULT 'text-embedding-004',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Enable RLS
ALTER TABLE public.user_settings ENABLE ROW LEVEL SECURITY;

-- 3. Create RLS policy
DROP POLICY IF EXISTS "Users can manage their settings" ON public.user_settings;
CREATE POLICY "Users can manage their settings"
    ON public.user_settings
    FOR ALL
    USING (auth.uid() = user_id);

-- 4. Create trigger for updated_at  
CREATE TRIGGER update_user_settings_modtime
    BEFORE UPDATE ON public.user_settings
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

-- 5. Ensure search function exists
CREATE OR REPLACE FUNCTION search_documents(
    p_user_id UUID,
    p_notebook_id UUID DEFAULT NULL,
    p_query_embedding VECTOR(768),
    p_match_count INTEGER DEFAULT 5,
    p_min_similarity FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE SQL
STABLE
AS $$
    SELECT
        d.id,
        d.content,
        d.metadata,
        1 - (d.embedding <=> p_query_embedding) as similarity
    FROM public.documents d
    WHERE d.user_id = p_user_id
        AND (p_notebook_id IS NULL OR d.notebook_id = p_notebook_id)
        AND (d.embedding <=> p_query_embedding) <= (1 - p_min_similarity)
    ORDER BY d.embedding <=> p_query_embedding
    LIMIT LEAST(p_match_count, 100);
$$;
""")

def main():
    """Main schema fixing routine"""
    print("🔧 Med-Bot Schema Fix")
    print("=" * 30)
    
    # Try automatic fixing
    success = apply_missing_schema_components()
    
    # Always provide manual instructions
    provide_manual_sql_instructions()
    
    if success:
        print("\n✅ Schema components applied!")
        print("Run python setup_storage_and_db.py to continue setup")
    else:
        print("\n⚠️  Manual SQL execution required")
        print("Copy the SQL above into Supabase Dashboard > SQL Editor")

if __name__ == "__main__":
    main()