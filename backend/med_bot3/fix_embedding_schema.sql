-- Fix embedding column to match all-mpnet-base-v2 (768 dimensions)
-- Run this in your Supabase SQL Editor
-- This will update the schema and optionally clear old embeddings

-- Step 1: Ensure pgvector extension is enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Check current column type
SELECT column_name, data_type, udt_name 
FROM information_schema.columns 
WHERE table_name = 'chunks' AND column_name = 'embedding';

-- Step 3: Drop the old embedding column (this will delete all existing embeddings)
-- If you want to keep old embeddings, comment out this line and use the ALTER COLUMN approach below
ALTER TABLE public.chunks DROP COLUMN IF EXISTS embedding;

-- Step 4: Add the new embedding column with correct dimension (768)
ALTER TABLE public.chunks ADD COLUMN embedding VECTOR(768);

-- ALTERNATIVE: If you want to keep old embeddings (they'll be incompatible but won't cause errors):
-- ALTER TABLE public.chunks ALTER COLUMN embedding TYPE VECTOR(768) USING embedding::text::vector(768);
-- Note: This will fail if there are existing 384-dim vectors, so dropping is safer

-- Step 5: Drop old index if it exists
DROP INDEX IF EXISTS chunks_embedding_idx;

-- Step 6: Create the HNSW index for vector similarity search (768 dimensions)
CREATE INDEX chunks_embedding_idx 
ON chunks USING hnsw (embedding vector_cosine_ops);

-- Step 7: Create other useful indexes (if they don't exist)
CREATE INDEX IF NOT EXISTS chunks_user_notebook_idx 
ON chunks (user_id, notebook_id);

-- Step 8: Verify the fix
SELECT column_name, data_type, udt_name 
FROM information_schema.columns 
WHERE table_name = 'chunks' AND column_name = 'embedding';

-- Step 9: Test that we can insert a vector (768 dimensions)
-- This is just a test - we'll delete it immediately
DO $$
DECLARE
    test_user_id uuid := gen_random_uuid();
    test_notebook_id uuid := gen_random_uuid();
BEGIN
    INSERT INTO public.chunks (content, embedding, user_id, notebook_id) 
    VALUES (
        'test content - will be deleted', 
        ARRAY_FILL(0.1, ARRAY[768])::vector(768), 
        test_user_id, 
        test_notebook_id
    );
    
    -- Verify it was inserted
    IF EXISTS (SELECT 1 FROM chunks WHERE content = 'test content - will be deleted') THEN
        RAISE NOTICE 'Test insert successful!';
    END IF;
    
    -- Clean up test data
    DELETE FROM public.chunks WHERE content = 'test content - will be deleted';
    RAISE NOTICE 'Test data cleaned up';
END $$;

-- Step 10: Show current chunk count
SELECT COUNT(*) as total_chunks, 
       COUNT(embedding) as chunks_with_embeddings,
       COUNT(*) - COUNT(embedding) as chunks_without_embeddings
FROM public.chunks;

-- Done! Your schema is now ready for 768-dim embeddings from all-mpnet-base-v2

