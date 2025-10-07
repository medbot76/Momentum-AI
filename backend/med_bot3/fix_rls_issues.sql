-- Fix RLS issues for notebooks table and storage

-- 1. Disable RLS on notebooks table
ALTER TABLE notebooks DISABLE ROW LEVEL SECURITY;

-- 2. Disable RLS on documents table  
ALTER TABLE documents DISABLE ROW LEVEL SECURITY;

-- 3. Create storage bucket if it doesn't exist
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'documents',
    'documents', 
    false,
    52428800, -- 50MB limit
    ARRAY['application/pdf', 'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'text/plain', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
)
ON CONFLICT (id) DO NOTHING;

-- 4. Create storage policy for documents bucket
CREATE POLICY "Allow authenticated users to upload documents" ON storage.objects
FOR INSERT WITH CHECK (
    bucket_id = 'documents' AND 
    (auth.role() = 'authenticated' OR auth.role() = 'service_role')
);

CREATE POLICY "Allow users to view their own documents" ON storage.objects
FOR SELECT USING (
    bucket_id = 'documents' AND 
    auth.uid()::text = (storage.foldername(name))[2]
);

CREATE POLICY "Allow users to delete their own documents" ON storage.objects
FOR DELETE USING (
    bucket_id = 'documents' AND 
    auth.uid()::text = (storage.foldername(name))[2]
);

-- 5. Verify tables don't have RLS enabled
SELECT schemaname, tablename, rowsecurity 
FROM pg_tables 
WHERE tablename IN ('notebooks', 'documents', 'chunks')
ORDER BY tablename;



