/*
  # Storage Buckets Setup

  1. Create storage buckets for file uploads
  2. Set up RLS policies for secure file access
*/

-- Create storage buckets
INSERT INTO storage.buckets (id, name, public) VALUES 
  ('documents', 'documents', false),
  ('generated-files', 'generated-files', false)
ON CONFLICT (id) DO NOTHING;

-- Documents bucket policies
CREATE POLICY "Users can upload own documents"
  ON storage.objects FOR INSERT
  TO authenticated
  WITH CHECK (bucket_id = 'documents' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can view own documents"
  ON storage.objects FOR SELECT
  TO authenticated
  USING (bucket_id = 'documents' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can update own documents"
  ON storage.objects FOR UPDATE
  TO authenticated
  USING (bucket_id = 'documents' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can delete own documents"
  ON storage.objects FOR DELETE
  TO authenticated
  USING (bucket_id = 'documents' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Generated files bucket policies
CREATE POLICY "Users can upload own generated files"
  ON storage.objects FOR INSERT
  TO authenticated
  WITH CHECK (bucket_id = 'generated-files' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can view own generated files"
  ON storage.objects FOR SELECT
  TO authenticated
  USING (bucket_id = 'generated-files' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can update own generated files"
  ON storage.objects FOR UPDATE
  TO authenticated
  USING (bucket_id = 'generated-files' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Users can delete own generated files"
  ON storage.objects FOR DELETE
  TO authenticated
  USING (bucket_id = 'generated-files' AND auth.uid()::text = (storage.foldername(name))[1]);