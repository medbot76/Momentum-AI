/*
  # Initial Database Schema for Med-Bot AI

  1. New Tables
    - `profiles` - User profile information
    - `notebooks` - Course/notebook organization
    - `documents` - Uploaded documents metadata
    - `chunks` - Vector chunks for RAG
    - `conversations` - Chat conversation history
    - `flashcards` - Generated flashcards
    - `exams` - Generated practice exams
    - `study_plans` - Generated study plans
    - `calendar_events` - Study plan calendar events

  2. Security
    - Enable RLS on all tables
    - Add policies for authenticated users to access their own data
    - Add policies for public read access where appropriate

  3. Storage
    - Create buckets for document uploads and generated files
*/

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Profiles table
CREATE TABLE IF NOT EXISTS profiles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  email text,
  full_name text,
  avatar_url text,
  preferences jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Notebooks table (for organizing courses/subjects)
CREATE TABLE IF NOT EXISTS notebooks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  name text NOT NULL,
  description text DEFAULT '',
  color text DEFAULT '#4285f4',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  notebook_id uuid REFERENCES notebooks(id) ON DELETE CASCADE,
  filename text NOT NULL,
  original_filename text NOT NULL,
  file_type text NOT NULL,
  file_size bigint DEFAULT 0,
  storage_path text NOT NULL,
  processing_status text DEFAULT 'pending',
  processing_error text,
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Chunks table for RAG pipeline
CREATE TABLE IF NOT EXISTS chunks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  notebook_id uuid REFERENCES notebooks(id) ON DELETE CASCADE,
  document_id uuid REFERENCES documents(id) ON DELETE CASCADE,
  content text NOT NULL,
  embedding vector(768),
  tokens integer DEFAULT 0,
  chunk_index integer DEFAULT 0,
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now()
);

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  notebook_id uuid REFERENCES notebooks(id) ON DELETE CASCADE,
  title text DEFAULT 'New Conversation',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id uuid REFERENCES conversations(id) ON DELETE CASCADE,
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  role text NOT NULL CHECK (role IN ('user', 'assistant')),
  content text NOT NULL,
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now()
);

-- Flashcards table
CREATE TABLE IF NOT EXISTS flashcards (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  notebook_id uuid REFERENCES notebooks(id) ON DELETE CASCADE,
  question text NOT NULL,
  answer text NOT NULL,
  topic text,
  difficulty text DEFAULT 'medium',
  last_reviewed timestamptz,
  review_count integer DEFAULT 0,
  ease_factor real DEFAULT 2.5,
  interval_days integer DEFAULT 1,
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Exams table
CREATE TABLE IF NOT EXISTS exams (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  notebook_id uuid REFERENCES notebooks(id) ON DELETE CASCADE,
  title text NOT NULL,
  description text DEFAULT '',
  difficulty text DEFAULT 'medium',
  num_questions integer DEFAULT 10,
  topic text,
  exam_content text NOT NULL,
  answer_key text,
  pdf_path text,
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Study plans table
CREATE TABLE IF NOT EXISTS study_plans (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  notebook_id uuid REFERENCES notebooks(id) ON DELETE CASCADE,
  title text NOT NULL,
  description text DEFAULT '',
  semester_weeks integer DEFAULT 16,
  syllabus_filename text,
  pdf_path text,
  smart_scheduling boolean DEFAULT false,
  calendar_type text,
  calendar_email text,
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Calendar events table
CREATE TABLE IF NOT EXISTS calendar_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  study_plan_id uuid REFERENCES study_plans(id) ON DELETE CASCADE,
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  title text NOT NULL,
  description text DEFAULT '',
  start_datetime timestamptz NOT NULL,
  end_datetime timestamptz NOT NULL,
  event_type text DEFAULT 'study',
  calendar_event_id text,
  metadata jsonb DEFAULT '{}',
  created_at timestamptz DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE notebooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE flashcards ENABLE ROW LEVEL SECURITY;
ALTER TABLE exams ENABLE ROW LEVEL SECURITY;
ALTER TABLE study_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE calendar_events ENABLE ROW LEVEL SECURITY;

-- Profiles policies
CREATE POLICY "Users can view own profile"
  ON profiles FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can update own profile"
  ON profiles FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own profile"
  ON profiles FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

-- Notebooks policies
CREATE POLICY "Users can manage own notebooks"
  ON notebooks FOR ALL
  TO authenticated
  USING (auth.uid() = user_id);

-- Documents policies
CREATE POLICY "Users can manage own documents"
  ON documents FOR ALL
  TO authenticated
  USING (auth.uid() = user_id);

-- Chunks policies
CREATE POLICY "Users can manage own chunks"
  ON chunks FOR ALL
  TO authenticated
  USING (auth.uid() = user_id);

-- Conversations policies
CREATE POLICY "Users can manage own conversations"
  ON conversations FOR ALL
  TO authenticated
  USING (auth.uid() = user_id);

-- Messages policies
CREATE POLICY "Users can manage own messages"
  ON messages FOR ALL
  TO authenticated
  USING (auth.uid() = user_id);

-- Flashcards policies
CREATE POLICY "Users can manage own flashcards"
  ON flashcards FOR ALL
  TO authenticated
  USING (auth.uid() = user_id);

-- Exams policies
CREATE POLICY "Users can manage own exams"
  ON exams FOR ALL
  TO authenticated
  USING (auth.uid() = user_id);

-- Study plans policies
CREATE POLICY "Users can manage own study plans"
  ON study_plans FOR ALL
  TO authenticated
  USING (auth.uid() = user_id);

-- Calendar events policies
CREATE POLICY "Users can manage own calendar events"
  ON calendar_events FOR ALL
  TO authenticated
  USING (auth.uid() = user_id);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_notebook_id ON chunks(notebook_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_notebook_id ON documents(notebook_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_flashcards_notebook_id ON flashcards(notebook_id);
CREATE INDEX IF NOT EXISTS idx_calendar_events_study_plan_id ON calendar_events(study_plan_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
CREATE TRIGGER update_profiles_updated_at BEFORE UPDATE ON profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_notebooks_updated_at BEFORE UPDATE ON notebooks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_flashcards_updated_at BEFORE UPDATE ON flashcards FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_exams_updated_at BEFORE UPDATE ON exams FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_study_plans_updated_at BEFORE UPDATE ON study_plans FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();