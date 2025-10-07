-- Database Schema for Med-Bot Application
-- This script sets up all necessary tables, indexes, and security policies

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create enum types
CREATE TYPE user_role AS ENUM ('student', 'instructor', 'admin');
CREATE TYPE message_role AS ENUM ('user', 'assistant', 'system');

-- Users profile (extends Supabase auth.users)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID REFERENCES auth.users ON DELETE CASCADE PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    avatar_url TEXT,
    role user_role DEFAULT 'student',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Notebooks for organizing content
CREATE TABLE IF NOT EXISTS public.notebooks (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_user_notebook UNIQUE (user_id, title)
);

-- Document chunks with vector embeddings
CREATE TABLE IF NOT EXISTS public.documents (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    notebook_id UUID REFERENCES public.notebooks(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    chunk_type TEXT DEFAULT 'text',
    tokens INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT unique_chunk_per_notebook UNIQUE (user_id, notebook_id, chunk_id)
);

-- Chat conversations
CREATE TABLE IF NOT EXISTS public.conversations (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    notebook_id UUID REFERENCES public.notebooks(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat messages
CREATE TABLE IF NOT EXISTS public.messages (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    conversation_id UUID REFERENCES public.conversations(id) ON DELETE CASCADE,
    role message_role NOT NULL,
    content TEXT NOT NULL,
    tokens INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User settings and preferences
CREATE TABLE IF NOT EXISTS public.user_settings (
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE PRIMARY KEY,
    theme TEXT DEFAULT 'light',
    default_model TEXT DEFAULT 'gemini-pro',
    default_embedding_model TEXT DEFAULT 'text-embedding-004',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_documents_user_notebook ON documents(user_id, notebook_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at);

-- Enable Row Level Security
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notebooks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_settings ENABLE ROW LEVEL SECURITY;

-- Row Level Security Policies
-- Profiles
CREATE POLICY "Users can view their own profile" 
    ON public.profiles FOR SELECT 
    USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id);

-- Notebooks
CREATE POLICY "Users can manage their notebooks"
    ON public.notebooks
    FOR ALL
    USING (auth.uid() = user_id);

-- Documents
CREATE POLICY "Users can manage their documents"
    ON public.documents
    FOR ALL
    USING (auth.uid() = user_id);

-- Conversations
CREATE POLICY "Users can manage their conversations"
    ON public.conversations
    FOR ALL
    USING (auth.uid() = user_id);

-- Messages
CREATE POLICY "Users can manage their messages"
    ON public.messages
    FOR ALL
    USING (EXISTS (
        SELECT 1 FROM public.conversations c 
        WHERE c.id = conversation_id AND c.user_id = auth.uid()
    ));

-- User Settings
CREATE POLICY "Users can manage their settings"
    ON public.user_settings
    FOR ALL
    USING (auth.uid() = user_id);

-- Helper function to update timestamps
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_profiles_modtime
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_notebooks_modtime
    BEFORE UPDATE ON public.notebooks
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_documents_modtime
    BEFORE UPDATE ON public.documents
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

CREATE TRIGGER update_conversations_modtime
    BEFORE UPDATE ON public.conversations
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

-- Search function for documents
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

-- Function to create a new user profile when a user signs up
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name)
    VALUES (NEW.id, NEW.email, NEW.raw_user_meta_data->>'full_name');
    
    INSERT INTO public.user_settings (user_id)
    VALUES (NEW.id);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create the trigger
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();
