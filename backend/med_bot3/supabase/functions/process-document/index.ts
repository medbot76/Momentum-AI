import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
}

interface ProcessDocumentRequest {
  documentId: string
  notebookId: string
}

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders })
  }

  try {
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    const authHeader = req.headers.get('Authorization')!
    const token = authHeader.replace('Bearer ', '')
    const { data: { user } } = await supabaseClient.auth.getUser(token)

    if (!user) {
      throw new Error('Unauthorized')
    }

    const { documentId, notebookId }: ProcessDocumentRequest = await req.json()

    // Get document details
    const { data: document, error: docError } = await supabaseClient
      .from('documents')
      .select('*')
      .eq('id', documentId)
      .eq('user_id', user.id)
      .single()

    if (docError || !document) {
      throw new Error('Document not found')
    }

    // Update processing status
    await supabaseClient
      .from('documents')
      .update({ processing_status: 'processing' })
      .eq('id', documentId)

    // Download file from storage
    const { data: fileData, error: downloadError } = await supabaseClient.storage
      .from('documents')
      .download(document.storage_path)

    if (downloadError || !fileData) {
      throw new Error('Failed to download file')
    }

    // Convert file to base64 for processing
    const arrayBuffer = await fileData.arrayBuffer()
    const base64File = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)))

    // Process document based on type
    let chunks: any[] = []
    
    if (document.file_type === 'application/pdf') {
      chunks = await processPDF(base64File, documentId, notebookId, user.id)
    } else if (document.file_type.startsWith('image/')) {
      chunks = await processImage(base64File, documentId, notebookId, user.id)
    } else if (document.file_type === 'text/plain') {
      chunks = await processText(base64File, documentId, notebookId, user.id)
    }

    // Store chunks in database
    if (chunks.length > 0) {
      const { error: chunksError } = await supabaseClient
        .from('chunks')
        .insert(chunks)

      if (chunksError) {
        throw new Error('Failed to store chunks')
      }
    }

    // Update processing status to completed
    await supabaseClient
      .from('documents')
      .update({ 
        processing_status: 'completed',
        metadata: { chunks_count: chunks.length }
      })
      .eq('id', documentId)

    return new Response(
      JSON.stringify({ 
        success: true, 
        chunks_count: chunks.length,
        message: 'Document processed successfully'
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error('Error processing document:', error)
    
    return new Response(
      JSON.stringify({ 
        success: false, 
        error: error.message 
      }),
      { 
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' }
      }
    )
  }
})

async function processPDF(base64File: string, documentId: string, notebookId: string, userId: string) {
  // This would integrate with your existing PDF processing logic
  // For now, return a placeholder
  return [{
    id: crypto.randomUUID(),
    user_id: userId,
    notebook_id: notebookId,
    document_id: documentId,
    content: 'PDF content placeholder',
    tokens: 100,
    chunk_index: 0,
    metadata: { type: 'text', page: 1 }
  }]
}

async function processImage(base64File: string, documentId: string, notebookId: string, userId: string) {
  // This would integrate with your existing image processing logic
  return [{
    id: crypto.randomUUID(),
    user_id: userId,
    notebook_id: notebookId,
    document_id: documentId,
    content: 'Image analysis placeholder',
    tokens: 50,
    chunk_index: 0,
    metadata: { type: 'image' }
  }]
}

async function processText(base64File: string, documentId: string, notebookId: string, userId: string) {
  // Decode base64 text
  const text = atob(base64File)
  
  // Simple text chunking
  const chunks = []
  const maxTokens = 500
  const sentences = text.split('. ')
  let currentChunk = ''
  let chunkIndex = 0
  
  for (const sentence of sentences) {
    if ((currentChunk + sentence).length > maxTokens && currentChunk) {
      chunks.push({
        id: crypto.randomUUID(),
        user_id: userId,
        notebook_id: notebookId,
        document_id: documentId,
        content: currentChunk.trim(),
        tokens: Math.ceil(currentChunk.length / 4), // Rough token estimate
        chunk_index: chunkIndex++,
        metadata: { type: 'text' }
      })
      currentChunk = sentence + '. '
    } else {
      currentChunk += sentence + '. '
    }
  }
  
  // Add remaining chunk
  if (currentChunk.trim()) {
    chunks.push({
      id: crypto.randomUUID(),
      user_id: userId,
      notebook_id: notebookId,
      document_id: documentId,
      content: currentChunk.trim(),
      tokens: Math.ceil(currentChunk.length / 4),
      chunk_index: chunkIndex,
      metadata: { type: 'text' }
    })
  }
  
  return chunks
}