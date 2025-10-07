import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
}

interface ChatQueryRequest {
  question: string
  notebookId: string
  conversationId?: string
  model?: string
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

    const { question, notebookId, conversationId, model = 'gemini-2.0-flash' }: ChatQueryRequest = await req.json()

    // Create or get conversation
    let currentConversationId = conversationId
    if (!currentConversationId) {
      const { data: conversation, error: convError } = await supabaseClient
        .from('conversations')
        .insert({
          user_id: user.id,
          notebook_id: notebookId,
          title: question.substring(0, 50) + '...'
        })
        .select()
        .single()

      if (convError) throw convError
      currentConversationId = conversation.id
    }

    // Store user message
    await supabaseClient
      .from('messages')
      .insert({
        conversation_id: currentConversationId,
        user_id: user.id,
        role: 'user',
        content: question
      })

    // Get relevant chunks using vector similarity
    const { data: chunks, error: chunksError } = await supabaseClient.rpc(
      'match_chunks',
      {
        query_embedding: await generateEmbedding(question),
        match_threshold: 0.7,
        match_count: 5,
        notebook_id: notebookId
      }
    )

    if (chunksError) {
      console.error('Error fetching chunks:', chunksError)
    }

    // Build context from chunks
    const context = chunks?.map((chunk: any) => chunk.content).join('\n\n') || ''

    // Generate response using AI
    const answer = await generateAnswer(question, context, model)

    // Store assistant message
    await supabaseClient
      .from('messages')
      .insert({
        conversation_id: currentConversationId,
        user_id: user.id,
        role: 'assistant',
        content: answer,
        metadata: {
          model,
          chunks_used: chunks?.length || 0,
          context_length: context.length
        }
      })

    return new Response(
      JSON.stringify({
        success: true,
        answer,
        conversationId: currentConversationId,
        chunks: chunks || [],
        metadata: {
          model,
          chunks_used: chunks?.length || 0
        }
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error('Error in chat query:', error)
    
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

async function generateEmbedding(text: string): Promise<number[]> {
  // This would integrate with your embedding generation logic
  // For now, return a placeholder embedding
  return new Array(768).fill(0).map(() => Math.random())
}

async function generateAnswer(question: string, context: string, model: string): Promise<string> {
  // This would integrate with your AI model logic
  // For now, return a placeholder response
  const prompt = context 
    ? `Based on the following context, answer the question: ${question}\n\nContext: ${context}`
    : `Answer the question: ${question}`
  
  return `This is a placeholder response for: "${question}". In a real implementation, this would use ${model} to generate a proper answer based on the provided context.`
}