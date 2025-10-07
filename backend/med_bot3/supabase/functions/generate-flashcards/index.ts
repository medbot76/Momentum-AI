import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
}

interface GenerateFlashcardsRequest {
  topic: string
  notebookId: string
  numCards: number
  difficulty?: string
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

    const { topic, notebookId, numCards, difficulty = 'medium' }: GenerateFlashcardsRequest = await req.json()

    // Get relevant chunks for the topic
    const { data: chunks, error: chunksError } = await supabaseClient.rpc(
      'match_chunks',
      {
        query_embedding: await generateEmbedding(topic),
        match_threshold: 0.6,
        match_count: 10,
        notebook_id: notebookId
      }
    )

    if (chunksError) {
      throw new Error('Failed to retrieve relevant content')
    }

    if (!chunks || chunks.length === 0) {
      throw new Error('No relevant content found for this topic')
    }

    // Build context from chunks
    const context = chunks.map((chunk: any) => chunk.content).join('\n\n')

    // Generate flashcards using AI
    const flashcards = await generateFlashcardsWithAI(topic, context, numCards, difficulty)

    // Store flashcards in database
    const flashcardsToInsert = flashcards.map((card: any) => ({
      user_id: user.id,
      notebook_id: notebookId,
      question: card.question,
      answer: card.answer,
      topic,
      difficulty,
      metadata: {
        generated_at: new Date().toISOString(),
        context_chunks: chunks.length
      }
    }))

    const { data: insertedCards, error: insertError } = await supabaseClient
      .from('flashcards')
      .insert(flashcardsToInsert)
      .select()

    if (insertError) {
      throw new Error('Failed to save flashcards')
    }

    return new Response(
      JSON.stringify({
        success: true,
        flashcards: insertedCards,
        count: insertedCards.length,
        topic,
        difficulty
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error('Error generating flashcards:', error)
    
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
  // Placeholder embedding generation
  return new Array(768).fill(0).map(() => Math.random())
}

async function generateFlashcardsWithAI(topic: string, context: string, numCards: number, difficulty: string) {
  // This would integrate with your AI model
  // For now, return placeholder flashcards
  const flashcards = []
  
  for (let i = 0; i < numCards; i++) {
    flashcards.push({
      question: `Sample question ${i + 1} about ${topic}?`,
      answer: `Sample answer ${i + 1} for ${topic} at ${difficulty} difficulty level.`
    })
  }
  
  return flashcards
}