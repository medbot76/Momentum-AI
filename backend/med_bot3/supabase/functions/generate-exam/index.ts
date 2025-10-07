import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
}

interface GenerateExamRequest {
  notebookId: string
  title: string
  difficulty: string
  numQuestions: number
  topic?: string
  includeAnswerKey?: boolean
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

    const { 
      notebookId, 
      title, 
      difficulty, 
      numQuestions, 
      topic,
      includeAnswerKey = false 
    }: GenerateExamRequest = await req.json()

    // Get relevant content
    const searchQuery = topic || 'general course content'
    const { data: chunks, error: chunksError } = await supabaseClient.rpc(
      'match_chunks',
      {
        query_embedding: await generateEmbedding(searchQuery),
        match_threshold: 0.5,
        match_count: 15,
        notebook_id: notebookId
      }
    )

    if (chunksError) {
      throw new Error('Failed to retrieve course content')
    }

    if (!chunks || chunks.length === 0) {
      throw new Error('No course content found')
    }

    // Build context from chunks
    const context = chunks.map((chunk: any) => chunk.content).join('\n\n')

    // Generate exam content
    const examContent = await generateExamWithAI(context, difficulty, numQuestions, topic)
    
    // Generate answer key if requested
    let answerKey = null
    if (includeAnswerKey) {
      answerKey = await generateAnswerKeyWithAI(examContent)
    }

    // Store exam in database
    const { data: exam, error: examError } = await supabaseClient
      .from('exams')
      .insert({
        user_id: user.id,
        notebook_id: notebookId,
        title,
        difficulty,
        num_questions: numQuestions,
        topic,
        exam_content: examContent,
        answer_key: answerKey,
        metadata: {
          generated_at: new Date().toISOString(),
          context_chunks: chunks.length
        }
      })
      .select()
      .single()

    if (examError) {
      throw new Error('Failed to save exam')
    }

    return new Response(
      JSON.stringify({
        success: true,
        exam,
        examContent,
        answerKey,
        metadata: {
          chunks_used: chunks.length,
          difficulty,
          num_questions: numQuestions
        }
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error('Error generating exam:', error)
    
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

async function generateExamWithAI(context: string, difficulty: string, numQuestions: number, topic?: string) {
  // This would integrate with your AI model
  // For now, return placeholder exam content
  return `
# Practice Exam${topic ? ` - ${topic}` : ''}

**Difficulty:** ${difficulty.toUpperCase()}
**Questions:** ${numQuestions}

## Instructions
Answer all questions clearly and completely.

## Multiple Choice Questions

**Question 1:** What is the main concept discussed in the course material?
A) Option A
B) Option B  
C) Option C
D) Option D

**Question 2:** Which of the following best describes...?
A) Option A
B) Option B
C) Option C
D) Option D

## Short Answer Questions

**Question 3:** Explain the key principles covered in the course material.

**Question 4:** Describe the relationship between the main concepts.

## Problem-Solving Questions

**Question 5:** Apply the concepts to solve the following scenario...

---
*This is a placeholder exam. In a real implementation, this would be generated using AI based on the course content.*
  `.trim()
}

async function generateAnswerKeyWithAI(examContent: string) {
  // This would integrate with your AI model
  // For now, return placeholder answer key
  return `
# Answer Key

**Question 1:** C) Option C
*Explanation: This is the correct answer because...*

**Question 2:** A) Option A  
*Explanation: This option best describes...*

**Question 3:** Key points should include: concept A, concept B, and their relationship...

**Question 4:** The relationship can be described as...

**Question 5:** Solution approach: Step 1... Step 2... Final answer...

---
*This is a placeholder answer key. In a real implementation, this would be generated using AI.*
  `.trim()
}