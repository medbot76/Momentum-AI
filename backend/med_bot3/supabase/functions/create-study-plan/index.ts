import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
}

interface CreateStudyPlanRequest {
  notebookId: string
  title: string
  semesterWeeks: number
  syllabusContent: string
  smartScheduling?: boolean
  calendarType?: string
  calendarEmail?: string
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
      semesterWeeks,
      syllabusContent,
      smartScheduling = false,
      calendarType,
      calendarEmail
    }: CreateStudyPlanRequest = await req.json()

    // Generate study plan events using AI
    const events = await generateStudyPlanWithAI(syllabusContent, semesterWeeks)

    // Create study plan record
    const { data: studyPlan, error: planError } = await supabaseClient
      .from('study_plans')
      .insert({
        user_id: user.id,
        notebook_id: notebookId,
        title,
        semester_weeks: semesterWeeks,
        smart_scheduling: smartScheduling,
        calendar_type: calendarType,
        calendar_email: calendarEmail,
        metadata: {
          generated_at: new Date().toISOString(),
          events_count: events.length
        }
      })
      .select()
      .single()

    if (planError) {
      throw new Error('Failed to create study plan')
    }

    // Store calendar events
    const eventsToInsert = events.map((event: any) => ({
      study_plan_id: studyPlan.id,
      user_id: user.id,
      title: event.title,
      description: event.description,
      start_datetime: event.start_datetime,
      end_datetime: event.end_datetime,
      event_type: event.type || 'study',
      metadata: event.metadata || {}
    }))

    const { data: calendarEvents, error: eventsError } = await supabaseClient
      .from('calendar_events')
      .insert(eventsToInsert)
      .select()

    if (eventsError) {
      throw new Error('Failed to create calendar events')
    }

    return new Response(
      JSON.stringify({
        success: true,
        studyPlan,
        events: calendarEvents,
        eventsCount: calendarEvents.length
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error('Error creating study plan:', error)
    
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

async function generateStudyPlanWithAI(syllabusContent: string, semesterWeeks: number) {
  // This would integrate with your AI model to generate study plan events
  // For now, return placeholder events
  const events = []
  const startDate = new Date()
  startDate.setDate(startDate.getDate() + (7 - startDate.getDay())) // Next Monday
  
  for (let week = 0; week < semesterWeeks; week++) {
    const weekStart = new Date(startDate)
    weekStart.setDate(weekStart.getDate() + (week * 7))
    
    // Study session
    const studyStart = new Date(weekStart)
    studyStart.setHours(10, 0, 0, 0) // 10 AM
    const studyEnd = new Date(studyStart)
    studyEnd.setHours(12, 0, 0, 0) // 12 PM
    
    events.push({
      title: `Week ${week + 1} Study Session`,
      description: `Study session for week ${week + 1} content`,
      start_datetime: studyStart.toISOString(),
      end_datetime: studyEnd.toISOString(),
      type: 'study',
      metadata: { week: week + 1 }
    })
    
    // Assignment (every 2 weeks)
    if (week % 2 === 1) {
      const assignmentDate = new Date(weekStart)
      assignmentDate.setDate(assignmentDate.getDate() + 4) // Friday
      assignmentDate.setHours(23, 59, 0, 0) // Due end of day
      
      events.push({
        title: `Assignment ${Math.floor(week / 2) + 1} Due`,
        description: `Submit assignment covering weeks ${week} and ${week + 1}`,
        start_datetime: assignmentDate.toISOString(),
        end_datetime: assignmentDate.toISOString(),
        type: 'assignment',
        metadata: { week: week + 1, assignment_number: Math.floor(week / 2) + 1 }
      })
    }
  }
  
  return events
}