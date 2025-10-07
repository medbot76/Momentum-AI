# backend/app.py
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from med_bot3.chatbot import ChatBot
from med_bot3.tts_elevenlabs import generate_teen_voice
from med_bot3.stt_whisper import transcribe_audio
from med_bot3.exam_feature.exam_generator import ExamGenerator, ExamConfig, Difficulty
from med_bot3.rag_flashcards import RAGFlashcards
from med_bot3.study_planner import StudyPlanner
from supabase import create_client, Client
import os
import asyncio
import io
import json
import datetime
import uuid
from functools import wraps
from pathlib import Path

app = Flask(__name__)

# Configure CORS to allow all origins for development
CORS(app, 
     origins="*",
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
     supports_credentials=False)

# Initialize components
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Initialize Supabase client
supabase_url = os.environ.get('SUPABASE_URL')
supabase_key = os.environ.get('SUPABASE_KEY')
if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")

supabase: Client = create_client(supabase_url, supabase_key)

chatbot = ChatBot(api_key)
exam_generator = ExamGenerator(api_key)
# Add StudyPlanner instance for API use
study_planner = StudyPlanner(api_key)

# Helper functions for user/notebook management
def get_or_create_notebook(user_id: str, notebook_name: str = "Default Notebook") -> str:
    """Get or create a notebook for the user"""
    try:
        # Try to find existing notebook
        result = supabase.table('notebooks').select('id').eq('user_id', user_id).eq('name', notebook_name).execute()
        
        if result.data:
            return result.data[0]['id']
        
        # Create new notebook
        notebook_data = {
            'user_id': user_id,
            'name': notebook_name,
            'description': f'Default notebook for user {user_id}',
            'color': '#4285f4'
        }
        result = supabase.table('notebooks').insert(notebook_data).execute()
        return result.data[0]['id']
    except Exception as e:
        print(f"Error creating notebook: {e}")
        # Return a default UUID if creation fails
        return "65100e0f-0045-415f-a98a-c30180f2fc52"

def upload_to_supabase_storage(file_path: str, user_id: str, filename: str) -> str:
    """Upload file to Supabase storage and return the storage path"""
    try:
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Create storage path: {user_id}/{filename} (bucket name 'documents' is already specified)
        storage_path = f"{user_id}/{filename}"
        
        # Upload to Supabase storage
        result = supabase.storage.from_('documents').upload(storage_path, file_data)
        
        if result.get('error'):
            print(f"Storage upload warning: {result['error']}")
            # Don't raise exception, just log the warning and continue
            # The file might still be uploaded successfully
        
        print(f"Storage upload result: {result}")
        return storage_path
    except Exception as e:
        print(f"Error uploading to storage: {e}")
        # Still return the storage path so processing can continue
        storage_path = f"{user_id}/{filename}"
        print(f"Continuing with storage path: {storage_path}")
        return storage_path

def create_document_record(user_id: str, notebook_id: str, filename: str, original_filename: str, 
                          file_type: str, file_size: int, storage_path: str) -> str:
    """Create a document record in the documents table"""
    try:
        # Generate the public URL for the file
        public_url = f"{supabase_url}/storage/v1/object/public/documents/{storage_path}"
        
        document_data = {
            'user_id': user_id,
            'notebook_id': notebook_id,
            'filename': filename,
            'original_filename': original_filename,
            'file_type': file_type,
            'file_size': file_size,
            'storage_path': storage_path,
            'file_path': public_url,  # Add the required file_path field
            'processing_status': 'pending',
            'metadata': {}
        }
        
        result = supabase.table('documents').insert(document_data).execute()
        return result.data[0]['id']
    except Exception as e:
        print(f"Error creating document record: {e}")
        raise e

def update_document_status(document_id: str, status: str, error: str = None):
    """Update document processing status"""
    try:
        update_data = {'processing_status': status}
        if error:
            update_data['processing_error'] = error
        
        supabase.table('documents').update(update_data).eq('id', document_id).execute()
    except Exception as e:
        print(f"Error updating document status: {e}")

async def initialize_exam_generator():
    """Initialize the exam generator by ingesting all existing course content files."""
    course_content_dir = Path("course_content")
    if not course_content_dir.exists():
        print("Course content directory not found, creating it...")
        course_content_dir.mkdir(exist_ok=True)
        return
    
    # Get all files in the course content directory
    files = list(course_content_dir.glob("*"))
    pdf_files = [f for f in files if f.suffix.lower() == '.pdf']
    
    if not pdf_files:
        print("No PDF files found in course_content directory")
        return
    
    print(f"Found {len(pdf_files)} PDF files to ingest...")
    
    # Ingest each PDF file
    for pdf_file in pdf_files:
        try:
            print(f"Ingesting {pdf_file.name}...")
            await exam_generator.rag.ingest_pdf(str(pdf_file), notebook_id=None)
            print(f"✓ Successfully ingested {pdf_file.name}")
        except Exception as e:
            print(f"✗ Failed to ingest {pdf_file.name}: {str(e)}")
    
    print(f"Initialization complete. Ingested {len(pdf_files)} files.")

# Initialize the exam generator (without auto-ingesting course content)
print("Exam generator ready (course content will be ingested on-demand).")

def async_route(f):
    """Decorator to handle async routes in Flask"""
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    wrapper.__name__ = f.__name__
    return wrapper

@app.route('/api/chat', methods=['POST'])
@async_route
async def chat():
    try:
        data = request.json
        question = data.get('message')
        user_id = data.get('user_id')
        notebook_id = data.get('notebook_id')
        
        if not question:
            return jsonify({'error': 'No message provided'}), 400
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Get or create notebook if not provided
        if not notebook_id:
            notebook_id = get_or_create_notebook(user_id)
        
        # Get response from chatbot with user isolation
        response = await chatbot.ask_question(question, notebook_id=notebook_id)
        
        # Ensure the response is always a dict with 'answer' and 'videos'
        if isinstance(response, dict) and 'answer' in response:
            return jsonify(response)
        else:
            return jsonify({'answer': response, 'videos': []})
    except Exception as e:
        print(f"Chat Error: {e}")  # Add logging
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/stream', methods=['POST'])
@async_route
async def chat_stream():
    """
    Streaming chat endpoint using Server-Sent Events (SSE)
    Streams real-time updates from the multi-hop RAG pipeline
    """
    try:
        data = request.json
        question = data.get('message')
        user_id = data.get('user_id')
        notebook_id = data.get('notebook_id')
        
        if not question:
            return jsonify({'error': 'No message provided'}), 400
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Get or create notebook if not provided
        if not notebook_id:
            notebook_id = get_or_create_notebook(user_id)
        
        # For now, just use the regular chat endpoint
        # In the future, this can be enhanced with real streaming
        response = await chatbot.ask_question(question, notebook_id=notebook_id)
        
        # Create a simple streaming response
        def generate_stream():
            # Send initial update
            yield f"data: {json.dumps({'type': 'query_start', 'question': question})}\n\n"
            
            # Send completion update
            if isinstance(response, dict) and 'answer' in response:
                yield f"data: {json.dumps({'type': 'query_complete', 'answer': response['answer']})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'query_complete', 'answer': str(response)})}\n\n"
            
            # Send final completion signal
            yield f"data: {json.dumps({'type': 'stream_complete'})}\n\n"
        
        return Response(
            generate_stream(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )
        
    except Exception as e:
        print(f"Stream Chat Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
@async_route
async def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get user_id and notebook_id from form data
        user_id = request.form.get('user_id')
        notebook_id = request.form.get('notebook_id')
        notebook_name = request.form.get('notebook_name', 'Default Notebook')
        
        print(f"/api/upload user_id: {user_id}, notebook_id: {notebook_id}")
        
        if not user_id:
            return jsonify({'error': 'Missing user_id (authentication required)'}), 401
        
        # Get or create notebook
        if not notebook_id:
            notebook_id = get_or_create_notebook(user_id, notebook_name)
            print(f"Using notebook ID: {notebook_id}")
        
        # Generate unique filename to avoid conflicts
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Save file temporarily for processing
        temp_dir = 'temp_uploads'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, unique_filename)
        file.save(temp_path)
        
        try:
            # Get file info
            file_size = os.path.getsize(temp_path)
            file_type = file.content_type or 'application/octet-stream'
            
            # Upload to Supabase storage
            print(f"Uploading to Supabase storage: {unique_filename}")
            storage_path = upload_to_supabase_storage(temp_path, user_id, unique_filename)
            
            # Create document record in database
            print(f"Creating document record in database")
            document_id = create_document_record(
                user_id=user_id,
                notebook_id=notebook_id,
                filename=unique_filename,
                original_filename=file.filename,
                file_type=file_type,
                file_size=file_size,
                storage_path=storage_path
            )
            
            # Process the file with RAG pipeline
            print(f"Processing file with RAG pipeline: {temp_path}")
            update_document_status(document_id, 'processing')
            
            # Process based on file type
            print(f"Processing file type: {file_extension.lower()}")
            print(f"Using notebook_id: {notebook_id}, user_id: {user_id}")
            
            if file_extension.lower() == '.pdf':
                print("Processing as PDF...")
                success = await chatbot.upload_document(temp_path, notebook_id=notebook_id, user_id=user_id)
            elif file_extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                print("Processing as image...")
                from PIL import Image
                image = Image.open(temp_path)
                await chatbot.rag.analyze_image(image, notebook_id=notebook_id, user_id=user_id)
                success = True
            else:
                # For other file types, try text processing
                print("Processing as text...")
                success = await chatbot.upload_document(temp_path, notebook_id=notebook_id, user_id=user_id)
            
            print(f"RAG processing result: {success}")
            
            if success:
                update_document_status(document_id, 'completed')
                return jsonify({
                    'message': 'File uploaded and processed successfully',
                    'filename': file.filename,
                    'document_id': document_id,
                    'storage_path': storage_path
                })
            else:
                update_document_status(document_id, 'failed', 'RAG processing failed')
                return jsonify({'error': 'Failed to process file with RAG pipeline'}), 500
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"Upload Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Update document status if we have a document_id
        if 'document_id' in locals():
            update_document_status(document_id, 'failed', str(e))
        
        return jsonify({'error': 'Failed to process file', 'detail': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    try:
        # Get user_id from query parameters
        user_id = request.args.get('user_id')
        notebook_id = request.args.get('notebook_id')
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Query documents from database
        query = supabase.table('documents').select('*').eq('user_id', user_id)
        
        if notebook_id:
            query = query.eq('notebook_id', notebook_id)
        
        result = query.order('created_at', desc=True).execute()
        
        return jsonify({'documents': result.data})
    except Exception as e:
        print(f"Documents Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts', methods=['POST'])
@async_route
async def tts():
    try:
        data = request.json
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Generate audio using TTS
        audio_bytes = await asyncio.to_thread(generate_teen_voice, text)
        
        return send_file(
            io.BytesIO(audio_bytes),
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name='output.mp3'
        )
    except Exception as e:
        print(f"TTS Error: {e}")  # Add logging
        return jsonify({'error': str(e)}), 500

@app.route('/api/stt', methods=['POST'])
@async_route
async def stt():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Get the audio format from the filename or content type
        audio_format = 'webm'  # Default format from browser recording
        if audio_file.content_type:
            if 'webm' in audio_file.content_type:
                audio_format = 'webm'
            elif 'wav' in audio_file.content_type:
                audio_format = 'wav'
            elif 'mp3' in audio_file.content_type:
                audio_format = 'mp3'
        
        # Read the audio data
        audio_data = audio_file.read()
        
        # Transcribe the audio using async wrapper
        transcribed_text = await asyncio.to_thread(transcribe_audio, audio_data, audio_format)
        
        return jsonify({'text': transcribed_text})
    except Exception as e:
        print(f"STT Error: {e}")  # Add logging
        return jsonify({'error': str(e)}), 500

@app.route('/api/exam-pdf', methods=['POST'])
def exam_pdf():
    try:
        data = request.json
        difficulty_str = data.get('difficulty')
        num_questions = data.get('num_questions')
        topic = data.get('topic')
        format_type = data.get('format', 'PDF')  # Default to PDF for backward compatibility
        example_exam_filename = data.get('example_exam_filename')
        example_exam_text = data.get('example_exam_text')
        
        if not (difficulty_str and num_questions and topic):
            return jsonify({'error': 'Missing required parameters: difficulty, num_questions, topic'}), 400
        difficulty_str = difficulty_str.upper()
        if difficulty_str not in Difficulty.__members__:
            return jsonify({'error': 'Invalid difficulty'}), 400
        difficulty = Difficulty[difficulty_str]
        try:
            num_questions = int(num_questions)
        except Exception:
            return jsonify({'error': 'num_questions must be an integer'}), 400

        # --- NEW: Handle example exam upload if provided ---
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if example_exam_text:
                # Direct text input for example exam
                loop.run_until_complete(exam_generator.upload_extra_files(text_content=example_exam_text, file_type="example"))
            elif example_exam_filename:
                # File-based example exam (now from example_exams directory)
                file_path = os.path.join('exam_generation_feature', 'example_exams', example_exam_filename)
                if not os.path.exists(file_path):
                    return jsonify({'error': f'Example exam file {example_exam_filename} not found in example_exams/'}), 400
                loop.run_until_complete(exam_generator.upload_extra_files(file_path=file_path, file_type="example"))
        finally:
            loop.close()
        # --- END NEW ---

        # Use the same logic as the CLI: get content via RAG pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            content = loop.run_until_complete(exam_generator.get_content(topic))
        finally:
            loop.close()
        if not content:
            return jsonify({'error': 'No content found for the topic or files'}), 400
        # Optionally truncate content to avoid token overflow
        MAX_TOKENS = 1000000
        words = content.split()
        if len(words) > MAX_TOKENS:
            content = ' '.join(words[:MAX_TOKENS])

        # --- NEW: Set use_example_questions based on presence of example exam ---
        example_files = [f.name for f in exam_generator.example_exams_dir.glob("*")]
        use_example_questions = bool(example_files) or bool(exam_generator.example_exam_text)
        # --- END NEW ---

        config = ExamConfig(
            difficulty=difficulty,
            num_questions=num_questions,
            topic=topic,
            use_example_questions=use_example_questions,
            use_study_guide=False  # Could be extended for study guide support
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            exam = loop.run_until_complete(exam_generator.generate_exam(config, content))
        finally:
            loop.close()
        
        # Return based on format type
        if format_type.upper() == 'TEXT':
            # Return exam as plain text
            return exam, 200, {'Content-Type': 'text/plain'}
        else:
            # Return PDF (default behavior)
            pdf_path = exam_generator._generate_pdf(exam, config)
            return send_file(pdf_path, mimetype='application/pdf', as_attachment=True, download_name='exam.pdf')
    except Exception as e:
        print(f"Exam PDF Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/flashcards', methods=['POST'])
@async_route
async def flashcards():
    try:
        data = request.json
        print(f"Flashcards request data: {data}")  # Debug log
        topic = data.get('topic')
        filename = data.get('filename')
        num_cards = int(data.get('num_cards', 8))
        notebook_id = data.get('notebook_id')
        user_id = data.get('user_id', 'web-user')
        
        print(f"Topic: {topic}, Filename: {filename}, Num cards: {num_cards}, Notebook: {notebook_id}")  # Debug log
        
        if not topic or not filename:
            print(f"Missing required fields - Topic: {topic}, Filename: {filename}")  # Debug log
            return jsonify({'error': 'Missing topic or filename'}), 400
        
        if not notebook_id:
            print("Missing notebook_id")  # Debug log
            return jsonify({'error': 'Missing notebook_id'}), 400
        
        # Use the provided notebook_id directly (it's already a UUID)
        notebook_uuid = notebook_id
        
        # Check if document exists in database
        try:
            # Find the document in Supabase
            result = supabase.table('documents').select('*').eq('filename', filename).eq('notebook_id', notebook_uuid).execute()
            
            if not result.data:
                print(f"Document {filename} not found in database for notebook {notebook_id}")  # Debug log
                return jsonify({'error': f'File {filename} not found in your notebooks'}), 400
            
            document = result.data[0]
            print(f"Found document: {document['filename']} in notebook {notebook_id}")  # Debug log
            
            # Check if document has been processed (has chunks)
            chunks_result = supabase.table('chunks').select('id').eq('document_id', document['id']).limit(1).execute()
            
            if not chunks_result.data:
                print(f"No chunks found for document {filename}")  # Debug log
                return jsonify({'error': f'File {filename} has not been processed yet. Please wait a moment and try again.'}), 400
            
            print(f"Found {len(chunks_result.data)} chunks for document {filename}")  # Debug log
            
        except Exception as db_error:
            print(f"Database error: {db_error}")  # Debug log
            return jsonify({'error': f'Error accessing file information: {str(db_error)}'}), 400
        
        # Generate flashcards using existing chunks
        try:
            # Create RAG instance
            rag = RAGFlashcards()
            
            # Generate flashcards using existing chunks (no need to re-ingest)
            print(f"Generating flashcards for topic: {topic}")  # Debug log
            cards = await rag.generate_flashcards(topic=topic, notebook_id=notebook_uuid, num_cards=num_cards, user_id=user_id)
            print(f"Generated {len(cards)} flashcards")  # Debug log
            
            return jsonify({'flashcards': cards})
            
        except Exception as rag_error:
            print(f"RAG error: {rag_error}")  # Debug log
            return jsonify({'error': f'Failed to generate flashcards: {str(rag_error)}'}), 500
            
    except Exception as e:
        print(f"Flashcards Error: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload-example-exam', methods=['POST'])
def upload_example_exam():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        # Save the file to the example_exams directory
        os.makedirs('exam_generation_feature/example_exams', exist_ok=True)
        save_path = os.path.join('exam_generation_feature', 'example_exams', file.filename)
        file.save(save_path)
        return jsonify({'filename': file.filename})
    except Exception as e:
        print(f"Upload Example Exam Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/studyplan', methods=['POST'])
@async_route
async def studyplan():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        # Save the file to course_content directory (consistent with other endpoints)
        os.makedirs('course_content', exist_ok=True)
        save_path = os.path.join('course_content', file.filename)
        file.save(save_path)
        # Get weeks parameter (optional)
        weeks = request.form.get('weeks', 16)
        try:
            weeks = int(weeks)
        except Exception:
            weeks = 16
        # Generate study plan (PDF + events)
        result = await study_planner.create_study_plan_for_api(save_path, semester_weeks=weeks)
        print(f"Study plan result: {result}")
        if not result or not isinstance(result, tuple) or len(result) != 2:
            print(f"Invalid result format: {result}")
            return jsonify({'error': 'Failed to generate study plan'}), 500
        pdf_path, events = result
        print(f"PDF path: {pdf_path}")
        print(f"Events type: {type(events)}")
        print(f"Events count: {len(events) if events else 0}")
        if not pdf_path or not events:
            print(f"Missing data - PDF path: {pdf_path}, Events: {events}")
            return jsonify({'error': 'Failed to generate study plan'}), 500
        
        # Debug: Check if PDF file exists and has content
        if pdf_path and os.path.exists(pdf_path):
            file_size = os.path.getsize(pdf_path)
            print(f"Generated PDF: {pdf_path}, size: {file_size} bytes")
        else:
            print(f"PDF file not found or empty: {pdf_path}")
            return jsonify({'error': 'PDF file not generated correctly'}), 500
        
        # Return both PDF and events (PDF as file, events as JSON)
        # Option 1: Return JSON with download URL and events
        # Option 2: Return PDF directly (not suitable for frontend display + events)
        # We'll return JSON with a download URL and the events
        pdf_url = f"/api/download-studyplan/{os.path.basename(pdf_path)}"
        print(f"PDF URL: {pdf_url}")
        print(f"Returning {len(events)} events to frontend")
        print(f"Sample event structure: {events[0] if events else 'No events'}")
        return jsonify({'pdf_url': pdf_url, 'events': events})
    except Exception as e:
        print(f"StudyPlan Error: {e}")
        return jsonify({'error': str(e)}), 500
'''    
{
    "notebookId": "string",
    "title": "string",
    "semesterWeeks": 0,
    "syllabusContent": "string",
    "smartScheduling": false,
    "calendarType": "string",
    "calendarEmail": "string", 
    {
        "success": true,
        "events": [
            {
                "title": "string",
                "description": "string",
                "start_datetime": "string",
                "end_datetime": "string"
            }
        ],
        "event_count": 0,
        "filename": "string",
        "pdf_path": "string",
        "pdf_available": true
    }
}
'''

@app.route('/api/studyplan-upload', methods=['POST'])
@async_route
async def studyplan_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        os.makedirs('course_content', exist_ok=True)
        save_path = os.path.join('course_content', file.filename)
        file.save(save_path)
        # Use default 16 weeks for now
        result = await study_planner.create_study_plan_for_api(save_path, semester_weeks=16)
        print(f"Study plan upload result: {result}")
        if not result or not isinstance(result, tuple) or len(result) != 2:
            print(f"Invalid result format: {result}")
            return jsonify({'error': 'Failed to generate study plan'}), 500
        pdf_path, events = result
        print(f"PDF path: {pdf_path}")
        print(f"Events count: {len(events) if events else 0}")
        if not pdf_path or not events:
            print(f"Missing data - PDF path: {pdf_path}, Events: {events}")
            return jsonify({'error': 'Failed to generate study plan'}), 500
        pdf_url = f"/api/download-studyplan/{os.path.basename(pdf_path)}"
        print(f"PDF URL (upload): {pdf_url}")
        print(f"Returning {len(events)} events to frontend")
        return jsonify({'pdf_url': pdf_url, 'events': events})
    except Exception as e:
        print(f"StudyPlanUpload Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-studyplan/<filename>', methods=['GET'])
def download_studyplan(filename):
    try:
        file_path = os.path.join('study_plans', filename)
        print(f"Attempting to download file: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        # Check file size
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")
        
        if file_size == 0:
            print(f"File is empty: {file_path}")
            return jsonify({'error': 'File is empty'}), 500
        
        # Read the file content to verify it's a valid PDF
        try:
            with open(file_path, 'rb') as f:
                content = f.read(1024)  # Read first 1KB
                if not content.startswith(b'%PDF'):
                    print(f"File is not a valid PDF: {file_path}")
                    return jsonify({'error': 'File is not a valid PDF'}), 500
        except Exception as e:
            print(f"Error reading file: {e}")
            return jsonify({'error': 'Error reading file'}), 500
        
        # Set proper headers for PDF download
        response = send_file(
            file_path, 
            mimetype='application/pdf', 
            as_attachment=True, 
            download_name=filename
        )
        
        # Add additional headers for better compatibility
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        print(f"Successfully serving PDF: {filename}")
        return response
        
    except Exception as e:
        print(f"Download StudyPlan Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/studyplan/add-to-calendar', methods=['POST'])
@async_route
async def add_study_plan_to_calendar():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        events = data.get('events', [])
        calendar_type = data.get('calendar_type', 'gmail')  # Default to gmail
        email = data.get('email', '')
        
        if not events:
            return jsonify({'error': 'No events provided'}), 400
        
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        if calendar_type not in ['gmail', 'outlook']:
            return jsonify({'error': 'Invalid calendar type. Use "gmail" or "outlook"'}), 400
        
        # Authenticate with the calendar service
        auth_success = await study_planner._ensure_calendar_auth(calendar_type, email)
        if not auth_success:
            return jsonify({'error': f'Failed to authenticate with {calendar_type.title()} calendar'}), 401
        
        # Process and add events to calendar
        processed_calendar_events = []
        for event_template in events:
            try:
                if calendar_type == "gmail":
                    # Validate datetime format
                    datetime.datetime.fromisoformat(event_template["start_datetime"].replace("Z", "+00:00"))
                    datetime.datetime.fromisoformat(event_template["end_datetime"].replace("Z", "+00:00"))
                    calendar_event_input = {
                        "calendar_id": "primary",
                        "summary": event_template["summary"],
                        "description": event_template.get("description", ""),
                        "start_datetime": event_template["start_datetime"],
                        "end_datetime": event_template["end_datetime"],
                    }
                else:  # outlook
                    calendar_event_input = {
                        "subject": event_template["summary"],
                        "body": event_template.get("description", ""),
                        "start_date_time": event_template["start_datetime"].split(".")[0],
                        "end_date_time": event_template["end_datetime"].split(".")[0],
                    }
                processed_calendar_events.append(calendar_event_input)
            except KeyError as e:
                print(f"Skipping calendar event due to missing key: {e}. Event data: {event_template}")
            except ValueError as e:
                print(f"Skipping calendar event due to invalid datetime format: {e}. Event data: {event_template}")
        
        if not processed_calendar_events:
            return jsonify({'error': 'No valid events to add to calendar after processing'}), 400
        
        # Add events to calendar
        successful_adds = 0
        failed_events = []
        
        for event_data in processed_calendar_events:
            try:
                print(f"Adding event to {calendar_type.title()} Calendar: {event_data.get('summary', event_data.get('subject', ''))}")
                if await study_planner._add_event_to_calendar(event_data, calendar_type):
                    successful_adds += 1
                else:
                    failed_events.append(event_data.get('summary', event_data.get('subject', 'Unknown')))
            except Exception as e:
                print(f"Error adding event to calendar: {str(e)}")
                failed_events.append(event_data.get('summary', event_data.get('subject', 'Unknown')))
        
        return jsonify({
            'success': True,
            'message': f'Successfully added {successful_adds} out of {len(processed_calendar_events)} events to your {calendar_type.title()} Calendar.',
            'successful_adds': successful_adds,
            'total_events': len(processed_calendar_events),
            'failed_events': failed_events
        })
        
    except Exception as e:
        print(f"Add to Calendar Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Notebook management endpoints
@app.route('/api/notebooks', methods=['GET'])
def list_notebooks():
    """List all notebooks for a user"""
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        result = supabase.table('notebooks').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
        return jsonify({'notebooks': result.data})
    except Exception as e:
        print(f"Notebooks Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/notebooks', methods=['POST'])
def create_notebook():
    """Create a new notebook"""
    try:
        data = request.json
        user_id = data.get('user_id')
        name = data.get('name')
        description = data.get('description', '')
        color = data.get('color', '#4285f4')
        
        if not user_id or not name:
            return jsonify({'error': 'user_id and name are required'}), 400
        
        notebook_data = {
            'user_id': user_id,
            'name': name,
            'description': description,
            'color': color
        }
        
        result = supabase.table('notebooks').insert(notebook_data).execute()
        return jsonify({'notebook': result.data[0]})
    except Exception as e:
        print(f"Create Notebook Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/notebooks/<notebook_id>', methods=['DELETE'])
def delete_notebook(notebook_id):
    """Delete a notebook and all its documents"""
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Delete notebook (cascade will handle documents and chunks)
        result = supabase.table('notebooks').delete().eq('id', notebook_id).eq('user_id', user_id).execute()
        
        if not result.data:
            return jsonify({'error': 'Notebook not found or access denied'}), 404
        
        return jsonify({'message': 'Notebook deleted successfully'})
    except Exception as e:
        print(f"Delete Notebook Error: {e}")
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Backend is running'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
