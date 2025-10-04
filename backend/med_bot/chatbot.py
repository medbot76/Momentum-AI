"""
Med‑Bot AI Chatbot
-----------------
A document-aware chatbot that can process PDFs, images, and text files and answers questions given the content. 

Quick Start
----------
1. Set your Gemini API key:
   $ export GEMINI_API_KEY="your-api-key-here"

2. Place your test documents in the test_docs directory:
   $ mkdir -p test_docs
   $ cp your_document.pdf test_docs/

3. Run the chatbot:
   $ python chatbot.py

4. Available commands:
   - upload <file_path>  : Upload and process a PDF, image, or text file
   - list               : List all uploaded documents
   - ask <question>     : Ask a question about your documents
   - voice              : Ask a question by speaking
   - exit              : Exit the chatbot

Example Usage
------------
$ python chatbot.py
> upload biology_notes.pdf
> ask "What is the Krebs cycle?"
> upload cell_division.png
> ask "Explain the stages of mitosis"
"""

import os
import asyncio
import shutil
from typing import Optional, List, Union
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from med_bot.rag_pipeline import RAGPipeline
from med_bot.multi_hop_rag_pipeline import MultiHopRAGPipeline
import docx2txt
from arcadepy import Arcade
import speech_recognition as sr
import queue
import threading
import time
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import re
import openai
import tempfile
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False
    print("Warning: playsound not available. Audio playback will be disabled.")
import anthropic
import multiprocessing

def _play_audio(audio_path):
    if PLAYSOUND_AVAILABLE:
        try:
            from playsound import playsound
            playsound(audio_path)
        except Exception as e:
            print(f"Error playing audio: {e}")
    else:
        print("Audio playback not available (playsound not installed)")

class ChatBot:
    # Supported models registry
    MODEL_REGISTRY = {
        "gemini-2.0-pro": {
            "provider": "google",
            "display_name": "Gemini 2.0 Pro",
            "model_id": "models/gemini-2.0-pro"
        },
        "gemini-2.0-flash": {
            "provider": "google",
            "display_name": "Gemini 2.0 Flash",
            "model_id": "models/gemini-2.0-flash"
        },
        "claude-3.5-sonnet": {
            "provider": "anthropic",
            "display_name": "Claude 3.5 Sonnet",
            "model_id": "claude-3.5-sonnet-20240620"
        },
        "claude-3.7-sonnet": {
            "provider": "anthropic",
            "display_name": "Claude 3.7 Sonnet",
            "model_id": "claude-3.7-sonnet-20240620"
        }
    }

    def __init__(self, api_key: str, claude_api_key: str = None):
        # Default to Gemini 2.0 Flash
        self.current_model = "gemini-2.0-flash"
        self.api_key = api_key
        self.claude_api_key = claude_api_key or os.getenv("CLAUDE_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.0-flash")
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # Initialize RAG pipeline
        # Use MultiHopRAGPipeline by default; switch to RAGPipeline() for single-hop
        self.rag = MultiHopRAGPipeline()

        # Arcade client for YouTube search
        self.arcade_client = Arcade()
        self._youtube_authed = False  # Track YouTube tool authorization

        # Create course_content directory if it doesn't exist
        self.course_content_dir = Path("course_content")
        self.course_content_dir.mkdir(exist_ok=True)

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()

        # Conversation history for prompt chaining
        self.conversation_history = []
        self.MAX_HISTORY = 2
        # Track last main topic for video search context
        self.last_main_topic = None

    # --- Keyword extraction and follow-up detection helpers ---
    STOPWORDS = set([
        "the", "is", "to", "me", "in", "of", "a", "an", "and", "or", "for", "on", "with", "at", "by", "from", "up", "about", "into", "over", "after", "it", "as", "but", "be", "if", "so", "than", "then", "too", "very", "can", "will", "just", "should", "could", "would", "explain", "please"
    ])

    def extract_keywords(self, query):
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in self.STOPWORDS]
        return " ".join(keywords)

    def is_followup_query(self, query):
        # Use conversation history to improve follow-up detection
        if not self.conversation_history:
            return False

        followup_words = {"it", "this", "that", "them", "those", "these", "its", "their", "his", "her"}
        generic_phrases = [
            "what about", "explain more", "explain further", "and then", "next step", "continue", "more details", "what else", "what next", "steps", "and", "then"
        ]
        words = set(re.findall(r'\b\w+\b', query.lower()))
        q = query.lower().strip()

        # 1. Pronoun-based
        if words & followup_words:
            return True

        # 2. Generic phrase-based
        for phrase in generic_phrases:
            if phrase in q:
                return True

        # 3. Short question after a previous question
        if len(q.split()) <= 4:
            return True

        # 4. Keyword overlap with last question
        last_question = self.conversation_history[-1]['question']
        query_keywords = set(self.extract_keywords(query).split())
        last_keywords = set(self.extract_keywords(last_question).split())
        if query_keywords and last_keywords:
            overlap = len(query_keywords & last_keywords) / len(query_keywords)
            if overlap > 0.5:
                return True

        return False

    async def upload_document(self, file_path: str, notebook_id: str = "default") -> bool:
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                print(f"Error: File {file_path} does not exist.")
                return False
            dest_path = self.course_content_dir / source_path.name
            shutil.copy2(source_path, dest_path)

            # Process the uploaded file
            if source_path.suffix.lower() == ".pdf":
                try:
                    await self.rag.ingest_pdf(str(source_path), notebook_id=notebook_id)
                    print(f"  Processed PDF: {source_path.name}")
                except Exception as e:
                    print(f"Error processing PDF {source_path.name}: {str(e)}")
                    dest_path.unlink()
                    return False
            
            elif source_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                try:
                    image = Image.open(dest_path)
                    await self.rag.analyze_image(image, notebook_id=notebook_id)
                    print(f"Processed image: {source_path.name}")
                except Exception as e:
                    print(f"Error processing image {source_path.name}: {str(e)}")
                    dest_path.unlink()
                    return False
            
            elif source_path.suffix.lower() == ".txt":
                try:
                    with open(dest_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    await self.rag.ingest_txt(text, notebook_id=notebook_id)
                    print(f"Processed text file: {source_path.name}")
                except Exception as e:
                    print(f"Error processing text file {source_path.name}: {str(e)}")
                    dest_path.unlink()
                    return False
            
            elif source_path.suffix.lower() == ".docx":
                try:
                    # Convert docx to text
                    text = docx2txt.process(str(dest_path))
                    # Process the extracted text
                    await self.rag.ingest_txt(text, notebook_id=notebook_id)
                    print(f"Processed Word document: {source_path.name}")
                except Exception as e:
                    print(f"Error processing Word document {source_path.name}: {str(e)}")
                    dest_path.unlink()
                    return False
            
            else:
                print(f"Unsupported file type: {source_path.suffix}")
                dest_path.unlink()
                return False
            return True
        except Exception as e:
            print(f"Error uploading document: {str(e)}")
            if dest_path.exists():
                dest_path.unlink()
            if dest_path.exists():
                dest_path.unlink()
            return False

    async def _call_llm(self, question: str, context: str = "", conversation_history: str = "") -> str:
        model_info = self.MODEL_REGISTRY[self.current_model]
        provider = model_info["provider"]
        model_id = model_info["model_id"]
        prompt = (
            "You are a professional, helpful and knowledgeable educator powered by advanced language models. "
            + (f"Here is the conversation so far:\n{conversation_history}\n\n" if conversation_history else "")
            + ("Using only the CONTEXT provided, " if context else "")
            + "answer the QUESTION in a clear, engaging, and accurate way. Break down complex "
            "concepts into simpler terms. Maintain accuracy at all times. \n\n"
            + (f"CONTEXT:\n{context}\n\n" if context else "")
            + f"QUESTION: {question}\n\n"
            "Please structure your answer with the following format but don't mention it in your answer:\n"
            "1. A clear, concise main answer\n"
            "2. Simple explanations of key concepts\n"
            "3. Relevant examples or explanations\n"
            + ("4. Citations to specific pages where the information comes from\n\n" if context else "\n")
            + "ANSWER:"
        )
        print(f"[DEBUG] Using model: {self.current_model}")
        if provider == "google":
            model = genai.GenerativeModel(model_id)
            resp = await asyncio.to_thread(model.generate_content, prompt)
            return resp.text.strip()
        elif provider == "anthropic":
            if not self.claude_api_key:
                raise ValueError("Claude API key not set. Please set the CLAUDE_API_KEY environment variable.")
            client = anthropic.Anthropic(api_key=self.claude_api_key)
            # Claude expects a list of messages (system, user, etc.)
            # We'll use the prompt as a single user message
            response = await asyncio.to_thread(
                lambda: client.messages.create(
                    model=model_id,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            # Claude's response is in response.content (a list of content blocks)
            if hasattr(response, "content") and response.content:
                # Join all text blocks
                return " ".join([block.text for block in response.content if hasattr(block, "text")]).strip()
            else:
                return "[No response from Claude]"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _call_voicechat_llm(self, question: str, context: str = "", conversation_history: str = "") -> str:
        model_info = self.MODEL_REGISTRY[self.current_model]
        provider = model_info["provider"]
        model_id = model_info["model_id"]
        prompt = (
            "You are a professional, helpful, and friendly educator powered by advanced language models. "
            + (f"Here is the conversation so far:\n{conversation_history}\n\n" if conversation_history else "")
            + ("Using only the CONTEXT provided, " if context else "")
            + "answer the QUESTION in a clear, natural, and conversational way, as if you are speaking to the user. "
            "Do not include references, citations, or complicated punctuation. Avoid lists, bullet points, or formatting that is awkward when spoken. "
            "Make your answer fluent, easy to follow, and engaging when read out loud.\n\n"
            + (f"CONTEXT:\n{context}\n\n" if context else "")
            + f"QUESTION: {question}\n\n"
            "Please provide only the answer, in a way that sounds natural when spoken."
        )
        print(f"[DEBUG] Using model: {self.current_model} (voicechat mode)")
        if provider == "google":
            model = genai.GenerativeModel(model_id)
            resp = await asyncio.to_thread(model.generate_content, prompt)
            return resp.text.strip()
        elif provider == "anthropic":
            if not self.claude_api_key:
                raise ValueError("Claude API key not set. Please set the CLAUDE_API_KEY environment variable.")
            client = anthropic.Anthropic(api_key=self.claude_api_key)
            response = await asyncio.to_thread(
                lambda: client.messages.create(
                    model=model_id,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            if hasattr(response, "content") and response.content:
                return " ".join([block.text for block in response.content if hasattr(block, "text")]).strip()
            else:
                return "[No response from Claude]"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def _search_youtube_videos(self, keywords: str, max_results: int = 3):
        """Search YouTube for videos related to the keywords using Arcade.dev. No authorization required."""
        try:
            tool_response = self.arcade_client.tools.execute(
                tool_name="Search.SearchYoutubeVideos",
                input={"keywords": keywords,
                "language_code": "en"}
            )
            
            vids = tool_response.output.value['videos']
            links = []
            for i in range(min(max_results, len(vids))):
                links.append(vids[i]['link'])

            return links
        except Exception as e:
            print(f"Error searching YouTube videos: {str(e)}")
            return []

    async def _get_youtube_transcript(self, video_id: str):
        """Fetch the transcript for a YouTube video. Returns list of transcript entries or None if unavailable."""
        try:
            # This returns a list of dicts with 'text', 'start', 'duration'
            transcript = await asyncio.to_thread(YouTubeTranscriptApi.get_transcript, video_id)
            return transcript
        except NoTranscriptFound:
            return None
        except Exception as e:
            print(f"Error fetching transcript for video {video_id}: {str(e)}")
            return None

    def _chunk_transcript(self, transcript, max_seconds=15, max_words=30):
        """Chunk transcript into segments of up to max_seconds or max_words."""
        chunks = []
        current_chunk = []
        start_time = None
        for entry in transcript:
            if not current_chunk:
                start_time = entry['start']
            current_chunk.append(entry)
            duration = entry['start'] + entry['duration'] - start_time
            words = sum(len(e['text'].split()) for e in current_chunk)
            if duration >= max_seconds or words >= max_words:
                end_time = entry['start'] + entry['duration']
                text = " ".join(e['text'] for e in current_chunk)
                chunks.append({'start': start_time, 'end': end_time, 'text': text})
                current_chunk = []
        # Add any remaining chunk
        if current_chunk:
            end_time = current_chunk[-1]['start'] + current_chunk[-1]['duration']
            text = " ".join(e['text'] for e in current_chunk)
            chunks.append({'start': start_time, 'end': end_time, 'text': text})
        return chunks

    async def _process_single_video(self, link: str, question: str):
        """Process a single YouTube video in parallel."""
        # Extract video ID
        if "v=" in link:
            video_id = link.split("v=")[1].split("&")[0]
        elif "youtu.be/" in link:
            video_id = link.split("youtu.be/")[1].split("?")[0]
        else:
            video_id = None
            
        if not video_id:
            return {
                "link": link,
                "thumbnail": None,
                "timestamp": None,
                "transcript_snippet": None
            }
            
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        transcript = await self._get_youtube_transcript(video_id)
        
        if transcript:
            chunks = self._chunk_transcript(transcript)
            best_chunk = await self._find_relevant_transcript_chunk(chunks, question)
            
            if best_chunk:
                def fmt_time(seconds):
                    m, s = divmod(int(seconds), 60)
                    return f"{m:02d}:{s:02d}"
                timestamp = {
                    "start": fmt_time(best_chunk["start"]),
                    "end": fmt_time(best_chunk["end"])
                }
                snippet = best_chunk["text"]
            else:
                timestamp = None
                snippet = None
        else:
            timestamp = None
            snippet = None
            
        return {
            "link": link,
            "thumbnail": thumbnail_url,
            "timestamp": timestamp,
            "transcript_snippet": snippet
        }

    async def _find_relevant_transcript_chunk(self, chunks, question, similarity_threshold=0.3, max_chunks_to_embed=5):
        """Find the most relevant transcript chunk using faster keyword matching first, then embeddings (limited to top N chunks)."""
        # First try simple keyword matching for speed
        question_words = set(question.lower().split())
        scored_chunks = []
        for chunk in chunks:
            chunk_words = set(chunk["text"].lower().split())
            # Simple word overlap score
            overlap = len(question_words.intersection(chunk_words)) / (len(question_words) or 1)
            scored_chunks.append((overlap, chunk))
        # Sort by overlap
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        # If a chunk has high enough overlap, return it immediately
        if scored_chunks and scored_chunks[0][0] > 0.2:
            return scored_chunks[0][1]
        # Otherwise, embed only the top N chunks
        top_chunks = [chunk for _, chunk in scored_chunks[:max_chunks_to_embed]]
        q_embed = await asyncio.to_thread(
            lambda: genai.embed_content(
                model="models/embedding-001",
                content=question,
                task_type="retrieval_document",
            )["embedding"]
        )
        best_chunk = None
        best_score = -1
        for chunk in top_chunks:
            chunk_embed = await asyncio.to_thread(
                lambda: genai.embed_content(
                    model="models/embedding-001",
                    content=chunk["text"],
                    task_type="retrieval_document",
                )["embedding"]
            )
            sim = np.dot(q_embed, chunk_embed) / (np.linalg.norm(q_embed) * np.linalg.norm(chunk_embed))
            if sim > best_score and sim > similarity_threshold:
                best_score = sim
                best_chunk = chunk
        return best_chunk

    async def ask_question(self, question: str, notebook_id: str = "default", voicechat: bool = False) -> dict:
        try:
            # Get answer using RAG pipeline
            rag_result = await self.rag.query(question=question, notebook_id=notebook_id)
            if rag_result["chunks"]:
                context_parts = []
                for chunk in rag_result["chunks"]:
                    if chunk.metadata.get("type") == "image":
                        context_parts.append(f"[Image Analysis]\n{chunk.text}")
                    else:
                        context_parts.append(f"[PDF Content - Page {chunk.metadata.get('page_end', 'N/A')}]\n{chunk.text}")
                context = "\n\n".join(context_parts)
            else:
                print("\nNo relevant information found in your uploaded documents. Using general knowledge to answer...")
                context = ""

            # Build conversation history string
            conversation_history = "\n".join([
                f"Q: {entry['question']}\nA: {entry['answer']}" 
                for entry in self.conversation_history
            ])

            # --- Video search context logic ---
            is_followup = self.is_followup_query(question)
            if not is_followup:
                video_search_query = self.extract_keywords(question)
                self.last_main_topic = question
            else:
                video_search_query = None

            if voicechat:
                llm_task = asyncio.create_task(self._call_voicechat_llm(question, context, conversation_history))
            else:
                llm_task = asyncio.create_task(self._call_llm(question, context, conversation_history))

            if video_search_query:
                youtube_links = await self._search_youtube_videos(video_search_query)
                video_tasks = [self._process_single_video(link, question) for link in youtube_links[:3]]
                video_results_task = asyncio.gather(*video_tasks, return_exceptions=True)
            else:
                video_results_task = asyncio.gather(*[], return_exceptions=True)

            answer, video_results = await asyncio.gather(llm_task, video_results_task)
            video_results = [result for result in video_results if not isinstance(result, Exception)]
            # Update conversation history
            self.conversation_history.append({"question": question, "answer": answer})
            if len(self.conversation_history) > self.MAX_HISTORY:
                self.conversation_history = self.conversation_history[-self.MAX_HISTORY:]

            return {
                "answer": answer,
                "videos": video_results
            }
        except Exception as e:
            return {"error": f"Error generating response: {str(e)}"}

    def listen_to_question(self) -> str:
        """Listen to user's voice input and convert to text with manual stop option."""
        with sr.Microphone() as source:
            print("\nListening... (speak your question)")
            print("Type 'stop' and press Enter when you're done speaking")
            print("(Recording will automatically stop after 60 seconds if not stopped manually)")
            
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source)
            
            # Create a queue for the audio data
            audio_queue = queue.Queue()
            stop_recording = threading.Event()
            
            def record_audio():
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=10,  # 10 seconds to start speaking
                        phrase_time_limit=60  # 60 seconds maximum recording time
                    )
                    audio_queue.put(audio)
                except sr.WaitTimeoutError:
                    print("No speech detected within 10 seconds")
                except Exception as e:
                    print(f"Error during recording: {e}")
                finally:
                    stop_recording.set()
            
            # Start recording in a separate thread
            record_thread = threading.Thread(target=record_audio)
            record_thread.start()
            
            # Wait for either 'stop' command or timeout
            while not stop_recording.is_set():
                if input() == "stop":
                    stop_recording.set()
                    break
                time.sleep(0.1)  # Small delay to prevent high CPU usage
            
            # Wait for the recording thread to finish
            record_thread.join()
            
            try:
                # Get the audio from the queue
                audio = audio_queue.get_nowait()
                print("Processing speech...")
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except queue.Empty:
                print("No speech was recorded")
                return ""
            except sr.UnknownValueError:
                print("Could not understand audio")
                return ""
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return ""

    def speak_text(self, text: str, voice: str = "nova"):
        audio_path = None
        try:
            if not openai.api_key:
                print("Cannot generate speech, OPENAI_API_KEY is not set.")
                return

            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,  # Options: "alloy", "echo", "fable", "onyx", "nova", "shimmer"
                input=text
            )

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(response.content)
                audio_path = tmp_file.name

            if PLAYSOUND_AVAILABLE:
                print(f"\n🔊 Playing response... (type 'stop' and press Enter to stop playback)")
                p = multiprocessing.Process(target=_play_audio, args=(audio_path,))
                p.start()

                while p.is_alive():
                    user_input = input()
                    if user_input.strip().lower() == "stop":
                        p.terminate()
                        print("Playback stopped.")
                        break
                p.join()
            else:
                print(f"\n🔊 Audio generated and saved to: {audio_path}")
                print("Audio playback not available (playsound not installed)")

        except Exception as e:
            print(f"Error generating speech: {e}")
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

    async def voice_ask(self) -> None:
        """Handle voice input for questions."""
        question = self.listen_to_question()
        if not question:
            print("No question detected. Please try again.")
            return
        
        print("\nThinking...")
        result = await self.ask_question(question, voicechat=True)
        if "error" in result:
            print(result["error"])
            return
        
        answer_text = result['answer']
        print(f"\nGemini Answer:\n{answer_text}")

        # Speak the text
        if openai.api_key:
            self.speak_text(answer_text)
        else:
            print("\n(Skipping voice output: OPENAI_API_KEY not set)")

        if result["videos"]:
            print("\nRelated YouTube Videos:")
            for idx, vid in enumerate(result["videos"], 1):
                print(f"\nVideo {idx}:")
                print(f"  Link: {vid['link']}")
                print(f"  Thumbnail: {vid['thumbnail']}")
                if vid["timestamp"]:
                    print(f"  Timestamp: {vid['timestamp']['start']} - {vid['timestamp']['end']}")
                if vid["transcript_snippet"]:
                    print(f"  Transcript Snippet: {vid['transcript_snippet'][:200]}{'...' if len(vid['transcript_snippet']) > 200 else ''}")
                if not vid["timestamp"]:
                    print("  No relevant transcript segment found or transcript unavailable.")
        else:
            print("No related YouTube videos found.")

    def list_documents(self) -> List[str]:
        return [f.name for f in self.course_content_dir.glob("*")]

    def cleanup(self) -> None:
        """Remove all files from the course_content directory."""
        try:
            for file_path in self.course_content_dir.glob("*"):
                file_path.unlink()
            print("\nCleaned up course_content directory.")
        except Exception as e:
            print(f"\nError during cleanup: {str(e)}")

    def set_model(self, model_name: str) -> bool:
        """Switch the current LLM model."""
        if model_name in self.MODEL_REGISTRY:
            self.current_model = model_name
            # For Gemini, update the model instance
            if self.MODEL_REGISTRY[model_name]["provider"] == "google":
                self.model = genai.GenerativeModel(self.MODEL_REGISTRY[model_name]["model_id"])
            return True
        return False

    def get_available_models(self):
        """Return a list of available model names and display names."""
        return [
            {"name": k, "display_name": v["display_name"]}
            for k, v in self.MODEL_REGISTRY.items()
        ]

    def get_current_model(self):
        """Return the current model name and display name."""
        m = self.MODEL_REGISTRY[self.current_model]
        return {"name": self.current_model, "display_name": m["display_name"]}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

async def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set. Text-to-speech will be disabled.")
    
    if not os.getenv("ARCADE_API_KEY"):
        print("Warning: ARCADE_API_KEY environment variable not set. YouTube search may fail.")
        
    chatbot = ChatBot(api_key)
    print("\nWelcome to Med-Bot AI Chatbot (Optimized Version)!")
    print("------------------------------------------------")
    print("Commands:")
    print("  upload <file_path> - Upload and process a document")
    print("  list - List all uploaded documents")
    print("  ask <question> - Ask a question")
    print("  voice - Ask a question by speaking")
    print("  models - List available models")
    print("  model <model_name> - Switch to a specific model")
    print("  exit - Exit the chatbot")
    print("--------------------------------\n")
    while True:
        try:
            user_input = input("\nEnter command: ").strip()
            if user_input.lower() == "exit":
                print("Goodbye!")
                chatbot.cleanup()  # Clean up before exiting
                break
            elif user_input.lower() == "list":
                documents = chatbot.list_documents()
                if documents:
                    print("\nUploaded documents:")
                    for doc in documents:
                        print(f"  • {doc}")
                else:
                    print("No documents uploaded yet.")
            elif user_input.lower().startswith("upload "):
                file_path = user_input[7:].strip()
                if await chatbot.upload_document(file_path):
                    print("  Document uploaded and processed successfully!")
                else:
                    print("Failed to upload document.")
            elif user_input.lower().startswith("ask "):
                question = user_input[4:].strip()
                if question:
                    print("\nThinking...")
                    result = await chatbot.ask_question(question)
                    if "error" in result:
                        print(result["error"])
                    else:
                        print(f"\nGemini Answer:\n{result['answer']}")
                        if result["videos"]:
                            print("\nRelated YouTube Videos:")
                            for idx, vid in enumerate(result["videos"], 1):
                                print(f"\nVideo {idx}:")
                                print(f"  Link: {vid['link']}")
                                print(f"  Thumbnail: {vid['thumbnail']}")
                                if vid["timestamp"]:
                                    print(f"  Timestamp: {vid['timestamp']['start']} - {vid['timestamp']['end']}")
                                if vid["transcript_snippet"]:
                                    print(f"  Transcript Snippet: {vid['transcript_snippet'][:200]}{'...' if len(vid['transcript_snippet']) > 200 else ''}")
                                if not vid["timestamp"]:
                                    print("  No relevant transcript segment found or transcript unavailable.")
                        else:
                            print("No related YouTube videos found.")
                else:
                    print("Please provide a question.")
            elif user_input.lower() == "voice":
                await chatbot.voice_ask()
            elif user_input.lower() == "models":
                print("\nAvailable models:")
                for m in chatbot.get_available_models():
                    print(f"  {m['name']}: {m['display_name']}")
                current = chatbot.get_current_model()
                print(f"\nCurrent model: {current['name']} ({current['display_name']})")
            elif user_input.lower().startswith("model "):
                model_name = user_input[6:].strip()
                if chatbot.set_model(model_name):
                    current = chatbot.get_current_model()
                    print(f"Switched to model: {current['name']} ({current['display_name']})")
                else:
                    print(f"Model '{model_name}' not found. Use 'models' to list available models.")
            else:
                print("Unknown command. Available commands: upload, list, ask, voice, models, model <model_name>, exit")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 