"""
Med‑Bot AI Exam Generator
------------------------
Generates practice exams based on uploaded course content, sample exams, and study guides.

Core Features:

1. Example Exam & Study Guide Support
   - Upload example exams for question style reference
   - Include study guides for topic focus
   - Direct text input support for both
   - Automatic text extraction from various file formats

2. Exam Generation
   - Multiple difficulty levels (easy, medium, hard)
   - Configurable number of questions (1-20)
   - Balanced question types (multiple choice, short answer, problem-solving)
   - Topic-focused exam generation
   - PDF output with professional formatting

3. File Management
   - Automatic cleanup of temporary files
   - Organized directory structure
   - Support for multiple file formats
   - Efficient text extraction and processing

"""

import os
import asyncio
from typing import List, Dict, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import google.generativeai as genai
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
from rag_pipeline import RAGPipeline
from multi_hop_rag_pipeline import MultiHopRAGPipeline

from pathlib import Path
import shutil
import docx2txt
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER
from datetime import datetime
import re
import PyPDF2
import anthropic

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class ExamConfig(BaseModel):
    difficulty: Difficulty
    num_questions: int = Field(ge=1, le=20)
    topic: Optional[str] = None
    use_example_questions: bool = False
    use_study_guide: bool = False

class ExamGenerator:
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
        }
    }

    def __init__(self, api_key: str, claude_api_key: str = None):
        """Initialize the exam generator with API key."""
        self.current_model = "gemini-2.0-flash"
        self.api_key = api_key
        self.claude_api_key = claude_api_key or os.getenv("CLAUDE_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.0-flash")
        # Use MultiHopRAGPipeline by default; switch to RAGPipeline() for single-hop
        self.rag = MultiHopRAGPipeline()
        self.course_content_dir = Path("exam-feature/course_content")
        self.example_exams_dir = Path("exam-feature/example_exams")
        self.study_guides_dir = Path("exam-feature/study_guides")
        self.generated_exams_dir = Path("exam-feature/generated_exams")
        
        # Create necessary directories
        for directory in [self.course_content_dir, self.example_exams_dir, self.study_guides_dir, self.generated_exams_dir]:
            directory.mkdir(exist_ok=True)
        
        # Store processed example exam and study guide text
        self.example_exam_text: Optional[str] = None
        self.study_guide_text: Optional[str] = None

    def set_model(self, model_name: str) -> bool:
        if model_name in self.MODEL_REGISTRY:
            self.current_model = model_name
            if self.MODEL_REGISTRY[model_name]["provider"] == "google":
                self.model = genai.GenerativeModel(self.MODEL_REGISTRY[model_name]["model_id"])
            return True
        return False

    def get_available_models(self):
        return [
            {"name": k, "display_name": v["display_name"]}
            for k, v in self.MODEL_REGISTRY.items()
        ]

    def get_current_model(self):
        m = self.MODEL_REGISTRY[self.current_model]
        return {"name": self.current_model, "display_name": m["display_name"]}

    async def upload_course_content(self, file_path: str) -> bool:
        """Upload and process course content file (For RAG pipeline).
        
        Args:
            file_path: Path to the course content file
        """
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                print(f"Error: File {file_path} does not exist.")
                return False

            dest_dir = self.course_content_dir
            dest_path = dest_dir / source_path.name
            shutil.copy2(source_path, dest_path)
            
            if source_path.suffix.lower() == ".pdf":
                await self.rag.ingest_pdf(str(source_path))
                print(f"  Processed PDF: {source_path.name}")
            elif source_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                image = Image.open(dest_path)
                await self.rag.analyze_image(image)
                print(f"  Processed image: {source_path.name}")
            elif source_path.suffix.lower() == ".txt":
                with open(dest_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                await self.rag.ingest_txt(text)
                print(f"  Processed text file: {source_path.name}")
            elif source_path.suffix.lower() == ".docx":
                text = docx2txt.process(str(dest_path))
                await self.rag.ingest_txt(text)
                print(f"  Processed Word document: {source_path.name}")
            else:
                print(f"Unsupported file type: {source_path.suffix}")
                dest_path.unlink()
                return False
            return True

        except Exception as e:
            print(f"Error uploading course content: {str(e)}")
            if dest_path.exists():
                dest_path.unlink()
            return False

    #Optional: Upload Example Exam or Study Guide
    async def upload_extra_files(self, file_path: str = None, file_type: str = None, text_content: str = None) -> bool:
        """Upload and process example exam or study guide files.
        
        Args:
            file_path: Path to the file to upload (optional)
            file_type: Type of file ("example" or "study_guide")
            text_content: Direct text content to use (optional)
        """
        try:
            if text_content:
                # Handle direct text input
                if file_type == "example":
                    self.example_exam_text = text_content
                    print("  Processed example exam text input")
                elif file_type == "study_guide":
                    self.study_guide_text = text_content
                    print("  Processed study guide text input")
                else:
                    print(f"Invalid file type: {file_type}")
                    return False
                return True

            # Handle file upload
            source_path = Path(file_path)
            if not source_path.exists():
                print(f"Error: File {file_path} does not exist.")
                return False

            # Determine target directory and validate file types
            if file_type == "example":
                dest_dir = self.example_exams_dir
                valid_extensions = [".pdf", ".docx", ".txt"]
                if source_path.suffix.lower() not in valid_extensions:
                    print("Error: Example exam files must be PDF, DOCX, or TXT format.")
                    return False
                
                # Clean up any existing example exam
                for existing_file in dest_dir.glob("*"):
                    existing_file.unlink()
                
                # Copy file and process content
                dest_path = dest_dir / source_path.name
                shutil.copy2(source_path, dest_path)
                await self._process_example_exam(str(source_path))
                print(f"  Processed example exam: {source_path.name}")
                
            elif file_type == "study_guide":
                dest_dir = self.study_guides_dir
                valid_extensions = [".pdf", ".docx", ".txt"]
                if source_path.suffix.lower() not in valid_extensions:
                    print("Error: Study guide files must be PDF, DOCX, or TXT format.")
                    return False
                
                # Clean up any existing study guide
                for existing_file in dest_dir.glob("*"):
                    existing_file.unlink()
                
                # Copy file and process content
                dest_path = dest_dir / source_path.name
                shutil.copy2(source_path, dest_path)
                await self._process_study_guide(str(source_path))
                print(f"  Processed study guide: {source_path.name}")
            else:
                print(f"Invalid file type: {file_type}")
                return False
            
            return True

        except Exception as e:
            print(f"Error uploading {file_type} file: {str(e)}")
            if 'dest_path' in locals() and dest_path.exists():
                dest_path.unlink()
            return False

    async def _process_example_exam(self, file_path: str) -> None:
        """Process an example exam file to extract text."""
        try:
            if file_path.lower().endswith('.pdf'):
                text = self._extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.docx'):
                text = docx2txt.process(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            self.example_exam_text = text
            
        except Exception as e:
            print(f"Error processing example exam: {str(e)}")

    async def _process_study_guide(self, file_path: str) -> None:
        """Process a study guide file to extract text."""
        try:
            if file_path.lower().endswith('.pdf'):
                text = self._extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.docx'):
                text = docx2txt.process(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            self.study_guide_text = text
            
        except Exception as e:
            print(f"Error processing study guide: {str(e)}")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using PyPDF2."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    # Clean up the text
                    page_text = re.sub(r'\s+', ' ', page_text)
                    page_text = re.sub(r'\n+', '\n', page_text)
                    text += page_text + "\n"
                
            text = text.strip()
            # Ensure proper paragraph spacing
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""

    async def generate_exam(self, config: ExamConfig, content: str) -> str:
        """Generate a practice exam based on the configuration and content."""
        
        # Calculate number of questions for each type
        num_mc = int(config.num_questions * 0.4)  # 40% multiple choice
        num_short = int(config.num_questions * 0.3)  # 30% short answer
        num_problem = config.num_questions - num_mc - num_short  # Remaining for problem-solving
        
        # Prepare example questions and study guide topics if requested
        example_prompt = ""
        if config.use_example_questions and self.example_exam_text:
            example_prompt = f"""
            Use this example exam as inspiration for generating similar questions:
            {self.example_exam_text}
            
            Guidelines for question variation:
            1. Maintain ~70% similarity in style and difficulty
            2. For numerical questions, vary values by ±20%
            3. For conceptual questions, use different scenarios or examples
            4. Preserve the core concepts being tested
            5. Extract the question format and style from the examples
            6. For multiple choice questions, maintain similar option structure
            7. Keep similar length and complexity in answers
            8. Make sure the questions are accurate and relevant to the course material
            """
        
        study_guide_prompt = ""
        if config.use_study_guide and self.study_guide_text:
            study_guide_prompt = f"""
            Use this study guide to identify key topics and concepts to cover in the exam:
            {self.study_guide_text}
            
            Guidelines for topic coverage:
            1. Focus on the main concepts and topics from the study guide
            2. Ensure questions test understanding of these key areas
            3. Include both basic and advanced concepts mentioned
            4. Make sure questions are relevant to the study guide content
            5. If a topic seems unrelated to the course material, do not include it
            """
        
        prompt = f"""\
        You are an advanced AI exam generator.
        Your task is to create a {config.difficulty.value} difficulty exam with EXACTLY {config.num_questions} questions.
        
        Create a variety of question types that test different levels of understanding:
        1. Multiple Choice Questions: EXACTLY {num_mc} questions
           - Format: "Question X: [question text]"
           - Four options labeled A), B), C), D)
           - One correct answer
           - Test recall and basic understanding
        
        2. Short Answer Questions: EXACTLY {num_short} questions
           - Format: "Question X: [question text]"
           - Require 2-3 sentence answers
           - Test explanation ability and comprehension
        
        3. Problem-Solving Questions: EXACTLY {num_problem} questions
           - Format: "Question X: [question text]"
           - Require detailed analysis and application
           - Test critical thinking and problem-solving
        
        Guidelines:
        - Vary difficulty within the specified level
        - Include clear instructions for each section
        - Format each question type distinctly
        - Base all questions on this content:
        {content}
        
        {example_prompt}
        
        {study_guide_prompt}
        
        Format the exam with:
        - Clear section headers
        - Each question starting with "Question [number]"
        - Point values for each question
        
        Do NOT include answers in the main exam text.
        Do NOT reveal chain-of-thought; output only the final exam text.
        Do NOT generate more than {config.num_questions} questions total.
        """
        
        model_info = self.MODEL_REGISTRY[self.current_model]
        provider = model_info["provider"]
        model_id = model_info["model_id"]

        if provider == "google":
            model = genai.GenerativeModel(model_id)
            response = await asyncio.to_thread(model.generate_content, prompt)
            return response.text
        elif provider == "anthropic":
            if not self.claude_api_key:
                raise ValueError("Claude API key not set. Please set the CLAUDE_API_KEY environment variable.")
            client = anthropic.Anthropic(api_key=self.claude_api_key)
            response = await asyncio.to_thread(
                lambda: client.messages.create(
                    model=model_id,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return " ".join([block.text for block in response.content if hasattr(block, "text")]).strip()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    

    async def get_content(self, topic: Optional[str] = None) -> str:
        """Get relevant content from the RAG pipeline"""

        # Temporarily lower the similarity threshold for better matching
        original_threshold = self.rag.similarity_threshold
        self.rag.similarity_threshold = 0.5
        try:
            if topic:
                # Query for specific topic
                result = await self.rag.query(
                    question=f"What is {topic}? Explain {topic} in detail.",
                    top_k=4
                )
                if not result["chunks"]:
                    return ""
                return "\n\n".join(chunk.text for chunk in result["chunks"])
            else:
                # Get general content
                result = await self.rag.query(
                    question="What are the main concepts and topics covered in this course?",
                    top_k=5
                )
                if not result["chunks"]:
                    return ""
                return "\n\n".join(chunk.text for chunk in result["chunks"])
        finally:
            self.rag.similarity_threshold = original_threshold    


    def _generate_pdf(self, exam: str, config: ExamConfig) -> str:
        """Convert the exam text to a properly formatted PDF."""

        pdf_path = self.generated_exams_dir / f"exam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='CenteredTitle',
            parent=styles['Title'],
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        styles.add(ParagraphStyle(
            name='CenteredHeader',
            parent=styles['Heading1'],
            alignment=TA_CENTER,
            spaceAfter=12
        ))
        
        content = []
        
        content.extend([
            Paragraph("PRACTICE EXAM", styles['CenteredTitle']),
            Paragraph(f"Difficulty: {config.difficulty.value.upper()}", styles['CenteredHeader']),
            Paragraph(f"Questions: {config.num_questions}", styles['CenteredHeader']),
            Spacer(1, 20)
        ])
        
        # Process exam content
        for line in exam.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Section headers
            if line.startswith('##'):
                content.append(Paragraph(line[2:].strip(), styles['Heading2']))
                content.append(Spacer(1, 12))
            
            # Instructions
            elif line.startswith('**Instructions:'):
                content.append(Paragraph(line.replace('**', ''), styles['Normal']))
                content.append(Spacer(1, 12))
            
            # Questions
            elif line.startswith('**Question') or line.startswith('Question'):
                # Add extra space before each question
                content.append(Spacer(1, 12))
                # Split question number and text
                question_text = line.replace('**', '')
                if ':' in question_text:
                    question_num, question_content = question_text.split(':', 1)
                    content.append(Paragraph(f'<b>{question_num}:</b>{question_content}', styles['Normal']))
                else:
                    content.append(Paragraph(question_text, styles['Normal']))
            
            # Multiple choice options
            elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                content.append(Paragraph(line, styles['Normal']))
            
            # Handle other content
            else:
                content.append(Paragraph(line, styles['Normal']))
        doc.build(content)
        return str(pdf_path)

    async def generate_exam_with_config(self, difficulty: str, num_questions: int, topic: str = None) -> Tuple[str, ExamConfig]:
        """Generate an exam with the given configuration.
        
        Args:
            difficulty: Difficulty level ("easy", "medium", "hard")
            num_questions: Number of questions (1-20)
            topic: Optional topic to focus on
            
        Returns:
            Tuple of (exam text, exam config)
        """
        if not 1 <= num_questions <= 20:
            print("Invalid number of questions. Using default of 10.")
            num_questions = 10
            
        # Convert difficulty string to enum
        diff_map = {"easy": Difficulty.EASY, "medium": Difficulty.MEDIUM, "hard": Difficulty.HARD}
        difficulty_enum = diff_map.get(difficulty.lower(), Difficulty.MEDIUM)
        
        # Check if example exams and study guides exist
        example_files = [f.name for f in self.example_exams_dir.glob("*")]
        study_files = [f.name for f in self.study_guides_dir.glob("*")]
        
        use_examples = bool(example_files) or bool(self.example_exam_text)
        use_study_guide = bool(study_files) or bool(self.study_guide_text)
        
        # Check if we have content for the topic
        if topic:
            content = await self.get_content(topic)
            if not content:
                return None, None
        else:
            content = await self.get_content()
            if not content:
                return None, None
        
        config = ExamConfig(
            difficulty=difficulty_enum,
            num_questions=num_questions,
            topic=topic,
            use_example_questions=use_examples,
            use_study_guide=use_study_guide
        )
        
        # Generate exam using the content
        exam = await self.generate_exam(config, content)
        
        return exam, config
    

# ---------------------------------------------------------------------------
# Testing 
# ---------------------------------------------------------------------------


    def cleanup(self) -> None:
        """Remove all uploaded files."""
        try:
            # Clean up course content directory
            for file_path in self.course_content_dir.glob("*"):
                file_path.unlink()
            print("\nCleaned up course_content directory.")
            
            # Clean up example exams directory
            for file_path in self.example_exams_dir.glob("*"):
                file_path.unlink()
            print("Cleaned up example_exams directory.")
            
            # Clean up study guides directory
            for file_path in self.study_guides_dir.glob("*"):
                file_path.unlink()
            print("Cleaned up study_guides directory.")
            
            # Clear stored questions and topics
            self.example_exam_text = None
            self.study_guide_text = None
            
        except Exception as e:
            print(f"\nError during cleanup: {str(e)}")

    async def generate_answer_key(self, exam_text: str) -> str:
        """Generate an answer key for the given exam text."""
        prompt = f"""You are an expert exam grader.\nGiven the following exam, generate a clear answer key.\nFor multiple choice, indicate the correct option (A, B, C, or D) and a brief explanation if possible.\nFor short answer and problem-solving, provide concise model answers.\n\nExam:\n{exam_text}\n\nFormat the answer key clearly, with question numbers matching the exam.\nDo NOT include the questions themselves, only the answers.\n"""
        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt
        )
        return response.text

async def main():
    """Interactive exam generator interface."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable")
    
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    generator = ExamGenerator(api_key, claude_api_key)
    
    print("\nWelcome to Med-Bot AI Exam Generator!")
    print("-------------------------------------")
    
    while True:
        print("\nOptions:")
        print("1. Upload course content")
        print("2. Upload example exam")
        print("3. Upload study guide")
        print("4. List uploaded files")
        print("5. Generate exam")
        print("6. Exit")
        print("models: List available models")
        print("model <model_name>: Switch to a specific model")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice.lower() == "models":
            print("\nAvailable models:")
            for m in generator.get_available_models():
                print(f"  {m['name']}: {m['display_name']}")
            current = generator.get_current_model()
            print(f"\nCurrent model: {current['name']} ({current['display_name']})")
        elif choice.lower().startswith("model "):
            model_name = choice[6:].strip()
            if generator.set_model(model_name):
                current = generator.get_current_model()
                print(f"Switched to model: {current['name']} ({current['display_name']})")
            else:
                print(f"Model '{model_name}' not found. Use 'models' to list available models.")

        #Upload Course Content
        if choice == "1":
            file_path = input("\nEnter the path to your course content file: ").strip()
            if await generator.upload_course_content(file_path):
                print("Course content uploaded successfully!")
            else:
                print("Failed to upload course content.")
        
        #Upload or Paste Sample Exam
        elif choice == "2":
            print("\nHow would you like to provide the example exam?")
            print("1. Upload a file")
            print("2. Paste text directly")
            sub_choice = input("Enter choice (1-2): ").strip()
            
            if sub_choice == "1":
                file_path = input("\nEnter the path to your example exam file: ").strip()
                if await generator.upload_extra_files(file_path, "example"):
                    print("Example exam uploaded successfully!")
                else:
                    print("Failed to upload example exam.")
            else:
                print("\nPaste your example exam text (press Ctrl+D or Ctrl+Z when done):")
                text_content = ""
                try:
                    while True:
                        line = input()
                        text_content += line + "\n"
                except EOFError:
                    pass
                if await generator.upload_extra_files(text_content=text_content, file_type="example"):
                    print("Example exam text processed successfully!")
                else:
                    print("Failed to process example exam text.")
        
        #Upload or Paste Study Guide
        elif choice == "3":
            print("\nHow would you like to provide the study guide?")
            print("1. Upload a file")
            print("2. Paste text directly")
            sub_choice = input("Enter choice (1-2): ").strip()
            
            if sub_choice == "1":
                file_path = input("\nEnter the path to your study guide file: ").strip()
                if await generator.upload_extra_files(file_path, "study_guide"):
                    print("Study guide uploaded successfully!")
                else:
                    print("Failed to upload study guide.")
            else:
                print("\nPaste your study guide text (press Ctrl+D or Ctrl+Z when done):")
                text_content = ""
                try:
                    while True:
                        line = input()
                        text_content += line + "\n"
                except EOFError:
                    pass
                if await generator.upload_extra_files(text_content=text_content, file_type="study_guide"):
                    print("Study guide text processed successfully!")
                else:
                    print("Failed to process study guide text.")
                    
        #List All Uploaded Files
        elif choice == "4":
            print("\nCourse Content Files:")
            course_files = [f.name for f in generator.course_content_dir.glob("*")]
            if course_files:
                for f in course_files:
                    print(f"  • {f}")
            else:
                print("  No course content files uploaded yet.")
            
            print("\nExample Exam Files:")
            example_files = [f.name for f in generator.example_exams_dir.glob("*")]
            if example_files:
                for f in example_files:
                    print(f"  • {f}")
            else:
                print("  No example exam files uploaded yet.")
            
            print("\nStudy Guide Files:")
            study_files = [f.name for f in generator.study_guides_dir.glob("*")]
            if study_files:
                for f in study_files:
                    print(f"  • {f}")
            else:
                print("  No study guide files uploaded yet.")
        
        #Generate Exam
        elif choice == "5":
            print("\nSelect difficulty:")
            print("1. Easy")
            print("2. Medium")
            print("3. Hard")
            diff_choice = input("Enter choice (1-3): ").strip()
            difficulty = "easy" if diff_choice == "1" else "medium" if diff_choice == "2" else "hard"
            
            num_questions = int(input("\nEnter number of questions (1-20): ").strip())
            topic = input("\nEnter topic to cover (or press Enter for all topics): ").strip()
            topic = topic if topic else None
            
            exam, config = await generator.generate_exam_with_config(difficulty, num_questions, topic)
            if exam:
                # Format and print exam with better spacing
                print("\n" + "="*50)
                print(f"PRACTICE EXAM")
                print(f"Difficulty: {config.difficulty.value.upper()}")
                print(f"Number of Questions: {config.num_questions}")
                if topic:
                    print(f"Topic: {topic}")
                if config.use_example_questions:
                    print("Using example questions as inspiration")
                if config.use_study_guide:
                    print("Focusing on study guide topics")
                print("="*50 + "\n")
                
                exam_lines = exam.split('\n')
                for line in exam_lines:
                    line = line.strip()
                    if not line:
                        print()  # Add extra line for empty lines
                    elif line.startswith('##'):
                        print(f"\n{line[2:].strip()}")
                        print("-" * len(line[2:].strip()))
                    elif line.startswith('**Instructions:'):
                        print(f"\n{line.replace('**', '')}")
                    elif line.startswith('**Question') or line.startswith('Question'):
                        print(f"\n{line.replace('**', '')}")
                    elif line.startswith(('A)', 'B)', 'C)', 'D)')):
                        print(f"    {line}")  # Indent options
                    else:
                        print(line)
                
                print("\n" + "="*50)
                
                # Ask if user wants to download PDF
                pdf_choice = input("\nWould you like to download a PDF version of this exam? (y/n): ").strip().lower()
                if pdf_choice == 'y':
                    pdf_path = generator._generate_pdf(exam, config)
                    print(f"\nPDF exam has been generated: {pdf_path}")
                
                # Ask if user wants an answer key
                answer_key_choice = input("\nWould you like to generate an answer key for this exam? (y/n): ").strip().lower()
                if answer_key_choice == 'y':
                    answer_key = await generator.generate_answer_key(exam)
                    print("\n" + "="*50)
                    print("ANSWER KEY")
                    print("="*50 + "\n")
                    print(answer_key)
                    # Optionally, ask if they want a PDF
                    pdf_key_choice = input("\nWould you like to download a PDF version of the answer key? (y/n): ").strip().lower()
                    if pdf_key_choice == 'y':
                        key_pdf_path = generator._generate_pdf(answer_key, config)
                        print(f"\nPDF answer key has been generated: {key_pdf_path}")
                
                # Ask if user wants to generate another exam
                another = input("\nWould you like to generate another exam? (y/n): ").strip().lower()
                if another != 'y':
                    break
        
        #Exit
        elif choice == "6":
            generator.cleanup()
            return
        
        else:
            print("Invalid choice. Please try again.")
    
    # Clean up after we're done
    generator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
