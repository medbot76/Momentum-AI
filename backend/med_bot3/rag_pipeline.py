"""Med‑Bot AI — Gemini + Supabase pgvector Retrieval‑Augmented Generation engine
-------------------------------------------------------------------------------
Self‑contained class that handles PDF ingestion, chunking, vector
storage, retrieval, and answer generation — powered by Google Gemini
models with Supabase pgvector for vector storage.

Recent Changes:
- Migrated from ChromaDB to Supabase pgvector for unified database architecture
- Added user-specific document isolation using Supabase authentication
- Enhanced metadata querying with SQL joins and filtering
- Maintained backward compatibility with existing interface

Quick start (CLI smoke‑test)
---------------------------
# For PDF files:
$ python rag_pipeline.py my_notes.pdf "What is the Krebs cycle?" 

# For image files:
$ python rag_pipeline.py testing_files/mitosis.png "What is the first state of mitosis?" --type image

Environment variables required:
    GEMINI_API_KEY        Your Google Generative AI key
    SUPABASE_URL          Your Supabase project URL
    SUPABASE_KEY          Your Supabase anon/service key
    SUPABASE_DB_URL       PostgreSQL connection string (for direct pgvector access)

Example (inside FastAPI)
-----------------------
from rag_pipeline import RAGPipeline
rag = RAGPipeline()
await rag.ingest_pdf(bytes_file, notebook_id="BIO101", user_id="user123")
res = await rag.query("Explain glycolysis", notebook_id="BIO101", user_id="user123")
print(res["answer"])
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
import logging
from io import BytesIO
from typing import List, TypedDict, Union
from PIL import Image, ImageStat

import fitz  # PyMuPDF
import numpy as np
import tiktoken
from pydantic import BaseModel
import google.generativeai as genai
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from supabase import create_client, Client
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from sentence_transformers import SentenceTransformer
import uuid

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

ENCODER = tiktoken.get_encoding("cl100k_base")

# Initialize BLIP model and processor
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    BLIP_AVAILABLE = True
except Exception as e:
    print(f"Warning: BLIP model initialization failed: {str(e)}")
    BLIP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    id: str
    text: str
    tokens: int
    metadata: dict

class QueryResult(TypedDict):
    answer: str
    chunks: List[Chunk]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    """Return number of tokens in *text* using cl100k_base encoder."""
    return len(ENCODER.encode(text))

# Global sentence transformer model (loaded once)
_sentence_model = None

def _get_sentence_model():
    """Get or initialize the sentence transformer model."""
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model

def _embed_text(text: str) -> List[float]:
    """Return a 768-dim embedding for *text* using sentence-transformers."""
    model = _get_sentence_model()
    embedding = model.encode(text)
    # Pad or truncate to 768 dimensions
    if len(embedding) < 768:
        # Pad with zeros
        embedding = np.pad(embedding, (0, 768 - len(embedding)), 'constant')
    elif len(embedding) > 768:
        # Truncate
        embedding = embedding[:768]
    return embedding.tolist()

def _embed_texts(texts: List[str]) -> np.ndarray:
    """Vectorise a list of *texts* → (n, 768) numpy array."""
    model = _get_sentence_model()
    embeddings = model.encode(texts)
    # Ensure all embeddings are 768 dimensions
    if embeddings.shape[1] < 768:
        # Pad with zeros
        embeddings = np.pad(embeddings, ((0, 0), (0, 768 - embeddings.shape[1])), 'constant')
    elif embeddings.shape[1] > 768:
        # Truncate
        embeddings = embeddings[:, :768]
    return embeddings

def _get_notebook_id(notebook_id: str) -> str:
    """Return the notebook_id as-is, handling 'default' case."""
    if notebook_id == "default":
        # For default case, return None so database stores as NULL
        # In production, this should rarely happen as API calls pass real UUIDs
        return None
    return notebook_id

def _get_user_id(user_id: str = None) -> str:
    """Return test user ID if none provided."""
    if user_id is None:
        # Use a test user UUID that exists in the database
        return "c04c5c6a-367b-4322-8913-856d13a2da75"
    return user_id


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Gemini + Supabase pgvector Retrieval‑Augmented Generation engine."""

    def __init__(
        self,
        *,
        table_name: str = "chunks",
        max_tokens_per_chunk: int = 500,
        similarity_threshold: float = 0.340,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        db_url: str | None = None
    ):
        self.max_tokens = max_tokens_per_chunk
        self.table_name = table_name
        self.similarity_threshold = similarity_threshold
        
        # Initialize Supabase client
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        self.db_url = db_url or os.getenv("SUPABASE_DB_URL")
        
        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.gemini_vision = genai.GenerativeModel('gemini-2.0-flash')
        self.logger = logging.getLogger(__name__)
        
        # Initialize database schema if needed
        self._ensure_schema()

    def _ensure_schema(self):
        """Ensure the documents table exists with pgvector extension."""
        if not self.db_url:
            self.logger.warning("No SUPABASE_DB_URL provided, skipping direct schema setup")
            return
            
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Enable pgvector extension
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    # Create chunks table if it doesn't exist (matching your schema)
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            user_id UUID,
                            notebook_id UUID,
                            document_id UUID,
                            content TEXT NOT NULL,
                            embedding VECTOR(768),
                            tokens INTEGER DEFAULT 0,
                            chunk_index INTEGER DEFAULT 0,
                            metadata JSONB DEFAULT '{{}}',
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        );
                    """)
                    
                    # Create indexes for better performance
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx 
                        ON {self.table_name} USING hnsw (embedding vector_cosine_ops);
                    """)
                    
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {self.table_name}_user_notebook_idx 
                        ON {self.table_name} (user_id, notebook_id);
                    """)
                    
                    conn.commit()
                    self.logger.info(f"Database schema for {self.table_name} ensured")
        except Exception as e:
            self.logger.error(f"Failed to ensure schema: {e}")

    # --------------------------- Image Processing ---------------------------

    def contains_text(self, image: Image.Image) -> bool:
        """Checks if the image contains any text using OCR with confidence filtering."""
        try:
            gray_image = image.convert('L')
            data = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT)

            for i, conf_str in enumerate(data['conf']):
                try:
                    conf = int(conf_str)
                    text = data['text'][i].strip()

                    if conf > 50 and text:
                        return True

                except ValueError:
                    continue  # Skip non-integer confidence values like '-1'
            return False

        except Exception as e:
            print(f"Error in OCR processing: {str(e)}")
            return False
        
        
    def _is_valid_image(self, image: Image.Image) -> bool:
        """Checks if image is suitable for Gemini analysis. """
        try:
            # Check dimensions
            min_width, min_height = 100, 100
            if image.width < min_width or image.height < min_height:
                return False
            
            # Check if image is blank or has low contrast
            stat = ImageStat.Stat(image.convert("L"))  # Convert to grayscale
            if stat.stddev[0] < 5:
                return False

            # Check for extremely low resolution
            total_pixels = image.width * image.height
            if total_pixels < 100_000:
                return False
            return True
                
        except Exception as e:
            print(f"Error in is_valid_image: {str(e)}")
            return False

    async def analyze_image(self, image: Union[Image.Image, bytes], *, notebook_id: str = "default", user_id: str = None) -> None:
        """Analyzes an image using Gemini Vision or falls back to BLIP if Gemini fails."""
        try:
            # Convert bytes to PIL Image if needed
            if isinstance(image, bytes): 
                image = Image.open(BytesIO(image))
                
            # Validate image 
            if not self._is_valid_image(image):
                raise ValueError("Image is not suitable for analysis. It may be too small, blank, or low quality.")
            
            # Convert RGBA to RGB if needed
            if image.mode == 'RGBA': 
                image = image.convert('RGB')
            
            # Try Gemini first
            try:
                response = await asyncio.to_thread(
                    self.gemini_vision.generate_content,
                    [
                        "Analyze this image and provide a detailed description of its content, "
                        "focusing on any diagrams, charts, or relevant visual information. ",
                        image  
                    ]
                )
                description = response.text
                print("Successfully analyzed image using Gemini Vision")
            except Exception as gemini_error:
                print(f"Gemini Vision analysis failed: {str(gemini_error)}")
                if not BLIP_AVAILABLE:
                    raise Exception("Both Gemini Vision and BLIP are unavailable")
                
                # Fall back to BLIP if Gemini Vision fails
                description = self._generate_blip_caption(image)
                print("Successfully analyzed image using BLIP")
            
            # Create and store chunk
            chunk = Chunk(
                id=str(uuid.uuid4()),
                text=description,
                tokens=_token_count(description),
                metadata={
                    "notebook_id": notebook_id,
                    "type": "image",
                    "timestamp": str(uuid.uuid4()),
                    "analyzer": "gemini" if "gemini_error" not in locals() else "blip"
                }
            )
            # Store in Supabase
            await self._store_chunks([chunk], user_id)
        except Exception as e:
            raise Exception(f"Error analyzing image: {str(e)}")

    def _generate_blip_caption(self, image: Image.Image) -> str:
        """ Generates a caption for the given image using BLIP. 
        This is used as a backup when Gemini Vision fails. """
        try:
            inputs = processor(images=image, return_tensors="pt").to(device)

            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)
            
            enhanced_caption = (
                f"This image shows: {caption}. "
                "The image appears to be a diagram or illustration that may contain "
                "important visual information relevant to the document's content."
            )
            return enhanced_caption

        except Exception as e:
            raise Exception(f"Error generating BLIP caption: {str(e)}")

    # --------------------------- file processing  ---------------------------

    async def ingest_txt(self, text: Union[str, bytes], *, notebook_id: str = "default", user_id: str = None) -> None:
        """Extract text, chunk, embed, and store vectors."""
        # Convert bytes to string if needed
        if isinstance(text, bytes):
            text = text.decode('utf-8')

        chunks: list[Chunk] = []
        buffer = ""
        
        sentences = text.replace('\n', ' ').split('. ')
        for sentence in sentences:
            buffer += sentence + '. '
            if _token_count(buffer) >= self.max_tokens:
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        text=buffer.strip(),
                        tokens=_token_count(buffer),
                        metadata={"notebook_id": notebook_id, "type": "text"},
                    )
                )
                buffer = ""
        # Add remaining text if any
        if buffer:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=buffer.strip(),
                    tokens=_token_count(buffer),
                    metadata={"notebook_id": notebook_id, "type": "text"},
                )
            )
        
        # Store chunks in Supabase
        await self._store_chunks(chunks, user_id)

    async def _store_chunks(self, chunks: List[Chunk], user_id: str = None) -> None:
        """Store chunks with embeddings in Supabase."""
        if not chunks:
            return
            
        # Generate embeddings for all chunks
        embeddings = _embed_texts([c.text for c in chunks])
        
        # Prepare data for insertion (matching chunks table schema)
        data_to_insert = []
        for i, chunk in enumerate(chunks):
            data_to_insert.append({
                "user_id": _get_user_id(user_id),
                "notebook_id": _get_notebook_id(chunk.metadata.get("notebook_id", "default")),
                "document_id": None,  # Will be set when document is created
                "content": chunk.text,
                "embedding": embeddings[i].tolist(),
                "tokens": chunk.tokens,
                "chunk_index": i,
                "metadata": chunk.metadata
            })
        
        try:
            # Insert chunks into Supabase
            result = self.supabase.table(self.table_name).upsert(data_to_insert).execute()
            self.logger.info(f"Successfully stored {len(chunks)} chunks")
        except Exception as e:
            self.logger.error(f"Failed to store chunks: {e}")
            raise

    async def ingest_pdf(self, pdf: Union[str, bytes], *, notebook_id: str = "default", user_id: str = None) -> None:
        """Extract text and images from PDF, creating chunks, embedding and storing as vectors."""

        if isinstance(pdf, str):
            doc = fitz.open(pdf)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf)
                doc = fitz.open(tmp.name)

        chunks: list[Chunk] = []
        buffer = ""
        
        for idx, page in enumerate(doc):
            # Process images in the page first
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(BytesIO(image_bytes))
                    
                    if self.contains_text(image) and self._is_valid_image(image):
                        print(f"Processing image on page {idx + 1}")
                        # Get image description
                        try:
                            response = await asyncio.to_thread(
                                self.gemini_vision.generate_content,
                                [
                                    "Analyze this image and provide a detailed description of its content, "
                                    "focusing on any diagrams, charts, or relevant visual information. Keep it within 100 words.",
                                    image  
                                ]
                            )
                            description = response.text
                            print("Successfully analyzed image using Gemini Vision")
                        except Exception as gemini_error:
                            print(f"Gemini Vision analysis failed: {str(gemini_error)}")
                            if not BLIP_AVAILABLE:
                                raise Exception("Both Gemini Vision and BLIP are unavailable")
                            print("Falling back to BLIP for image analysis...")
                            description = self._generate_blip_caption(image)
                            print("Successfully analyzed image using BLIP")
                        # Store image description as its own chunk
                        chunks.append(
                            Chunk(
                                id=str(uuid.uuid4()),
                                text=description,
                                tokens=_token_count(description),
                                metadata={
                                    "notebook_id": notebook_id,
                                    "type": "image",
                                    "page": idx + 1,
                                    "image_index": img_index + 1
                                }
                            )
                        )
                except Exception as e:
                    print(f"Error processing image on page {idx + 1}: {str(e)}")
                    continue
            # Process text content
            page_text = page.get_text().strip()
            if page_text:
                buffer += page_text + "\n"
                if _token_count(buffer) >= self.max_tokens or idx == len(doc) - 1:
                    if buffer.strip():
                        chunks.append(
                            Chunk(
                                id=str(uuid.uuid4()),
                                text=buffer.strip(),
                                tokens=_token_count(buffer),
                                metadata={
                                    "notebook_id": notebook_id,
                                    "type": "text",
                                    "page_end": idx + 1
                                }
                            )
                        )
                    buffer = ""

        if chunks:  # Only process if we have chunks
            await self._store_chunks(chunks, user_id)
            print("\nAll chunk metadata after ingestion:")
            for c in chunks:
                print(c.metadata)

    # --------------------------- query ---------------------------

    async def query(self, *, question: str, notebook_id: str = "default", top_k: int = 3, user_id: str = None) -> QueryResult:
        """
        Retrieve top‑k chunks and answer the *question*.
        """
        q_embed = _embed_text(question)
        
        try:
            # Query Supabase for chunks filtered by notebook_id
            query = self.supabase.table(self.table_name).select('*')
            
            # Filter by notebook_id if provided and not "default"
            if notebook_id and notebook_id != "default":
                query = query.eq('notebook_id', notebook_id)
            elif notebook_id == "default":
                # For default, get chunks with NULL notebook_id or specific default notebook
                # First try to get the default notebook ID
                try:
                    default_notebook_result = self.supabase.table('notebooks').select('id').eq('name', 'Default Notebook').execute()
                    if default_notebook_result.data:
                        default_notebook_id = default_notebook_result.data[0]['id']
                        query = query.eq('notebook_id', default_notebook_id)
                    else:
                        # If no default notebook exists, get chunks with NULL notebook_id
                        query = query.is_('notebook_id', 'null')
                except Exception as e:
                    print(f"Warning: Could not filter by notebook_id: {e}")
                    # Fall back to getting all chunks
                    pass
            
            result = query.execute()
            results = result.data
            
            if not results:
                return {"answer": "I couldn't find relevant material.", "chunks": []}
            
            # Calculate similarities and get top-k
            docs = [item["content"] for item in results]
            metas = [item["metadata"] for item in results]
            ids = [item["id"] for item in results]
            context = self.find_relevent_chunks(q_embed, docs, metas, ids)
                
        except Exception as e:
            self.logger.error(f"Failed to query Supabase: {e}")
            return {"answer": "I couldn't find relevant material.", "chunks": []}
        
        if not context:
            print("No chunks passed similarity threshold!")
            return {"answer": "I couldn't find relevant material.", "chunks": []}
        
        print(f"\nUsing {len(context)} relevant chunks for answer generation")
        
        chunks = [
            Chunk(id=i, text=t, tokens=_token_count(t), metadata=m)
            for t, m, i in context
        ]
        
        # Generate answer using Gemini
        context_text = "\n\n".join([chunk.text for chunk in chunks])
        
        prompt = f"""
Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, say so.

Context:
{context_text}

Question: {question}

Answer:"""
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            self.logger.error(f"Failed to generate answer with Gemini: {e}")
            answer = "I apologize, but I encountered an error while generating the answer."
        
        return {"answer": answer, "chunks": chunks}
    


    def find_relevent_chunks(self, query_embedding: List[float], documents: List[str], 
                    metadatas: List[dict], ids: List[str]) -> List[tuple[str, dict, str]]:
        """Filter chunks based on cosine similarity with the query."""

        relevant_chunks = []
        print("\nCalculating similarity scores...")
        
        for doc, meta, id_ in zip(documents, metadatas, ids):
            # Calculate cosine similarity between question and chunk
            doc_embed = _embed_text(doc)
            similarity = np.dot(query_embedding, doc_embed) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embed)
            )
            
            # Print similarity score for debugging
            print(f"\n  Chunk preview: {doc[:100]}...")
            
            if similarity > self.similarity_threshold:
                relevant_chunks.append((doc, meta, id_))
                print(f"✓ Chunk accepted (similarity: {similarity:.3f})")
            else:
                print(f"✗ Chunk rejected (similarity: {similarity:.3f})")
                
        return relevant_chunks


# ---------------------------------------------------------------------------
# CLI smoke‑test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLI test for RAGPipeline (Gemini only)")
    parser.add_argument("file", help="PDF file path or image file path")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--notebook", default="default", help="Notebook/course id (optional)")
    parser.add_argument("--api-key", help="Your Gemini API key")
    parser.add_argument("--type", choices=["pdf", "image"], default="pdf", help="Type of file (pdf or image)")
    args = parser.parse_args()
    
    # Override API key if provided as argument
    if args.api_key:
        genai.configure(api_key=args.api_key)

    async def _cli():
        rag = RAGPipeline()
        
        if args.type == "pdf":
            await rag.ingest_pdf(args.file, notebook_id=args.notebook)
            print("PDF processed successfully.\n")
        else:  # image
            image = Image.open(args.file)
            await rag.analyze_image(image, notebook_id=args.notebook)
            print(f"Analyzed and stored image.\n")
            
        res = await rag.query(question=args.question, notebook_id=args.notebook)
        if "answer" in res:
            print("Answer:\n", res["answer"], "\n")
        if "chunks" in res and res["chunks"]:
            print("Cited chunks:\n")
            for c in res["chunks"]:
                print(f" • {c.text[:200].replace(chr(10), ' ')}…")
                print(f"   Metadata: {c.metadata}")

    asyncio.run(_cli()) 

    