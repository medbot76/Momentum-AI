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
import ast
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
        _sentence_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return _sentence_model

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize a numpy array."""
    if x.ndim == 1:
        denom = max(np.linalg.norm(x), eps)
        return (x / denom).astype(np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return (x / denom).astype(np.float32)

def _embed_text(text: str) -> List[float]:
    """Return a 768-dim embedding for *text* using sentence-transformers."""
    model = _get_sentence_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
    embedding = _l2_normalize(embedding)
    return embedding.tolist()

def _embed_texts(texts: List[str]) -> np.ndarray:
    """Vectorise a list of *texts* → (n, 768) numpy array."""
    model = _get_sentence_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    return _l2_normalize(embeddings)

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
        similarity_threshold: float = 0.30,
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
        
        user_id = _get_user_id(user_id)
        
        # Try using direct PostgreSQL connection if available (better for VECTOR types)
        if self.db_url:
            try:
                await asyncio.to_thread(self._store_chunks_via_db, chunks, embeddings, user_id)
                self.logger.info(f"Successfully stored {len(chunks)} chunks via direct DB connection")
                return
            except Exception as e:
                self.logger.warning(f"Direct DB connection failed, falling back to REST API: {e}")
        
        # Fallback: Use Supabase REST API
        data_to_insert = []
        for i, chunk in enumerate(chunks):
            data_to_insert.append({
                "user_id": user_id,
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
            self.logger.info(f"Successfully stored {len(chunks)} chunks via REST API")
        except Exception as e:
            self.logger.error(f"Failed to store chunks: {e}")
            raise
    
    def _store_chunks_via_db(self, chunks: List[Chunk], embeddings: np.ndarray, user_id: str) -> None:
        """Store chunks using direct PostgreSQL connection (handles VECTOR types correctly)."""
        import json
        
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cur:
                for i, chunk in enumerate(chunks):
                    notebook_id = _get_notebook_id(chunk.metadata.get("notebook_id", "default"))
                    
                    # Convert embedding to PostgreSQL vector format
                    embedding_list = embeddings[i].tolist()
                    embedding_str = '[' + ','.join([str(x) for x in embedding_list]) + ']'
                    
                    # Insert chunk with proper vector type
                    cur.execute("""
                        INSERT INTO chunks (
                            user_id, notebook_id, document_id, content, 
                            embedding, tokens, chunk_index, metadata
                        ) VALUES (%s, %s, %s, %s, %s::vector(768), %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            tokens = EXCLUDED.tokens,
                            metadata = EXCLUDED.metadata
                    """, (
                        user_id,
                        notebook_id,
                        None,  # document_id
                        chunk.text,
                        embedding_str,
                        chunk.tokens,
                        i,
                        json.dumps(chunk.metadata)
                    ))
                
                conn.commit()

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
        Retrieve top‑k chunks and answer the *question* using pgvector native similarity search.
        """
        q_embed = _embed_text(question)
        user_id = _get_user_id(user_id)
        
        try:
            # Use pgvector native similarity search if DB connection available
            if self.db_url:
                try:
                    context = await self._query_with_pgvector(q_embed, notebook_id, top_k, user_id)
                except Exception as pg_error:
                    # If direct DB connection fails, fall back to REST API
                    self.logger.warning(f"Direct DB connection failed ({pg_error}), falling back to REST API")
                    context = await self._query_with_supabase_client(q_embed, notebook_id, top_k, user_id)
            else:
                # Fallback to Supabase client method
                context = await self._query_with_supabase_client(q_embed, notebook_id, top_k, user_id)
                
        except Exception as e:
            self.logger.error(f"Failed to query: {e}")
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
    
    async def _query_with_pgvector(self, q_embed: List[float], notebook_id: str, top_k: int, user_id: str) -> List[tuple[str, dict, str]]:
        """Use pgvector native similarity search for fast, accurate retrieval."""
        try:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Build notebook filter
                    notebook_filter = ""
                    notebook_params = []
                    
                    if notebook_id and notebook_id != "default":
                        notebook_filter = "AND c.notebook_id = %s"
                        notebook_params = [notebook_id]
                    elif notebook_id == "default":
                        try:
                            default_notebook_result = self.supabase.table('notebooks').select('id').eq('name', 'Default Notebook').execute()
                            if default_notebook_result.data:
                                default_notebook_id = default_notebook_result.data[0]['id']
                                notebook_filter = "AND c.notebook_id = %s"
                                notebook_params = [default_notebook_id]
                            else:
                                notebook_filter = "AND c.notebook_id IS NULL"
                        except Exception:
                            notebook_filter = "AND c.notebook_id IS NULL"
                    
                    # Use pgvector cosine similarity (<=> operator)
                    # 1 - (embedding <=> query) gives cosine similarity
                    # Get top-k results sorted by similarity
                    sql = f"""
                        SELECT 
                            c.id,
                            c.content,
                            c.metadata,
                            c.tokens,
                            1 - (c.embedding <=> %s::vector) as similarity
                        FROM {self.table_name} c
                        WHERE c.user_id = %s
                            AND c.embedding IS NOT NULL
                            {notebook_filter}
                            AND (1 - (c.embedding <=> %s::vector)) >= %s
                        ORDER BY c.embedding <=> %s::vector
                        LIMIT %s
                    """
                    
                    params = [q_embed, user_id] + notebook_params + [q_embed, self.similarity_threshold, q_embed, top_k * 2]
                    
                    cur.execute(sql, params)
                    results = cur.fetchall()
                    
                    if not results:
                        print(f"No chunks found with similarity >= {self.similarity_threshold}")
                        return []
                    
                    # Convert to expected format
                    context = []
                    for row in results[:top_k]:
                        similarity = float(row['similarity'])
                        print(f"✓ Chunk accepted (similarity: {similarity:.3f}): {row['content'][:100]}...")
                        context.append((
                            row['content'],
                            row['metadata'] or {},
                            str(row['id'])
                        ))
                    
                    print(f"\n📊 Found {len(context)} chunks using pgvector native search")
                    return context
                    
        except Exception as e:
            self.logger.error(f"pgvector query failed: {e}")
            raise
    
    async def _query_with_supabase_client(self, q_embed: List[float], notebook_id: str, top_k: int, user_id: str) -> List[tuple[str, dict, str]]:
        """Fallback query method using Supabase client."""
        try:
            # Query Supabase for chunks filtered by notebook_id and user_id
            query = self.supabase.table(self.table_name).select('id, content, metadata, tokens, embedding, user_id, notebook_id')
            
            # Filter by user_id
            query = query.eq('user_id', user_id)
            
            # Filter by notebook_id if provided and not "default"
            if notebook_id and notebook_id != "default":
                query = query.eq('notebook_id', notebook_id)
            elif notebook_id == "default":
                try:
                    default_notebook_result = self.supabase.table('notebooks').select('id').eq('name', 'Default Notebook').execute()
                    if default_notebook_result.data:
                        default_notebook_id = default_notebook_result.data[0]['id']
                        query = query.eq('notebook_id', default_notebook_id)
                    else:
                        query = query.is_('notebook_id', 'null')
                except Exception as e:
                    print(f"Warning: Could not filter by notebook_id: {e}")
                    pass
            
            # Only get chunks that have embeddings
            query = query.not_.is_('embedding', 'null')
            
            result = query.execute()
            results = result.data
            
            if not results:
                return []
            
            # Extract data and parse embeddings
            docs = [item["content"] for item in results]
            metas = [item.get("metadata", {}) for item in results]
            ids = [item["id"] for item in results]
            
            # Parse embeddings
            stored_embeddings = []
            for item in results:
                embedding = item.get("embedding")
                if embedding is None:
                    stored_embeddings.append(None)
                elif isinstance(embedding, str):
                    try:
                        stored_embeddings.append(json.loads(embedding))
                    except (json.JSONDecodeError, TypeError):
                        try:
                            stored_embeddings.append(ast.literal_eval(embedding))
                        except (ValueError, SyntaxError):
                            stored_embeddings.append(None)
                elif isinstance(embedding, list):
                    stored_embeddings.append(embedding)
                else:
                    stored_embeddings.append(None)
            
            # Filter by similarity using stored embeddings
            return self.find_relevent_chunks(q_embed, docs, metas, ids, stored_embeddings=stored_embeddings)
            
        except Exception as e:
            self.logger.error(f"Supabase client query failed: {e}")
            raise

    def find_relevent_chunks(self, query_embedding: List[float], documents: List[str], 
                    metadatas: List[dict], ids: List[str], stored_embeddings: List[List[float]] = None) -> List[tuple[str, dict, str]]:
        """Filter chunks based on cosine similarity with the query using stored embeddings."""
        
        relevant_chunks = []
        similarity_scores = []  # Track all scores for debugging
        print("\nCalculating similarity scores using stored embeddings...")
        
        # Convert query embedding to numpy for efficient computation
        q_embed_np = np.array(query_embedding, dtype=np.float32)
        
        # Always normalize query embedding (ensures consistency)
        q_norm = np.linalg.norm(q_embed_np)
        if q_norm > 1e-8:  # Avoid division by very small numbers
            q_embed_np = q_embed_np / q_norm
        else:
            self.logger.error("Query embedding has zero norm!")
            return []
        
        # Debug: Check query embedding norm (should be ~1.0 after normalization)
        if abs(q_norm - 1.0) > 0.01:
            print(f"Debug: Query embedding norm after normalization: {np.linalg.norm(q_embed_np):.6f}")
        
        # Expected embedding dimension for current model (all-mpnet-base-v2 = 768)
        EXPECTED_DIM = 768
        
        # First pass: Check how many chunks need re-embedding
        mismatches = 0
        embedding_dims = []
        if stored_embeddings:
            for idx, emb in enumerate(stored_embeddings[:min(10, len(stored_embeddings))]):  # Sample first 10
                if emb and isinstance(emb, (list, tuple)):
                    try:
                        emb_array = np.array(emb, dtype=np.float32)
                        dim = len(emb_array)
                        embedding_dims.append(dim)
                        if dim != EXPECTED_DIM:
                            mismatches += 1
                    except:
                        pass
        
        # Debug: Report embedding dimensions found
        if embedding_dims:
            unique_dims = set(embedding_dims)
            print(f"Debug: Found embedding dimensions in sample: {unique_dims} (expected: {EXPECTED_DIM})")
            if mismatches > 0:
                print(f"Debug: {mismatches} out of {len(embedding_dims)} sampled chunks have wrong dimension")
        
        # If most chunks have mismatches, batch re-embed for efficiency
        if mismatches > 5 and len(documents) > 10:
            print(f"Detected {mismatches} dimension mismatches in sample. Batch re-embedding all chunks for accuracy...")
            # Batch re-embed all documents
            doc_embeddings = _embed_texts(documents)
            stored_embeddings = [emb.tolist() for emb in doc_embeddings]
        
        for idx, (doc, meta, id_) in enumerate(zip(documents, metadatas, ids)):
            # Check if we should use stored embedding or re-embed
            should_reembed = False
            doc_embed = None
            
            if stored_embeddings and idx < len(stored_embeddings) and stored_embeddings[idx]:
                # Ensure embedding is a list/array before converting
                if isinstance(stored_embeddings[idx], (list, tuple)):
                    try:
                        doc_embed = np.array(stored_embeddings[idx], dtype=np.float32)
                        
                        # Check dimension - if it doesn't match expected, re-embed
                        # This handles cases where old chunks used different models
                        if len(doc_embed) != EXPECTED_DIM:
                            should_reembed = True
                            if idx < 3:  # Debug first few
                                print(f"Debug: Chunk {idx} has dimension {len(doc_embed)}, expected {EXPECTED_DIM}, re-embedding...")
                        elif len(doc_embed) != len(q_embed_np):
                            # Dimension mismatch with query - re-embed to be safe
                            should_reembed = True
                            if idx < 3:
                                print(f"Debug: Chunk {idx} dimension {len(doc_embed)} != query {len(q_embed_np)}, re-embedding...")
                                
                    except (ValueError, TypeError) as e:
                        should_reembed = True
                        if idx < 3:
                            print(f"Warning: Failed to convert embedding for chunk {idx}: {e}, re-embedding...")
                else:
                    should_reembed = True
                    if idx < 3:
                        print(f"Warning: Invalid embedding format for chunk {idx}, re-embedding...")
            else:
                should_reembed = True
                if idx < 3:
                    print(f"Warning: No stored embedding for chunk {idx}, re-embedding...")
            
            # Re-embed if needed (ensures same model is used)
            if should_reembed:
                doc_embed = np.array(_embed_text(doc), dtype=np.float32)
            else:
                # Use stored embedding - ensure it matches query dimension exactly
                if len(doc_embed) != len(q_embed_np):
                    # Final safety check - re-embed if dimensions still don't match
                    doc_embed = np.array(_embed_text(doc), dtype=np.float32)
            
            # Normalize document embedding (always normalize to ensure consistency)
            doc_norm_before = np.linalg.norm(doc_embed)
            if doc_norm_before > 1e-8:  # Avoid division by very small numbers
                doc_embed = doc_embed / doc_norm_before
                doc_norm_after = np.linalg.norm(doc_embed)
            else:
                # Embedding is essentially zero, skip this chunk
                print(f"Warning: Zero or near-zero embedding for chunk {idx} (norm={doc_norm_before:.6f}), skipping...")
                continue
            
            # Cosine similarity: dot product of normalized vectors
            # Both vectors should be normalized, so dot product gives cosine similarity
            similarity = float(np.dot(q_embed_np, doc_embed))
            
            # Clamp similarity to [-1, 1] range (should already be in this range for normalized vectors)
            similarity = max(-1.0, min(1.0, similarity))
            similarity_scores.append(similarity)
            
            # Debug output for first few chunks
            if idx < 3:
                print(f"Debug chunk {idx}: doc_dim={len(doc_embed)}, doc_norm_before={doc_norm_before:.6f}, doc_norm_after={doc_norm_after:.6f}, similarity={similarity:.6f}")
            
            # Print similarity score for debugging
            print(f"\n  Chunk preview: {doc[:100]}...")
            
            if similarity >= self.similarity_threshold:
                relevant_chunks.append((doc, meta, id_))
                print(f"✓ Chunk accepted (similarity: {similarity:.3f})")
            else:
                print(f"✗ Chunk rejected (similarity: {similarity:.3f})")
        
        # Print summary of similarity scores
        if similarity_scores:
            scores_array = np.array(similarity_scores)
            print(f"\n📊 Similarity Score Summary:")
            print(f"   Total chunks: {len(similarity_scores)}")
            print(f"   Accepted (≥{self.similarity_threshold:.3f}): {len(relevant_chunks)}")
            print(f"   Rejected: {len(similarity_scores) - len(relevant_chunks)}")
            print(f"   Score range: [{scores_array.min():.3f}, {scores_array.max():.3f}]")
            print(f"   Mean score: {scores_array.mean():.3f}")
            print(f"   Median score: {np.median(scores_array):.3f}")
            if len(similarity_scores) > 0:
                top_5_indices = np.argsort(scores_array)[-5:][::-1]
                print(f"   Top 5 scores: {[f'{scores_array[i]:.3f}' for i in top_5_indices]}")
                
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

    