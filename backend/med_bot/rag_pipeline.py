"""
Med‑Bot AI — Gemini‑only Retrieval‑Augmented Generation engine
----------------------------------------------------------------
Self‑contained class that handles PDF ingestion, chunking, vector
storage, retrieval, and answer generation — all powered solely by
Google Gemini models.

Recent Changes:
- Refactored image handling to store image descriptions as separate chunks
- Each image chunk now has its own metadata with type='image'
- Simplified chunk creation logic by separating image and text processing

Quick start (CLI smoke‑test)
---------------------------
# For PDF files:
$ python rag_pipeline.py my_notes.pdf "What is the Krebs cycle?" 

# For image files:
$ python rag_pipeline.py testing_files/mitosis.png "What is the first state of mitosis?" --type image

Environment variables required:
    GEMINI_API_KEY        Your Google Generative AI key
    CHROMA_PERSIST_DIR    (optional) path to store vectors, default=".chroma"

Example (inside FastAPI)
-----------------------
from rag_pipeline import RAGPipeline
rag = RAGPipeline()
await rag.ingest_pdf(bytes_file, notebook_id="BIO101")
res = await rag.query("Explain glycolysis", notebook_id="BIO101")
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

import chromadb
import fitz  # PyMuPDF
import numpy as np
import tiktoken
from chromadb.config import Settings
from pydantic import BaseModel
import google.generativeai as genai
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

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

def _embed_text(text: str) -> List[float]:
    """Return a 768‑dim embedding for *text* using Gemini embeddings."""
    resp = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
    )
    return resp["embedding"]

def _embed_texts(texts: List[str]) -> np.ndarray:
    """Vectorise a list of *texts* → (n, 768) numpy array."""
    return np.array([_embed_text(t) for t in texts])


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Gemini‑powered Retrieval‑Augmented Generation engine."""

    def __init__(
        self,
        *,
        collection_name: str = "rag_chunks",
        max_tokens_per_chunk: int = 500,
        persist_dir: str | None = None,
        similarity_threshold: float = 0.675
    ):
        self.max_tokens = max_tokens_per_chunk
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", ".chroma")
        self._client = chromadb.Client(
            Settings(persist_directory=self.persist_dir, anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(collection_name)
        self.gemini_vision = genai.GenerativeModel('gemini-2.0-flash')
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(__name__)


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

    async def analyze_image(self, image: Union[Image.Image, bytes], *, notebook_id: str = "default") -> None:
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
            # Add to vector database
            embedding = _embed_text(description)
            self._col.add(
                ids=[chunk.id],
                embeddings=[embedding],
                documents=[chunk.text],
                metadatas=[chunk.metadata]
            )
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

    async def ingest_txt(self, text: Union[str, bytes], *, notebook_id: str = "default") -> None:
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
        embeddings = _embed_texts([c.text for c in chunks])
        self._col.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    async def ingest_pdf(self, pdf: Union[str, bytes], *, notebook_id: str = "default") -> None:
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
            embeddings = _embed_texts([c.text for c in chunks])
            self._col.add(
                ids=[c.id for c in chunks],
                embeddings=embeddings.tolist(),
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )
            print("\nAll chunk metadata after ingestion:")
            for c in chunks:
                print(c.metadata)

    # --------------------------- query ---------------------------

    async def query(self, *, question: str, notebook_id: str = "default", top_k: int = 4) -> QueryResult:
        """
        Retrieve top‑k chunks and answer the *question*.
        Searches across both PDF and image chunks.
        """
        # Verify we have documents in the collection
        collection_count = self._col.count()
        if collection_count == 0:
            return {"answer": "I couldn't find relevant material.", "chunks": []}

        q_embed = _embed_text(question)
        
        res = self._col.query(
            query_embeddings=[q_embed],
            n_results=min(top_k, collection_count),  
            where={"notebook_id": notebook_id},
        )

        if not res["ids"] or not res["ids"][0]:
            return {"answer": "I couldn't find relevant material.", "chunks": []}

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        ids = res["ids"][0]
        
        # Filter relevant chunks based on similarity
        context = self.find_relevent_chunks(q_embed, docs, metas, ids)
        
        if not context:
            print("No chunks passed similarity threshold!")
            return {"answer": "I couldn't find relevant material.", "chunks": []}
        
        print(f"\nUsing {len(context)} relevant chunks for answer generation")
        
        chunks = [
            Chunk(id=i, text=t, tokens=_token_count(t), metadata=m)
            for t, m, i in context
        ]
        return {"chunks": chunks}
    


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

    