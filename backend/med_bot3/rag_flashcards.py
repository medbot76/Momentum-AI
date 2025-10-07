from __future__ import annotations

"""RAG‑powered flashcard generation module.

Extend the existing ``RAGPipeline`` with a ``generate_flashcards`` method
that produces question‑answer pairs for spaced‑repetition apps like Anki.

Usage example
-------------
>>> from rag_flashcards import RAGFlashcards
>>> rag = RAGFlashcards()
>>> await rag.ingest_pdf("my_notes.pdf", notebook_id="BIO101")
>>> cards = await rag.generate_flashcards(topic="Glycolysis", notebook_id="BIO101", num_cards=10)
>>> print(cards[0]["question"], "->", cards[0]["answer"])

The CLI at the bottom shows a quick manual smoke‑test.
"""

import asyncio
import json
from typing import Dict, List
import sys

import google.generativeai as genai
from multi_hop_rag_pipeline import MultiHopRAGPipeline

sys.path.append('.')
from rag_pipeline import RAGPipeline

__all__ = ["RAGFlashcards"]


class RAGFlashcards(MultiHopRAGPipeline):  # Use MultiHopRAGPipeline by default; switch to RAGPipeline for single-hop
    """RAGPipeline → flashcard generator mix‑in."""

    async def generate_flashcards(
        self,
        *,
        topic: str,
        notebook_id: str,
        num_cards: int = 10,
        top_k: int = 6,
        user_id: str = None,
    ) -> List[Dict[str, str]]:
        """Return *num_cards* flashcards on *topic* from *notebook_id*.

        Each flashcard is a ``{"question": str, "answer": str}`` dict.
        """
        if num_cards < 1:
            raise ValueError("num_cards must be ≥ 1")

        # 1) Pull context from the vector store
        search_q = f"Give a concise yet complete explanation of {topic}."
        res = await self.query(question=search_q, notebook_id=notebook_id, top_k=top_k, user_id=user_id)
        context = "\n\n".join(c.text for c in res["chunks"]) if res["chunks"] else ""
        if not context:
            return []

        # 2) Ask Gemini to craft JSON flashcards
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        prompt = (
            "You are an expert tutor. Using only the provided CONTEXT, "
            f"create {num_cards} short flashcards that help a student self‑test on the topic \"{topic}\". "
            "Return ONLY a JSON array where each element has \"question\" and \"answer\" keys. "
            "No markdown, no extra commentary.\n\n"
            "CONTEXT:\n" + context + "\n\nJSON:"
        )
        resp = await asyncio.to_thread(model.generate_content, prompt)
        raw = resp.text.strip()

        try:
            cards: List[Dict[str, str]] = json.loads(raw)
        except json.JSONDecodeError:
            # Try to salvage by locating JSON substring
            start, end = raw.find("["), raw.rfind("]")
            if start != -1 and end != -1:
                cards = json.loads(raw[start : end + 1])
            else:
                raise RuntimeError("Could not parse flashcards JSON:\n" + raw[:400])
        return cards


# ---------------------------------------------------------------------------
# CLI smoke‑test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os
    import textwrap

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Quick test for RAGFlashcards:
              1. Ingest the PDF.
              2. Generate flashcards for a given topic.
            """
        ),
    )
    parser.add_argument("pdf", help="Path to PDF to ingest")
    parser.add_argument("topic", help="Flashcard topic, e.g. 'Krebs cycle'")
    parser.add_argument("-n", "--num", type=int, default=8, help="# flashcards to create")
    parser.add_argument("--notebook", default="cli-test", help="Notebook/course id")
    args = parser.parse_args()

    # Allow overriding the API key via env for quick testing
    if "GEMINI_API_KEY" not in os.environ:
        parser.error("GEMINI_API_KEY env var not set — export it before running.")

    async def _cli():
        rag = RAGFlashcards()
        count = await rag.ingest_pdf(args.pdf, notebook_id=args.notebook)
        print(f"Ingested {count} chunks.\n")
        cards = await rag.generate_flashcards(
            topic=args.topic, notebook_id=args.notebook, num_cards=args.num
        )
        if not cards:
            print("No flashcards generated. :(")
            return
        for i, card in enumerate(cards, 1):
            print(f"\nCard {i}/{len(cards)}")
            print("Q:", card["question"])
            print("A:", card["answer"])

    asyncio.run(_cli()) 