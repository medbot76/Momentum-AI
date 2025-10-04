"""
MultiHopRAGPipeline: Multi-hop Retrieval-Augmented Generation engine
-------------------------------------------------------------------
Extends RAGPipeline to support multi-hop (compositional) question answering.
All ingestion and image analysis features (Gemini/BLIP) are inherited from RAGPipeline.
Override the query() method to implement multi-hop logic.
"""

from med_bot.rag_pipeline import RAGPipeline
import re
import google.generativeai as genai
import asyncio
import time

class MultiHopRAGPipeline(RAGPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose_mode = True  # Can be toggled for demos
    
    def print_step(self, step_num, sub_question):
        """Visual output for demo purposes"""
        if self.verbose_mode:
            print(f"\n{'='*60}")
            print(f"🔍 STEP {step_num}: Analyzing sub-question")
            print(f"❓ Question: {sub_question}")
            print(f"{'='*60}")
    
    def print_synthesis(self):
        """Visual output for synthesis step"""
        if self.verbose_mode:
            print(f"\n{'🔄'*20}")
            print("🧠 SYNTHESIZING FINAL ANSWER FROM ALL STEPS...")
            print(f"{'🔄'*20}")
    
    def print_results(self, chunks_found, similarity_scores):
        """Visual output for results"""
        if self.verbose_mode:
            print(f"✅ Found {chunks_found} relevant chunks")
            if similarity_scores:
                print(f"📊 Similarity scores: {', '.join([f'{score:.3f}' for score in similarity_scores[:3]])}...")

    async def query(self, *, question: str, notebook_id: str = "default", top_k: int = 4, verbose: bool = False) -> dict:
        """
        Multi-hop retrieval with optional verbose output for demos and Flask integration.
        
        Returns:
            dict: Contains answer, chunks, and optionally detailed step information
        """
        self.verbose_mode = verbose
        start_time = time.time()
        
        if self.is_multihop_question(question):
            if verbose:
                print(f"\n🚀 MULTI-HOP RAG DETECTED")
                print(f"📝 Original question: {question}")
            
            # Decompose question
            sub_questions = self.heuristic_decompose_question(question)
            decomposition_method = "heuristic"
            
            # If heuristics only find one sub-question, use LLM-based decomposition
            if len(sub_questions) <= 1:
                sub_questions = await self.llm_decompose_question(question)
                decomposition_method = "llm"
            
            if verbose:
                print(f"🔧 Decomposition method: {decomposition_method}")
                print(f"📋 Sub-questions ({len(sub_questions)}):")
                for i, sq in enumerate(sub_questions, 1):
                    print(f"    {i}. {sq}")
            
            # Process each sub-question
            steps = []
            all_contexts = []
            all_answers = []
            
            for i, sub_q in enumerate(sub_questions):
                self.print_step(i+1, sub_q)
                
                step_start = time.time()
                res = await super().query(question=sub_q, notebook_id=notebook_id, top_k=top_k)
                step_time = time.time() - step_start
                
                chunks = res.get("chunks", [])
                answer = res.get("answer", "")
                
                # Calculate similarity scores if available
                similarity_scores = []
                chunk_previews = []
                for chunk in chunks:
                    chunk_previews.append({
                        "id": chunk.id,
                        "preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                        "tokens": chunk.tokens,
                        "metadata": chunk.metadata
                    })
                    # You could add similarity calculation here if needed
                
                all_contexts.extend(chunks)
                all_answers.append(answer)
                
                step_info = {
                    "step": i+1,
                    "sub_question": sub_q,
                    "answer": answer,
                    "chunks_found": len(chunks),
                    "chunk_previews": chunk_previews,
                    "similarity_scores": similarity_scores,
                    "processing_time": round(step_time, 3)
                }
                steps.append(step_info)
                
                self.print_results(len(chunks), similarity_scores)
                
                if verbose and answer:
                    print(f"💡 Sub-answer: {answer[:150]}{'...' if len(answer) > 150 else ''}")
            
            # Synthesize final answer
            self.print_synthesis()
            synthesis_start = time.time()
            final_answer = await self.synthesize_answer(question, all_answers, all_contexts)
            synthesis_time = time.time() - synthesis_start
            
            total_time = time.time() - start_time
            
            if verbose:
                print(f"\n🎯 FINAL ANSWER:")
                print(f"{final_answer}")
                print(f"\n⏱️  Total processing time: {total_time:.3f}s")
                print(f"📊 Total chunks used: {len(all_contexts)}")
            
            result = {
                "answer": final_answer,
                "chunks": all_contexts,
                "is_multihop": True,
                "decomposition_method": decomposition_method,
                "steps": steps,
                "total_steps": len(sub_questions),
                "total_chunks": len(all_contexts),
                "processing_time": round(total_time, 3),
                "synthesis_time": round(synthesis_time, 3)
            }
            
            return result
        else:
            # Single-hop processing
            if verbose:
                print(f"\n📝 SINGLE-HOP RAG (simple question)")
                print(f"❓ Question: {question}")
            
            start_time = time.time()
            result = await super().query(question=question, notebook_id=notebook_id, top_k=top_k)
            processing_time = time.time() - start_time
            
            if verbose:
                chunks = result.get("chunks", [])
                print(f"✅ Found {len(chunks)} relevant chunks")
                print(f"⏱️  Processing time: {processing_time:.3f}s")
                if result.get("answer"):
                    print(f"💡 Answer: {result['answer'][:150]}{'...' if len(result.get('answer', '')) > 150 else ''}")
            
            # Add metadata for consistency
            result.update({
                "is_multihop": False,
                "total_steps": 1,
                "processing_time": round(processing_time, 3)
            })
            
            return result

    def is_multihop_question(self, question: str) -> bool:
        # Improved heuristic: look for multiple question marks, conjunctions, or multi-part instructions
        if question.count("?") > 1:
            return True
        if re.search(r" and | then | as well as | in addition to |, |; | after | before | because | so that | affect | relationship | both | compare | contrast ", question, re.I):
            return True
        if "," in question and any(word in question for word in ["how", "what", "why", "when"]):
            return True
        return False

    def heuristic_decompose_question(self, question: str) -> list:
        # Improved: split on more conjunctions and phrases
        splitters = r" and | then | as well as | in addition to |, |; | after | before | because | so that "
        sub_questions = [q.strip() for q in re.split(splitters, question) if q.strip()]
        return sub_questions

    async def llm_decompose_question(self, question: str) -> list:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        prompt = (
            f"Decompose the following question into a list of simpler sub-questions needed to answer it. "
            f"Return only a numbered list, one sub-question per line.\n\nQuestion: {question}\n\nList:"
        )
        resp = await asyncio.to_thread(model.generate_content, prompt)
        lines = [line.strip(" .") for line in resp.text.strip().split("\n") if line.strip()]
        # Remove numbering if present
        sub_questions = [re.sub(r'^\d+\.\s*', '', line) for line in lines]
        return sub_questions

    async def synthesize_answer(self, original_question, sub_answers, contexts):
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        prompt = (
            f"Original question: {original_question}\n"
            f"Sub-answers:\n" + "\n".join(f"- {a}" for a in sub_answers) +
            "\n\nCombine these into a single, clear, and complete answer."
        )
        resp = await asyncio.to_thread(model.generate_content, prompt)
        return resp.text.strip() 