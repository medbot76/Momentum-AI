#!/usr/bin/env python3
"""
Simple RAG test without database dependency
"""
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

def test_simple_rag():
    """Test RAG functionality without database"""
    load_dotenv()
    
    print("🧪 Testing Simple RAG (no database)...")
    
    try:
        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not set")
            return False
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        
        # Initialize sentence transformer
        print("📥 Loading sentence transformer model...")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer loaded")
        
        # Test embedding generation
        test_text = "This is a test document about machine learning and artificial intelligence."
        print(f"🔤 Testing embedding for: '{test_text[:50]}...'")
        
        embedding = sentence_model.encode(test_text)
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        
        # Test Gemini generation
        print("🤖 Testing Gemini generation...")
        response = model.generate_content("What is machine learning?")
        print(f"✅ Gemini response: {response.text[:100]}...")
        
        print("\n🎉 All tests passed! The core RAG components are working.")
        print("The issue is with database access, not the core functionality.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_simple_rag()
