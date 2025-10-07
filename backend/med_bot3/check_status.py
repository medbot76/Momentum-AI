#!/usr/bin/env python3
"""
Quick status check for Med-Bot database and API connections
"""
import os
from dotenv import load_dotenv

def check_status():
    """Check the current status of all required components"""
    load_dotenv()
    
    print("🔍 Med-Bot Status Check")
    print("=" * 40)
    
    # Check environment variables
    env_vars = {
        'SUPABASE_URL': 'Supabase Project URL',
        'SUPABASE_KEY': 'Supabase API Key',
        'GEMINI_API_KEY': 'Google Gemini API Key',
        'CLAUDE_API_KEY': 'Anthropic Claude API Key (optional)'
    }
    
    print("📋 Environment Variables:")
    all_env_ok = True
    for var, description in env_vars.items():
        value = os.getenv(var)
        status = "✅ FOUND" if value else "❌ MISSING"
        if not value and var != 'CLAUDE_API_KEY':  # Claude is optional
            all_env_ok = False
        print(f"   {description}: {status}")
    
    # Check if we can import required modules
    print("\n📦 Python Dependencies:")
    try:
        import supabase
        print("   ✅ supabase: OK")
    except ImportError:
        print("   ❌ supabase: Missing - run 'pip install supabase'")
        all_env_ok = False
    
    try:
        import google.generativeai
        print("   ✅ google-generativeai: OK")
    except ImportError:
        print("   ❌ google-generativeai: Missing - run 'pip install google-generativeai'")
        all_env_ok = False
    
    # Check if virtual environment is active
    print(f"\n🐍 Virtual Environment: {'✅ ACTIVE' if 'VIRTUAL_ENV' in os.environ else '❌ NOT ACTIVE'}")
    
    print("\n" + "=" * 40)
    
    if all_env_ok:
        print("🎉 All basic requirements are met!")
        print("\n📝 Next steps:")
        print("1. Make sure database migrations are complete in Supabase Dashboard")
        print("2. Run: python chatbot.py")
        print("3. The schema cache will refresh automatically when you use the app")
    else:
        print("❌ Some requirements are missing.")
        print("\nFix the issues above before running the chatbot.")
    
    return all_env_ok

if __name__ == "__main__":
    check_status()