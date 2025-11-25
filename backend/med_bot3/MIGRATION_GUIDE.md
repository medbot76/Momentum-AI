# Embedding Model Migration Guide

## Problem
Your database contains chunks with embeddings from the old `all-MiniLM-L6-v2` model (384 dimensions), but your RAG pipeline now uses `all-mpnet-base-v2` (768 dimensions). This mismatch causes:
- Negative or very low similarity scores
- Poor retrieval performance
- Chunks being rejected even when they're relevant

## Solution
We've updated the codebase to use `all-mpnet-base-v2` consistently everywhere and created a migration script to re-embed all existing chunks.

## Changes Made

### 1. Code Updates
- ✅ `rag_pipeline.py`: Already using `all-mpnet-base-v2` (no changes needed)
- ✅ `chatbot.py`: Updated to use `all-mpnet-base-v2` instead of `all-MiniLM-L6-v2`
- ✅ `chatbot.py`: Fixed embedding normalization for YouTube transcript matching

### 2. Migration Script
Created `migrate_embeddings_to_mpnet.py` to batch re-embed all chunks.

## How to Run the Migration

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Migration (Dry Run First)
```bash
# Test run - see what would be migrated (no changes made)
python migrate_embeddings_to_mpnet.py --dry-run
```

### Step 3: Run Actual Migration
```bash
# Migrate all chunks
python migrate_embeddings_to_mpnet.py

# Or migrate specific user's chunks
python migrate_embeddings_to_mpnet.py --user-id YOUR_USER_ID

# Or migrate specific notebook
python migrate_embeddings_to_mpnet.py --notebook-id YOUR_NOTEBOOK_ID

# Adjust batch size if needed (default: 50)
python migrate_embeddings_to_mpnet.py --batch-size 100
```

### Migration Options
- `--user-id`: Filter by specific user ID
- `--notebook-id`: Filter by specific notebook ID
- `--batch-size`: Number of chunks to process at once (default: 50)
- `--dry-run`: Preview what would be migrated without making changes

## What the Migration Does

1. **Fetches all chunks** from Supabase (or filtered by user/notebook)
2. **Identifies chunks** that need migration:
   - Chunks with no embedding
   - Chunks with embeddings that aren't 768 dimensions
3. **Re-embeds** chunks in batches using `all-mpnet-base-v2`
4. **Updates** the embeddings in the database
5. **Reports** progress and results

## Expected Results

After migration:
- ✅ All chunks will have 768-dimension embeddings
- ✅ Similarity scores will be in the normal range (0.3-0.9 for relevant content)
- ✅ Better retrieval performance
- ✅ No more negative similarity scores

## Performance Notes

- **Batch processing**: Processes chunks in batches of 50 (configurable) for efficiency
- **Progress bar**: Shows real-time progress
- **Error handling**: Continues processing even if individual chunks fail
- **Time estimate**: ~1-2 seconds per batch (depends on chunk size and hardware)

## Troubleshooting

### If migration fails:
1. Check your Supabase credentials are set:
   ```bash
   export SUPABASE_URL="your-url"
   export SUPABASE_KEY="your-key"
   ```

2. Check database connection:
   ```bash
   python test_db_connection.py
   ```

3. Try smaller batch size:
   ```bash
   python migrate_embeddings_to_mpnet.py --batch-size 25
   ```

### If some chunks fail to update:
- The script will continue and report errors
- You can re-run the migration - it will only update chunks that need it
- Check the error messages for specific issues

## Verification

After migration, test with a query:
```python
from med_bot3.rag_pipeline import RAGPipeline
import asyncio

async def test():
    rag = RAGPipeline()
    result = await rag.query(question="Your test question", notebook_id="your-notebook-id")
    print(f"Found {len(result['chunks'])} relevant chunks")
    for chunk in result['chunks']:
        print(f"Chunk: {chunk.text[:100]}...")

asyncio.run(test())
```

You should see:
- Positive similarity scores
- Relevant chunks being retrieved
- No "re-embedding" warnings in the logs









