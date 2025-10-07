#!/bin/bash
# Med-Bot Database Recovery Script
# Usage: ./restore_database.sh <backup_name>

set -e

if [ -z "$1" ]; then
    echo "❌ Usage: $0 <backup_name>"
    echo "📋 Available backups:"
    ls -la backups/*_manifest.json 2>/dev/null | sed 's/.*backups\///' | sed 's/_manifest.json//' || echo "No backups found"
    exit 1
fi

BACKUP_NAME="$1"
BACKUP_DIR="./backups"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "🔄 Starting Med-Bot Database Recovery: $BACKUP_NAME"
echo "📅 Timestamp: $TIMESTAMP"

# Load environment variables
source .env

# Verify backup files exist
if [ ! -f "$BACKUP_DIR/${BACKUP_NAME}_manifest.json" ]; then
    echo "❌ Backup manifest not found: $BACKUP_DIR/${BACKUP_NAME}_manifest.json"
    exit 1
fi

echo "📋 Backup manifest found:"
cat "$BACKUP_DIR/${BACKUP_NAME}_manifest.json"

read -p "⚠️  This will OVERWRITE the current database. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Recovery cancelled"
    exit 1
fi

# 1. Drop and recreate schema (DANGEROUS!)
echo "🗑️  WARNING: Dropping existing data..."
echo "📋 Restoring schema..."
psql "$SUPABASE_DB_URL" -c "
    DROP SCHEMA IF EXISTS public CASCADE;
    CREATE SCHEMA public;
    GRANT ALL ON SCHEMA public TO postgres;
    GRANT ALL ON SCHEMA public TO public;
"

# 2. Restore schema
echo "📋 Restoring database schema..."
psql "$SUPABASE_DB_URL" < "$BACKUP_DIR/${BACKUP_NAME}_schema.sql"

# 3. Restore data
echo "💾 Restoring database data..."
psql "$SUPABASE_DB_URL" < "$BACKUP_DIR/${BACKUP_NAME}_data.sql"

# 4. Verify restoration
echo "✅ Verifying restoration..."
python3 -c "
import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

tables = ['profiles', 'notebooks', 'documents', 'conversations', 'messages']
for table in tables:
    try:
        result = supabase.table(table).select('*').limit(1).execute()
        print(f'✅ {table}: Table accessible')
    except Exception as e:
        print(f'❌ {table}: {e}')
"

echo "✅ Database recovery completed!"
echo "📝 Remember to:"
echo "   - Test application functionality"
echo "   - Verify user access permissions"
echo "   - Check storage bucket contents"
