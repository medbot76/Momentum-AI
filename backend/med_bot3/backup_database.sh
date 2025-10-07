#!/bin/bash
# Med-Bot Database Backup Script
# Usage: ./backup_database.sh [backup_name]

set -e

BACKUP_NAME=${1:-"medbot_backup_$(date +%Y%m%d_%H%M%S)"}
BACKUP_DIR="./backups"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "🚀 Starting Med-Bot Database Backup: $BACKUP_NAME"
echo "📅 Timestamp: $TIMESTAMP"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Load environment variables
source .env

# 1. Export schema
echo "📋 Backing up database schema..."
pg_dump "$SUPABASE_DB_URL" --schema-only --no-owner --no-privileges \
    > "$BACKUP_DIR/${BACKUP_NAME}_schema.sql"

# 2. Export data only  
echo "💾 Backing up database data..."
pg_dump "$SUPABASE_DB_URL" --data-only --no-owner --no-privileges \
    --exclude-table=auth.* --exclude-table=storage.* \
    > "$BACKUP_DIR/${BACKUP_NAME}_data.sql"

# 3. Export specific tables as JSON (for easier inspection)
echo "📊 Exporting key tables as JSON..."
python3 -c "
import os
import json
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

tables = ['profiles', 'notebooks', 'conversations']
for table in tables:
    try:
        result = supabase.table(table).select('*').execute()
        with open('$BACKUP_DIR/${BACKUP_NAME}_${table}.json', 'w') as f:
            json.dump(result.data, f, indent=2, default=str)
        print(f'✅ Exported {table}: {len(result.data)} records')
    except Exception as e:
        print(f'❌ Failed to export {table}: {e}')
"

# 4. Create backup manifest
echo "📝 Creating backup manifest..."
cat > "$BACKUP_DIR/${BACKUP_NAME}_manifest.json" << EOF
{
  "backup_name": "$BACKUP_NAME",
  "timestamp": "$TIMESTAMP",
  "database_url": "$(echo $SUPABASE_DB_URL | sed 's/:[^@]*@/:***@/')",
  "files": [
    "${BACKUP_NAME}_schema.sql",
    "${BACKUP_NAME}_data.sql",
    "${BACKUP_NAME}_profiles.json",
    "${BACKUP_NAME}_notebooks.json", 
    "${BACKUP_NAME}_conversations.json",
    "${BACKUP_NAME}_manifest.json"
  ],
  "backup_size": "$(du -sh $BACKUP_DIR/${BACKUP_NAME}* | tail -1 | cut -f1)"
}
EOF

echo "✅ Backup completed successfully!"
echo "📁 Backup files saved to: $BACKUP_DIR"
echo "📋 Manifest: $BACKUP_DIR/${BACKUP_NAME}_manifest.json"

# Optional: Compress backup
# echo "🗜️  Compressing backup..."
# tar -czf "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" -C "$BACKUP_DIR" ${BACKUP_NAME}*
# echo "✅ Compressed backup: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"
