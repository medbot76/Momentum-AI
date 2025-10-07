#!/usr/bin/env python3
"""
Backup and recovery procedures for Med-Bot Supabase database
"""
import os
import json
import datetime
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

def create_backup_procedures():
    """Generate backup and recovery procedures documentation"""
    
    print("💾 Med-Bot Backup & Recovery Procedures")
    print("=" * 45)
    
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    
    # Generate backup procedure script
    backup_script = """#!/bin/bash
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
pg_dump "$SUPABASE_DB_URL" --schema-only --no-owner --no-privileges \\
    > "$BACKUP_DIR/${BACKUP_NAME}_schema.sql"

# 2. Export data only  
echo "💾 Backing up database data..."
pg_dump "$SUPABASE_DB_URL" --data-only --no-owner --no-privileges \\
    --exclude-table=auth.* --exclude-table=storage.* \\
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
"""
    
    # Write backup script
    backup_script_path = Path("backup_database.sh")
    with open(backup_script_path, 'w') as f:
        f.write(backup_script)
    
    # Make executable
    os.chmod(backup_script_path, 0o755)
    
    print(f"✅ Created backup script: {backup_script_path}")
    
    return True

def create_recovery_procedures():
    """Generate recovery procedures documentation"""
    
    recovery_script = """#!/bin/bash
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
"""
    
    # Write recovery script
    recovery_script_path = Path("restore_database.sh")
    with open(recovery_script_path, 'w') as f:
        f.write(recovery_script)
    
    # Make executable
    os.chmod(recovery_script_path, 0o755)
    
    print(f"✅ Created recovery script: {recovery_script_path}")
    
    return True

def create_maintenance_scripts():
    """Create database maintenance scripts"""
    
    # Health check script
    health_check_script = """#!/usr/bin/env python3
\"\"\"
Database health check and maintenance script
\"\"\"
import os
import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

def health_check():
    load_dotenv()
    
    print("🏥 Med-Bot Database Health Check")
    print("=" * 35)
    
    try:
        supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))
        
        # Check table row counts
        tables = ['profiles', 'notebooks', 'documents', 'conversations', 'messages', 'user_settings']
        total_records = 0
        
        print("📊 Table Statistics:")
        for table in tables:
            try:
                result = supabase.table(table).select('*', count='exact').execute()
                count = result.count if hasattr(result, 'count') else len(result.data)
                total_records += count
                print(f"   {table}: {count:,} records")
            except Exception as e:
                print(f"   {table}: Error - {str(e)}")
        
        print(f"\\n📈 Total Records: {total_records:,}")
        
        # Check storage usage
        print("\\n💾 Storage Check:")
        try:
            buckets = supabase.storage.list_buckets()
            print(f"   Storage buckets: {len(buckets)}")
            for bucket in buckets:
                print(f"   - {bucket.name} ({'public' if bucket.public else 'private'})")
        except Exception as e:
            print(f"   Storage error: {str(e)}")
        
        print(f"\\n✅ Health check completed at {datetime.datetime.now()}")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False

if __name__ == "__main__":
    health_check()
"""
    
    # Write health check script
    health_script_path = Path("health_check.py")
    with open(health_script_path, 'w') as f:
        f.write(health_check_script)
    
    print(f"✅ Created health check script: {health_script_path}")
    
    return True

def create_documentation():
    """Create comprehensive backup/recovery documentation"""
    
    doc_content = """# Med-Bot Database Backup & Recovery Procedures

## Overview
This document outlines backup and recovery procedures for the Med-Bot Supabase database.

## Prerequisites
- Supabase CLI installed
- PostgreSQL client tools (pg_dump, psql)
- Python environment with supabase-py
- Proper environment variables configured

## Backup Procedures

### 1. Automated Backup
```bash
# Run automated backup
./backup_database.sh

# Run backup with custom name
./backup_database.sh medbot_backup_before_migration
```

### 2. Manual Backup
```bash
# Schema only
pg_dump "$SUPABASE_DB_URL" --schema-only > schema_backup.sql

# Data only
pg_dump "$SUPABASE_DB_URL" --data-only > data_backup.sql

# Full backup
pg_dump "$SUPABASE_DB_URL" > full_backup.sql
```

### 3. Application-Level Backup
```bash
# Export key data as JSON
python3 -c "
from supabase import create_client
import json, os
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# Export profiles
profiles = supabase.table('profiles').select('*').execute()
with open('profiles_backup.json', 'w') as f:
    json.dump(profiles.data, f, indent=2, default=str)
"
```

## Recovery Procedures

### 1. Full Database Restore
```bash
# Restore from backup
./restore_database.sh medbot_backup_20240101_120000

# Verify restoration
python health_check.py
```

### 2. Partial Restore (Specific Tables)
```bash
# Restore only specific tables
psql "$SUPABASE_DB_URL" -c "TRUNCATE profiles CASCADE;"
psql "$SUPABASE_DB_URL" -c "\\copy profiles FROM 'profiles_backup.csv' WITH CSV HEADER;"
```

### 3. Schema-Only Restore
```bash
# Restore schema without data
psql "$SUPABASE_DB_URL" < schema_backup.sql
```

## Monitoring & Maintenance

### Daily Checks
- Run health check: `python health_check.py`
- Monitor storage usage
- Check connection pool status

### Weekly Tasks
- Full database backup
- Review slow query logs
- Clean up old backup files

### Monthly Tasks
- VACUUM ANALYZE on all tables
- Review and optimize indexes
- Capacity planning review

## Emergency Procedures

### Data Corruption
1. Stop application immediately
2. Assess corruption scope
3. Restore from latest good backup
4. Verify data integrity
5. Resume operations

### Complete Database Loss
1. Create new Supabase project if necessary
2. Apply schema from migration files
3. Restore data from latest backup
4. Update environment variables
5. Test all functionality

## Backup Retention Policy

- Daily backups: Keep for 30 days
- Weekly backups: Keep for 6 months  
- Monthly backups: Keep for 2 years
- Pre-migration backups: Keep indefinitely

## Storage Requirements

- Estimated backup size: ~10MB per 10K documents
- Compressed backups: ~50% reduction
- Vector embeddings: Largest data component

## Testing Recovery

### Monthly Recovery Test
1. Create test environment
2. Restore latest backup
3. Verify all functionality
4. Document any issues
5. Update procedures as needed

## Contacts & Escalation

- Database Admin: [Your Email]
- Supabase Support: support@supabase.io
- Emergency: [Emergency Contact]

## Related Files

- `backup_database.sh` - Automated backup script
- `restore_database.sh` - Recovery script  
- `health_check.py` - Health monitoring
- `optimize_database.py` - Performance tuning
- `supabase/migrations/` - Schema migrations
"""
    
    # Write documentation
    doc_path = Path("BACKUP_RECOVERY.md")
    with open(doc_path, 'w') as f:
        f.write(doc_content)
    
    print(f"✅ Created documentation: {doc_path}")
    
    return True

def main():
    """Main backup/recovery setup routine"""
    print("💾 Setting up Backup & Recovery Procedures")
    print("=" * 45)
    
    success = all([
        create_backup_procedures(),
        create_recovery_procedures(), 
        create_maintenance_scripts(),
        create_documentation()
    ])
    
    if success:
        print("\n🎉 Backup & Recovery setup completed!")
        print("\n📝 Next steps:")
        print("   1. Test backup: ./backup_database.sh test_backup")
        print("   2. Run health check: python health_check.py") 
        print("   3. Schedule automated backups")
        print("   4. Review BACKUP_RECOVERY.md documentation")
    else:
        print("\n❌ Some setup steps failed")
    
    return success

if __name__ == "__main__":
    main()