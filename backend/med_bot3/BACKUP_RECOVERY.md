# Med-Bot Database Backup & Recovery Procedures

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
psql "$SUPABASE_DB_URL" -c "\copy profiles FROM 'profiles_backup.csv' WITH CSV HEADER;"
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
