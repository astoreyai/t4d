# Migrations
**Path**: `/mnt/projects/t4d/t4dm/migrations/`

## What
Cypher migration scripts for Neo4j schema changes, versioned and tracked via SchemaVersion nodes.

## How
- Files named `{version}_{description}.cypher` (e.g., `001_initial_schema.cypher`)
- All migrations are idempotent (`CREATE INDEX IF NOT EXISTS`)
- Applied via `scripts/migrate.sh` which tracks SchemaVersion in Neo4j

## Why
Provides incremental, safe schema evolution for Neo4j graph database as the memory model evolves (new indexes, constraints, node types).

## Key Files
| File | Purpose |
|------|---------|
| `README.md` | Naming conventions, format, and best practices |
| `*.cypher` | Individual migration scripts (versioned) |

## Data Flow
```
*.cypher files → scripts/migrate.sh → Neo4j (schema changes)
                                     → SchemaVersion nodes (tracking)
```

## Integration Points
- **Scripts**: `scripts/migrate.sh` applies pending migrations
- **Deploy**: Run before deployment to update schema
- **Backup**: Always run `scripts/backup.sh` before migrating
