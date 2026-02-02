# T4DM Database Migrations

This directory contains Cypher migration scripts for Neo4j schema changes.

## Naming Convention

Migration files should be named: `{version}_{description}.cypher`

Example:
- `001_initial_schema.cypher`
- `002_add_fsrs_indexes.cypher`
- `003_add_entity_types.cypher`

## Format

Each migration file should:
1. Be idempotent (safe to run multiple times)
2. Include comments explaining the changes
3. Use `CREATE INDEX IF NOT EXISTS` and similar safe operations

## Example Migration

```cypher
// 001_initial_schema.cypher
// Initial schema setup for T4DM

// Create indexes for Episode nodes
CREATE INDEX episode_id_idx IF NOT EXISTS FOR (e:Episode) ON (e.id);
CREATE INDEX episode_timestamp_idx IF NOT EXISTS FOR (e:Episode) ON (e.timestamp);

// Create indexes for Entity nodes
CREATE INDEX entity_name_idx IF NOT EXISTS FOR (ent:Entity) ON (ent.name);
CREATE INDEX entity_type_idx IF NOT EXISTS FOR (ent:Entity) ON (ent.type);

// Create indexes for Procedure nodes
CREATE INDEX procedure_name_idx IF NOT EXISTS FOR (p:Procedure) ON (p.name);
CREATE INDEX procedure_version_idx IF NOT EXISTS FOR (p:Procedure) ON (p.version);

// Create constraints
CREATE CONSTRAINT episode_id_unique IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (ent:Entity) REQUIRE ent.name IS UNIQUE;
```

## Running Migrations

Use the migration script:

```bash
# Apply all pending migrations
./scripts/migrate.sh

# Check current schema version
docker exec ww-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "MATCH (v:SchemaVersion) RETURN v.version ORDER BY v.applied DESC LIMIT 1"
```

## Schema Version Tracking

The migration script automatically creates SchemaVersion nodes:

```cypher
(:SchemaVersion {
  version: "001",
  applied: datetime()
})
```

## Best Practices

1. **Test in development first** - Always test migrations on dev/staging before production
2. **Backup before migrating** - Run `./scripts/backup.sh` before migrations
3. **One migration per version** - Keep migrations focused and incremental
4. **Document breaking changes** - Add comments for any schema changes that affect queries
5. **Use transactions** - Wrap large migrations in transactions when possible
