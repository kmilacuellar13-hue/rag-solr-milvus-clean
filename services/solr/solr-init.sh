#!/usr/bin/env sh
set -e
SOLR="http://solr:8983/solr/rag2/schema"
curl -s -X POST "$SOLR" -H 'Content-Type: application/json' --data @/schema.json || true
echo "Schema aplicado en Solr (campo 'text')."

