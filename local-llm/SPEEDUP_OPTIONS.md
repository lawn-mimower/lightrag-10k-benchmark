# LightRAG Indexing Speed Options

## Current Setup (Optimized Graph Mode)
- **Mode**: Full knowledge graph extraction
- **Speed**: ~48x faster than original
- **Accuracy**: High (entity + relationship extraction)

## Ultra-Fast Option: Naive Mode

If indexing is STILL too slow, use **naive mode** which skips graph extraction entirely:

```python
# When querying, use mode="naive" instead of "hybrid"
response = await rag.aquery(
    flagged_query,
    param=QueryParam(mode="naive")  # Skip graph, use vector search only
)
```

### Mode Comparison:

| Mode | Entity Extraction | Relationship Extraction | Speed | Accuracy |
|------|------------------|------------------------|-------|----------|
| `hybrid` (default) | ✅ Yes | ✅ Yes | Slow | Highest |
| `local` | ✅ Yes | ✅ Yes | Slow | High |
| `global` | ✅ Yes | ✅ Yes | Slow | High |
| `naive` | ❌ No | ❌ No | **FASTEST** | Good |

### When to Use Naive Mode:
- ✅ Quick benchmarking
- ✅ Dense documents (like 10-Ks) where entities are hard to extract
- ✅ Limited GPU/CPU resources
- ❌ Not ideal for final evaluation (lower accuracy)

## Alternative: Reduce Document Count

Set a limit in the config:
```python
LIMIT_DOCS = 100  # Process only 100 documents instead of 5832
```

## Monitoring Progress

LightRAG prints these during indexing:
```
INFO: Processing X document(s)
INFO: Extracting stage Y/Z
```

Watch for:
- **Stage count increasing** = Progress is being made
- **Same stage stuck** = Possible LLM formatting error (check logs)
