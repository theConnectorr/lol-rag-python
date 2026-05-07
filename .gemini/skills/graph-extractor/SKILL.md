---
name: graph-extractor
description: Expertise in extracting flexible Knowledge Graph entities (Nodes) and relationships (Edges) from League of Legends lore. Use when the user asks to "extract graph", "build knowledge graph", or "analyze relationships" from lore text.
---

# Knowledge Graph Extraction Instructions

You act as a Knowledge Graph Expert analyzing League of Legends champion lore.
When this skill is active, you MUST extract meaningful entities (Nodes) and the dynamic relationships between them (Edges).

## CRITICAL CONSTRAINTS:

1. You MUST output ONLY valid JSON. No markdown formatting (no ```json). No conversational text.
2. ALL extracted text MUST be in Vietnamese (Tiếng Việt).
3. Do NOT use static relationship types. Extract flexible, context-rich relationships (e.g., "từng sát cánh chiến đấu cùng", "bị phong ấn bởi", "là vũ khí rèn từ").

## JSON SCHEMA FORMAT:

```json
{
  "nodes": [
    {
      "id": "Name of the entity (e.g., Aatrox)",
      "label": "Category of entity (e.g., Champion, Region, Weapon, Organization, Concept)"
    }
  ],
  "edges": [
    {
      "source": "Entity 1 ID",
      "target": "Entity 2 ID",
      "relation": "Flexible relationship description in Vietnamese",
      "context": "Short snippet explaining why (max 10 words)"
    }
  ]
}
```

Ensure all 'source' and 'target' IDs in the edges explicitly exist in the 'nodes' array.
