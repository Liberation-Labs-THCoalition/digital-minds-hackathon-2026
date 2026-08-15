#!/usr/bin/env python3
"""Convert Expedition JSONL memories to Variable Landing JSON format.

The Expedition stores memories as one JSON object per line in
mnemosyne_memories.jsonl. The Variable Landing experiment expects
a JSON array via --memories.

Usage:
    python convert_expedition_memories.py \
        --input data/orientation/mnemosyne_memories.jsonl \
        --output expedition_memories.json
"""
import argparse
import json
from pathlib import Path


def convert(input_path: str, output_path: str):
    memories = []
    seen_ids = set()

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            mem_id = raw.get("id", f"mem_{len(memories)}")
            if mem_id in seen_ids:
                continue
            seen_ids.add(mem_id)

            content = raw.get("content", "")
            if "\n\nAgent response: " in content:
                parts = content.split("\n\nAgent response: ", 1)
                content = parts[1] if len(parts) > 1 else content

            memories.append({
                "id": mem_id,
                "content": content[:500],
                "entity": raw.get("entity", "expedition_agent"),
                "task_prompt": "What do you remember about this?",
                "marker_tokens": raw.get("marker_tokens", []),
                "scene": raw.get("scene", ""),
            })

    Path(output_path).write_text(json.dumps(memories, indent=2))
    print(f"Converted {len(memories)} memories: {input_path} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="expedition_memories.json")
    args = parser.parse_args()
    convert(args.input, args.output)
