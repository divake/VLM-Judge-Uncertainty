#!/usr/bin/env python3
"""Download and inspect the MLLM-as-a-Judge dataset.

Usage:
    python scripts/download_data.py [--inspect-only]
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Download MLLM-Judge dataset")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Only inspect existing data, don't download")
    parser.add_argument("--data-dir", type=str, default="data/mllm_judge",
                        help="Directory to store dataset")
    args = parser.parse_args()

    if not args.inspect_only:
        download_mllm_judge(args.data_dir)

    inspect_data(args.data_dir)


def download_mllm_judge(data_dir):
    """Clone the MLLM-Judge repository."""
    raw_dir = os.path.join(data_dir, "raw")

    if os.path.exists(raw_dir) and os.listdir(raw_dir):
        print(f"[download] {raw_dir} already exists, skipping clone")
    else:
        print("[download] Cloning MLLM-Judge repo (text only, shallow)...")
        os.makedirs(data_dir, exist_ok=True)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Dongping-Chen/MLLM-Judge",
            raw_dir,
        ], check=True)

    print(f"\n[download] Contents of {raw_dir}:")
    for item in sorted(os.listdir(raw_dir)):
        path = os.path.join(raw_dir, item)
        if os.path.isdir(path):
            n_files = len(os.listdir(path))
            print(f"  {item}/ ({n_files} items)")
        else:
            size = os.path.getsize(path) / 1024
            print(f"  {item} ({size:.1f} KB)")


def inspect_data(data_dir):
    """Find and inspect JSON/JSONL files in the data directory."""
    print(f"\n{'='*60}")
    print(f"Inspecting data in {data_dir}")
    print(f"{'='*60}")

    json_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(('.json', '.jsonl')):
                json_files.append(os.path.join(root, f))

    if not json_files:
        print("No JSON/JSONL files found!")
        return

    print(f"\nFound {len(json_files)} JSON/JSONL files:")
    for fp in sorted(json_files):
        size = os.path.getsize(fp) / 1024
        print(f"  {os.path.relpath(fp, data_dir)} ({size:.1f} KB)")

    # Inspect scoring-related files
    for fp in sorted(json_files):
        name = os.path.basename(fp).lower()
        if "scor" in name or "data" in name or "eval" in name:
            print(f"\n--- Inspecting: {os.path.relpath(fp, data_dir)} ---")
            _inspect_file(fp, max_samples=3)


def _inspect_file(filepath, max_samples=3):
    """Print structure and sample entries from a JSON/JSONL file."""
    try:
        if filepath.endswith('.jsonl'):
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line for line in f if line.strip()]
            data = [json.loads(line) for line in lines[:max_samples]]
            total = len(lines)
            print(f"  Format: JSONL, {total} lines")
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            if isinstance(raw, list):
                total = len(raw)
                data = raw[:max_samples]
                print(f"  Format: JSON list, {total} items")
            elif isinstance(raw, dict):
                print(f"  Format: JSON dict, keys: {list(raw.keys())}")
                # Try to find the data array
                for key in ['data', 'samples', 'scoring', 'annotations', 'results']:
                    if key in raw and isinstance(raw[key], list):
                        total = len(raw[key])
                        data = raw[key][:max_samples]
                        print(f"  Using key '{key}': {total} items")
                        break
                else:
                    data = [raw]
                    total = 1

        if data:
            first = data[0]
            print(f"  Fields: {list(first.keys())}")

            # Look for score-like fields
            score_fields = [k for k in first.keys()
                           if any(s in k.lower() for s in ['score', 'rating', 'label', 'human'])]
            if score_fields:
                print(f"  Score fields: {score_fields}")
                for sf in score_fields:
                    vals = [item.get(sf) for item in data if sf in item]
                    print(f"    {sf} sample values: {vals}")

            # Look for image fields
            img_fields = [k for k in first.keys()
                         if any(s in k.lower() for s in ['image', 'img', 'photo', 'picture'])]
            if img_fields:
                print(f"  Image fields: {img_fields}")
                for imf in img_fields:
                    print(f"    {imf} example: {first.get(imf, '')[:100]}")

            print(f"\n  First sample (truncated):")
            for k, v in first.items():
                val_str = str(v)
                if len(val_str) > 120:
                    val_str = val_str[:120] + "..."
                print(f"    {k}: {val_str}")

    except Exception as e:
        print(f"  Error reading file: {e}")


if __name__ == "__main__":
    main()
