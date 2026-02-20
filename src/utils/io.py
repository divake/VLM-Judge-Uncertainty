"""Atomic save/load utilities with crash-safe writing and resumability checks."""

import json
import os
import tempfile

import torch


def atomic_save_json(data, path):
    """Write JSON atomically: write to temp file, then rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.rename(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_save_pt(tensor_dict, path):
    """Write PyTorch tensors atomically: write to temp file, then rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), suffix=".tmp"
    )
    os.close(fd)
    try:
        torch.save(tensor_dict, tmp_path)
        os.rename(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pt(path):
    """Load a PyTorch file."""
    return torch.load(path, map_location="cpu", weights_only=False)


def sample_exists(output_dir, sample_id):
    """Check if both .json and .pt outputs exist for a sample (resumability)."""
    json_path = os.path.join(output_dir, f"sample_{sample_id:04d}.json")
    pt_path = os.path.join(output_dir, f"sample_{sample_id:04d}.pt")
    return os.path.exists(json_path) and os.path.exists(pt_path)


def get_sample_paths(output_dir, sample_id):
    """Return (json_path, pt_path) for a given sample ID."""
    json_path = os.path.join(output_dir, f"sample_{sample_id:04d}.json")
    pt_path = os.path.join(output_dir, f"sample_{sample_id:04d}.pt")
    return json_path, pt_path


def load_yaml(path):
    """Load a YAML config file."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
