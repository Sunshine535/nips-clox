"""Monkey-patch huggingface_hub.file_exists to handle offline/local paths.

vllm 0.6.3 calls file_exists() unconditionally, which hits the network
even when model is a local path. This patch short-circuits for local
paths and honors HF_HUB_OFFLINE for cached repos.

Import this BEFORE importing vllm.
"""
from __future__ import annotations
import os
from pathlib import Path


def _patched_file_exists(repo_id, filename, *args, **kwargs):
    """Return True if the file exists locally; bypass network if offline."""
    # Local directory path
    if repo_id.startswith("/") or repo_id.startswith("."):
        p = Path(repo_id) / filename
        return p.exists()

    # Check HF cache for cached repo_id
    cache_dirs = [
        os.environ.get("HF_HUB_CACHE", ""),
        os.environ.get("TRANSFORMERS_CACHE", ""),
        os.path.join(os.environ.get("HF_HOME", ""), "hub"),
    ]
    for cache in cache_dirs:
        if not cache or not os.path.isdir(cache):
            continue
        repo_dir = os.path.join(cache, f"models--{repo_id.replace('/', '--')}")
        snaps = os.path.join(repo_dir, "snapshots")
        if os.path.isdir(snaps):
            for snap in os.listdir(snaps):
                cand = os.path.join(snaps, snap, filename)
                if os.path.exists(cand):
                    return True

    # If offline mode is on, don't try network
    if os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true", "True"):
        return False

    # Fall through to real implementation
    from huggingface_hub.hf_api import file_exists as _real
    return _real(repo_id, filename, *args, **kwargs)


def apply():
    import huggingface_hub
    huggingface_hub.file_exists = _patched_file_exists
    from huggingface_hub import hf_api
    hf_api.file_exists = _patched_file_exists


apply()
