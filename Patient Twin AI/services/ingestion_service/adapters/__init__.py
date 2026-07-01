"""Ingest adapters. Each maps a source payload to CANONICAL reading dicts, then
everything funnels through the one normaliser — adapters never build/validate
Readings themselves (docs/02 §2). Adding a device is a mapping change here, not a
new call site (respecting the stable normalisation boundary).
"""
