"""Ingestion Service (docs/02 §2, docs/07 §3).

Responsibility: adapter normalisation → per-reading schema + consent check, then
hand accepted readings to the SQI stage. It does NOT compute SQI/baselines, write
the PSG, or call the LLM — those are downstream modules.
"""
