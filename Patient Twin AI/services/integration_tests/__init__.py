"""Cross-service integration tests (docs/15 T6.1; docs/11 §2 rung 2).

Unlike the per-service test packages, these drive a synthetic patient through the
*real* engines end to end — ingestion → SQI/features → baseline → PSG commit →
event/forecast → copilot answer — over one shared audit chain, and assert the
final OutputContract is Policy-approved and the chain reconstructs the whole path.
"""
