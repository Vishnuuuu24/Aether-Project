# core — shared governance & platform libraries

No business decisions live here (those belong to the engines and the Policy
Engine). `core` provides the cross-cutting primitives every service depends on:

| package | purpose | spec |
|---|---|---|
| `core.auth` | JWT verification, RBAC, **consent enforcement** (`schemas.consent`) | docs/07 §1–2, docs/06 |
| `core.audit` | **hash-chained** append-only audit writer + verifier (`schemas.audit`) | docs/04 §7, docs/06 §8 |
| `core.versioning` | active model/ruleset/prompt/baseline-engine/schema version registry | docs/04, docs/06 §8 |
| `core.db` | SQLAlchemy models + alembic baseline migration for `schemas/` | docs/04 §3 |

## Decisions taken for T0.2 (confirm at review)

These were under-specified in the docs; chosen defaults are recorded here so they
are not silently re-litigated. (CLAUDE.md flags consent + audit as do-not-guess.)

1. **Audit chain scope = single global append-only chain.** One `prev_hash → hash`
   chain across all records, totally ordered by an insertion sequence. Strongest
   tamper-evidence; the audit record holds only refs/versions/hashes, never raw PHI,
   so a global chain does not conflict with per-patient data erasure. Per-patient
   sub-chains remain a possible future enhancement (the writer is store-abstracted).

2. **Audit hash = SHA-256 over the full record minus `hash`.** Preimage is canonical
   compact JSON: keys sorted, UTC RFC3339 timestamps, UUIDs/enums stringified,
   `separators=(",",":")`. The preimage includes `prev_hash` (this is what chains the
   record to its predecessor) and every other field except `hash` itself. Genesis
   `prev_hash` is 64 zeros. Tamper-evident on id and timestamp as well as payload.

3. **Consent gate = deny-by-default, all actors including `system`.** A data-processing
   operation declares a required `ConsentScope`; it proceeds only if the patient's
   current, non-revoked `Consent` covers it. The internal pipeline (`system` actor) is
   **not** exempt — internal processing is still processing (CLAUDE.md: "no processing
   without valid consent"). The gate guards data-processing ops (vitals / documents /
   copilot / forecast); consent-management and audit/governance endpoints are governed
   by RBAC, not by the consent gate (a patient must be able to revoke even when revoked).
