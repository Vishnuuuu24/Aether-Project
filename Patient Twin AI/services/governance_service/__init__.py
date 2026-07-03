"""governance-service — consent lifecycle, audit query, outcome capture, version
registry (docs/07 §2 & §7; T5.1).

This is the production backing for consent (the other services carry only a local
`ConsentProvider` port), the append-only audit trail's read side, and the
outer-loop outcome store (docs/11 §3). Every mutation here is itself audited into
the same hash-chain, and nothing in this service is writable by the LLM
(CLAUDE.md principle 5).
"""
