"""Fine-tune the cross-encoder reranker on nfcorpus (docs/16 Sprint 11).

    python -m ai.training.train_reranker            # full-quality run (all train queries)
    python -m ai.training.train_reranker --smoke    # tiny end-to-end validation

Full fine-tune of `bge-reranker-v2-m3` on the BEIR nfcorpus medical-IR benchmark, using
real qrels relevance (never fabricated labels). Positives come from the qrels; negatives
are BM25 hard negatives (`ai/training/rerank_pairs.py`). Best-validation checkpoint is
kept via a reranking evaluator (MRR@10) on the DEV split — the TEST split is held out
and only touched by the final honest lift eval (`ai/training/rerank_eval.py`), which
reports Recall@10 / MRR / nDCG@10 for BM25-only vs base bge-reranker vs the fine-tuned
model. Writes a content-addressed checkpoint (`checkpoints/<version>/model/` +
`manifest.json`) and an advisory `promotion.json`; promotion stays human-gated (§5).

Serving: the reranker ships as HF weights loaded by `FineTunedReranker` (the seam's new
implementation) — the NumPy-serving rule is for biosignal encoders, not this text model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ai.eval_datasets.nfcorpus import (
    DEFAULT_NFCORPUS_ROOT,
    IrSplit,
    NfcorpusLayoutError,
    load_nfcorpus_split,
    nfcorpus_available,
)
from ai.retrieval.bm25 import BM25
from ai.retrieval.reranker import FINETUNED_RERANKER_VERSION, FineTunedReranker
from ai.training.checkpoints import DEFAULT_CHECKPOINT_ROOT
from ai.training.promotion import Bar, evaluate_promotion, write_promotion_recommendation
from ai.training.rerank_eval import LiftResult, evaluate_rerank_lift
from ai.training.rerank_pairs import build_rerank_examples

RERANKER_NAME = "bge-reranker-v2-m3-nfcorpus"
BASE_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RerankTrainConfig:
    base_model: str = BASE_MODEL
    epochs: int = 2
    batch_size: int = 16
    negatives_per_positive: int = 4
    max_positives_per_query: int | None = 10  # cap the noisy broad-query tail (quality)
    lr: float = 2e-5
    max_length: int = 512
    eval_k: int = 10
    pool_size: int = 100
    seed: int = 0
    max_train_queries: int | None = None  # None = full-quality (every train query)
    max_dev_queries: int | None = 200  # bound best-val selection cost (subset of dev)
    max_eval_queries: int | None = None  # cap held-out TEST eval queries (smoke only)


def _content_version(config: RerankTrainConfig, provenance: dict[str, object]) -> str:
    identity = {"name": RERANKER_NAME, "config": asdict(config), "provenance": provenance}
    blob = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    return f"{RERANKER_NAME}@{hashlib.sha256(blob).hexdigest()[:12]}"


def _reranking_eval_samples(
    split: IrSplit, *, negatives: int, max_positives: int, seed: int
) -> list[dict[str, object]]:
    """Build CERerankingEvaluator samples for best-checkpoint selection (MRR@10 on DEV).

    Positives (BM25-ordered) are capped at `max_positives` and negatives at a small
    multiple — otherwise broad queries produce a huge, mostly-positive candidate list
    that makes the selection metric meaningless. This is the *selection* signal only; the
    authoritative held-out lift is measured on TEST with the full relevant set."""
    import random

    doc_ids = list(split.corpus.keys())
    bm25 = BM25([split.corpus[d] for d in doc_ids])
    rng = random.Random(seed)
    samples: list[dict[str, object]] = []
    for qid in split.query_ids:
        query = split.queries[qid]
        judged = set(split.qrels.get(qid, {}))
        relevant = split.relevant_ids(qid)
        if not relevant:
            continue
        scores = bm25.scores(query)
        ranked = sorted(range(len(doc_ids)), key=lambda i: (scores[i], doc_ids[i]), reverse=True)
        positives = [doc_ids[i] for i in ranked if doc_ids[i] in relevant][:max_positives]
        n_neg = negatives * len(positives)
        negs = [doc_ids[i] for i in ranked[: n_neg + 20] if doc_ids[i] not in judged][:n_neg]
        if not negs:
            pool = [d for d in doc_ids if d not in judged]
            rng.shuffle(pool)
            negs = pool[:n_neg]
        samples.append(
            {
                "query": query,
                "positive": [split.corpus[d] for d in positives],
                "negative": [split.corpus[d] for d in negs],
            }
        )
    return samples


def _limit_queries(split: IrSplit, n: int | None) -> IrSplit:
    if n is None or n >= len(split.query_ids):
        return split
    keep = set(sorted(split.query_ids)[:n])
    return IrSplit(
        split=split.split,
        queries={q: t for q, t in split.queries.items() if q in keep},
        corpus=split.corpus,
        qrels={q: v for q, v in split.qrels.items() if q in keep},
    )


def _lift_line(tag: str, lift: LiftResult) -> str:
    b, r = lift.baseline, lift.reranked
    return (
        f"  {tag:<22} Recall@{r.k} {r.recall_at_k:.4f}  MRR {r.mrr:.4f}  "
        f"nDCG@{r.k} {r.ndcg_at_k:.4f}   (BM25-only nDCG {b.ndcg_at_k:.4f})"
    )


def run(
    *,
    nfcorpus_root: Path = DEFAULT_NFCORPUS_ROOT,
    config: RerankTrainConfig | None = None,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    device: str | None = None,
) -> int:
    config = config or RerankTrainConfig()
    if not nfcorpus_available(nfcorpus_root):
        print(f"nfcorpus not found under {nfcorpus_root} — nothing to train.", file=sys.stderr)
        return 2

    from sentence_transformers import CrossEncoder, InputExample
    from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
    from torch.utils.data import DataLoader

    train = _limit_queries(load_nfcorpus_split("train", nfcorpus_root), config.max_train_queries)
    dev = _limit_queries(load_nfcorpus_split("dev", nfcorpus_root), config.max_dev_queries)
    test = _limit_queries(load_nfcorpus_split("test", nfcorpus_root), config.max_eval_queries)
    print(
        f"train queries: {len(train.query_ids)}  dev: {len(dev.query_ids)}  "
        f"test: {len(test.query_ids)}"
    )

    examples = build_rerank_examples(
        train,
        negatives_per_positive=config.negatives_per_positive,
        max_positives_per_query=config.max_positives_per_query,
        seed=config.seed,
    )
    n_pos = sum(1 for e in examples if e.label == 1.0)
    print(f"train pairs: {len(examples)}  (positives {n_pos}, negatives {len(examples) - n_pos})")

    train_samples = [InputExample(texts=[e.query, e.passage], label=e.label) for e in examples]
    loader = DataLoader(train_samples, shuffle=True, batch_size=config.batch_size)
    steps_per_epoch = max(1, len(loader))
    warmup = int(0.1 * steps_per_epoch * config.epochs)
    dev_samples = _reranking_eval_samples(
        dev,
        negatives=config.negatives_per_positive,
        max_positives=config.max_positives_per_query or 10,
        seed=config.seed,
    )
    evaluator = CERerankingEvaluator(
        dev_samples, at_k=config.eval_k, name="nfcorpus-dev", write_csv=False
    )

    provenance: dict[str, object] = {
        "dataset": "BEIR nfcorpus (medical IR)",
        "base_model": config.base_model,
        "negatives": "BM25 hard negatives",
        "train_queries": len(train.query_ids),
        "train_pairs": len(examples),
        "split": "BEIR train/dev/test qrels (test held out)",
    }
    version = _content_version(config, provenance)
    out_dir = checkpoint_root / version
    model_dir = out_dir / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline arm: the *un-fine-tuned* bge-reranker, before we touch it (held-out test).
    base_model = CrossEncoder(
        config.base_model, num_labels=1, max_length=config.max_length, device=device
    )
    print("scoring base bge-reranker on held-out test (pre-fine-tune)…", flush=True)
    base_lift = evaluate_rerank_lift(
        test,
        FineTunedReranker(base_model, version="base"),
        k=config.eval_k,
        pool_size=config.pool_size,
    )
    print(_lift_line("base bge-reranker", base_lift))

    print(f"\nfine-tuning ({config.epochs} epochs, batch {config.batch_size}, "
          f"{steps_per_epoch} steps/epoch, warmup {warmup}) → {model_dir}", flush=True)
    base_model.fit(
        train_dataloader=loader,
        evaluator=evaluator,
        epochs=config.epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": config.lr},
        output_path=str(model_dir),
        save_best_model=True,
        evaluation_steps=steps_per_epoch,  # once per epoch (dev eval is itself costly)
        use_amp=False,
        show_progress_bar=True,
    )

    # Reload the best-validation checkpoint via the seam and score it on held-out test.
    tuned = FineTunedReranker.from_checkpoint(out_dir)
    if tuned.is_fallback:
        print("ERROR: fine-tuned checkpoint did not load back through the seam.", file=sys.stderr)
        return 1
    print("\nscoring fine-tuned reranker on held-out test…", flush=True)
    tuned_lift = evaluate_rerank_lift(test, tuned, k=config.eval_k, pool_size=config.pool_size)
    print(_lift_line("fine-tuned", tuned_lift))

    bm25_only = base_lift.baseline  # identical BM25 candidate order in both arms
    metrics = {
        "bm25_ndcg": bm25_only.ndcg_at_k,
        "base_ndcg": base_lift.reranked.ndcg_at_k,
        "tuned_ndcg": tuned_lift.reranked.ndcg_at_k,
        "base_mrr": base_lift.reranked.mrr,
        "tuned_mrr": tuned_lift.reranked.mrr,
        "tuned_recall": tuned_lift.reranked.recall_at_k,
        "ndcg_lift_vs_base": tuned_lift.reranked.ndcg_at_k - base_lift.reranked.ndcg_at_k,
        "ndcg_lift_vs_bm25": tuned_lift.reranked.ndcg_at_k - bm25_only.ndcg_at_k,
    }

    manifest = {
        "name": RERANKER_NAME,
        "version": version,
        "kind": "cross_encoder_reranker",
        "base_version": FINETUNED_RERANKER_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "provenance": provenance,
        "metrics": metrics,
        "held_out": {"split": "test", "n_queries": _n_queries(test)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    recommendation = evaluate_promotion(
        version,
        [
            Bar("ndcg@10_vs_base", metrics["tuned_ndcg"], metrics["base_ndcg"],
                higher_is_better=True),
            Bar("ndcg@10_vs_bm25", metrics["tuned_ndcg"], metrics["bm25_ndcg"],
                higher_is_better=True),
        ],
    )
    write_promotion_recommendation(recommendation, out_dir)
    rec = "✅ RECOMMENDED" if recommendation.recommended else "not recommended"

    print(f"\nBM25-only nDCG@{config.eval_k} {metrics['bm25_ndcg']:.4f}  |  "
          f"base {metrics['base_ndcg']:.4f}  |  fine-tuned {metrics['tuned_ndcg']:.4f}")
    print(
        f"lift vs base: {metrics['ndcg_lift_vs_base']:+.4f}   "
        f"lift vs BM25: {metrics['ndcg_lift_vs_bm25']:+.4f}"
    )
    print(f"promotion: {rec} (advisory; human-gated) — {recommendation.rationale}")
    print(f"checkpoint: {out_dir}")
    print("\n── log entry (paste into docs/17_Training_Log.md, then add judgement) ──")
    print(f"| run | `{version}` |")
    print(f"| held-out | nfcorpus test · n={_n_queries(test)} queries |")
    print(f"| nDCG@10 BM25 / base / tuned | {metrics['bm25_ndcg']:.4f} / "
          f"{metrics['base_ndcg']:.4f} / **{metrics['tuned_ndcg']:.4f}** |")
    print(f"| MRR base / tuned | {metrics['base_mrr']:.4f} / **{metrics['tuned_mrr']:.4f}** |")
    return 0


def _n_queries(split: IrSplit) -> int:
    return len(split.query_ids)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sprint 11 cross-encoder reranker fine-tune.")
    parser.add_argument("--root", default=str(DEFAULT_NFCORPUS_ROOT))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--max-positives", type=int, default=10,
                        help="cap positives/query (quality: drops noisy broad-query tail)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None, help="torch device (default: auto/mps)")
    parser.add_argument("--max-train-queries", type=int, default=None,
                        help="cap train queries (debug only; omit for full-quality)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny end-to-end validation: 20 train queries, 1 epoch")
    args = parser.parse_args(argv)

    if args.smoke:
        config = RerankTrainConfig(
            epochs=1, batch_size=args.batch_size, max_train_queries=20,
            max_dev_queries=10, max_eval_queries=25, pool_size=40,
        )
    else:
        config = RerankTrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, negatives_per_positive=args.negatives,
            max_positives_per_query=args.max_positives, lr=args.lr, max_length=args.max_length,
            max_train_queries=args.max_train_queries,
        )
    try:
        return run(nfcorpus_root=Path(args.root), config=config, device=args.device)
    except (NfcorpusLayoutError, FileNotFoundError, ValueError) as exc:
        print(f"reranker training failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
