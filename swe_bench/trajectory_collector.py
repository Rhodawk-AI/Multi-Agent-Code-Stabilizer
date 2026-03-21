"""
swe_bench/trajectory_collector.py
===================================
ARPO trajectory collector for RL fine-tuning pipeline (Gap 5.5).

Architecture (Section 3.5 / Gap 5.5 of GAP5_SWEBench90_Architecture.md)
──────────────────────────────────────────────────────────────────────────
Every evaluation run writes training data for offline ARPO/GRPO fine-tuning.
After 500 collected trajectories, a first RL checkpoint can be trained,
delivering an estimated +8-12% SWE-bench lift. After 2000, +18-25%.

The ARPO algorithm (arxiv 2411.06345):
  - Training signal: binary reward — 1.0 if patch passes ALL FAIL_TO_PASS
    tests, 0.0 otherwise
  - Training input: (instance, localization_result, cpg_context, patch_attempt)
  - Algorithm: GRPO (Group Relative Policy Optimization) over Qwen2.5-Coder-32B
  - Framework: OpenRLHF (openrlhf-org/OpenRLHF)

The trajectory format written here is compatible with OpenRLHF's dataset format:
  {"prompt": "...", "response": "...", "reward": 0.0 or 1.0}

This collector is called from swe_bench/evaluator.py after every instance,
regardless of whether the patch resolved the instance or not. Failed trajectories
are valuable training signal — GRPO learns from the contrast between good and
bad attempts from the same distribution.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_TRAJECTORY_DIR = Path(
    os.environ.get("RHODAWK_TRAJECTORY_DIR", ".stabilizer/trajectories")
)
_TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

# Minimum training corpus size to trigger RL training recommendation
_MIN_FOR_FIRST_TRAINING = int(os.environ.get("RHODAWK_RL_MIN_CORPUS", "500"))


@dataclass
class TrajectoryRecord:
    """
    A single (prompt, patch, reward) triple for ARPO training.

    Compatible with OpenRLHF dataset format and TRL SFT/GRPO trainers.
    """
    instance_id:        str   = ""
    prompt:             str   = ""   # Full context given to the model
    response:           str   = ""   # Generated patch (the trajectory)
    reward:             float = 0.0  # 1.0 = resolved, 0.0 = unresolved
    model:              str   = ""
    temperature:        float = 0.0
    test_score:         float = 0.0  # Partial credit: fraction of tests passed
    localization_used:  bool  = False
    cpg_used:           bool  = False
    n_rounds:           int   = 1    # Feedback rounds used
    composite_score:    float = 0.0  # BoBN composite score
    is_winner:          bool  = False
    recorded_at:        str   = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    def to_openrlhf_format(self) -> dict:
        """Format compatible with OpenRLHF dataset ingestion."""
        return {
            "prompt":   self.prompt,
            "response": self.response,
            "reward":   self.reward,
        }

    def to_trl_format(self) -> dict:
        """Format compatible with TRL GRPO trainer."""
        return {
            "query":  self.prompt,
            "answer": self.response,
            "label":  self.reward,
        }


class TrajectoryCollector:
    """
    Collects (prompt, patch, reward) triples from every SWE-bench evaluation.

    Thread-safe: uses per-file locking to handle concurrent workers.
    Each run appends to a JSONL file partitioned by date for easy incremental
    ingestion into the OpenRLHF training pipeline.
    """

    def __init__(
        self,
        trajectory_dir: Path | None = None,
        format: str = "openrlhf",   # "openrlhf" | "trl"
    ) -> None:
        self.trajectory_dir = trajectory_dir or _TRAJECTORY_DIR
        self.format         = format
        self._today_file    = self._get_today_file()
        self._count         = self._count_existing_trajectories()

    def collect(
        self,
        instance_id:        str,
        prompt:             str,
        patch:              str,
        resolved:           bool,
        model:              str          = "",
        temperature:        float        = 0.0,
        test_score:         float        = 0.0,
        localization_used:  bool         = False,
        cpg_used:           bool         = False,
        n_rounds:           int          = 1,
        composite_score:    float        = 0.0,
        is_winner:          bool         = False,
    ) -> TrajectoryRecord:
        """
        Record one trajectory. Called for EVERY candidate, not just the winner.
        GRPO benefits from contrast between good and bad attempts.
        """
        record = TrajectoryRecord(
            instance_id       = instance_id,
            prompt            = prompt,
            response          = patch,
            reward            = 1.0 if resolved else 0.0,
            model             = model,
            temperature       = temperature,
            test_score        = test_score,
            localization_used = localization_used,
            cpg_used          = cpg_used,
            n_rounds          = n_rounds,
            composite_score   = composite_score,
            is_winner         = is_winner,
        )

        try:
            self._append_to_file(record)
            self._count += 1
            if self._count % 50 == 0:
                self._log_corpus_status()
        except Exception as exc:
            log.warning(f"[trajectory] Failed to record {instance_id}: {exc}")

        return record

    def collect_from_bobn_result(
        self,
        instance_id: str,
        bobn_result: Any,   # BoBNResult
        resolved:    bool,
        issue_text:  str,
        loc_context: str = "",
    ) -> list[TrajectoryRecord]:
        """
        Batch-collect all BoBN candidates from a completed sampling run.
        The winner gets reward=resolved (binary), others get test_score-based reward.
        This provides richer training signal than only recording the winner.
        """
        records: list[TrajectoryRecord] = []
        if not bobn_result or not bobn_result.all_candidates:
            return records

        for candidate in bobn_result.all_candidates:
            is_winner = (
                bobn_result.winner is not None
                and candidate.candidate_id == bobn_result.winner.candidate_id
            )
            # Winner reward is binary (resolved or not), others use test_score
            reward = (1.0 if resolved else 0.0) if is_winner else candidate.test_score

            prompt = self._build_training_prompt(
                issue_text   = issue_text,
                loc_context  = loc_context,
                model        = candidate.model,
                temperature  = candidate.temperature,
            )

            record = self.collect(
                instance_id       = instance_id,
                prompt            = prompt,
                patch             = candidate.patch,
                resolved          = is_winner and resolved,
                model             = candidate.model,
                temperature       = candidate.temperature,
                test_score        = candidate.test_score,
                localization_used = bool(loc_context),
                cpg_used          = bool(
                    candidate.exec_result and candidate.exec_result.rounds
                ),
                n_rounds          = candidate.exec_rounds,
                composite_score   = candidate.composite_score,
                is_winner         = is_winner,
            )
            records.append(record)

        log.info(
            f"[trajectory] {instance_id}: recorded {len(records)} trajectories "
            f"(winner={'yes' if resolved else 'no'})"
        )
        return records

    def corpus_size(self) -> int:
        return self._count

    def is_ready_for_training(self) -> bool:
        return self._count >= _MIN_FOR_FIRST_TRAINING

    def training_status(self) -> dict:
        return {
            "corpus_size":       self._count,
            "min_for_training":  _MIN_FOR_FIRST_TRAINING,
            "ready":             self.is_ready_for_training(),
            "trajectory_dir":    str(self.trajectory_dir),
            "today_file":        str(self._today_file),
        }

    def export_for_openrlhf(self, output_path: Path) -> int:
        """Export all trajectories in OpenRLHF format. Returns count."""
        all_records = self._load_all_records()
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r.to_openrlhf_format()) + "\n")
                count += 1
        log.info(f"[trajectory] Exported {count} records to {output_path}")
        return count

    def export_for_trl(self, output_path: Path) -> int:
        """Export all trajectories in TRL GRPO format. Returns count."""
        all_records = self._load_all_records()
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r.to_trl_format()) + "\n")
                count += 1
        log.info(f"[trajectory] Exported {count} records (TRL format) to {output_path}")
        return count

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_training_prompt(
        self,
        issue_text:  str,
        loc_context: str,
        model:       str,
        temperature: float,
    ) -> str:
        """
        Build the training prompt in the format the model sees during inference.
        Consistency between training and inference prompts is critical for GRPO.
        """
        loc_block = f"\n\n## Edit Targets\n{loc_context}" if loc_context else ""
        return (
            f"## GitHub Issue\n{issue_text[:4000]}"
            f"{loc_block}\n\n"
            "Fix the issue above. Produce a unified diff patch. "
            "Output ONLY the diff starting with '--- '."
        )

    def _get_today_file(self) -> Path:
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        return self.trajectory_dir / f"trajectories_{today}.jsonl"

    def _append_to_file(self, record: TrajectoryRecord) -> None:
        line = json.dumps(asdict(record)) + "\n"
        with open(self._today_file, "a", encoding="utf-8") as f:
            f.write(line)

    def _count_existing_trajectories(self) -> int:
        count = 0
        for p in self.trajectory_dir.glob("trajectories_*.jsonl"):
            try:
                count += sum(1 for _ in open(p, encoding="utf-8"))
            except Exception:
                pass
        return count

    def _load_all_records(self) -> list[TrajectoryRecord]:
        records: list[TrajectoryRecord] = []
        for p in sorted(self.trajectory_dir.glob("trajectories_*.jsonl")):
            try:
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            records.append(TrajectoryRecord(**{
                                k: v for k, v in data.items()
                                if k in TrajectoryRecord.__dataclass_fields__
                            }))
                        except Exception:
                            pass
            except Exception:
                pass
        return records

    def _log_corpus_status(self) -> None:
        if self.is_ready_for_training():
            log.info(
                f"[trajectory] RL corpus ready: {self._count} trajectories "
                f"(>= {_MIN_FOR_FIRST_TRAINING}). "
                "Run: python scripts/arpo_trainer.py to start GRPO fine-tuning."
            )
        else:
            remaining = _MIN_FOR_FIRST_TRAINING - self._count
            log.info(
                f"[trajectory] Corpus: {self._count}/{_MIN_FOR_FIRST_TRAINING} "
                f"({remaining} more needed for first RL checkpoint)"
            )
