"""LLM-assisted components of LoGe Planner (``v1.1.0``).

This package implements the *executable* LLM paths referenced by the manuscript:

* :mod:`src.agent_llm.decompose_cli` -- real OpenAI-compatible cloud macro-task
  decomposition that emits a ``checkpoint_clusters_<method>.csv`` file directly
  consumable by Stage 2 (``stage2_plan_from_clusters``) and Stage 3
  (``stage3_plan_with_pddl``).
* :mod:`src.agent_llm.terminal_llm` -- local ``transformers`` + ``peft``
  Qwen2.5-0.5B (+LoRA adapter) inference used by
  :mod:`src.validation.terminal_monitor_eval`.
* :mod:`src.agent_llm.train_lora` -- LoRA supervised fine-tuning CLI over
  external JSONL chat data (correct chat template + assistant-only label mask).

Design rules honoured by this package:

* No fabricated results.  An API error, an unparsable response, a missing
  adapter, or an invalid decomposition raises instead of returning a default.
* Heavy ML dependencies (``transformers`` / ``peft`` / ``accelerate`` /
  ``bitsandbytes``) are declared in ``requirements-llm.txt`` and imported
  lazily, so the symbolic pipeline keeps running without them.
* The public version stays ``1.1.0``; these are implementation-completing
  modules, not a new release.
"""

from __future__ import annotations

PLANNER_VERSION = "1.1.0"
LLM_PROTOCOL_VERSION = "1.0"

__all__ = ["PLANNER_VERSION", "LLM_PROTOCOL_VERSION"]
