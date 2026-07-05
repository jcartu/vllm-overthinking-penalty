#!/usr/bin/env python3
"""
Overthinking Penalty Benchmark — Before/After Comparison

Runs a suite of prompts against the GLM-5.2 vLLM endpoint and collects:
  - Reasoning token count (via tokenizer)
  - Content token count (via tokenizer)
  - Total completion tokens (from API usage)
  - Time to first token (TTFT) via streaming
  - Total generation latency
  - Finish reason (stop vs length)
  - Correctness (regex match on content)
  - Hesitation marker word count in reasoning
  - Hesitation token ID count (via tokenizer on reasoning text)

Usage:
  python3 benchmark.py --label before --output results_before.json
  python3 benchmark.py --label after  --output results_after.json
  python3 benchmark.py --compare results_before.json results_after.json
"""

import argparse
import json
import re
import subprocess
import sys
import time
import os
from pathlib import Path

import requests

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "glm-5.2"
CONTAINER = "glm52-reap-term-v13-594b"
PROMPTS_FILE = Path(__file__).parent / "prompts.json"

# Hesitation marker words from the paper — these trigger self-correction loops
HESITATION_WORDS = [
    "wait", "but", "alternatively", "however", "hmm", "actually",
    "let me", "i should", "on the other hand", "but wait", "no,",
    "hold on", "let me think", "i need to", "actually,", "wait,",
    "but,", "however,", "alternatively,", "hmm,", "on second thought",
    "let me reconsider", "i realize", "but actually", "wait —",
    "but then", "though", "although", "nevertheless", "nonetheless",
]

# The 43 hesitation token IDs from the paper
HESITATION_TOKEN_IDS = {
    11, 67, 71, 83, 265, 552, 1347, 1419, 1975, 2028,
    2152, 2371, 2753, 3821, 3983, 4331, 5482, 5569, 6282, 7615,
    7887, 8087, 10857, 11484, 12440, 13123, 14181, 24636, 26779, 27356,
    32618, 33141, 34696, 36569, 40190, 49893, 52246, 63108, 64796, 72465,
    79380, 91243, 97009,
}


def tokenize(text: str) -> list[int]:
    """Tokenize text using the model's tokenizer inside the container."""
    result = subprocess.run(
        ["docker", "exec", "-i", CONTAINER, "/opt/venv/bin/python", "-c", """
import sys, json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data1/GLM-5.2-Int8Mix-NVFP4-REAP-594B", trust_remote_code=True)
text = sys.stdin.read()
ids = tok.encode(text, add_special_tokens=False)
sys.stdout.write(json.dumps(ids))
"""],
        input=text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(f"  [WARN] tokenization failed: {result.stderr[:200]}", file=sys.stderr)
        return []
    return json.loads(result.stdout)


def count_hesitation_words(reasoning: str) -> int:
    """Count hesitation marker words in reasoning text."""
    reasoning_lower = reasoning.lower()
    count = 0
    for word in HESITATION_WORDS:
        count += len(re.findall(r'\b' + re.escape(word) + r'\b', reasoning_lower))
    return count


def count_hesitation_tokens(token_ids: list[int]) -> int:
    """Count how many generated tokens are in the hesitation set."""
    return sum(1 for tid in token_ids if tid in HESITATION_TOKEN_IDS)

def batch_tokenize(texts: dict[str, str]) -> dict[str, list[int]]:
    """Tokenize multiple texts in a single docker exec call.

    Args:
        texts: dict mapping key -> text (e.g. {"math-1_reasoning": "...", "math-1_content": "..."})

    Returns:
        dict mapping key -> list of token IDs
    """
    if not texts:
        return {}

    input_json = json.dumps(texts)
    result = subprocess.run(
        ["docker", "exec", "-i", CONTAINER, "/opt/venv/bin/python", "-c", """
import sys, json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data1/GLM-5.2-Int8Mix-NVFP4-REAP-594B", trust_remote_code=True)
texts = json.loads(sys.stdin.read())
result = {}
for key, text in texts.items():
    if text:
        result[key] = tok.encode(text, add_special_tokens=False)
    else:
        result[key] = []
sys.stdout.write(json.dumps(result))
"""],
        input=input_json,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"  [WARN] batch tokenization failed: {result.stderr[:200]}", file=sys.stderr)
        return {k: [] for k in texts}
    return json.loads(result.stdout)


def run_single_prompt(prompt_data: dict, max_tokens: int = 8192) -> dict:
    """Run a single prompt and collect all metrics."""
    prompt = prompt_data["prompt"]
    prompt_id = prompt_data["id"]

    print(f"  [{prompt_id}] {prompt[:70]}...", flush=True)

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    reasoning_parts = []
    content_parts = []
    ttft = None
    t_start = time.perf_counter()
    finish_reason = None
    usage = None
    chunk_count = 0

    try:
        resp = requests.post(VLLM_URL, json=payload, stream=True, timeout=300)
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            if ttft is None:
                ttft = time.perf_counter() - t_start

            chunk_count += 1
            choices = chunk.get("choices", [])
            if choices:
                choice = choices[0]
                delta = choice.get("delta", {})

                if "reasoning" in delta and delta["reasoning"]:
                    reasoning_parts.append(delta["reasoning"])
                if "content" in delta and delta["content"]:
                    content_parts.append(delta["content"])

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

            if chunk.get("usage"):
                usage = chunk["usage"]

    except Exception as e:
        return {
            "id": prompt_id,
            "error": str(e),
            "latency_total": time.perf_counter() - t_start,
        }

    t_total = time.perf_counter() - t_start

    reasoning_text = "".join(reasoning_parts)
    content_text = "".join(content_parts)

    # Tokenization deferred to batch_tokenize in run_benchmark

    # Count hesitation words (text-level, no tokenizer needed)
    hes_words = count_hesitation_words(reasoning_text)

    # Check correctness
    answer_regex = prompt_data.get("answer_regex", "")
    correct = False
    if answer_regex:
        correct = bool(re.search(answer_regex, content_text, re.IGNORECASE))

    result = {
        "id": prompt_id,
        "category": prompt_data.get("category", "unknown"),
        "prompt": prompt,
        "expected_answer": prompt_data.get("answer_description", ""),
        "reasoning_text": reasoning_text,
        "content_text": content_text,
        "reasoning_char_count": len(reasoning_text),
        "content_char_count": len(content_text),
        "reasoning_token_count": 0,  # filled in by batch_tokenize
        "content_token_count": 0,    # filled in by batch_tokenize
        "total_completion_tokens": usage.get("completion_tokens", 0) if usage else 0,
        "prompt_tokens": usage.get("prompt_tokens", 0) if usage else 0,
        "ttft_seconds": round(ttft, 4) if ttft else None,
        "latency_total_seconds": round(t_total, 4),
        "finish_reason": finish_reason,
        "correct": correct,
        "hesitation_word_count": hes_words,
        "hesitation_token_count": 0,  # filled in by batch_tokenize
        "chunk_count": chunk_count,
    }

    status = "✓" if correct else "✗"
    print(f"    → {status} chars_reasoning={len(reasoning_text)} chars_content={len(content_text)} "
          f"latency={t_total:.2f}s ttft={ttft:.3f}s hes_words={hes_words}",
          flush=True)

    return result


def run_benchmark(label: str, output_file: str, max_tokens: int = 8192):
    """Run the full benchmark suite."""
    with open(PROMPTS_FILE) as f:
        suite = json.load(f)

    prompts = suite["prompts"]
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: {label}")
    print(f"  Prompts: {len(prompts)} | Max tokens: {max_tokens} | Temperature: 0.0")
    print(f"{'='*70}\n")

    results = []
    for i, prompt_data in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}]", flush=True)
        result = run_single_prompt(prompt_data, max_tokens)
        results.append(result)
        print(flush=True)
    # Batch tokenize all reasoning and content texts in a single docker exec
    print(f"  Tokenizing all texts (single batch)...", flush=True)
    texts_to_tokenize = {}
    for r in results:
        if "error" in r:
            continue
        texts_to_tokenize[f"{r['id']}_reasoning"] = r["reasoning_text"]
        texts_to_tokenize[f"{r['id']}_content"] = r["content_text"]

    tokenized = batch_tokenize(texts_to_tokenize)

    for r in results:
        if "error" in r:
            continue
        r_tokens = tokenized.get(f"{r['id']}_reasoning", [])
        c_tokens = tokenized.get(f"{r['id']}_content", [])
        r["reasoning_token_count"] = len(r_tokens)
        r["content_token_count"] = len(c_tokens)
        r["hesitation_token_count"] = count_hesitation_tokens(r_tokens)
    print(f"  Tokenization complete.\n", flush=True)

    # Compute aggregates
    valid = [r for r in results if "error" not in r]
    agg = compute_aggregates(valid)

    output = {
        "label": label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "num_prompts": len(prompts),
        "num_valid": len(valid),
        "aggregates": agg,
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS: {label}")
    print(f"{'='*70}")
    print_aggregates(agg)
    print(f"\n  Results saved to: {output_file}")

    return output


def compute_aggregates(results: list[dict]) -> dict:
    """Compute aggregate statistics."""
    if not results:
        return {}

    def stats(values):
        values = sorted(values)
        n = len(values)
        mean = sum(values) / n
        median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
        return {
            "mean": round(mean, 2),
            "median": round(median, 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "sum": round(sum(values), 2),
        }

    reasoning_tokens = [r["reasoning_token_count"] for r in results]
    content_tokens = [r["content_token_count"] for r in results]
    total_tokens = [r["total_completion_tokens"] for r in results]
    ttfts = [r["ttft_seconds"] for r in results if r["ttft_seconds"]]
    latencies = [r["latency_total_seconds"] for r in results]
    hes_words = [r["hesitation_word_count"] for r in results]
    hes_tokens = [r["hesitation_token_count"] for r in results]
    correct = [r["correct"] for r in results]
    finish_reasons = [r["finish_reason"] for r in results]

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    cat_stats = {}
    for cat, cat_results in categories.items():
        cat_stats[cat] = {
            "count": len(cat_results),
            "correct": sum(1 for r in cat_results if r["correct"]),
            "accuracy": round(sum(1 for r in cat_results if r["correct"]) / len(cat_results), 4),
            "reasoning_tokens": stats([r["reasoning_token_count"] for r in cat_results]),
            "total_tokens": stats([r["total_completion_tokens"] for r in cat_results]),
            "latency": stats([r["latency_total_seconds"] for r in cat_results]),
            "hesitation_words": stats([r["hesitation_word_count"] for r in cat_results]),
            "hesitation_tokens": stats([r["hesitation_token_count"] for r in cat_results]),
        }

    return {
        "num_results": len(results),
        "accuracy": round(sum(correct) / len(correct), 4),
        "num_correct": sum(correct),
        "finish_reasons": {fr: finish_reasons.count(fr) for fr in set(finish_reasons)},
        "reasoning_tokens": stats(reasoning_tokens),
        "content_tokens": stats(content_tokens),
        "total_tokens": stats(total_tokens),
        "ttft_seconds": stats(ttfts),
        "latency_seconds": stats(latencies),
        "hesitation_words": stats(hes_words),
        "hesitation_tokens": stats(hes_tokens),
        "by_category": cat_stats,
    }


def print_aggregates(agg: dict):
    """Print aggregate statistics in a readable format."""
    print(f"  Accuracy:              {agg['accuracy']*100:.1f}% ({agg['num_correct']}/{agg['num_results']})")
    print(f"  Finish reasons:        {agg['finish_reasons']}")
    print(f"  Reasoning tokens:      mean={agg['reasoning_tokens']['mean']} median={agg['reasoning_tokens']['median']} "
          f"min={agg['reasoning_tokens']['min']} max={agg['reasoning_tokens']['max']}")
    print(f"  Content tokens:        mean={agg['content_tokens']['mean']} median={agg['content_tokens']['median']}")
    print(f"  Total completion toks: mean={agg['total_tokens']['mean']} median={agg['total_tokens']['median']}")
    print(f"  TTFT (seconds):        mean={agg['ttft_seconds']['mean']} median={agg['ttft_seconds']['median']}")
    print(f"  Latency (seconds):     mean={agg['latency_seconds']['mean']} median={agg['latency_seconds']['median']}")
    print(f"  Hesitation words:      mean={agg['hesitation_words']['mean']} median={agg['hesitation_words']['median']} "
          f"sum={agg['hesitation_words']['sum']}")
    print(f"  Hesitation tokens:     mean={agg['hesitation_tokens']['mean']} median={agg['hesitation_tokens']['median']} "
          f"sum={agg['hesitation_tokens']['sum']}")

    print(f"\n  By category:")
    for cat, cs in agg["by_category"].items():
        print(f"    {cat:12s} acc={cs['accuracy']*100:.0f}%  "
              f"reasoning={cs['reasoning_tokens']['mean']:.0f}tok  "
              f"latency={cs['latency']['mean']:.2f}s  "
              f"hes_words={cs['hesitation_words']['mean']:.1f}")


def compare_results(before_file: str, after_file: str):
    """Compare two benchmark result files."""
    with open(before_file) as f:
        before = json.load(f)
    with open(after_file) as f:
        after = json.load(f)

    ba = before["aggregates"]
    aa = after["aggregates"]

    print(f"\n{'='*78}")
    print(f"  BEFORE/AFTER COMPARISON")
    print(f"  Before: {before['label']} ({before['timestamp']})")
    print(f"  After:  {after['label']} ({after['timestamp']})")
    print(f"{'='*78}\n")

    def delta(before_val, after_val, unit="", lower_is_better=False):
        if before_val == 0:
            return "N/A"
        pct = (after_val - before_val) / before_val * 100
        arrow = "↓" if pct < 0 else "↑" if pct > 0 else "→"
        good = (pct < 0) if lower_is_better else (pct > 0)
        marker = " ✓" if (good and abs(pct) > 1) else " ✗" if (not good and abs(pct) > 1) else ""
        return f"{before_val:.2f} → {after_val:.2f}{unit} ({arrow}{abs(pct):.1f}%{marker})"

    print(f"  {'Metric':<30s} {'Before → After':<50s}")
    print(f"  {'-'*80}")
    print(f"  {'Accuracy':<30s} {ba['accuracy']*100:.1f}% → {aa['accuracy']*100:.1f}%")
    print(f"  {'Reasoning tokens (mean)':<30s} {delta(ba['reasoning_tokens']['mean'], aa['reasoning_tokens']['mean'], 'tok', lower_is_better=True)}")
    print(f"  {'Reasoning tokens (median)':<30s} {delta(ba['reasoning_tokens']['median'], aa['reasoning_tokens']['median'], 'tok', lower_is_better=True)}")
    print(f"  {'Content tokens (mean)':<30s} {delta(ba['content_tokens']['mean'], aa['content_tokens']['mean'], 'tok')}")
    print(f"  {'Total completion tokens':<30s} {delta(ba['total_tokens']['mean'], aa['total_tokens']['mean'], 'tok', lower_is_better=True)}")
    print(f"  {'TTFT (mean)':<30s} {delta(ba['ttft_seconds']['mean'], aa['ttft_seconds']['mean'], 's', lower_is_better=True)}")
    print(f"  {'Latency (mean)':<30s} {delta(ba['latency_seconds']['mean'], aa['latency_seconds']['mean'], 's', lower_is_better=True)}")
    print(f"  {'Latency (median)':<30s} {delta(ba['latency_seconds']['median'], aa['latency_seconds']['median'], 's', lower_is_better=True)}")
    print(f"  {'Hesitation words (sum)':<30s} {delta(ba['hesitation_words']['sum'], aa['hesitation_words']['sum'], '', lower_is_better=True)}")
    print(f"  {'Hesitation tokens (sum)':<30s} {delta(ba['hesitation_tokens']['sum'], aa['hesitation_tokens']['sum'], '', lower_is_better=True)}")
    print(f"  {'Finish reasons':<30s} {ba['finish_reasons']} → {aa['finish_reasons']}")

    print(f"\n  Per-prompt comparison:")
    print(f"  {'ID':<14s} {'Correct':<10s} {'Reasoning tokens':<25s} {'Latency':<25s} {'Hes words':<20s}")
    print(f"  {'-'*94}")

    before_by_id = {r["id"]: r for r in before["results"] if "error" not in r}
    after_by_id = {r["id"]: r for r in after["results"] if "error" not in r}

    for pid in before_by_id:
        b = before_by_id[pid]
        a = after_by_id.get(pid, {})
        if not a:
            print(f"  {pid:<14s} MISSING in after")
            continue

        b_corr = "✓" if b["correct"] else "✗"
        a_corr = "✓" if a["correct"] else "✗"
        corr = f"{b_corr}→{a_corr}"

        r_tok = f"{b['reasoning_token_count']}→{a['reasoning_token_count']}"
        lat = f"{b['latency_total_seconds']:.2f}→{a['latency_total_seconds']:.2f}s"
        hw = f"{b['hesitation_word_count']}→{a['hesitation_word_count']}"

        print(f"  {pid:<14s} {corr:<10s} {r_tok:<25s} {lat:<25s} {hw:<20s}")

    print(f"\n  By category:")
    print(f"  {'Category':<14s} {'Acc':<12s} {'Reasoning tok':<25s} {'Latency':<25s} {'Hes words':<20s}")
    print(f"  {'-'*96}")
    for cat in ba["by_category"]:
        bc = ba["by_category"][cat]
        ac = aa["by_category"].get(cat, {})
        if not ac:
            continue
        acc = f"{bc['accuracy']*100:.0f}%→{ac['accuracy']*100:.0f}%"
        rtok = f"{bc['reasoning_tokens']['mean']:.0f}→{ac['reasoning_tokens']['mean']:.0f}"
        lat = f"{bc['latency']['mean']:.2f}→{ac['latency']['mean']:.2f}s"
        hw = f"{bc['hesitation_words']['mean']:.1f}→{ac['hesitation_words']['mean']:.1f}"
        print(f"  {cat:<14s} {acc:<12s} {rtok:<25s} {lat:<25s} {hw:<20s}")


def main():
    parser = argparse.ArgumentParser(description="Overthinking penalty benchmark")
    parser.add_argument("--label", help="Label for this run (e.g. 'before', 'after')")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="Compare two result files")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens per response")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    elif args.label and args.output:
        run_benchmark(args.label, args.output, args.max_tokens)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
