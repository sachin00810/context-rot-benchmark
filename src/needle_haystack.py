#!/usr/bin/env python3
"""
Needle in a Haystack Benchmark Runner.

Evaluates LLM retrieval performance across varying context lengths and depths.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default needle and query
DEFAULT_NEEDLE = "The secret code to unlock the vault is 849204."
DEFAULT_QUESTION = "What is the secret code to unlock the vault? Reply with the code."
DEFAULT_EXPECTED = "849204"

# Fallback filler paragraph when data/haystack.txt is not present
FALLBACK_FILLER = (
    "The atmospheric pressure at sea level is approximately 101.3 kilopascals. "
    "Observations across different weather patterns indicate that localized low-pressure systems "
    "often correlate with cloud cover and precipitation. Data recorded from multiple oceanic stations "
    "reinforce the understanding of planetary boundary layer dynamics and thermal circulation cycles. "
    "Long-term analysis of meteorological trends requires consistent calibration of barometric equipment "
    "and continuous logging of ambient temperature, humidity ratios, and wind vectors across coordinates. "
)


def load_haystack_text(data_path: Path, target_words: int) -> str:
    """Loads background haystack text from file or generates filler if absent."""
    if data_path.exists() and data_path.is_file():
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        words = raw_text.split()
        if len(words) == 0:
            print(f"[!] Warning: {data_path} is empty. Using synthetic fallback text.")
            return generate_fallback_text(target_words)
        
        # Loop text if shorter than target length
        repeated_words = []
        while len(repeated_words) < target_words:
            repeated_words.extend(words)
        return " ".join(repeated_words[:target_words])
    else:
        print(f"[*] Notice: '{data_path}' not found. Using synthetic filler text for this run.")
        print("    Tip: Add a real book or text as data/haystack.txt for realistic benchmark results.")
        return generate_fallback_text(target_words)


def generate_fallback_text(target_words: int) -> str:
    """Generates synthetic repeating filler text to reach the desired word count."""
    filler_words = FALLBACK_FILLER.split()
    words = []
    while len(words) < target_words:
        words.extend(filler_words)
    return " ".join(words[:target_words])


def insert_needle(haystack_text: str, needle: str, depth_percent: float) -> str:
    """Inserts the needle into the haystack at the specified depth percentage (0 - 100)."""
    words = haystack_text.split()
    total_words = len(words)
    
    # Calculate insertion index based on depth percentage
    depth_clamped = max(0.0, min(100.0, float(depth_percent)))
    insert_index = int(total_words * (depth_clamped / 100.0))
    
    words_before = words[:insert_index]
    words_after = words[insert_index:]
    
    combined = " ".join(words_before) + f"\n\n{needle}\n\n" + " ".join(words_after)
    return combined.strip()


def query_gemini(prompt: str, model_name: str) -> str:
    """Queries Gemini model via REST API or Google GenAI SDK."""
    import urllib.request
    import urllib.error

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set. Please copy .env.example to .env and add your valid Gemini API key."
        )

    # Models to try in order
    candidate_models = [model_name]
    for alt in ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-flash-latest"]:
        if alt not in candidate_models:
            candidate_models.append(alt)

    # Primary method: Direct REST API (fast, robust, no SDK AFC overhead)
    last_error = None
    for current_model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={api_key}"
        payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")

        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                    return ""
            except urllib.error.HTTPError as e:
                last_error = e
                code = e.code
                if code in (503, 429):
                    time.sleep(2.0)
                    continue
                elif code == 404:
                    # Model not supported, try next model candidate
                    break
                else:
                    err_body = e.read().decode("utf-8", errors="ignore")
                    raise RuntimeError(f"Gemini API error (HTTP {code}): {err_body}")
            except Exception as e:
                last_error = e
                time.sleep(1.0)
                continue

    if last_error:
        raise last_error
    return ""


def query_mock(prompt: str, needle: str, expected: str, depth_percent: float) -> str:
    """Mock provider for dry-run validation without API calls."""
    time.sleep(0.05)
    # Simulate high retrieval accuracy
    return f"Based on the text, {expected} is the secret code."


def evaluate_response(response_text: str, expected_answer: str) -> bool:
    """Checks if expected needle fact is present in the response."""
    return expected_answer.lower() in response_text.lower()


def run_benchmark(
    provider: str,
    model: str,
    lengths: List[int],
    depths: List[float],
    needle: str,
    question: str,
    expected: str,
    data_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Runs the benchmark over the grid of context lengths and depths."""
    print("=" * 65)
    print(f" Needle in a Haystack Benchmark: {provider.upper()} ({model})")
    print("=" * 65)
    print(f" Context Lengths : {lengths}")
    print(f" Depths (%)      : {depths}")
    print(f" Target Needle   : {needle}")
    print(f" Question        : {question}")
    print(f" Expected Answer : {expected}")
    print(f" Total Tests     : {len(lengths) * len(depths)}")
    print("=" * 65)

    results = []
    total_runs = len(lengths) * len(depths)
    current_idx = 0

    for length in lengths:
        # Load or generate haystack for this context length (approximating 1 token ≈ 0.75 words)
        target_words = int(length * 0.75)
        base_haystack = load_haystack_text(data_path, target_words)

        for depth in depths:
            current_idx += 1
            print(f"[{current_idx}/{total_runs}] Testing length={length} | depth={depth}% ... ", end="", flush=True)

            document = insert_needle(base_haystack, needle, depth)
            prompt = (
                f"<document>\n{document}\n</document>\n\n"
                f"Based on the document provided above, please answer this question accurately:\n"
                f"{question}"
            )

            start_time = time.time()
            error_msg = None
            response_text = ""
            success = False

            try:
                if provider.lower() == "gemini":
                    response_text = query_gemini(prompt, model)
                elif provider.lower() == "mock":
                    response_text = query_mock(prompt, needle, expected, depth)
                else:
                    raise ValueError(f"Unsupported provider: {provider}")

                latency = round(time.time() - start_time, 2)
                success = evaluate_response(response_text, expected)
                score = 1.0 if success else 0.0
                status_label = "PASS (1.0)" if success else "FAIL (0.0)"
                print(f"{status_label} in {latency}s")

            except Exception as e:
                latency = round(time.time() - start_time, 2)
                error_msg = str(e)
                score = 0.0
                print(f"ERROR: {error_msg}")

            results.append({
                "context_length": length,
                "depth_percent": depth,
                "score": score,
                "success": success,
                "latency_seconds": latency,
                "response": response_text[:300],  # truncated for brevity
                "error": error_msg,
            })

    output_data = {
        "provider": provider,
        "model": model,
        "needle": needle,
        "question": question,
        "expected": expected,
        "lengths": lengths,
        "depths": depths,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 65)
    print(f"[✓] Benchmark run complete! Results saved to:")
    print(f"    -> {output_path}")
    print("=" * 65)

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Needle in a Haystack LLM Benchmark")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "mock"], help="LLM provider")
    parser.add_argument("--model", type=str, default="gemini-3.5-flash", help="Model identifier")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2000, 8000], help="Context lengths to test")
    parser.add_argument("--depths", type=float, nargs="+", default=[0, 50, 100], help="Depths in %% to place needle (0 - 100)")
    parser.add_argument("--needle", type=str, default=DEFAULT_NEEDLE, help="Needle string to insert")
    parser.add_argument("--question", type=str, default=DEFAULT_QUESTION, help="Question to ask")
    parser.add_argument("--expected", type=str, default=DEFAULT_EXPECTED, help="Expected answer string")
    parser.add_argument("--haystack", type=str, default="data/haystack.txt", help="Path to haystack text file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / args.haystack

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / "results" / f"{args.provider}_results.json"

    run_benchmark(
        provider=args.provider,
        model=args.model,
        lengths=args.lengths,
        depths=args.depths,
        needle=args.needle,
        question=args.question,
        expected=args.expected,
        data_path=data_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
