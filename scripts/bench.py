#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request


def post_json(url: str, data: dict, api_key: str | None = None) -> tuple[int, float, str]:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("X-API-Key", api_key)

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            latency_ms = (time.perf_counter() - start) * 1000
            return resp.status, latency_ms, text
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")
        latency_ms = (time.perf_counter() - start) * 1000
        return e.code, latency_ms, text


def pctl(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def run_bench(args: argparse.Namespace) -> int:
    url = args.base_url.rstrip("/") + args.path
    latencies: list[float] = []
    failures = 0

    start_all = time.perf_counter()
    for i in range(args.requests):
        payload = {
            "collection": args.collection,
            "query": f"benchmark query {i}",
            "prefix": "query: ",
            "top_k": args.top_k,
            "with_payload": False,
            "hnsw_ef": args.hnsw_ef,
        }
        status, latency_ms, body = post_json(url, payload, args.api_key)
        if 200 <= status < 300:
            latencies.append(latency_ms)
        else:
            failures += 1
            if args.verbose_fail:
                print(f"FAIL {status}: {body}")

    duration = time.perf_counter() - start_all
    ok = len(latencies)
    total = args.requests
    qps = ok / duration if duration > 0 else 0.0

    latencies_sorted = sorted(latencies)
    p50 = pctl(latencies_sorted, 0.50)
    p95 = pctl(latencies_sorted, 0.95)
    p99 = pctl(latencies_sorted, 0.99)
    avg = statistics.mean(latencies) if latencies else 0.0

    print(f"url={url}")
    print(f"requests={total} success={ok} fail={failures}")
    print(f"duration_s={duration:.3f} qps={qps:.2f}")
    print(f"latency_ms avg={avg:.2f} p50={p50:.2f} p95={p95:.2f} p99={p99:.2f}")

    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple search benchmark for embedding service")
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--path", default="/search")
    parser.add_argument("--collection", default="documents")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--hnsw-ef", type=int, default=64)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--verbose-fail", action="store_true")
    args = parser.parse_args()
    return run_bench(args)


if __name__ == "__main__":
    raise SystemExit(main())
