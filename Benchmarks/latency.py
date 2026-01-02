import time
import numpy as np

def benchmark_latency(model_wrapper, texts, runs=10):
    """
    Measures encoding latency.
    """
    # Warmup (crucial for fair local comparison)
    print(f"  ...Warming up {model_wrapper.name}...")
    model_wrapper.encode(["warmup"] * 2)

    latencies = []
    
    print(f"  ...Running latency test ({runs} runs)...")
    for _ in range(runs):
        start_time = time.time()
        # Encode batch
        model_wrapper.encode(texts)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000) # Convert to ms

    results = {
        "mean_ms": np.mean(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99)
    }
    return results