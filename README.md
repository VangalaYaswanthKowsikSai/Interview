# Text Embeddings Benchmark

This project benchmarks various text embedding models (local and API-based) to compare their latency, cost, and retrieval quality. It simulates a production RAG (Retrieval-Augmented Generation) scenario to help teams make data-driven decisions between "Build vs. Buy".

## Project Overview

The benchmark executes a series of tests on 5 different models:
1.  `sentence-transformers/all-MiniLM-L6-v2` (Speed Standard)
2.  `sentence-transformers/all-mpnet-base-v2` (Quality Standard)
3.  `BAAI/bge-small-en-v1.5` (Modern Efficient)
4.  `BAAI/bge-base-en-v1.5` (Modern Balanced)
5.  `intfloat/e5-small-v2` (Competitor)

It generates a detailed report (`results/article.md`) and rich visualisations to demonstrate the trade-offs between speed, accuracy, and cost.

## Requirements

### Prerequisites
- Python 3.8+
- Internet connection (to download models from Hugging Face)

### Dependencies
Install the required packages using pip:

```bash
pip install pandas matplotlib seaborn sentence-transformers openai pyyaml
```

## How to Run

1.  **Clone the repository** (or navigate to the project directory).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    # OR run manually:
    pip install pandas matplotlib seaborn sentence-transformers openai pyyaml
    ```
3.  **Run the benchmark**:
    ```bash
    python run_benchmark.py
    ```

## Output

After execution, check the `results/` directory for:
- `article.md`: The final benchmark report.
- `chart_*.png`: Visualizations of latency, cost, and efficiency.

## Author

**Yaswanth**
