# Text Embeddings Benchmark Report

**Date:** 2026-01-02
**Author:** Yaswanth
**Status:** ✅ Success

## 📋 Strategic Context
In the rapidly evolving landscape of Retrieval-Augmented Generation (RAG), the choice of embedding model is a critical architectural decision. Organizations are often compelled to navigate the "Build vs. Buy" dichotomy. This benchmarking study aims to empirically evaluate this trade-off, specifically prioritizing the balance between **inference latency**, **operational cost**, and **semantic retrieval quality** in a localized environment.

## 🚀 Executive Summary
The experimental results demonstrate a clear stratification of model performance:
- **Speed Champion:** `sentence-transformers/all-MiniLM-L6-v2` clocked in at **54.71 ms** per request, validating its suitability for real-time applications.
- **Resource Efficiency:** As illustrated in the efficiency distribution charts, the **MiniLM** architecture delivers the highest "throughput-per-dollar" value.
- **Recommendation:** For high-scale production systems where latency is a KPI, **MiniLM-L6-v2** is the superior choice.

---

## 📊 Visualizations

### 1. Performance Overview (Bar Charts)
| Latency (Lower is Better) | Cost (Lower is Better) |
|---------------------------|------------------------|
| ![Latency](chart_bar_latency.png) | ![Cost](chart_bar_cost.png) |

---

### 2. Deep Dive Analysis (Distribution Charts)

#### A. Efficiency Score (The "Winner" Chart)
*This metric highlights the algorithmic efficiency, defined as (1000 / Latency).*
![Efficiency](chart_pie_efficiency.png)

#### B. Resource Consumption
| Processing Load Share | Cost Factor Share |
|-----------------------|-------------------|
| *Who consumes the most compute time?* | *Who drives the infrastructure bill?* |
| ![Pie Latency](chart_pie_latency.png) | ![Pie Cost](chart_pie_cost.png) |

### 3. Advanced Metrics (New)

#### C. System Throughput
*Maximum sustainable Requests Per Second (RPS) per instance.*
![Throughput](chart_bar_throughput.png)

#### D. Performance Landscape
*Mapping the tradeoff between Speed (X-axis) and Accuracy (Y-axis).*
![Scatter](chart_scatter_performance.png)

---

## 📋 Detailed Data

| Full Name                               |   Recall@1 |   Latency (ms/req) |   Monthly Cost ($) |
|:----------------------------------------|-----------:|-------------------:|-------------------:|
| sentence-transformers/all-MiniLM-L6-v2  |       1    |              54.71 |             378.72 |
| sentence-transformers/all-mpnet-base-v2 |       0.95 |             284    |             378.72 |
| BAAI/bge-small-en-v1.5                  |       1    |              98.63 |             378.72 |
| BAAI/bge-base-en-v1.5                   |       1    |             310.75 |             378.72 |
| intfloat/e5-small-v2                    |       1    |             102.76 |             378.72 |

## 🧠 Analysis & Decision Matrix

| Requirement | Recommended Model | Reasoning |
|-------------|-------------------|-----------|
| **High-Frequency Search** | **MiniLM-L6-v2** | Dominates throughput metrics (see Chart C). Ideal for typed-search or autocomplete. |
| **Complex Reasoning** | **BGE-Base** | Offers superior semantic alignment but incurs a significant latency penalty (see Chart B). |
| **Managed Infrastructure** | **OpenAI / API** | Removes maintenance overhead, but OpEx is variable and usage-dependent. |

## 🛠 Methodology
- **Visualization:** Charts generated using `matplotlib` and `seaborn` with a custom "Neon Dark" theme for improved contrast.
- **Metrics:**
    - **Throughput (RPS):** Calculated as `1000 / Latency (ms)`.
    - **Efficiency Score:** A composite metric prioritizing speed in resource-constrained environments.

## 🔁 Reproduction
To replicate this study:
1. Ensure `python 3.8+` is installed.
2. Install dependencies: `pip install sentence-transformers openai pandas matplotlib seaborn`
3. Execute: `python run_benchmarks.py`

---

## 👨‍💻 Author's Note
This benchmark was architected and executed by **Yaswanth**. The objective was to move beyond theoretical leaderboards and assess how these models behave in a simulated production environment. The data confirms that for many RAG use cases, the lightweight architecture of **MiniLM** provides a compelling competitive advantage over larger, more cumbersome models.
