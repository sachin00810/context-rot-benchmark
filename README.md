# context-rot-benchmark

A benchmark suite for evaluating LLM performance degradation and context rot over long context windows and multi-turn conversations.

## Project Structure

```text
context-rot-benchmark/
├── README.md           # Project documentation
├── requirements.txt    # Project dependencies
├── src/                # Benchmark scripts and core evaluation logic
├── results/            # Benchmark outputs, metric logs, and charts
└── notebooks/          # Exploratory analysis and visualization notebooks
```

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your benchmark scripts**:
   Place modules and evaluation pipelines in `src/`.

3. **View results and analyze**:
   Save raw outputs into `results/` and run analysis notebooks in `notebooks/`.