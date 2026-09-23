# Context Rot & Needle-in-a-Haystack Benchmark

A benchmark suite to evaluate LLM context retrieval and test for context degradation ("context rot") across varied document lengths and insertion depths.

---

## Repository Structure

```text
context-rot-benchmark/
├── .env.example          # Environment variable template for API keys
├── .gitignore            # Git exclusion rules for keys, data, and cache
├── README.md             # Project documentation and quick start guide
├── requirements.txt      # Project dependencies
├── data/                 # Directory for haystack text (e.g. haystack.txt)
├── notebooks/            # Exploratory analysis and notebooks
├── results/              # Evaluation outputs and generated heatmaps
└── src/
    ├── __init__.py
    ├── needle_haystack.py # Benchmark execution script
    └── plot_results.py    # Heatmap visualization generator
```

---

## Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Copy `.env.example` to `.env` and paste your Gemini API key:
```bash
cp .env.example .env
```
Open `.env` and set your key:
```env
GEMINI_API_KEY=your_real_gemini_api_key_here
```
*(The `.env` file is excluded in `.gitignore` to prevent committing secrets).*

### 3. Add Haystack Text (Optional but Recommended)
For realistic evaluation, download a long public-domain book (e.g., from [Project Gutenberg](https://www.gutenberg.org)) and save it as:
```text
data/haystack.txt
```
> **Note**: If `data/haystack.txt` is not provided, the benchmark will automatically fall back to repeating filler paragraphs so you can test the pipeline immediately.

---

## Running the Benchmark

### Small Initial Test Run
Run a small test grid to confirm the pipeline before larger evaluations:
```bash
python src/needle_haystack.py --provider gemini --lengths 2000 8000 --depths 0 50 100
```

### Dry Run (Mock Provider without API calls)
To test the full execution and pipeline without using API credits:
```bash
python src/needle_haystack.py --provider mock --lengths 2000 8000 --depths 0 50 100
```

---

## Plotting Results

Generate the retrieval heatmap from the benchmark results:
```bash
python src/plot_results.py --provider gemini
```

This creates a 2D performance heatmap saved to:
```text
results/gemini_heatmap.png
```

---

## Advanced CLI Options

### `needle_haystack.py`
| Flag | Default | Description |
|---|---|---|
| `--provider` | `gemini` | LLM provider (`gemini`, `mock`) |
| `--model` | `gemini-2.0-flash` | Model identifier |
| `--lengths` | `2000 8000` | Context lengths to test |
| `--depths` | `0 50 100` | Depths (%) to place needle (0 to 100) |
| `--needle` | Default secret code | Fact string inserted into text |
| `--question` | Default prompt | Retrieval query presented to model |
| `--expected` | Default code value | Keyword to verify retrieval success |
| `--haystack` | `data/haystack.txt` | Path to background text |
| `--output` | `results/{provider}_results.json` | JSON output destination |

### `plot_results.py`
| Flag | Default | Description |
|---|---|---|
| `--provider` | `gemini` | Target provider name |
| `--input` | `results/{provider}_results.json` | Path to benchmark JSON file |
| `--output` | `results/{provider}_heatmap.png` | Destination image path |