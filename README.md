# llms_final_project

Tone-bias benchmark for retrieval-augmented generation (RAG) systems.

## Implemented systems

- `standard_rag`: one-pass retrieve-then-generate baseline.
- `madam_rag`: multi-agent debate with a judge.
- `eajp`: MADAM-RAG with Evidence-Anchored Judge Prompting.

## EAJP protocol

EAJP forces the judge to do evidence-first reasoning:

1. Identify the most relevant claim in each retrieved document.
2. Record the stated credibility tier and supporting evidence spans.
3. Produce a preliminary evidence-only verdict.
4. Read the full debate arguments.
5. If the argument-based verdict conflicts with the scorecard, privilege the scorecard verdict.

This is meant to reduce style-driven source laundering in debate-based RAG.

## Running experiments

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add API keys.

Example run on the sample dataset:

```bash
python -m experiments.run_experiment \
  --dataset data/samples/sample_dataset.json \
  --benchmark custom \
  --systems standard_rag madam_rag eajp \
  --judge-suite both \
  --output results/sample_run
```

Benchmark families supported by the runner:

- `ambigdocs`
- `faitheval`
- `tcb`
- `custom`

Judge-model suites supported by the runner:

- `openai` → GPT-5.2
- `anthropic` → Claude Sonnet 4.6
- `both`

## Building TCB

To construct a tone-controlled benchmark from an existing dataset:

```bash
python -m experiments.build_tcb \
  --input data/samples/sample_dataset.json \
  --benchmark custom \
  --output data/tcb_sample.json
```

The builder will prefer Gemini if `GEMINI_API_KEY` is set and the
`google-genai` package is available; otherwise it falls back to a local
rewrite heuristic.

## Loading AmbigDocs directly

To download and convert AmbigDocs from Hugging Face into the local JSON
format used by this project:

```bash
python -m experiments.convert_ambigdocs \
  --split validation \
  --output data/ambigdocs_validation.json
```

The script uses the first document as the gold document by default. If you
want to test a different gold-document choice, pass `--gold-doc-index`.

## Loading FaithEval directly

To convert the FaithEval counterfactual and inconsistent datasets into the
local JSON format used by this project:

```bash
python -m experiments.convert_faitheval \
  --dataset-types counterfactual inconsistent \
  --split test \
  --output data/faitheval_combined.json
```

The converter loads both public Hugging Face datasets directly, infers the
correct answer from the dataset fields, and creates a two-document RAG-style
instance for each example.

Each run writes per-system metrics and details JSON files under the chosen output directory.

## One-script workflow

From PowerShell, you can run the full pipeline with:

```powershell
./run_experiments.ps1
```

This will:

1. install dependencies
2. convert AmbigDocs
3. convert FaithEval
4. build TCB from both datasets
5. run standard RAG, MADAM-RAG, and EAJP on each dataset

You can skip steps with switches such as `-BuildDatasets:$false` or
`-RunExperiments:$false`.