# AlphaGenome Variant Scoring

This workspace scores four resolved hg38 alleles with the AlphaGenome Python API.

## Setup

```bash
source .venv/bin/activate
export ALPHAGENOME_API_KEY="your_api_key_here"
```

## Run

```bash
python scripts/run_alphagenome.py --score-set batch --sequence-length 1MB
```

Outputs are written to `results/`:

- `variant_scores_batch.csv`: full tidy AlphaGenome scores.
- `splicing_summary_batch.csv`: merged splicing score per variant.
- `top_target_gene_scores_batch.csv`: strongest raw RNA/splicing effect for the named gene.
- `top_output_scores_batch.csv`: strongest quantile-score track per variant/output type.
- `alphagenome_interpretation.md`: compact report.

The API key is read from the environment and is intentionally not stored in this repo.
