#!/usr/bin/env python3
"""Run AlphaGenome variant scoring for a small VCF-like TSV."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from alphagenome.data import genome
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from tqdm import tqdm


BATCH_SCORER_KEYS = [
    "ATAC",
    "DNASE",
    "CHIP_TF",
    "CHIP_HISTONE",
    "CAGE",
    "PROCAP",
    "RNA_SEQ",
    "POLYADENYLATION",
    "SPLICE_SITES",
    "SPLICE_SITE_USAGE",
    "SPLICE_JUNCTIONS",
]

SPLICING_SCORER_KEYS = [
    "SPLICE_SITES",
    "SPLICE_SITE_USAGE",
    "SPLICE_JUNCTIONS",
]

SEQUENCE_LENGTHS = {
    "16KB": dna_client.SEQUENCE_LENGTH_16KB,
    "100KB": dna_client.SEQUENCE_LENGTH_100KB,
    "500KB": dna_client.SEQUENCE_LENGTH_500KB,
    "1MB": dna_client.SEQUENCE_LENGTH_1MB,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="variants.tsv")
    parser.add_argument("--outdir", default="results")
    parser.add_argument(
        "--score-set",
        choices=["splicing", "batch"],
        default="batch",
        help=(
            "'splicing' uses the three AlphaGenome splicing scorers. "
            "'batch' uses the scorer set from the AlphaGenome batch tutorial."
        ),
    )
    parser.add_argument(
        "--sequence-length",
        choices=sorted(SEQUENCE_LENGTHS),
        default="1MB",
    )
    parser.add_argument("--api-key-env", default="ALPHAGENOME_API_KEY")
    return parser.parse_args()


def selected_scorers(score_set: str) -> list[variant_scorers.VariantScorer]:
    keys = SPLICING_SCORER_KEYS if score_set == "splicing" else BATCH_SCORER_KEYS
    return [variant_scorers.RECOMMENDED_VARIANT_SCORERS[key] for key in keys]


def load_variants(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    required = ["variant_id", "gene", "transcript_hgvs", "grch38_hgvs", "CHROM", "POS", "REF", "ALT"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["POS"] = df["POS"].astype(int)
    return df


def score_variants(
    dna_model: dna_client.DnaClient,
    variants_df: pd.DataFrame,
    scorers: list[variant_scorers.VariantScorer],
    sequence_length: int,
) -> pd.DataFrame:
    scores = []
    for row in tqdm(variants_df.itertuples(index=False), total=len(variants_df)):
        variant = genome.Variant(
            chromosome=str(row.CHROM),
            position=int(row.POS),
            reference_bases=str(row.REF),
            alternate_bases=str(row.ALT),
            name=str(row.variant_id),
            info={
                "gene": row.gene,
                "transcript_hgvs": row.transcript_hgvs,
                "grch38_hgvs": row.grch38_hgvs,
            },
        )
        interval = variant.reference_interval.resize(sequence_length)
        scores.append(
            dna_model.score_variant(
                interval=interval,
                variant=variant,
                variant_scorers=scorers,
                organism=dna_client.Organism.HOMO_SAPIENS,
            )
        )
    return variant_scorers.tidy_scores(scores)


def alphagenome_variant_string(row: pd.Series) -> str:
    variant = genome.Variant(
        chromosome=str(row["CHROM"]),
        position=int(row["POS"]),
        reference_bases=str(row["REF"]),
        alternate_bases=str(row["ALT"]),
    )
    return str(variant)


def attach_input_metadata(df_scores: pd.DataFrame, variants_df: pd.DataFrame) -> pd.DataFrame:
    """Replace AlphaGenome's variant string with the input ID and add metadata."""
    scored = df_scores.copy()
    scored["alphagenome_variant"] = scored["variant_id"].map(str)

    metadata = variants_df.copy()
    metadata["alphagenome_variant"] = metadata.apply(alphagenome_variant_string, axis=1)
    metadata = metadata[
        ["alphagenome_variant", "variant_id", "gene", "transcript_hgvs", "grch38_hgvs"]
    ].rename(columns={"variant_id": "input_variant_id"})

    scored = scored.merge(metadata, on="alphagenome_variant", how="left")
    scored["variant_id"] = scored["input_variant_id"].fillna(scored["alphagenome_variant"])
    scored = scored.drop(columns=["input_variant_id"])
    return scored


def top_abs_by_group(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    work = df.copy()
    abs_col = f"abs_{value_col}"
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    work[abs_col] = work[value_col].abs()
    idx = work.groupby(group_cols)[abs_col].idxmax()
    return work.loc[idx].sort_values(group_cols + [abs_col], ascending=[True] * len(group_cols) + [False])


def summarize_splicing(df_scores: pd.DataFrame, variants_df: pd.DataFrame) -> pd.DataFrame:
    splicing = df_scores[df_scores["output_type"].isin(SPLICING_SCORER_KEYS)].copy()
    if splicing.empty:
        return pd.DataFrame()

    per_output = top_abs_by_group(splicing, ["variant_id", "output_type"], "raw_score")
    pivot = (
        per_output.pivot(index="variant_id", columns="output_type", values="abs_raw_score")
        .fillna(0.0)
        .rename_axis(None, axis=1)
    )
    for output_type in SPLICING_SCORER_KEYS:
        if output_type not in pivot.columns:
            pivot[output_type] = 0.0
    pivot["alphagenome_splicing"] = (
        pivot["SPLICE_SITES"]
        + pivot["SPLICE_SITE_USAGE"]
        + pivot["SPLICE_JUNCTIONS"] / 5.0
    )

    metadata = variants_df.set_index("variant_id")[["gene", "transcript_hgvs", "grch38_hgvs"]]
    summary = metadata.join(pivot, how="left").fillna(0.0).reset_index()
    summary["splicing_effect_call"] = summary["alphagenome_splicing"].map(call_splicing_effect)
    return summary.sort_values("alphagenome_splicing", ascending=False)


def call_splicing_effect(score: float) -> str:
    if score > 1.0:
        return "large predicted splicing effect"
    if score >= 0.5:
        return "moderate predicted splicing effect"
    return "low predicted splicing effect"


def call_abs_raw_effect(score: float) -> str:
    if score > 1.0:
        return "large predicted molecular effect"
    if score >= 0.5:
        return "moderate predicted molecular effect"
    return "low predicted molecular effect"


def call_raw_score_effect(score: float) -> str:
    abs_call = call_abs_raw_effect(abs(score)).replace(" molecular effect", "")
    if score > 0:
        return f"{abs_call} molecular increase"
    if score < 0:
        return f"{abs_call} molecular decrease"
    return "low predicted molecular effect"


def call_quantile_effect(score: float) -> str:
    abs_score = abs(score)
    if abs_score >= 0.99:
        strength = "extreme"
    elif abs_score >= 0.95:
        strength = "high"
    elif abs_score >= 0.8:
        strength = "moderate"
    else:
        strength = "low"

    if score > 0:
        return f"{strength} positive-effect quantile"
    if score < 0:
        return f"{strength} negative-effect quantile"
    return "low quantile effect"


def add_score_interpretations(summary: pd.DataFrame) -> pd.DataFrame:
    interpreted = summary.copy()
    if "raw_score" in interpreted.columns:
        interpreted["raw_score"] = pd.to_numeric(
            interpreted["raw_score"], errors="coerce"
        ).fillna(0.0)
        interpreted["abs_raw_score"] = interpreted["raw_score"].abs()
        interpreted["raw_score_call"] = interpreted["raw_score"].map(call_raw_score_effect)
        interpreted["abs_raw_score_call"] = interpreted["abs_raw_score"].map(call_abs_raw_effect)
    if "quantile_score" in interpreted.columns:
        interpreted["quantile_score"] = pd.to_numeric(
            interpreted["quantile_score"], errors="coerce"
        ).fillna(0.0)
        interpreted["abs_quantile_score"] = interpreted["quantile_score"].abs()
        interpreted["quantile_score_call"] = interpreted["quantile_score"].map(
            call_quantile_effect
        )
    return interpreted


def summarize_outputs(df_scores: pd.DataFrame) -> pd.DataFrame:
    top = top_abs_by_group(df_scores, ["variant_id", "output_type"], "quantile_score")
    top = add_score_interpretations(top)
    keep = [
        "variant_id",
        "output_type",
        "gene_name",
        "gene_id",
        "variant_scorer",
        "track_name",
        "biosample_name",
        "biosample_type",
        "transcription_factor",
        "histone_mark",
        "gtex_tissue",
        "raw_score",
        "raw_score_call",
        "quantile_score",
        "quantile_score_call",
        "abs_raw_score",
        "abs_raw_score_call",
        "abs_quantile_score",
    ]
    return top[[column for column in keep if column in top.columns]]


def summarize_target_gene_outputs(df_scores: pd.DataFrame) -> pd.DataFrame:
    if "gene" not in df_scores.columns or "gene_name" not in df_scores.columns:
        return pd.DataFrame()
    target = df_scores[df_scores["gene_name"].fillna("") == df_scores["gene"].fillna("")].copy()
    if target.empty:
        return pd.DataFrame()
    top = top_abs_by_group(target, ["variant_id", "output_type"], "raw_score")
    top = add_score_interpretations(top)
    keep = [
        "variant_id",
        "gene",
        "output_type",
        "gene_name",
        "variant_scorer",
        "track_name",
        "biosample_name",
        "biosample_type",
        "gtex_tissue",
        "raw_score",
        "raw_score_call",
        "quantile_score",
        "quantile_score_call",
        "abs_raw_score",
        "abs_raw_score_call",
    ]
    return top[[column for column in keep if column in top.columns]]


def write_report(
    outdir: Path,
    variants_df: pd.DataFrame,
    splicing_summary: pd.DataFrame,
    target_gene_summary: pd.DataFrame,
    output_summary: pd.DataFrame,
    score_set: str,
    sequence_length: str,
) -> None:
    lines = [
        "# AlphaGenome Variant Assessment",
        "",
        f"Score set: `{score_set}`",
        f"Sequence length: `{sequence_length}`",
        "",
        "AlphaGenome predicts molecular effects, not clinical pathogenicity by itself. "
        "For splice-adjacent variants, the merged splicing score follows the AlphaGenome "
        "tutorial formula: max(abs splice-sites) + max(abs splice-site-usage) + "
        "max(abs splice-junctions) / 5.",
        "",
        "Raw-score and abs-raw-score calls use the same low/moderate/large thresholds "
        "as the splicing summary (<0.5 low, 0.5-1.0 moderate, >1.0 large). "
        "Quantile-score calls summarize how outlier-like the signed score is "
        "(>=0.99 extreme, >=0.95 high, >=0.8 moderate). These are molecular-effect "
        "summaries, not standalone ACMG pathogenicity classifications.",
        "",
        "## Input Alleles",
        "",
        variants_df[["variant_id", "gene", "transcript_hgvs", "grch38_hgvs", "CHROM", "POS", "REF", "ALT"]]
        .to_markdown(index=False),
        "",
    ]

    if not splicing_summary.empty:
        lines.extend(
            [
                "## Splicing Summary",
                "",
                splicing_summary[
                    [
                        "variant_id",
                        "gene",
                        "alphagenome_splicing",
                        "splicing_effect_call",
                        "SPLICE_SITES",
                        "SPLICE_SITE_USAGE",
                        "SPLICE_JUNCTIONS",
                    ]
                ].to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )

    if not target_gene_summary.empty:
        lines.extend(
            [
                "## Target Gene Strongest Raw Effects",
                "",
                target_gene_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )

    if not output_summary.empty:
        lines.extend(
            [
                "## Strongest Track Per Output Type",
                "",
                output_summary.to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )

    (outdir / "alphagenome_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before running.")

    variants_path = Path(args.variants)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    variants_df = load_variants(variants_path)
    variants_df.to_csv(outdir / "input_variants.resolved_hg38.tsv", sep="\t", index=False)

    dna_model = dna_client.create(api_key)
    df_scores = score_variants(
        dna_model=dna_model,
        variants_df=variants_df,
        scorers=selected_scorers(args.score_set),
        sequence_length=SEQUENCE_LENGTHS[args.sequence_length],
    )
    df_scores = attach_input_metadata(df_scores, variants_df)

    score_path = outdir / f"variant_scores_{args.score_set}.csv"
    df_scores.to_csv(score_path, index=False)

    splicing_summary = summarize_splicing(df_scores, variants_df)
    if not splicing_summary.empty:
        splicing_summary.to_csv(outdir / f"splicing_summary_{args.score_set}.csv", index=False)

    output_summary = summarize_outputs(df_scores)
    output_summary.to_csv(outdir / f"top_output_scores_{args.score_set}.csv", index=False)

    target_gene_summary = summarize_target_gene_outputs(df_scores)
    if not target_gene_summary.empty:
        target_gene_summary.to_csv(
            outdir / f"top_target_gene_scores_{args.score_set}.csv", index=False
        )

    write_report(
        outdir=outdir,
        variants_df=variants_df,
        splicing_summary=splicing_summary,
        target_gene_summary=target_gene_summary,
        output_summary=output_summary,
        score_set=args.score_set,
        sequence_length=args.sequence_length,
    )
    print(f"Wrote {score_path}")
    print(f"Wrote {outdir / 'alphagenome_interpretation.md'}")


if __name__ == "__main__":
    main()
