#!/usr/bin/env python3
"""
Select claims for user study based on checkworthiness criteria.

Selects 3 claims representing different decision categories:
1. HIGH CONFIDENCE POSITIVE - clearly checkworthy
2. NEGATIVE RECOMMENDATION - not checkworthy
3. BORDERLINE CASE - mixed signals

Usage:
    python select_claims_for_study.py --input data/pipeline_output/expose_3day_full
    python select_claims_for_study.py --input data/pipeline_output/expose_3day_full --generate-reports
"""

import argparse
import sys
from pathlib import Path

import polars as pl

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))


def load_claims(input_dir: Path) -> pl.DataFrame:
    """Load claims from pipeline output."""
    claims_path = input_dir / "claims.parquet"
    if not claims_path.exists():
        raise FileNotFoundError(f"Claims file not found: {claims_path}")

    df = pl.read_parquet(claims_path)
    print(f"Loaded {len(df)} claims from {claims_path}")
    return df


def select_high_confidence_positive(df: pl.DataFrame, top_n: int = 3) -> pl.DataFrame:
    """
    Select claims that are clearly checkworthy.

    Criteria:
    - High checkworthiness probability (> 0.6)
    - High checkability score (> 60)
    - High verifiability score (> 60)
    - Elevated harm potential (> 40)
    - Bonus: viral prediction
    """
    # Filter for positive candidates
    candidates = df.filter(
        (pl.col("checkworthiness_prob").is_not_null()) &
        (pl.col("checkworthiness_prob") > 0.6)
    )

    # If we have scores, filter further
    if "checkability_score" in df.columns:
        candidates = candidates.filter(
            (pl.col("checkability_score").is_null()) |
            (pl.col("checkability_score") > 60)
        )

    if "verifiability_score" in df.columns:
        candidates = candidates.filter(
            (pl.col("verifiability_score").is_null()) |
            (pl.col("verifiability_score") > 60)
        )

    # Score candidates (higher is better)
    candidates = candidates.with_columns([
        (
            pl.col("checkworthiness_prob").fill_null(0) * 100 +
            pl.col("checkability_score").fill_null(50) +
            pl.col("verifiability_score").fill_null(50) +
            pl.col("harm_score").fill_null(30) +
            pl.when(pl.col("is_viral") == True).then(20).otherwise(0)
        ).alias("_selection_score")
    ])

    return candidates.sort("_selection_score", descending=True).head(top_n)


def select_negative_recommendation(df: pl.DataFrame, top_n: int = 3) -> pl.DataFrame:
    """
    Select claims that should NOT be fact-checked.

    Criteria:
    - Low checkworthiness probability (< 0.4)
    - Low checkability OR low verifiability (< 50)
    - Lower harm potential (< 50)
    """
    # Filter for negative candidates
    candidates = df.filter(
        (pl.col("checkworthiness_prob").is_not_null()) &
        (pl.col("checkworthiness_prob") < 0.4)
    )

    # If no checkworthiness_prob, use is_checkworthy = False
    if len(candidates) == 0:
        candidates = df.filter(pl.col("is_checkworthy") == False)

    # Score candidates (lower checkworthiness + clear reasons = better)
    candidates = candidates.with_columns([
        (
            (1 - pl.col("checkworthiness_prob").fill_null(0.5)) * 100 +
            (100 - pl.col("checkability_score").fill_null(50)) +
            (100 - pl.col("verifiability_score").fill_null(50)) +
            (100 - pl.col("harm_score").fill_null(50))
        ).alias("_selection_score")
    ])

    return candidates.sort("_selection_score", descending=True).head(top_n)


def select_borderline_case(df: pl.DataFrame, top_n: int = 3) -> pl.DataFrame:
    """
    Select claims with mixed signals (ambiguous cases).

    Criteria:
    - Moderate checkworthiness probability (0.35 - 0.65)
    - OR: High variance across dimensions (one high, one low)
    - These are the most interesting for testing explainability
    """
    # Filter for borderline candidates
    candidates = df.filter(
        (pl.col("checkworthiness_prob").is_not_null()) &
        (pl.col("checkworthiness_prob") > 0.35) &
        (pl.col("checkworthiness_prob") < 0.65)
    )

    # Score by how "borderline" they are (closest to 0.5 is best)
    # Also reward high variance across dimensions
    candidates = candidates.with_columns([
        (
            # Closer to 0.5 = more borderline
            100 - (pl.col("checkworthiness_prob") - 0.5).abs() * 200 +
            # High spread across dimensions is interesting
            (
                (pl.col("checkability_score").fill_null(50) - pl.col("harm_score").fill_null(50)).abs() +
                (pl.col("verifiability_score").fill_null(50) - pl.col("harm_score").fill_null(50)).abs()
            ) * 0.5
        ).alias("_selection_score")
    ])

    return candidates.sort("_selection_score", descending=True).head(top_n)


def print_claim_summary(claim: dict, category: str) -> None:
    """Print a formatted summary of a claim."""
    print(f"\n{'='*80}")
    print(f"  {category}")
    print(f"{'='*80}")
    print(f"  Claim ID: {claim.get('claim_id', 'N/A')[:16]}...")
    print(f"  Text: \"{claim.get('claim_text', 'N/A')[:100]}...\"" if len(claim.get('claim_text', '')) > 100 else f"  Text: \"{claim.get('claim_text', 'N/A')}\"")
    print()
    print(f"  CHECKWORTHINESS:")
    print(f"    Final Prob:    {claim.get('checkworthiness_prob', 'N/A'):.2%}" if claim.get('checkworthiness_prob') else "    Final Prob:    N/A")
    print(f"    Is Checkworthy: {claim.get('is_checkworthy', 'N/A')}")
    print()
    print(f"  MODULE SCORES:")
    print(f"    Checkability:  {claim.get('checkability_score', 'N/A'):.1f}/100" if claim.get('checkability_score') else "    Checkability:  N/A")
    print(f"    Verifiability: {claim.get('verifiability_score', 'N/A'):.1f}/100" if claim.get('verifiability_score') else "    Verifiability: N/A")
    print(f"    Harm Potential:{claim.get('harm_score', 'N/A'):.1f}/100" if claim.get('harm_score') else "    Harm Potential: N/A")
    print()
    print(f"  VIRALITY:")
    print(f"    Is Viral:      {claim.get('is_viral', 'N/A')}")
    print(f"    Confidence:    {claim.get('viral_confidence', 'N/A'):.2%}" if claim.get('viral_confidence') else "    Confidence:    N/A")
    print(f"    Total Tweets:  {claim.get('total_tweets', 'N/A')}")
    print(f"    Engagement:    {claim.get('total_engagement', 'N/A')}")
    print()

    # Print reasoning if available
    if claim.get('checkability_reasoning'):
        print(f"  CHECKABILITY REASONING:")
        print(f"    {claim['checkability_reasoning'][:200]}...")
    if claim.get('harm_reasoning'):
        print(f"  HARM REASONING:")
        print(f"    {claim['harm_reasoning'][:200]}...")


def generate_report_for_claim(claim: dict, output_dir: Path) -> Path:
    """Generate HTML explainability report for a claim."""
    from src.streaming.schemas import ClaimInfo
    from src.streaming.report_generator import create_report_from_claim

    # Convert dict back to ClaimInfo
    claim_obj = ClaimInfo(**claim)
    return create_report_from_claim(claim_obj, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Select claims for user study")
    parser.add_argument("--input", "-i", required=True, help="Pipeline output directory")
    parser.add_argument("--generate-reports", "-g", action="store_true", help="Generate HTML reports for selected claims")
    parser.add_argument("--output", "-o", default=None, help="Output directory for reports (default: input/study_claims)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "study_claims"

    # Load claims
    df = load_claims(input_dir)

    # Check if we have checkworthiness data
    has_checkworthiness = df["checkworthiness_prob"].null_count() < len(df)
    if not has_checkworthiness:
        print("\nWARNING: No checkworthiness scores found in claims!")
        print("Run the pipeline with enable_checkworthiness: true to get scores.")
        print("\nFalling back to virality-based selection...\n")

    # Select claims for each category
    print("\n" + "="*80)
    print("  CLAIM SELECTION FOR USER STUDY")
    print("="*80)

    selected_claims = []

    # 1. High Confidence Positive
    print("\n[1/3] Selecting HIGH CONFIDENCE POSITIVE claims...")
    positive_candidates = select_high_confidence_positive(df)
    if len(positive_candidates) > 0:
        best_positive = positive_candidates.head(1).to_dicts()[0]
        print_claim_summary(best_positive, "HIGH CONFIDENCE POSITIVE")
        selected_claims.append(("positive", best_positive))
    else:
        print("  No suitable candidates found!")

    # 2. Negative Recommendation
    print("\n[2/3] Selecting NEGATIVE RECOMMENDATION claims...")
    negative_candidates = select_negative_recommendation(df)
    if len(negative_candidates) > 0:
        best_negative = negative_candidates.head(1).to_dicts()[0]
        print_claim_summary(best_negative, "NEGATIVE RECOMMENDATION")
        selected_claims.append(("negative", best_negative))
    else:
        print("  No suitable candidates found!")

    # 3. Borderline Case
    print("\n[3/3] Selecting BORDERLINE CASE claims...")
    borderline_candidates = select_borderline_case(df)
    if len(borderline_candidates) > 0:
        best_borderline = borderline_candidates.head(1).to_dicts()[0]
        print_claim_summary(best_borderline, "BORDERLINE CASE")
        selected_claims.append(("borderline", best_borderline))
    else:
        print("  No suitable candidates found!")

    # Summary
    print("\n" + "="*80)
    print("  SELECTION SUMMARY")
    print("="*80)
    print(f"  Total claims analyzed: {len(df)}")
    print(f"  Claims with checkworthiness: {len(df) - df['checkworthiness_prob'].null_count()}")
    print(f"  Claims selected: {len(selected_claims)}")

    # Generate reports if requested
    if args.generate_reports and selected_claims:
        print(f"\n  Generating HTML reports to: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        for category, claim in selected_claims:
            try:
                report_path = generate_report_for_claim(claim, output_dir)
                # Rename to include category
                new_name = f"report_{category}_{claim['claim_id'][:8]}.html"
                new_path = output_dir / new_name
                report_path.rename(new_path)
                print(f"    {category}: {new_path}")
            except Exception as e:
                print(f"    {category}: Failed to generate report - {e}")

    print()


if __name__ == "__main__":
    main()
