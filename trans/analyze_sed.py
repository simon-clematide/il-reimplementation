"""Analyze source-target pairs with a fitted SED model."""
from typing import List, Optional
import argparse
import csv
import dataclasses
import logging
import math
import sys

from trans import sed
from trans.actions import Del, Ins, Sub
from trans import utils


@dataclasses.dataclass
class SedAnalysis:
    line_number: int
    source: str
    target: str
    stochastic_logp: float
    stochastic_surprisal: float
    max_length_surprisal: float
    target_length_surprisal: float
    viterbi_logp: float
    viterbi_surprisal: float
    alignment_ambiguity: float
    alignment: str


def read_tsv(path: str, normalize: bool = False) -> List[utils.Sample]:
    samples = []
    with utils.OpenNormalize(path, normalize) as f:
        for line_number, line in enumerate(f, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                raise ValueError(
                    f"Expected at least two TSV columns at line {line_number}: {line!r}")
            samples.append(utils.Sample(input=fields[0], target=fields[1]))
    return samples


def length_normalize(value: float, length: int) -> float:
    return value / max(length, 1)


def format_float(value: float) -> str:
    if math.isinf(value) or math.isnan(value):
        return str(value)
    return f"{value:.6f}"


def format_alignment(alignment) -> str:
    formatted = []
    for action in alignment:
        if isinstance(action, Sub):
            formatted.append(f"sub({action.old}->{action.new})")
        elif isinstance(action, Del):
            formatted.append(f"del({action.old})")
        elif isinstance(action, Ins):
            formatted.append(f"ins({action.new})")
        else:
            formatted.append(repr(action))
    return " ".join(formatted)


def analyze_sample(sed_: sed.StochasticEditDistance,
                   sample: utils.Sample,
                   line_number: int) -> SedAnalysis:
    stochastic_logp = sed_.stochastic_distance(sample.input, sample.target)
    alignment, viterbi_logp = sed_.viterbi_distance(
        sample.input,
        sample.target,
        with_alignment=True,
    )
    stochastic_surprisal = -stochastic_logp
    viterbi_surprisal = -viterbi_logp
    return SedAnalysis(
        line_number=line_number,
        source=sample.input,
        target=sample.target,
        stochastic_logp=stochastic_logp,
        stochastic_surprisal=stochastic_surprisal,
        max_length_surprisal=length_normalize(
            stochastic_surprisal,
            max(len(sample.input), len(sample.target)),
        ),
        target_length_surprisal=length_normalize(
            stochastic_surprisal,
            len(sample.target),
        ),
        viterbi_logp=viterbi_logp,
        viterbi_surprisal=viterbi_surprisal,
        alignment_ambiguity=stochastic_logp - viterbi_logp,
        alignment=format_alignment(alignment),
    )


def write_analyses(analyses: List[SedAnalysis], output: Optional[str]) -> None:
    fields = [field.name for field in dataclasses.fields(SedAnalysis)]
    rows = ["\t".join(fields)]
    for analysis in analyses:
        values = []
        for field in fields:
            value = getattr(analysis, field)
            if isinstance(value, float):
                value = format_float(value)
            values.append(str(value))
        rows.append("\t".join(values))

    text = "\n".join(rows) + "\n"
    if output is None or output == "-":
        sys.stdout.write(text)
    else:
        with open(output, "w", encoding="utf8") as w:
            w.write(text)
        logging.info("Wrote SED analysis to %s.", output)


def write_rows(rows, output: str, fields: List[str]) -> None:
    with open(output, "w", encoding="utf8", newline="") as w:
        writer = csv.DictWriter(w, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: format_float(row[field])
                if isinstance(row.get(field), float) else row.get(field, "")
                for field in fields
            })
    logging.info("Wrote %s.", output)


def mapping_summary_rows(sed_: sed.StochasticEditDistance,
                         limit: Optional[int] = None):
    rows = []
    for source in sed_.source_alphabet:
        all_mappings = sed_.top_mappings(source, limit=None)
        row_mass = sum(row["probability"] for row in all_mappings)
        identity_mass = sum(
            row["probability"] for row in all_mappings
            if row["identity"]
        )
        nonidentity_mass = row_mass - identity_mass
        mappings = sed_.top_mappings(source, limit=limit)
        for rank, row in enumerate(mappings, start=1):
            rows.append({
                "source": source,
                "rank": rank,
                "target": row["target"],
                "probability": row["probability"],
                "map_probability_given_source": (
                    row["probability"] / row_mass if row_mass else 0.
                ),
                "cost": row["cost"],
                "identity": row["identity"],
                "source_map_mass": row_mass,
                "identity_map_mass": identity_mass,
                "nonidentity_map_mass": nonidentity_mass,
                "identity_map_mass_given_source": (
                    identity_mass / row_mass if row_mass else 0.
                ),
                "nonidentity_map_mass_given_source": (
                    nonidentity_mass / row_mass if row_mass else 0.
                ),
            })
    return rows


def map_mass_summary_rows(sed_: sed.StochasticEditDistance):
    map_mass = sum(
        row["probability"] for row in sed_.event_table()
        if row["event"] == "map"
    )
    diagonal_mass = sum(
        probability for (source, target), log_probability in sed_.delta_sub.items()
        if source == target
        for probability in [math.exp(log_probability)]
    )
    nondiagonal_mass = map_mass - diagonal_mass
    total_mass = sum(row["probability"] for row in sed_.event_table())
    return [
        {
            "metric": "source_alphabet_size",
            "value": len(sed_.source_alphabet),
        },
        {
            "metric": "target_alphabet_size",
            "value": len(sed_.target_alphabet),
        },
        {
            "metric": "total_event_mass",
            "value": total_mass,
        },
        {
            "metric": "map_event_mass",
            "value": map_mass,
        },
        {
            "metric": "diagonal_map_mass",
            "value": diagonal_mass,
        },
        {
            "metric": "nondiagonal_map_mass",
            "value": nondiagonal_mass,
        },
        {
            "metric": "diagonal_map_mass_given_map",
            "value": diagonal_mass / map_mass if map_mass else 0.,
        },
        {
            "metric": "nondiagonal_map_mass_given_map",
            "value": nondiagonal_mass / map_mass if map_mass else 0.,
        },
    ]


def copy_preference_rows(sed_: sed.StochasticEditDistance):
    rows = []
    overlap = sorted(set(sed_.source_alphabet) & set(sed_.target_alphabet))
    for source in overlap:
        mappings = sed_.top_mappings(source, limit=None)
        row_mass = sum(row["probability"] for row in mappings)
        identity_rows = [
            row for row in mappings
            if row["target"] == source
        ]
        identity_row = identity_rows[0] if identity_rows else None
        nonidentity_rows = [
            row for row in mappings
            if row["target"] != source
        ]
        best_nonidentity = nonidentity_rows[0] if nonidentity_rows else None
        identity_probability = (
            identity_row["probability"] if identity_row is not None else 0.
        )
        best_nonidentity_probability = (
            best_nonidentity["probability"] if best_nonidentity is not None else 0.
        )
        rows.append({
            "source": source,
            "identity_target": source,
            "identity_probability": identity_probability,
            "source_map_mass": row_mass,
            "identity_probability_given_source": (
                identity_probability / row_mass if row_mass else 0.
            ),
            "best_nonidentity_target": (
                best_nonidentity["target"] if best_nonidentity is not None else ""
            ),
            "best_nonidentity_probability": best_nonidentity_probability,
            "best_nonidentity_probability_given_source": (
                best_nonidentity_probability / row_mass if row_mass else 0.
            ),
            "copy_preference_margin": (
                (identity_probability - best_nonidentity_probability) / row_mass
                if row_mass else 0.
            ),
            "best_target": mappings[0]["target"] if mappings else "",
            "best_target_is_identity": bool(mappings and mappings[0]["target"] == source),
        })
    rows.sort(key=lambda row: row["identity_probability_given_source"])
    return rows


def main(args: argparse.Namespace) -> None:
    sed_ = sed.StochasticEditDistance.from_pickle(args.sed_params)
    if args.events_output:
        write_rows(
            sed_.event_table(),
            args.events_output,
            ["event", "source", "target", "probability", "cost", "log_probability"],
        )
    if args.mappings_output:
        write_rows(
            mapping_summary_rows(sed_, limit=args.top_k_mappings),
            args.mappings_output,
            [
                "source", "rank", "target", "probability", "cost", "identity",
                "map_probability_given_source", "source_map_mass",
                "identity_map_mass", "nonidentity_map_mass",
                "identity_map_mass_given_source",
                "nonidentity_map_mass_given_source",
            ],
        )
    if args.map_mass_output:
        write_rows(
            map_mass_summary_rows(sed_),
            args.map_mass_output,
            ["metric", "value"],
        )
    if args.copy_preference_output:
        write_rows(
            copy_preference_rows(sed_),
            args.copy_preference_output,
            [
                "source", "identity_target", "identity_probability",
                "source_map_mass", "identity_probability_given_source",
                "best_nonidentity_target", "best_nonidentity_probability",
                "best_nonidentity_probability_given_source",
                "copy_preference_margin", "best_target",
                "best_target_is_identity",
            ],
        )
    if args.input is not None:
        samples = read_tsv(args.input, normalize=args.nfd)
        analyses = [
            analyze_sample(sed_, sample, line_number)
            for line_number, sample in enumerate(samples, 1)
        ]
        analyses.sort(
            key=lambda analysis: getattr(analysis, args.sort_by),
            reverse=args.descending,
        )
        if args.limit is not None:
            analyses = analyses[:args.limit]
        write_analyses(analyses, args.output)
    elif (not args.events_output and not args.mappings_output and
          not args.map_mass_output and not args.copy_preference_output):
        raise ValueError(
            "Provide --input for pair analysis or --events-output/"
            "--mappings-output/--map-mass-output/--copy-preference-output "
            "for SED parameter inspection.")


def cli_main():
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Rank source-target pairs by SED surprisal and alignment diagnostics.")
    parser.add_argument("--sed-params", required=True,
                        help="Path to a fitted sed.pkl file.")
    parser.add_argument("--input",
                        help="Input TSV with source and target in the first two columns.")
    parser.add_argument("--output",
                        help="Output TSV path. Defaults to stdout.")
    parser.add_argument("--sort-by", default="stochastic_surprisal",
                        choices=[field.name for field in dataclasses.fields(SedAnalysis)],
                        help="Column used for ranking.")
    parser.add_argument("--ascending", action="store_true",
                        help="Sort in ascending order instead of descending.")
    parser.add_argument("--limit", type=int,
                        help="Only write the top N rows after sorting.")
    parser.add_argument("--nfd", action="store_true",
                        help="Read input after NFD normalization.")
    parser.add_argument("--events-output",
                        help="Write the complete learned SED event table as TSV.")
    parser.add_argument("--mappings-output",
                        help="Write top target mappings per source symbol as TSV.")
    parser.add_argument("--map-mass-output",
                        help="Write diagonal/non-diagonal MAP event mass summary as TSV.")
    parser.add_argument("--copy-preference-output",
                        help="Write source-specific COPY preference summary for alphabet overlap.")
    parser.add_argument("--top-k-mappings", type=int, default=10,
                        help="Number of target mappings to report per source symbol.")

    args = parser.parse_args()
    args.descending = not args.ascending
    main(args)


if __name__ == "__main__":
    cli_main()
