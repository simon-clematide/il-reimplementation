"""Analyze source-target pairs with a fitted SED model."""
from typing import List, Optional
import argparse
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


def main(args: argparse.Namespace) -> None:
    sed_ = sed.StochasticEditDistance.from_pickle(args.sed_params)
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


def cli_main():
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Rank source-target pairs by SED surprisal and alignment diagnostics.")
    parser.add_argument("--sed-params", required=True,
                        help="Path to a fitted sed.pkl file.")
    parser.add_argument("--input", required=True,
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

    args = parser.parse_args()
    args.descending = not args.ascending
    main(args)


if __name__ == "__main__":
    cli_main()
