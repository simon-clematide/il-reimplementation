"""Diagnose greedy decoding errors against the expert policy."""
import argparse
import csv
import json
import logging
import math
import os
import pickle
from typing import Any, Iterable, Optional

import torch

from trans import optimal_expert_substitutions
from trans import sed
from trans import transducer
from trans import utils
from trans import vocabulary


def action_label(action: Any) -> str:
    if isinstance(action, vocabulary.ConditionalCopy):
        return "copy"
    if isinstance(action, vocabulary.ConditionalDel):
        return "del"
    if isinstance(action, vocabulary.ConditionalIns):
        return f"ins({action.new})"
    if isinstance(action, vocabulary.ConditionalSub):
        return f"sub({action.new})"
    if isinstance(action, vocabulary.EndOfSequence):
        return "end"
    if isinstance(action, vocabulary.BeginOfSequence):
        return "begin"
    return str(action)


def action_label_at_state(source: str, alignment: int, action: Any) -> str:
    old = source[alignment] if 0 <= alignment < len(source) else ""
    if isinstance(action, vocabulary.ConditionalCopy):
        return f"copy({old})"
    if isinstance(action, vocabulary.ConditionalDel):
        return f"del({old})"
    if isinstance(action, vocabulary.ConditionalSub):
        return f"sub({old}->{action.new})"
    return action_label(action)


def logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return -math.inf
    max_value = max(values)
    if max_value == -math.inf:
        return -math.inf
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def load_vocabularies(path: str):
    with open(path, "rb") as f:
        params = pickle.load(f)
    if "features" in params:
        return vocabulary.FeatureVocabularies(**params)
    return vocabulary.Vocabularies(**params)


def read_samples(path: str, vocabularies, device: str, normalize: bool) -> list[utils.Sample]:
    has_features = isinstance(vocabularies, vocabulary.FeatureVocabularies)
    source_tokenizer = utils.Tokenizer(vocabularies.source_separator)
    target_tokenizer = utils.Tokenizer(vocabularies.target_separator)
    samples = []
    with utils.OpenNormalize(path, normalize) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if has_features:
                if len(fields) < 3:
                    raise ValueError("Feature vocabularies require input rows with source, target, and features.")
                source_text, target_text, features = fields[:3]
                encoded_features = torch.tensor(
                    vocabularies.encode_unseen_features(features),
                    device=device,
                )
            else:
                if len(fields) < 2:
                    raise ValueError("Diagnostic input rows require at least source and target columns.")
                source_text, target_text = fields[:2]
                features = encoded_features = None
            source = source_tokenizer.tokenize(source_text)
            target = target_tokenizer.tokenize(target_text)
            encoded_input = torch.tensor(
                vocabularies.encode_unseen_input(source),
                device=device,
            )
            samples.append(utils.Sample(
                source,
                target,
                encoded_input,
                features=features,
                encoded_features=encoded_features,
            ))
    return samples


def load_model_args(metadata_path: str, device: str) -> argparse.Namespace:
    with open(metadata_path) as f:
        metadata = json.load(f)
    model_args = metadata["args"]
    model_args.setdefault("enc_output_dropout", 0.)
    model_args.setdefault("enc_output_dropout_type", "locked")
    model_args.setdefault("source_separator", None)
    model_args.setdefault("target_separator", None)
    model_args["device"] = device
    return argparse.Namespace(**model_args)


def top_action_items(vocabularies, log_probs: torch.Tensor, top_k: int) -> list[tuple[int, float]]:
    values, indices = torch.topk(log_probs, k=min(top_k, log_probs.numel()))
    return [
        (index.item(), value.item())
        for index, value in zip(indices, values)
        if value.item() != -math.inf
    ]


def format_action_distribution(
        vocabularies,
        action_logps: list[tuple[int, float]],
        source: Optional[str] = None,
        alignment: Optional[int] = None) -> str:
    def label(action_id):
        action = vocabularies.decode_action(action_id)
        if source is None or alignment is None:
            return action_label(action)
        return action_label_at_state(source, alignment, action)

    return " ".join(
        f"{label(action_id)}:{math.exp(logp):.6f}"
        for action_id, logp in action_logps
    )


def diagnose_sample(
        model: transducer.Transducer,
        sample: utils.Sample,
        top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    encoded_input = sample.encoded_input.unsqueeze(dim=0)
    encoded_features = (
        sample.encoded_features.unsqueeze(dim=0)
        if sample.encoded_features is not None else None
    )
    output = model.transduce([sample.input], encoded_input, encoded_features)
    prediction = output.output[0]
    source_text = model.source_tokenizer.untokenize(sample.input)
    gold_text = model.target_tokenizer.untokenize(sample.target)
    correct = prediction == gold_text

    model.h0_c0 = 1
    encoder_output = model.encoder_step(encoded_input)
    feature_embedding = model.feature_embedding(encoded_features)
    decoder = model.h0_c0
    alignment = torch.tensor([0], device=model.device)
    action_history = torch.tensor([[vocabulary.BEGIN_WORD]], device=model.device)
    prediction_so_far = []
    true_input_length = torch.tensor([len(sample.input) + 1], device=model.device)

    step_rows = []
    first_non_optimal_step = None
    first_error_model_action = ""
    first_error_model_prob = ""
    first_error_oracle_mass = ""
    first_error_margin = ""
    num_non_optimal = 0

    with torch.no_grad():
        for step, action_id in enumerate(output.action_history[0]):
            valid_actions_mask = model.valid_actions_for_suffixes(true_input_length - alignment)
            decoder_output, next_decoder = model.decoder_step(
                encoder_output,
                feature_embedding,
                decoder,
                alignment,
                action_history[-1].unsqueeze(dim=0),
            )
            logits = model.W(decoder_output)
            log_probs = model.log_softmax(logits, valid_actions_mask)[0, 0]

            oracle_action_ids = model.expert_rollout(
                sample.input,
                sample.target,
                alignment.item(),
                prediction_so_far,
            )
            alignment_int = alignment.item()
            oracle_logps = [(a, log_probs[a].item()) for a in oracle_action_ids]
            oracle_mass_logp = logsumexp(logp for _, logp in oracle_logps)
            model_logp = log_probs[action_id].item()
            oracle_optimal = action_id in oracle_action_ids
            margin = model_logp - oracle_mass_logp

            if not oracle_optimal:
                num_non_optimal += 1
                if first_non_optimal_step is None:
                    first_non_optimal_step = step
                    first_error_model_action = action_label_at_state(
                        sample.input,
                        alignment_int,
                        model.vocab.decode_action(action_id),
                    )
                    first_error_model_prob = math.exp(model_logp)
                    first_error_oracle_mass = math.exp(oracle_mass_logp)
                    first_error_margin = margin

            step_rows.append({
                "source": source_text,
                "gold": gold_text,
                "target": gold_text,
                "prediction": prediction,
                "correct": correct,
                "step": step,
                "alignment": alignment_int,
                "output_so_far": model.target_tokenizer.untokenize(prediction_so_far),
                "model_action": action_label_at_state(
                    sample.input,
                    alignment_int,
                    model.vocab.decode_action(action_id),
                ),
                "model_action_id": action_id,
                "model_logp": model_logp,
                "model_prob": math.exp(model_logp),
                "oracle_optimal": oracle_optimal,
                "oracle_actions": format_action_distribution(
                    model.vocab, oracle_logps, sample.input, alignment_int),
                "oracle_mass_logp": oracle_mass_logp,
                "oracle_mass_prob": math.exp(oracle_mass_logp),
                "oracle_margin_logp": margin,
                "top_actions": format_action_distribution(
                    model.vocab,
                    top_action_items(model.vocab, log_probs, top_k),
                    sample.input,
                    alignment_int,
                ),
            })

            action_tensor = torch.tensor([[action_id]], device=model.device)
            action = model.vocab.decode_action(action_id)
            char, alignment, stop = model.decode_single_action(
                sample.input, action, alignment)
            if char:
                prediction_so_far.append(char)
            action_history = torch.cat((action_history, action_tensor))
            decoder = next_decoder
            if stop:
                break

    num_actions = len(output.action_history[0])
    summary = {
        "source": source_text,
        "gold": gold_text,
        "target": gold_text,
        "prediction": prediction,
        "correct": correct,
        "num_actions": num_actions,
        "first_non_optimal_step": "" if first_non_optimal_step is None else first_non_optimal_step,
        "num_non_optimal": num_non_optimal,
        "fraction_oracle_optimal": (
            1.0 if num_actions == 0 else (num_actions - num_non_optimal) / num_actions
        ),
        "first_error_model_action": first_error_model_action,
        "first_error_model_prob": first_error_model_prob,
        "first_error_oracle_mass": first_error_oracle_mass,
        "first_error_margin": first_error_margin,
    }
    return summary, step_rows


def write_tsv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: format_tsv_value(row.get(key, ""))
                for key in fieldnames
            })


def format_tsv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output, exist_ok=True)
    vocabularies = load_vocabularies(args.vocabulary)
    model_args = load_model_args(args.metadata, args.device)
    sed_aligner = sed.StochasticEditDistance.from_pickle(args.sed_params)
    expert = optimal_expert_substitutions.OptimalSubstitutionExpert(sed_aligner)
    model = transducer.Transducer(vocabularies, expert, model_args)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.eval()

    samples = read_samples(args.input, vocabularies, args.device, args.nfd)
    summary_rows = []
    step_rows = []
    for line_number, sample in enumerate(samples, start=1):
        summary, steps = diagnose_sample(model, sample, args.top_k_actions)
        summary["line_number"] = line_number
        for step in steps:
            step["line_number"] = line_number
        if not args.errors_only or not summary["correct"]:
            summary_rows.append(summary)
            step_rows.extend(steps)

    summary_path = os.path.join(args.output, "diagnostics.tsv")
    steps_path = os.path.join(args.output, "diagnostic_steps.tsv")
    summary_fields = [
        "line_number", "source", "gold", "target", "prediction", "correct",
        "num_actions", "first_non_optimal_step", "num_non_optimal",
        "fraction_oracle_optimal", "first_error_model_action",
        "first_error_model_prob", "first_error_oracle_mass",
        "first_error_margin",
    ]
    step_fields = [
        "line_number", "source", "gold", "target", "prediction", "correct", "step",
        "alignment", "output_so_far", "model_action", "model_action_id",
        "model_logp", "model_prob", "oracle_optimal", "oracle_actions",
        "oracle_mass_logp", "oracle_mass_prob", "oracle_margin_logp",
        "top_actions",
    ]
    write_tsv(summary_path, summary_rows, summary_fields)
    write_tsv(steps_path, step_rows, step_fields)
    logging.info("Wrote %s.", summary_path)
    logging.info("Wrote %s.", steps_path)


def cli_main() -> None:
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Diagnose greedy transducer predictions against the expert policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Path to model checkpoint, usually best.model.")
    parser.add_argument("--metadata", required=True,
                        help="Path to checkpoint metadata, usually best.model.json.")
    parser.add_argument("--vocabulary", required=True,
                        help="Path to vocabulary.pkl.")
    parser.add_argument("--sed-params", required=True,
                        help="Path to sed.pkl used by the expert.")
    parser.add_argument("--input", required=True,
                        help="TSV file with source and target columns.")
    parser.add_argument("--output", required=True,
                        help="Output directory for diagnostics TSV files.")
    parser.add_argument("--top-k-actions", type=int, default=5,
                        help="Number of highest-probability actions to include per step.")
    parser.add_argument("--errors-only", action=argparse.BooleanOptionalAction, default=True,
                        help="Only write diagnostics for incorrect predictions.")
    parser.add_argument("--nfd", action="store_true", default=False,
                        help="Read input as NFD-normalized data.")
    parser.add_argument("--device", default="cpu",
                        help="Device to run diagnostics on.")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
