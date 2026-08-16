"""Diagnose greedy decoding errors against the expert policy."""
import argparse
import collections
import csv
import dataclasses
import json
import logging
import math
import os
import pickle
import itertools
from typing import Any, Iterable, Optional

import torch

from trans import optimal_expert_substitutions
from trans import optimal_expert
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


def action_type(action: Any) -> str:
    if isinstance(action, vocabulary.ConditionalCopy):
        return "COPY"
    if isinstance(action, vocabulary.ConditionalSub):
        return "SUB"
    if isinstance(action, vocabulary.ConditionalIns):
        return "INS"
    if isinstance(action, vocabulary.ConditionalDel):
        return "DEL"
    if isinstance(action, vocabulary.EndOfSequence):
        return "EOS"
    if action is None:
        return "NONE"
    return action.__class__.__name__.upper()


def emission_action_label(action: Any) -> str:
    if isinstance(action, vocabulary.ConditionalCopy):
        return "COPY"
    if isinstance(action, vocabulary.ConditionalDel):
        return "DEL"
    if isinstance(action, vocabulary.ConditionalIns):
        return f"INS({action.new})"
    if isinstance(action, vocabulary.ConditionalSub):
        return f"SUB({action.new})"
    if isinstance(action, vocabulary.EndOfSequence):
        return "EOS"
    if action is None:
        return "NONE"
    return str(action)


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
    model_args.setdefault("output_feedback_dim", 0)
    model_args.setdefault("expert_temperature", 0.)
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


def reference_action_sequence(
        model: transducer.Transducer,
        sample: utils.Sample) -> list[int]:
    alignment = 0
    prediction = []
    actions = []
    stop = False
    while not stop and len(actions) <= transducer.MAX_ACTION_SEQ_LEN:
        oracle_actions = model.expert_rollout(
            sample.input,
            sample.target,
            alignment,
            prediction,
        )
        action_id = oracle_actions[0]
        actions.append(action_id)
        char, alignment, stop = model.decode_single_action(
            sample.input,
            action_id,
            alignment,
        )
        if char:
            prediction.append(char)
    return actions


@dataclasses.dataclass
class ActionConfusionStats:
    type_counts: collections.Counter = dataclasses.field(default_factory=collections.Counter)
    emission_counts: collections.Counter = dataclasses.field(default_factory=collections.Counter)
    mapping_counts: dict[str, collections.Counter] = dataclasses.field(
        default_factory=lambda: {
            "reference_nonidentity": collections.Counter(),
            "reference_identity": collections.Counter(),
        })


@dataclasses.dataclass
class InferenceReplayStats:
    total_decisions: int = 0
    optimal_decisions: int = 0
    by_predicted_action: dict[str, collections.Counter] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(collections.Counter))
    expert_sub_model_action: collections.Counter = dataclasses.field(
        default_factory=collections.Counter)
    expert_sub_model_copy_pairs: collections.Counter = dataclasses.field(
        default_factory=collections.Counter)
    first_deviation: dict[str, collections.Counter] = dataclasses.field(
        default_factory=lambda: {
            "before_first_deviation": collections.Counter(),
            "after_first_deviation": collections.Counter(),
        })


def mapping_prediction_bucket(gold_action: Any, predicted_action: Any) -> Optional[str]:
    if isinstance(gold_action, vocabulary.ConditionalSub):
        if isinstance(predicted_action, vocabulary.ConditionalSub):
            if predicted_action.new == gold_action.new:
                return "predicted_correct_sub"
            return "predicted_wrong_sub"
        if isinstance(predicted_action, vocabulary.ConditionalCopy):
            return "predicted_copy"
        return "predicted_other"
    if isinstance(gold_action, vocabulary.ConditionalCopy):
        if isinstance(predicted_action, vocabulary.ConditionalCopy):
            return "predicted_copy"
        if isinstance(predicted_action, vocabulary.ConditionalSub):
            return "predicted_sub"
        return "predicted_other"
    return None


def update_action_confusions(
        stats: ActionConfusionStats,
        model: transducer.Transducer,
        sample: utils.Sample,
        predicted_action_ids: list[int]) -> None:
    reference_action_ids = reference_action_sequence(model, sample)
    for gold_id, predicted_id in itertools.zip_longest(
            reference_action_ids,
            predicted_action_ids,
            fillvalue=None):
        gold_action = model.vocab.decode_action(gold_id) if gold_id is not None else None
        predicted_action = (
            model.vocab.decode_action(predicted_id)
            if predicted_id is not None else None
        )
        gold_type = action_type(gold_action)
        predicted_type = action_type(predicted_action)
        stats.type_counts[(gold_type, predicted_type)] += 1
        stats.emission_counts[(
            emission_action_label(gold_action),
            emission_action_label(predicted_action),
        )] += 1
        bucket = mapping_prediction_bucket(gold_action, predicted_action)
        if bucket is not None:
            mapping_name = (
                "reference_nonidentity"
                if isinstance(gold_action, vocabulary.ConditionalSub)
                else "reference_identity"
            )
            stats.mapping_counts[mapping_name]["total"] += 1
            stats.mapping_counts[mapping_name][bucket] += 1


def update_inference_replay_stats(
        stats: InferenceReplayStats,
        source: list[str],
        alignment: int,
        model_action: Any,
        optimal_actions: list[Any],
        oracle_optimal: bool,
        phase: str) -> None:
    predicted_type = action_type(model_action)
    stats.total_decisions += 1
    stats.by_predicted_action[predicted_type]["total"] += 1
    if oracle_optimal:
        stats.optimal_decisions += 1
        stats.by_predicted_action[predicted_type]["optimal"] += 1
    else:
        stats.by_predicted_action[predicted_type]["nonoptimal"] += 1

    phase_counts = stats.first_deviation[phase]
    phase_counts["total"] += 1
    if oracle_optimal:
        phase_counts["optimal"] += 1

    optimal_subs = [
        action for action in optimal_actions
        if isinstance(action, vocabulary.ConditionalSub)
    ]
    if not optimal_subs:
        return

    stats.expert_sub_model_action[predicted_type] += 1
    phase_counts["expert_sub_total"] += 1
    if isinstance(model_action, vocabulary.ConditionalCopy):
        phase_counts["expert_sub_model_copy"] += 1
        source_symbol = source[alignment] if 0 <= alignment < len(source) else ""
        for action in optimal_subs:
            stats.expert_sub_model_copy_pairs[
                f"{source_symbol}->{action.new}"] += 1


def delete_regret_row(
        model: transducer.Transducer,
        sample: utils.Sample,
        source_text: str,
        gold_text: str,
        prediction: str,
        correct: bool,
        step: int,
        alignment: int,
        prediction_so_far: list[str],
        log_probs: torch.Tensor,
        valid_actions_mask: torch.Tensor) -> dict[str, Any]:
    source_symbol = sample.input[alignment] if 0 <= alignment < len(sample.input) else ""
    current_score = model.optimal_expert.score_decoder_state(
        sample.input,
        sample.target,
        alignment,
        prediction_so_far,
    )
    delete_action_id = vocabulary.DELETE
    delete_score = model.expert_score_decoder_action(
        sample.input,
        sample.target,
        alignment,
        prediction_so_far,
        delete_action_id,
    )
    map_candidates = []
    for action_id in range(model.number_actions):
        if not valid_actions_mask[0, 0, action_id].item():
            continue
        action = model.vocab.decode_action(action_id)
        if not isinstance(action, (
                vocabulary.ConditionalCopy,
                vocabulary.ConditionalSub,
        )):
            continue
        try:
            score = model.expert_score_decoder_action(
                sample.input,
                sample.target,
                alignment,
                prediction_so_far,
                action_id,
            )
        except ValueError:
            continue
        map_candidates.append((score.total, action_id, action, score))
    best_map = min(map_candidates, default=None, key=lambda item: item[0])
    aligner = model.optimal_expert.aligner
    sed_delete_cost = (
        aligner.delete_cost(source_symbol)
        if hasattr(aligner, "delete_cost") and source_symbol != ""
        else ""
    )
    top_mappings = (
        aligner.top_mappings(source_symbol, limit=1)
        if hasattr(aligner, "top_mappings") and source_symbol != ""
        else []
    )
    best_sed_map = top_mappings[0] if top_mappings else None
    final_prediction_tokens = model.target_tokenizer.tokenize(prediction)
    final_symbol_distance = optimal_expert.levenshtein_distance(
        final_prediction_tokens,
        sample.target,
    )[-1, -1]
    return {
        "source": source_text,
        "gold": gold_text,
        "target": gold_text,
        "prediction": prediction,
        "correct": correct,
        "step": step,
        "alignment": alignment,
        "source_symbol": source_symbol,
        "output_so_far": model.target_tokenizer.untokenize(prediction_so_far),
        "model_delete_prob": math.exp(log_probs[delete_action_id].item()),
        "sed_delete_cost": sed_delete_cost,
        "best_sed_map_target": (
            best_sed_map["target"] if best_sed_map is not None else ""
        ),
        "best_sed_map_cost": (
            best_sed_map["cost"] if best_sed_map is not None else ""
        ),
        "best_sed_map_probability_given_source": (
            best_sed_map["probability"] / sum(
                row["probability"]
                for row in aligner.top_mappings(source_symbol, limit=None)
            )
            if best_sed_map is not None and hasattr(aligner, "top_mappings")
            else ""
        ),
        "current_value": current_score.total,
        "after_delete_value": delete_score.total,
        "delete_damage": delete_score.total - current_score.total,
        "best_map_action": (
            action_label_at_state(sample.input, alignment, best_map[2])
            if best_map is not None else ""
        ),
        "best_map_value": best_map[0] if best_map is not None else "",
        "delete_regret": (
            delete_score.total - best_map[0]
            if best_map is not None else ""
        ),
        "final_symbol_distance": float(final_symbol_distance),
    }


def diagnose_sample(
        model: transducer.Transducer,
        sample: utils.Sample,
        top_k: int,
        replay_stats: Optional[InferenceReplayStats] = None
        ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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
    previous_output = torch.tensor(
        [[model.vocab.encode_output_symbol(vocabulary.BOS_OUTPUT)]],
        device=model.device,
        dtype=torch.long,
    )
    prediction_so_far = []
    true_input_length = torch.tensor([len(sample.input) + 1], device=model.device)

    step_rows = []
    delete_rows = []
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
                previous_output if model.output_lookup is not None else None,
            )
            logits = model.W(decoder_output)
            log_probs = model.log_softmax(logits, valid_actions_mask)[0, 0]

            oracle_action_ids = model.expert_rollout(
                sample.input,
                sample.target,
                alignment.item(),
                prediction_so_far,
            )
            action_scores = model.expert_action_scores(
                sample.input,
                sample.target,
                alignment.item(),
                prediction_so_far,
            )
            alignment_int = alignment.item()
            best_cost = min(action_scores.values())
            optimal_actions = [
                action for action, cost in action_scores.items()
                if cost == best_cost
            ]
            model_action = model.vocab.decode_action(action_id)
            model_cost_gap = action_scores.get(model_action, float("inf")) - best_cost
            non_optimal_gaps = [
                cost - best_cost
                for action, cost in action_scores.items()
                if cost > best_cost
            ]
            best_non_oracle_cost_gap = min(non_optimal_gaps) if non_optimal_gaps else ""
            oracle_logps = [(a, log_probs[a].item()) for a in oracle_action_ids]
            oracle_mass_logp = logsumexp(logp for _, logp in oracle_logps)
            model_logp = log_probs[action_id].item()
            oracle_optimal = action_id in oracle_action_ids
            margin = model_logp - oracle_mass_logp
            replay_phase = (
                "before_first_deviation"
                if first_non_optimal_step is None and oracle_optimal
                else "after_first_deviation"
            )
            if replay_stats is not None:
                update_inference_replay_stats(
                    replay_stats,
                    sample.input,
                    alignment_int,
                    model_action,
                    optimal_actions,
                    oracle_optimal,
                    replay_phase,
                )
            if isinstance(model_action, vocabulary.ConditionalDel):
                delete_rows.append(delete_regret_row(
                    model,
                    sample,
                    source_text,
                    gold_text,
                    prediction,
                    correct,
                    step,
                    alignment_int,
                    prediction_so_far,
                    log_probs,
                    valid_actions_mask,
                ))

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
                "previous_output": model.vocab.target_symbols.decode(
                    previous_output[0, 0].item()),
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
                "model_cost_gap": model_cost_gap,
                "best_non_oracle_cost_gap": best_non_oracle_cost_gap,
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
            if model.output_lookup is not None:
                previous_output = torch.tensor(
                    [[model.output_symbol_id_for_action(
                        sample.input,
                        action,
                        step_rows[-1]["alignment"],
                    )]],
                    device=model.device,
                    dtype=torch.long,
                )
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
    return summary, step_rows, delete_rows


def write_tsv(path: str, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: format_tsv_value(row.get(key, ""))
                for key in fieldnames
            })


def action_type_confusion_rows(stats: ActionConfusionStats) -> list[dict[str, Any]]:
    return [
        {
            "gold_action_type": gold_type,
            "predicted_action_type": predicted_type,
            "count": count,
        }
        for (gold_type, predicted_type), count in sorted(stats.type_counts.items())
    ]


def emission_confusion_rows(stats: ActionConfusionStats) -> list[dict[str, Any]]:
    return [
        {
            "gold_action": gold_action,
            "predicted_action": predicted_action,
            "count": count,
        }
        for (gold_action, predicted_action), count in sorted(stats.emission_counts.items())
    ]


def mapping_bias_rows(stats: ActionConfusionStats) -> list[dict[str, Any]]:
    rows = []
    bucket_names = [
        "predicted_correct_sub",
        "predicted_copy",
        "predicted_wrong_sub",
        "predicted_sub",
        "predicted_other",
    ]
    for mapping_name, counts in stats.mapping_counts.items():
        total = counts["total"]
        row = {
            "reference_mapping": mapping_name,
            "total": total,
        }
        for bucket in bucket_names:
            count = counts[bucket]
            row[bucket] = count
            row[f"{bucket}_pct"] = (100 * count / total) if total else 0.
        rows.append(row)
    return rows


def inference_action_summary_rows(stats: InferenceReplayStats) -> list[dict[str, Any]]:
    return [
        {
            "metric": "total_decisions",
            "value": stats.total_decisions,
        },
        {
            "metric": "optimal_decisions",
            "value": stats.optimal_decisions,
        },
        {
            "metric": "optimal_pct",
            "value": (
                100 * stats.optimal_decisions / stats.total_decisions
                if stats.total_decisions else 0.
            ),
        },
    ]


def inference_action_by_prediction_rows(
        stats: InferenceReplayStats) -> list[dict[str, Any]]:
    rows = []
    for predicted_type, counts in sorted(stats.by_predicted_action.items()):
        total = counts["total"]
        optimal = counts["optimal"]
        nonoptimal = counts["nonoptimal"]
        rows.append({
            "predicted_action_type": predicted_type,
            "total": total,
            "optimal": optimal,
            "nonoptimal": nonoptimal,
            "optimal_pct": 100 * optimal / total if total else 0.,
            "nonoptimal_pct": 100 * nonoptimal / total if total else 0.,
        })
    return rows


def expert_sub_model_action_rows(
        stats: InferenceReplayStats) -> list[dict[str, Any]]:
    return [
        {
            "predicted_action_type": predicted_type,
            "count": count,
        }
        for predicted_type, count in sorted(stats.expert_sub_model_action.items())
    ]


def expert_sub_model_copy_pair_rows(
        stats: InferenceReplayStats) -> list[dict[str, Any]]:
    return [
        {
            "source_to_required_phone": source_to_phone,
            "count": count,
        }
        for source_to_phone, count in stats.expert_sub_model_copy_pairs.most_common()
    ]


def first_deviation_rows(stats: InferenceReplayStats) -> list[dict[str, Any]]:
    rows = []
    for phase in ["before_first_deviation", "after_first_deviation"]:
        counts = stats.first_deviation[phase]
        total = counts["total"]
        expert_sub_total = counts["expert_sub_total"]
        rows.append({
            "phase": phase,
            "total": total,
            "optimal": counts["optimal"],
            "optimal_pct": (
                100 * counts["optimal"] / total if total else 0.
            ),
            "expert_sub_total": expert_sub_total,
            "expert_sub_model_copy": counts["expert_sub_model_copy"],
            "expert_sub_model_copy_pct": (
                100 * counts["expert_sub_model_copy"] / expert_sub_total
                if expert_sub_total else 0.
            ),
        })
    return rows


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
    delete_rows = []
    confusion_stats = ActionConfusionStats()
    replay_stats = InferenceReplayStats()
    for line_number, sample in enumerate(samples, start=1):
        summary, steps, deletes = diagnose_sample(
            model, sample, args.top_k_actions, replay_stats)
        summary["line_number"] = line_number
        for step in steps:
            step["line_number"] = line_number
        for delete in deletes:
            delete["line_number"] = line_number
        update_action_confusions(
            confusion_stats,
            model,
            sample,
            [step["model_action_id"] for step in steps],
        )
        if not args.errors_only or not summary["correct"]:
            summary_rows.append(summary)
            step_rows.extend(steps)
        delete_rows.extend(deletes)

    summary_path = os.path.join(args.output, "diagnostics.tsv")
    steps_path = os.path.join(args.output, "diagnostic_steps.tsv")
    action_type_confusion_path = os.path.join(args.output, "action_type_confusion.tsv")
    emission_confusion_path = os.path.join(args.output, "emission_confusion.tsv")
    mapping_bias_path = os.path.join(args.output, "mapping_bias.tsv")
    inference_summary_path = os.path.join(args.output, "inference_action_summary.tsv")
    inference_by_prediction_path = os.path.join(
        args.output, "inference_action_by_prediction.tsv")
    expert_sub_model_action_path = os.path.join(
        args.output, "expert_sub_model_action.tsv")
    expert_sub_model_copy_pairs_path = os.path.join(
        args.output, "expert_sub_model_copy_pairs.tsv")
    first_deviation_path = os.path.join(
        args.output, "first_deviation_diagnostics.tsv")
    delete_diagnostics_path = os.path.join(args.output, "delete_diagnostics.tsv")
    summary_fields = [
        "line_number", "source", "gold", "target", "prediction", "correct",
        "num_actions", "first_non_optimal_step", "num_non_optimal",
        "fraction_oracle_optimal", "first_error_model_action",
        "first_error_model_prob", "first_error_oracle_mass",
        "first_error_margin",
    ]
    step_fields = [
        "line_number", "source", "gold", "target", "prediction", "correct", "step",
        "alignment", "output_so_far", "previous_output", "model_action", "model_action_id",
        "model_logp", "model_prob", "oracle_optimal", "oracle_actions",
        "oracle_mass_logp", "oracle_mass_prob", "oracle_margin_logp",
        "model_cost_gap", "best_non_oracle_cost_gap",
        "top_actions",
    ]
    write_tsv(summary_path, summary_rows, summary_fields)
    write_tsv(steps_path, step_rows, step_fields)
    write_tsv(
        action_type_confusion_path,
        action_type_confusion_rows(confusion_stats),
        ["gold_action_type", "predicted_action_type", "count"],
    )
    write_tsv(
        emission_confusion_path,
        emission_confusion_rows(confusion_stats),
        ["gold_action", "predicted_action", "count"],
    )
    mapping_bias_fields = [
        "reference_mapping", "total",
        "predicted_correct_sub", "predicted_correct_sub_pct",
        "predicted_copy", "predicted_copy_pct",
        "predicted_wrong_sub", "predicted_wrong_sub_pct",
        "predicted_sub", "predicted_sub_pct",
        "predicted_other", "predicted_other_pct",
    ]
    write_tsv(mapping_bias_path, mapping_bias_rows(confusion_stats), mapping_bias_fields)
    write_tsv(
        inference_summary_path,
        inference_action_summary_rows(replay_stats),
        ["metric", "value"],
    )
    write_tsv(
        inference_by_prediction_path,
        inference_action_by_prediction_rows(replay_stats),
        [
            "predicted_action_type", "total", "optimal", "nonoptimal",
            "optimal_pct", "nonoptimal_pct",
        ],
    )
    write_tsv(
        expert_sub_model_action_path,
        expert_sub_model_action_rows(replay_stats),
        ["predicted_action_type", "count"],
    )
    write_tsv(
        expert_sub_model_copy_pairs_path,
        expert_sub_model_copy_pair_rows(replay_stats),
        ["source_to_required_phone", "count"],
    )
    write_tsv(
        first_deviation_path,
        first_deviation_rows(replay_stats),
        [
            "phase", "total", "optimal", "optimal_pct",
            "expert_sub_total", "expert_sub_model_copy",
            "expert_sub_model_copy_pct",
        ],
    )
    delete_fields = [
        "line_number", "source", "gold", "target", "prediction", "correct",
        "step", "alignment", "source_symbol", "output_so_far",
        "model_delete_prob", "sed_delete_cost", "best_sed_map_target",
        "best_sed_map_cost", "best_sed_map_probability_given_source",
        "current_value", "after_delete_value", "delete_damage",
        "best_map_action", "best_map_value", "delete_regret",
        "final_symbol_distance",
    ]
    write_tsv(delete_diagnostics_path, delete_rows, delete_fields)
    logging.info("Wrote %s.", summary_path)
    logging.info("Wrote %s.", steps_path)
    logging.info("Wrote %s.", action_type_confusion_path)
    logging.info("Wrote %s.", emission_confusion_path)
    logging.info("Wrote %s.", mapping_bias_path)
    logging.info("Wrote %s.", inference_summary_path)
    logging.info("Wrote %s.", inference_by_prediction_path)
    logging.info("Wrote %s.", expert_sub_model_action_path)
    logging.info("Wrote %s.", expert_sub_model_copy_pairs_path)
    logging.info("Wrote %s.", first_deviation_path)
    logging.info("Wrote %s.", delete_diagnostics_path)


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
