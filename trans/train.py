"""Trains a grapheme-to-phoneme neural transducer."""
import argparse
import collections
import copy
import dataclasses
import json
import logging
import os
import random
import subprocess
import sys
from typing import Optional

import progressbar

import torch
import numpy as np

from trans import optimal_expert_substitutions
from trans import sed
from trans import transducer
from trans import utils
from trans import vocabulary
from trans import ENCODER_MAPPING, OPTIMIZER_MAPPING, LR_SCHEDULER_MAPPING

random.seed(1)


def accumulation_loss_scale(batch_index: int, batch_count: int,
                            accumulation: int) -> int:
    if accumulation < 1:
        raise ValueError("Gradient accumulation must be at least 1.")
    group_start = batch_index - (batch_index % accumulation)
    return min(accumulation, batch_count - group_start)


def should_step(batch_index: int, batch_count: int, accumulation: int) -> bool:
    if accumulation < 1:
        raise ValueError("Gradient accumulation must be at least 1.")
    is_accumulated_batch = (batch_index + 1) % accumulation == 0
    is_final_batch = batch_index + 1 == batch_count
    return is_accumulated_batch or is_final_batch


def should_stop_for_patience(patience: int, max_patience: int) -> bool:
    if max_patience < 1:
        raise ValueError("Patience must be at least 1.")
    return patience >= max_patience


def model_selection_key(string_accuracy: float, symbol_accuracy: float) -> tuple[float, float]:
    return string_accuracy, symbol_accuracy


def optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> list[float]:
    return [param_group["lr"] for param_group in optimizer.param_groups]


def learning_rates_changed(before: list[float], after: list[float]) -> bool:
    return before != after


def log_learning_rate_change(
        before: list[float],
        optimizer: torch.optim.Optimizer,
        scheduler_name: str) -> None:
    after = optimizer_learning_rates(optimizer)
    if learning_rates_changed(before, after):
        logging.info(
            "Learning rate changed by %s scheduler: %s -> %s.",
            scheduler_name,
            before,
            after,
        )


def build_optimizer_and_scheduler(
        model: transducer.Transducer,
        args: argparse.Namespace) -> tuple[torch.optim.Optimizer, Optional[object]]:
    optimizer = OPTIMIZER_MAPPING[args.optimizer](model.parameters(), args)
    scheduler = None
    if args.scheduler is not None:
        scheduler = LR_SCHEDULER_MAPPING[args.scheduler](optimizer, args)
    return optimizer, scheduler


def set_optimizer_learning_rates(
        optimizer: torch.optim.Optimizer,
        learning_rates: list[float]) -> None:
    if len(learning_rates) != len(optimizer.param_groups):
        raise ValueError(
            "Cannot restore learning rates: number of learning rates does not "
            "match optimizer parameter groups."
        )
    for param_group, learning_rate in zip(optimizer.param_groups, learning_rates):
        param_group["lr"] = learning_rate


def reload_best_model_and_reset_optimizer(
        model: transducer.Transducer,
        args: argparse.Namespace,
        best_model_path: str,
        learning_rates: list[float]) -> tuple[torch.optim.Optimizer, Optional[object]]:
    if not os.path.exists(best_model_path):
        raise RuntimeError(
            f"Cannot reload best model after learning-rate reduction; "
            f"{best_model_path} does not exist."
        )
    model.load_state_dict(torch.load(best_model_path, map_location=model.device))
    optimizer, scheduler = build_optimizer_and_scheduler(model, args)
    set_optimizer_learning_rates(optimizer, learning_rates)
    logging.info(
        "Reloaded best model from %s and reset optimizer state at learning rates %s.",
        best_model_path,
        learning_rates,
    )
    return optimizer, scheduler


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_checkpoint_metadata(path: str, args: argparse.Namespace,
                              epoch: int, dev_string_accuracy: float,
                              dev_symbol_accuracy: float,
                              train_string_accuracy: float) -> None:
    metadata = {
        "epoch": epoch,
        "dev_string_accuracy": dev_string_accuracy,
        "dev_symbol_accuracy": dev_symbol_accuracy,
        "train_string_accuracy": train_string_accuracy,
        "git_commit": current_git_commit(),
        "args": vars(args),
    }
    with open(path, "w") as w:
        json.dump(metadata, w, indent=2, sort_keys=True)


def write_sed_metadata(path: str, args: argparse.Namespace,
                       training_data: utils.Dataset,
                       vocabulary_: vocabulary.Vocabularies) -> None:
    metadata = {
        "sed_params": os.path.basename(path).removesuffix(".json"),
        "git_commit": current_git_commit(),
        "args": vars(args),
        "train": args.train,
        "source_separator": vocabulary_.source_separator,
        "target_separator": vocabulary_.target_separator,
        "em_iterations": args.sed_em_iterations,
        "em_mode": args.sed_em_mode,
        "em_damping": args.sed_em_damping,
        "copy_probability": None,
        "num_samples": len(training_data.samples),
        "source_alphabet_size": len(vocabulary_.characters.to_i2w()),
        "target_alphabet_size": len(vocabulary_.target_characters),
        "source_alphabet": vocabulary_.characters.to_i2w(),
        "target_alphabet": sorted(vocabulary_.target_characters),
    }
    with open(path, "w") as w:
        json.dump(metadata, w, indent=2, sort_keys=True)


def configure_output_file_logging(output_dir: str,
                                  filename: str = "training.info.log") -> str:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, filename)
    root_logger = logging.getLogger()
    abs_log_path = os.path.abspath(log_path)
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and \
                os.path.abspath(handler.baseFilename) == abs_log_path:
            return log_path
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    root_logger.addHandler(file_handler)
    return log_path


def best_non_optimal_cost_gaps(training_data: utils.Dataset) -> list[float]:
    gaps = []
    for sample in training_data.samples:
        if sample.expert_action_costs is None:
            continue
        costs = sample.expert_action_costs.detach().cpu()
        for state_costs in costs:
            finite_costs = state_costs[torch.isfinite(state_costs)]
            if finite_costs.numel() < 2:
                continue
            best_cost = torch.min(finite_costs)
            nonzero_gaps = finite_costs - best_cost
            nonzero_gaps = nonzero_gaps[nonzero_gaps > 0]
            if nonzero_gaps.numel() > 0:
                gaps.append(torch.min(nonzero_gaps).item())
    return gaps


def soft_expert_distribution_statistics(
        training_data: utils.Dataset,
        temperature: float) -> tuple[list[float], list[float], list[float]]:
    if temperature <= 0:
        return [], [], []
    optimal_masses = []
    entropies = []
    effective_sizes = []
    for sample in training_data.samples:
        if sample.expert_action_costs is None:
            continue
        costs = sample.expert_action_costs.detach().cpu()
        for state_costs in costs:
            finite_costs = state_costs[torch.isfinite(state_costs)]
            if finite_costs.numel() == 0:
                continue
            best_cost = torch.min(finite_costs)
            gaps = finite_costs - best_cost
            weights = torch.exp(-gaps / temperature)
            probs = weights / torch.sum(weights)
            optimal_mass = torch.sum(probs[gaps == 0]).item()
            positive_probs = probs[probs > 0]
            entropy = -torch.sum(positive_probs * torch.log(positive_probs)).item()
            optimal_masses.append(optimal_mass)
            entropies.append(entropy)
            effective_sizes.append(float(np.exp(entropy)))
    return optimal_masses, entropies, effective_sizes


def log_expert_gap_statistics(training_data: utils.Dataset,
                              temperature: float) -> None:
    gaps = best_non_optimal_cost_gaps(training_data)
    total_states = sum(
        0 if sample.expert_action_costs is None else sample.expert_action_costs.size(0)
        for sample in training_data.samples
    )
    logging.info("Expert cost-gap states: %d.", total_states)
    logging.info("Expert cost-gap states with a non-optimal finite action: %d.", len(gaps))
    if not gaps:
        return
    quantiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 100]
    values = np.percentile(gaps, quantiles)
    logging.info("Best non-optimal expert cost-gap quantiles:")
    for quantile, value in zip(quantiles, values):
        label = "min" if quantile == 0 else "max" if quantile == 100 else f"p{quantile:02d}"
        if temperature > 0:
            logging.info("\t%s: %.4f (weight at tau=%.4f: %.4f)",
                         label, value, temperature, np.exp(-value / temperature))
        else:
            logging.info("\t%s: %.4f", label, value)
    if temperature > 0:
        weights = np.exp(-np.array(gaps) / temperature)
        logging.info("Best non-optimal expert weight coverage at tau=%.4f:", temperature)
        for threshold in (0.01, 0.05, 0.10, 0.25):
            logging.info(
                "\tweight >= %.2f: %.2f%% of states",
                threshold,
                100 * np.mean(weights >= threshold),
            )
        optimal_masses, entropies, effective_sizes = \
            soft_expert_distribution_statistics(training_data, temperature)
        if optimal_masses:
            logging.info(
                "Normalized soft expert at tau=%.4f: optimal mass mean %.4f "
                "median %.4f; entropy mean %.4f median %.4f; effective actions "
                "mean %.4f median %.4f.",
                temperature,
                np.mean(optimal_masses),
                np.median(optimal_masses),
                np.mean(entropies),
                np.median(entropies),
                np.mean(effective_sizes),
                np.median(effective_sizes),
            )


def log_margin_statistics(stats: dict[str, float]) -> None:
    states = stats["states"]
    if states == 0:
        return
    active = stats["active"]
    margin_count = stats["margin_count"]
    logging.info("Margin active: %.4f.", active / states)
    if margin_count > 0:
        logging.info("Mean oracle/nonoracle logit margin: %.4f.",
                     stats["margin_sum"] / margin_count)
    if active > 0:
        logging.info("Mean active margin loss: %.4f.",
                     stats["active_loss_sum"] / active)


def log_focal_statistics(stats: dict[str, list[torch.Tensor]]) -> None:
    oracle_masses = stats["oracle_masses"]
    focal_weights = stats["focal_weights"]
    if not oracle_masses:
        return
    oracle_mass = torch.cat(oracle_masses).numpy()
    focal_weight = torch.cat(focal_weights).numpy()
    logging.info("Focal states: %d.", len(oracle_mass))
    logging.info(
        "Focal oracle mass: mean %.4f p10 %.4f p50 %.4f p90 %.4f.",
        np.mean(oracle_mass),
        np.percentile(oracle_mass, 10),
        np.percentile(oracle_mass, 50),
        np.percentile(oracle_mass, 90),
    )
    logging.info(
        "Focal weight: mean %.4f p90 %.4f.",
        np.mean(focal_weight),
        np.percentile(focal_weight, 90),
    )


def log_contrastive_statistics(stats: dict) -> None:
    states = stats["states"]
    if states == 0:
        return
    logging.info("Contrastive states: %d.", states)
    logging.info("Contrastive active: %.4f.", stats["active"] / states)
    logging.info("Contrastive accuracy: %.4f.",
                 stats["ranking_correct"] / states)
    logging.info("Contrastive margin satisfied: %.4f.",
                 stats["margin_satisfied"] / states)
    decoder_top_states = stats["decoder_top_states"]
    if decoder_top_states > 0:
        logging.info(
            "Decoder top action: optimal %.4f finite-expert %.4f excluded %.4f.",
            stats["decoder_top_optimal"] / decoder_top_states,
            stats["decoder_top_finite"] / decoder_top_states,
            stats["decoder_top_excluded"] / decoder_top_states,
        )
    logging.info("Mean contrastive loss: %.4f.",
                 stats["loss_sum"] / states)
    if stats["positive_count_sum"] > 0:
        logging.info("Contrastive positive set size: mean %.4f max %d.",
                     stats["positive_count_sum"] / states,
                     stats["positive_count_max"])
    hard_negative_total = (
        stats["negative_finite_expert"] + stats["negative_expert_excluded"])
    if hard_negative_total > 0:
        logging.info(
            "Contrastive hard negatives: finite expert cost %d (%.2f%%), "
            "no expert cost %d (%.2f%%).",
            stats["negative_finite_expert"],
            100 * stats["negative_finite_expert"] / hard_negative_total,
            stats["negative_expert_excluded"],
            100 * stats["negative_expert_excluded"] / hard_negative_total,
        )
    if stats["gap_count"] > 0:
        logging.info("Mean hard-negative expert gap: %.4f.",
                     stats["gap_sum"] / stats["gap_count"])
    negative_type_counts = stats["negative_type_counts"]
    if negative_type_counts:
        total = sum(negative_type_counts.values())
        logging.info("Contrastive hard-negative action types:")
        for action_type, count in sorted(negative_type_counts.items()):
            logging.info("\t%s: %d (%.2f%%)",
                         action_type, count, 100 * count / total)
    decoder_error_type_counts = stats["decoder_error_type_counts"]
    if decoder_error_type_counts:
        total = sum(decoder_error_type_counts.values())
        logging.info("Decoder top nonoptimal action types:")
        for action_type, count in sorted(decoder_error_type_counts.items()):
            logging.info("\t%s: %d (%.2f%%)",
                         action_type, count, 100 * count / total)
    if stats["delete_gap_count"] > 0:
        delete_count = stats["delete_gap_count"]
        logging.info(
            "DELETE hard negatives: count %d; mean expert gap %.4f; "
            "mean logit advantage %.4f; ranking correct %.4f; "
            "margin satisfied %.4f.",
            delete_count,
            stats["delete_gap_sum"] / delete_count,
            stats["delete_logit_advantage_sum"] / delete_count,
            stats["delete_ranking_correct"] / delete_count,
            stats["delete_margin_satisfied"] / delete_count,
        )


def focal_gamma_schedule(epoch: int, target_gamma: float,
                         start: int, ramp: int) -> float:
    if epoch < start:
        return 0.
    if ramp <= 0:
        return target_gamma
    progress = min(1., (epoch - start + 1) / ramp)
    return target_gamma * progress


def levenshtein_distance(source: list[str], target: list[str]) -> int:
    previous = list(range(len(target) + 1))
    for i, source_symbol in enumerate(source, start=1):
        current = [i]
        for j, target_symbol in enumerate(target, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (source_symbol != target_symbol),
            ))
        previous = current
    return previous[-1]


def output_symbols(text: str, tokenizer: utils.Tokenizer) -> list[str]:
    if text == "":
        return []
    return tokenizer.tokenize(text)


def load_transducer_for_device(vocabulary_: vocabulary.Vocabularies,
                               expert,
                               args: argparse.Namespace,
                               model_path: str,
                               device: str) -> transducer.Transducer:
    model_args = copy.copy(args)
    model_args.device = device
    model = transducer.Transducer(vocabulary_, expert, model_args)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def decode(transducer_: transducer.Transducer, data_loader: torch.utils.data.DataLoader,
           beam_width: int = 1) -> utils.DecodingOutput:
    if beam_width == 1:
        decoding = lambda b: \
            transducer_.transduce(
                b.input,
                b.encoded_input.to(transducer_.device),
                b.encoded_features.to(transducer_.device)
                if b.encoded_features is not None else None,
            )
    else:
        def decoding(b):
            final_output = transducer.Output([], [], 0)
            for s in range(len(b.input)):
                encoded_features = b.encoded_features[s].unsqueeze(dim=0).to(transducer_.device)\
                    if b.encoded_features is not None else None
                o = transducer_.beam_search_decode(b.input[s],
                                                   b.encoded_input[s].unsqueeze(dim=0).to(transducer_.device),
                                                   encoded_features,
                                                   beam_width)[0]
                final_output.action_history.append(o.action_history)
                final_output.output.append(o.output)
                final_output.log_p += o.log_p
            final_output.log_p /= len(b.input)

            return final_output

    predictions = []
    loss = 0
    correct = 0
    symbol_edits = 0
    reference_symbols = 0
    j = 0
    for batch in data_loader:
        output = decoding(batch)
        inputs, features, targets = \
            batch.input, batch.features, batch.target
        for i, p in enumerate(output.output):
            input_text = transducer_.source_tokenizer.untokenize(inputs[i])
            target_text = (
                transducer_.target_tokenizer.untokenize(targets[i])
                if targets[i] is not None else None
            )
            if any(features):
                prediction = f"{input_text}\t{p}\t{features[i]}"
            else:
                prediction = f"{input_text}\t{p}"
            predictions.append(prediction)
            if target_text is not None:
                if p == target_text:
                    correct += 1
                predicted_symbols = output_symbols(p, transducer_.target_tokenizer)
                target_symbols = output_symbols(target_text, transducer_.target_tokenizer)
                symbol_edits += levenshtein_distance(predicted_symbols, target_symbols)
                reference_symbols += len(target_symbols)
        loss += output.log_p
        if j > 0 and j % 100 == 0:
            logging.info("\t\t...%d batches", j)
        j += 1
    logging.info("\t\t...%d batches", j)

    return utils.DecodingOutput(
        string_accuracy=correct / len(data_loader.dataset),
        symbol_accuracy=1 - (symbol_edits / reference_symbols) if reference_symbols else 0.,
        loss=-loss / len(data_loader.dataset),
        predictions=predictions,
    )


def inverse_sigmoid_schedule(k: int):
    """Probability of sampling an action from the model as function of epoch."""
    return lambda epoch: (1 - k / (k + np.exp(epoch / k)))


@dataclasses.dataclass
class RollinStats:
    states: int = 0
    model_controlled: int = 0
    model_expert_agree: int = 0
    model_non_optimal: int = 0
    critic_states: int = 0
    critic_new_actions: int = 0
    critic_better_actions: int = 0
    critic_equal_actions: int = 0
    critic_regret_sum: float = 0.
    critic_finite_gaps: list[float] = dataclasses.field(default_factory=list)
    critic_infinite_gaps: int = 0
    critic_gaps_by_action_type: dict[str, list[float]] = dataclasses.field(
        default_factory=dict)
    critic_infinite_gaps_by_action_type: dict[str, int] = dataclasses.field(
        default_factory=dict)
    critic_old_costs_by_action_type: dict[str, list[float]] = dataclasses.field(
        default_factory=dict)
    critic_prefix_costs_by_action_type: dict[str, list[float]] = dataclasses.field(
        default_factory=dict)
    critic_continuation_costs_by_action_type: dict[str, list[float]] = dataclasses.field(
        default_factory=dict)
    critic_new_totals_by_action_type: dict[str, list[float]] = dataclasses.field(
        default_factory=dict)
    truncated: int = 0
    trajectory_lengths: list[int] = dataclasses.field(default_factory=list)

    def update(self, other: "RollinStats") -> None:
        self.states += other.states
        self.model_controlled += other.model_controlled
        self.model_expert_agree += other.model_expert_agree
        self.model_non_optimal += other.model_non_optimal
        self.critic_states += other.critic_states
        self.critic_new_actions += other.critic_new_actions
        self.critic_better_actions += other.critic_better_actions
        self.critic_equal_actions += other.critic_equal_actions
        self.critic_regret_sum += other.critic_regret_sum
        self.critic_finite_gaps.extend(other.critic_finite_gaps)
        self.critic_infinite_gaps += other.critic_infinite_gaps
        for action_type, gaps in other.critic_gaps_by_action_type.items():
            self.critic_gaps_by_action_type.setdefault(action_type, []).extend(gaps)
        for action_type, count in other.critic_infinite_gaps_by_action_type.items():
            self.critic_infinite_gaps_by_action_type[action_type] = \
                self.critic_infinite_gaps_by_action_type.get(action_type, 0) + count
        for action_type, costs in other.critic_old_costs_by_action_type.items():
            self.critic_old_costs_by_action_type.setdefault(action_type, []).extend(costs)
        for action_type, costs in other.critic_prefix_costs_by_action_type.items():
            self.critic_prefix_costs_by_action_type.setdefault(action_type, []).extend(costs)
        for action_type, costs in other.critic_continuation_costs_by_action_type.items():
            self.critic_continuation_costs_by_action_type.setdefault(action_type, []).extend(costs)
        for action_type, costs in other.critic_new_totals_by_action_type.items():
            self.critic_new_totals_by_action_type.setdefault(action_type, []).extend(costs)
        self.truncated += other.truncated
        self.trajectory_lengths.extend(other.trajectory_lengths)


def log_rollin_stats(epoch: int, probability: float, stats: RollinStats) -> None:
    agreement = (
        100 * stats.model_expert_agree / stats.model_controlled
        if stats.model_controlled else 0.
    )
    non_optimal = (
        100 * stats.model_non_optimal / stats.model_controlled
        if stats.model_controlled else 0.
    )
    mean_length = (
        float(np.mean(stats.trajectory_lengths))
        if stats.trajectory_lengths else 0.
    )
    logging.info("IL trajectory refresh at epoch %d", epoch)
    logging.info("\tmodel roll-in probability: %.3f", probability)
    logging.info("\tstates: %d", stats.states)
    logging.info("\tmodel-controlled: %d", stats.model_controlled)
    logging.info("\tmodel/expert agreement: %.1f%%", agreement)
    logging.info("\tnon-optimal model actions: %d (%.1f%%)", stats.model_non_optimal, non_optimal)
    logging.info("\ttruncated trajectories: %d", stats.truncated)
    logging.info("\tmean trajectory length: %.1f", mean_length)
    log_critic_stats(stats)


def log_critic_stats(stats: RollinStats) -> None:
    if stats.critic_states:
        novel_pct = 100 * stats.critic_new_actions / stats.critic_states
        already_pct = 100 - novel_pct
        better_pct = 100 * stats.critic_better_actions / stats.critic_states
        mean_regret = (
            stats.critic_regret_sum / stats.critic_new_actions
            if stats.critic_new_actions else 0.
        )
        logging.info("Model-action critic:")
        logging.info("\tstates scored: %d", stats.critic_states)
        logging.info("\talready in expert set: %.1f%%", already_pct)
        logging.info("\tnovel model actions: %d (%.1f%%)",
                     stats.critic_new_actions, novel_pct)
        logging.info("\tnovel actions better than reference best: %d (%.1f%%)",
                     stats.critic_better_actions, better_pct)
        logging.info("\tnovel actions equal to SED best: %d",
                     stats.critic_equal_actions)
        logging.info("\tmean novel cost gap: %.4f", mean_regret)
        log_critic_gap_distribution(stats)


def critic_action_type(action: object) -> str:
    name = action.__class__.__name__
    for prefix in ("Conditional",):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name == "EndOfSequence":
        return "EOS"
    return name.upper()


def record_critic_gap(stats: RollinStats, action_type: str, gap: float) -> None:
    if np.isfinite(gap):
        stats.critic_finite_gaps.append(gap)
        stats.critic_gaps_by_action_type.setdefault(action_type, []).append(gap)
    else:
        stats.critic_infinite_gaps += 1
        stats.critic_infinite_gaps_by_action_type[action_type] = \
            stats.critic_infinite_gaps_by_action_type.get(action_type, 0) + 1


def record_decoder_state_critic(
        stats: RollinStats,
        action_type: str,
        old_cost: float,
        prefix_cost: float,
        continuation_cost: float,
        new_total: float) -> None:
    if np.isfinite(old_cost):
        stats.critic_old_costs_by_action_type.setdefault(action_type, []).append(old_cost)
    if np.isfinite(prefix_cost):
        stats.critic_prefix_costs_by_action_type.setdefault(action_type, []).append(prefix_cost)
    if np.isfinite(continuation_cost):
        stats.critic_continuation_costs_by_action_type.setdefault(action_type, []).append(
            continuation_cost)
    if np.isfinite(new_total):
        stats.critic_new_totals_by_action_type.setdefault(action_type, []).append(new_total)


def log_critic_gap_distribution(stats: RollinStats) -> None:
    if not stats.critic_new_actions:
        return
    finite_gaps = np.array(stats.critic_finite_gaps, dtype=float)
    logging.info("Novel critic cost-gap distribution:")
    if finite_gaps.size:
        for label, value in zip(
                ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max"),
                np.percentile(finite_gaps, [0, 25, 50, 75, 90, 95, 99, 100])):
            logging.info("\t%s: %.4f", label, value)
        buckets = [
            ("gap == 0", finite_gaps == 0),
            ("0 < gap <= 1", (finite_gaps > 0) & (finite_gaps <= 1)),
            ("1 < gap <= 2", (finite_gaps > 1) & (finite_gaps <= 2)),
            ("2 < gap <= 5", (finite_gaps > 2) & (finite_gaps <= 5)),
            ("5 < gap <= 10", (finite_gaps > 5) & (finite_gaps <= 10)),
            ("gap > 10", finite_gaps > 10),
        ]
        for label, mask in buckets:
            logging.info("\t%s: %d", label, int(np.sum(mask)))
    else:
        logging.info("\tfinite gaps: 0")
    logging.info("\tinvalid/infinite: %d", stats.critic_infinite_gaps)
    logging.info("Novel critic cost-gap by action type:")
    action_types = sorted(
        set(stats.critic_gaps_by_action_type) |
        set(stats.critic_infinite_gaps_by_action_type))
    for action_type in action_types:
        gaps = np.array(
            stats.critic_gaps_by_action_type.get(action_type, []),
            dtype=float,
        )
        invalid = stats.critic_infinite_gaps_by_action_type.get(action_type, 0)
        if gaps.size:
            le_one = int(np.sum(gaps <= 1))
            logging.info(
                "\t%s count=%d median=%.4f <=1=%d invalid=%d",
                action_type,
                gaps.size + invalid,
                np.percentile(gaps, 50),
                le_one,
                invalid,
            )
        else:
            logging.info(
                "\t%s count=%d median=nan <=1=0 invalid=%d",
                action_type,
                invalid,
                invalid,
            )
    if stats.critic_new_totals_by_action_type:
        logging.info("Decoder-state critic costs by action type:")
        action_types = sorted(
            set(stats.critic_old_costs_by_action_type) |
            set(stats.critic_prefix_costs_by_action_type) |
            set(stats.critic_continuation_costs_by_action_type) |
            set(stats.critic_new_totals_by_action_type))
        for action_type in action_types:
            old_costs = stats.critic_old_costs_by_action_type.get(action_type, [])
            prefix_costs = stats.critic_prefix_costs_by_action_type.get(action_type, [])
            continuation_costs = \
                stats.critic_continuation_costs_by_action_type.get(action_type, [])
            new_totals = stats.critic_new_totals_by_action_type.get(action_type, [])
            logging.info(
                "\t%s count=%d old-median=%s prefix-median=%s "
                "continuation-median=%s new-total-median=%s",
                action_type,
                len(new_totals),
                format_optional_median(old_costs),
                format_optional_median(prefix_costs),
                format_optional_median(continuation_costs),
                format_optional_median(new_totals),
            )


def format_optional_median(values: list[float]) -> str:
    if not values:
        return "nan"
    return f"{np.median(values):.4f}"



def should_refresh_rollin(epoch: int, start: int, refresh: int,
                          probability: float) -> bool:
    if probability <= 0:
        return False
    if refresh < 1:
        raise ValueError("rollin_refresh must be at least 1.")
    return epoch >= start and (epoch - start) % refresh == 0


def model_greedy_rollin_action(
        s: utils.Sample,
        transducer_: transducer.Transducer,
        alignment_history: list[int],
        action_history: list[list[int]],
        output_history: list[int]) -> int:
    encoded_input = s.encoded_input.to(transducer_.device).unsqueeze(dim=0)
    encoded_features = (
        s.encoded_features.to(transducer_.device).unsqueeze(dim=0)
        if s.encoded_features is not None else None
    )
    transducer_.h0_c0 = 1
    with torch.no_grad():
        encoder_output = transducer_.encoder_step(encoded_input)
        feature_embedding = transducer_.feature_embedding(encoded_features)
        alignment = torch.tensor(alignment_history, device=transducer_.device)
        previous_actions = torch.tensor(
            [actions[0] for actions in action_history],
            device=transducer_.device,
            dtype=torch.long,
        ).unsqueeze(dim=1)
        previous_outputs = None
        if transducer_.output_lookup is not None:
            previous_outputs = torch.tensor(
                output_history,
                device=transducer_.device,
                dtype=torch.long,
            ).unsqueeze(dim=1)
        decoder_output, _ = transducer_.decoder_step(
            encoder_output,
            feature_embedding,
            transducer_.h0_c0,
            alignment,
            previous_actions,
            previous_outputs,
        )
        valid_actions_mask = transducer_.valid_actions_for_suffixes(
            torch.tensor(
                [len(s.input) + 1 - alignment_history[-1]],
                device=transducer_.device,
            )
        )
        logits = transducer_.W(decoder_output[-1:])
        log_probs = transducer_.log_softmax(logits, valid_actions_mask)
        return int(torch.argmax(log_probs[0, 0]).item())


def precompute_from_expert(s: utils.Sample, transducer_: transducer.Transducer,
                           device: str = 'cpu', rollin_prob: float = 0.,
                           rollin_policy: str = "expert",
                           rollin_rng: Optional[random.Random] = None,
                           critic_model_action: bool = False,
                           critic_augment_model_action: bool = False) -> RollinStats:
    """ Precompute the optimal policy (optimal and valid actions as well as the alignment) from the expert.

    Args:
        s: A data sample.
        transducer_: The transducer object holding the expert.
        device: Device on which tensors are allocated.

    Returns:
        None
    """
    if rollin_prob < 0 or rollin_prob > 1:
        raise ValueError("rollin_prob must satisfy 0 <= p <= 1.")
    if rollin_policy not in {"expert", "greedy"}:
        raise ValueError("rollin_policy must be one of: expert, greedy.")
    if rollin_rng is None:
        rollin_rng = random

    stats = RollinStats()
    alignment_history = [0]
    action_history = [[vocabulary.BEGIN_WORD]]
    optimal_action_history = []
    output_history = [transducer_.vocab.encode_output_symbol(vocabulary.BOS_OUTPUT)]
    expert_action_costs = []
    output = []
    a = 0
    stop = False

    # continue until end-of-sequence-token is found
    # (or max seq len is reached)
    max_rollout_actions = min(
        transducer.MAX_ACTION_SEQ_LEN,
        2 * (len(s.input) + len(s.target or [])) + 10,
    )
    while not stop and len(action_history) <= max_rollout_actions:
        stats.states += 1
        action_scores = transducer_.expert_action_scores(s.input, s.target, a, output)
        original_action_scores = dict(action_scores)
        for action, value in action_scores.items():
            transducer_.encode_known_action(
                action,
                "precompute_from_expert: "
                f"input={s.input!r}, target={s.target!r}, "
                f"alignment={a}, prediction={output!r}, score={value}",
            )
        optimal_value = min(action_scores.values())
        actions = [
            transducer_.encode_known_action(
                action,
                "precompute_from_expert: "
                f"input={s.input!r}, target={s.target!r}, "
                f"alignment={a}, prediction={output!r}, score={value}",
            )
            for action, value in action_scores.items()
            if value == optimal_value
        ]
        optimal_action_history.append(actions)
        if critic_augment_model_action:
            action_scores = {
                action: transducer_.expert_score_decoder_action(
                    s.input,
                    s.target,
                    a,
                    output,
                    transducer_.encode_known_action(
                        action,
                        "precompute_from_expert critic augmentation: "
                        f"input={s.input!r}, target={s.target!r}, "
                        f"alignment={a}, prediction={output!r}, action={action}",
                    ),
                ).total
                for action in original_action_scores
            }
        # todo: allow optimization of multiple target actions
        model_action = None
        follow_model = rollin_policy == "greedy" and rollin_rng.random() < rollin_prob
        if critic_model_action or follow_model:
            model_action = model_greedy_rollin_action(
                s,
                transducer_,
                alignment_history,
                action_history,
                output_history,
            )
        if critic_model_action and model_action is not None:
            stats.critic_states += 1
            critic_action = transducer_.expert_action_from_id(s.input, model_action, a)
            if critic_action not in original_action_scores:
                old_model_cost = transducer_.expert_score_action(
                    s.input,
                    s.target,
                    a,
                    output,
                    model_action,
                )
                decoder_state_score = transducer_.expert_score_decoder_action(
                    s.input,
                    s.target,
                    a,
                    output,
                    model_action,
                )
                model_cost = decoder_state_score.total
                reference_best_cost = min(action_scores.values())
                if critic_augment_model_action:
                    action_scores[critic_action] = model_cost
                stats.critic_new_actions += 1
                regret = model_cost - reference_best_cost
                stats.critic_regret_sum += regret
                action_type = critic_action_type(critic_action)
                record_critic_gap(
                    stats,
                    action_type,
                    regret,
                )
                record_decoder_state_critic(
                    stats,
                    action_type,
                    old_model_cost,
                    decoder_state_score.prefix_cost,
                    decoder_state_score.continuation_cost,
                    decoder_state_score.total,
                )
                if model_cost < reference_best_cost:
                    stats.critic_better_actions += 1
                elif model_cost == reference_best_cost:
                    stats.critic_equal_actions += 1
        expert_action_costs.append(
            transducer_.encode_expert_action_costs(action_scores, device=device))
        rollout_action = actions[0]
        if follow_model:
            stats.model_controlled += 1
            rollout_action = model_action
            if rollout_action in actions:
                stats.model_expert_agree += 1
            else:
                stats.model_non_optimal += 1
        action_history.append([rollout_action])

        output_history.append(
            transducer_.output_symbol_id_for_action(s.input, rollout_action, a))
        char_, a, stop = transducer_.decode_single_action(s.input, rollout_action, a)
        alignment_history.append(a)
        if char_ != "":
            output.append(char_)

    if not stop and len(action_history) > max_rollout_actions:
        stats.truncated += 1

    stats.trajectory_lengths.append(len(action_history) - 1)

    optimal_actions_mask = torch.full(
        (len(optimal_action_history), transducer_.number_actions),
        False, dtype=torch.bool, device=device)
    invalid_actions = [
        action
        for history in action_history[1:]
        for action in history
        if action < 0 or action >= transducer_.number_actions
    ]
    if invalid_actions:
        raise RuntimeError(
            f"Invalid action history for input={s.input!r}, target={s.target!r}: "
            f"invalid_actions={invalid_actions}, action_history={action_history}, "
            f"num_actions={transducer_.number_actions}."
        )
    seq_pos, emb_pos = zip(*[(s, a) for s, actions in enumerate(optimal_action_history)
                             for a in actions])
    optimal_actions_mask[seq_pos, emb_pos] = True
    s.optimal_actions_mask = optimal_actions_mask

    # now this is a crucial part: the last alignment index as well as the last action
    # are irrelevant and changing these lines will mess up training
    s.alignment_history = torch.tensor(alignment_history[:-1], device=device)
    s.action_history = torch.tensor(action_history[:-1], device=device).squeeze(dim=1)
    s.output_history = torch.tensor(output_history[:-1], device=device)
    s.expert_action_costs = torch.stack(expert_action_costs, dim=0)

    valid_actions_mask = torch.stack(
        # + 1 is needed to compensate for lack of end-of-seq-token
        # :-1 for same reason as above
        [
            transducer_.compute_valid_actions(
                len(s.input) + 1 - a,
                device=device,
            )
            for a in alignment_history[:-1]
        ],
        dim=0)
    s.valid_actions_mask = valid_actions_mask
    return stats


def refresh_precomputed_training(
        training_data: utils.Dataset,
        transducer_: transducer.Transducer,
        args: argparse.Namespace,
        epoch: Optional[int] = None,
        rollin_prob: float = 0.,
        rollin_policy: str = "expert",
        rollin_rng: Optional[random.Random] = None) -> RollinStats:
    stats = RollinStats()
    transducer_was_training = transducer_.training
    transducer_.eval()
    with torch.no_grad():
        for s in training_data.samples:
            stats.update(precompute_from_expert(
                s,
                transducer_,
                device="cpu",
                rollin_prob=rollin_prob,
                rollin_policy=rollin_policy,
                rollin_rng=rollin_rng,
                critic_model_action=args.critic_model_action,
                critic_augment_model_action=args.critic_augment_model_action,
            ))
    if transducer_was_training:
        transducer_.train()
    if epoch is not None and (rollin_prob > 0 or args.critic_model_action):
        log_rollin_stats(epoch, rollin_prob, stats)
    return stats


def main(args: argparse.Namespace):
    args.source_separator = utils.Tokenizer.from_cli(args.source_separator).separator
    args.target_separator = utils.Tokenizer.from_cli(args.target_separator).separator

    log_path = configure_output_file_logging(args.output)
    logging.info("Writing detailed training log to %s.", log_path)

    for key, value in vars(args).items():
        logging.info("%s: %s", str(key).ljust(15), value)

    if args.pytorch_seed is not None:
        torch.manual_seed(args.pytorch_seed)

        train_generator = torch.Generator()
        train_generator.manual_seed(args.pytorch_seed)

        def train_worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
    else:
        train_generator, train_worker_init_fn = None, None
    rollin_seed = args.rollin_seed if args.rollin_seed is not None else args.pytorch_seed
    rollin_rng = random.Random(rollin_seed if rollin_seed is not None else 1)

    if args.nfd:
        logging.info("Will perform training on NFD-normalized data.")
    else:
        logging.info("Will perform training on unnormalized data.")

    has_features = (args.feat_dim is not None)
    source_tokenizer = utils.Tokenizer.from_cli(args.source_separator)
    target_tokenizer = utils.Tokenizer.from_cli(args.target_separator)
    args.source_separator = source_tokenizer.separator
    args.target_separator = target_tokenizer.separator
    if has_features:
        vocabulary_class = vocabulary.FeatureVocabularies
    else:
        vocabulary_class = vocabulary.Vocabularies

    if args.vocabulary is not None:
        vocabulary_ = vocabulary_class.from_pickle(args.vocabulary)
        args.source_separator = vocabulary_.source_separator
        args.target_separator = vocabulary_.target_separator
        source_tokenizer = utils.Tokenizer(args.source_separator)
        target_tokenizer = utils.Tokenizer(args.target_separator)
        logging.info("%d actions: %s", len(vocabulary_.actions),
                     vocabulary_.actions)
        logging.info("%d chars: %s", len(vocabulary_.characters),
                     vocabulary_.characters)
        if has_features:
            logging.info("%d features: %s", len(vocabulary_.features),
                         vocabulary_.features)
    else:
        vocabulary_ = vocabulary_class(
            source_separator=source_tokenizer.separator,
            target_separator=target_tokenizer.separator,
        )

    if args.precomputed_train is not None:
        training_data = utils.Dataset.from_pickle(args.precomputed_train, device="cpu")
        if args.output_feedback_dim > 0 and any(
                sample.output_history is None for sample in training_data.samples):
            raise ValueError(
                "Output feedback requires precomputed output histories. "
                "Regenerate precomputed training data.")
        if args.expert_temperature > 0 and any(
                sample.expert_action_costs is None for sample in training_data.samples):
            raise ValueError(
                "Soft oracle loss requires precomputed expert action costs. "
                "Regenerate precomputed training data.")
    else:
        training_data = utils.Dataset()

        with utils.OpenNormalize(args.train, args.nfd) as f:
            for line in f:
                if has_features:
                    input_text, target_text, features = line.rstrip().split("\t", 2)
                    encoded_features = torch.tensor(vocabulary_.encode_features(features))
                else:
                    input_text, target_text = line.rstrip().split("\t", 1)
                    features = encoded_features = None

                input_ = source_tokenizer.tokenize(input_text)
                target = target_tokenizer.tokenize(target_text)
                encoded_input = torch.tensor(vocabulary_.encode_input(input_))
                vocabulary_.encode_actions(target)
                sample = utils.Sample(
                    input_, target, encoded_input,
                    features=features,
                    encoded_features=encoded_features,
                )
                training_data.add_samples(sample)

        logging.info("%d actions: %s", len(vocabulary_.actions),
                     vocabulary_.actions)
        logging.info("%d chars: %s", len(vocabulary_.characters),
                     vocabulary_.characters)
        if has_features:
            logging.info("%d features: %s", len(vocabulary_.features),
                         vocabulary_.features)
        vocabulary_path = os.path.join(args.output, "vocabulary.pkl")
        vocabulary_.persist(vocabulary_path)
        logging.info("Wrote vocabulary to %s.", vocabulary_path)

    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    development_data = utils.Dataset()
    with utils.OpenNormalize(args.dev, args.nfd) as f:
        for line in f:
            if has_features:
                input_text, target_text, features = line.rstrip().split("\t", 2)
                encoded_features = torch.tensor(vocabulary_.encode_unseen_features(features))
            else:
                input_text, target_text = line.rstrip().split("\t", 1)
                features = encoded_features = None

            input_ = source_tokenizer.tokenize(input_text)
            target = target_tokenizer.tokenize(target_text)
            encoded_input = torch.tensor(vocabulary_.encode_unseen_input(input_))
            sample = utils.Sample(
                input_, target, encoded_input,
                features=features,
                encoded_features=encoded_features,
            )
            development_data.add_samples(sample)
    development_data_loader = development_data.get_data_loader(batch_size=eval_batch_size,
                                                               device=args.device)

    if args.test is not None:
        test_data = utils.Dataset()
        with utils.OpenNormalize(args.test, args.nfd) as f:
            for line in f:
                if has_features:
                    input_text, optional_target, features = line.rstrip().split(
                        "\t", 2)
                    encoded_features = torch.tensor(vocabulary_.encode_unseen_features(features))
                    target = target_tokenizer.tokenize(optional_target) if optional_target else None
                else:
                    input_text, *optional_target = line.rstrip().split("\t", 1)
                    features = encoded_features = None
                    target = target_tokenizer.tokenize(optional_target[0]) if optional_target else None

                input_ = source_tokenizer.tokenize(input_text)
                encoded_input = torch.tensor(vocabulary_.encode_unseen_input(input_))
                sample = utils.Sample(
                    input_, target, encoded_input,
                    features=features,
                    encoded_features=encoded_features,
                )
                test_data.add_samples(sample)
        test_data_loader = test_data.get_data_loader(batch_size=eval_batch_size,
                                                     device=args.device)

    if args.sed_params is not None:
        sed_aligner = sed.StochasticEditDistance.from_pickle(
            args.sed_params)
    else:
        sed_parameters_path = os.path.join(args.output, "sed.pkl")
        sed_aligner = sed.StochasticEditDistance.fit_from_data(
            training_data.samples, em_iterations=args.sed_em_iterations,
            output_path=sed_parameters_path, em_mode=args.sed_em_mode,
            em_damping=args.sed_em_damping)
        sed_metadata_path = f"{sed_parameters_path}.json"
        write_sed_metadata(
            sed_metadata_path,
            args,
            training_data,
            vocabulary_,
        )
        logging.info("Wrote SED metadata to %s.", sed_metadata_path)
    expert = optimal_expert_substitutions.OptimalSubstitutionExpert(sed_aligner)

    transducer_ = transducer.Transducer(vocabulary_, expert, args)

    widgets = [progressbar.Bar(">"), " ", progressbar.ETA()]

    # precompute from expert
    if not args.precomputed_train:
        logging.info("Precomputing optimal actions for training samples.")
        precompute_progress_bar = progressbar.ProgressBar(
            widgets=widgets, maxval=len(training_data.samples)
        ).start()
        precompute_stats = RollinStats()
        for i, s in enumerate(training_data.samples):
            precompute_stats.update(precompute_from_expert(
                s,
                transducer_,
                device="cpu",
                critic_model_action=args.critic_model_action,
                critic_augment_model_action=args.critic_augment_model_action,
            ))
            precompute_progress_bar.update(i)
        if args.critic_model_action:
            log_critic_stats(precompute_stats)

        if args.save_precomputed_train:
            precomputed_train_path = os.path.join(args.output, "precomputed_train.pkl")
            training_data.persist(precomputed_train_path)

    if args.expert_loss == "margin":
        logging.info(
            "Using fixed-margin expert loss with expert_margin=%.4f; "
            "expert_temperature is unused.",
            args.expert_margin,
        )
    elif args.expert_loss == "contrastive":
        logging.info(
            "Using contrastive expert loss with expert_margin=%.4f, "
            "contrastive_negative=%s; expert_temperature is unused.",
            args.expert_margin,
            args.contrastive_negative,
        )
    else:
        log_expert_gap_statistics(training_data, args.expert_temperature)

    def build_training_data_loader():
        return training_data.get_data_loader(
            is_training=True,
            batch_size=args.batch_size,
            device=args.device,
            shuffle=True,
            generator=train_generator,
            worker_init_fn=train_worker_init_fn,
        )

    def build_train_subset_loader():
        subset_size = int(len(training_data.samples) * args.train_subset_eval_size / 100)
        subset_size = max(1, subset_size)
        return utils.Dataset(random.sample(training_data.samples, subset_size)) \
            .get_data_loader(batch_size=eval_batch_size, device=args.device)

    training_data_loader = build_training_data_loader()

    train_progress_bar = progressbar.ProgressBar(
        widgets=widgets, maxval=args.epochs).start()

    train_log_path = os.path.join(args.output, "train.log")
    best_model_path = os.path.join(args.output, "best.model")
    best_model_metadata_path = os.path.join(args.output, "best.model.json")

    with open(train_log_path, "w") as w:
        w.write("epoch\tavg_loss\ttrain_string_accuracy\tdev_string_accuracy\tdev_symbol_accuracy\n")

    optimizer, scheduler = build_optimizer_and_scheduler(transducer_, args)
    train_subset_loader = build_train_subset_loader()
    # rollin_schedule = inverse_sigmoid_schedule(args.k)
    max_patience = args.patience

    if args.loss_reduction == "sum":
        reduce_loss = torch.sum
    else:
        reduce_loss = torch.mean

    logging.info("Training for a maximum of %d with a maximum patience of %d.",
                 args.epochs, max_patience)
    logging.info("Number of train batches: %d.", len(training_data_loader))

    best_train_accuracy = 0
    best_dev_accuracy = -float("inf")
    best_dev_symbol_accuracy = -float("inf")
    best_selection_key = (-float("inf"), -float("inf"))
    best_epoch = 0
    patience = 0

    for epoch in range(args.epochs):
        if args.expert_loss == "focal_marginal":
            transducer_.focal_gamma = focal_gamma_schedule(
                epoch,
                args.focal_gamma,
                args.focal_start,
                args.focal_ramp,
            )
            logging.info("Focal marginal gamma: %.4f.", transducer_.focal_gamma)
        if should_refresh_rollin(
                epoch,
                args.rollin_start,
                args.rollin_refresh,
                args.rollin_prob):
            refresh_precomputed_training(
                training_data,
                transducer_,
                args,
                epoch=epoch,
                rollin_prob=args.rollin_prob,
                rollin_policy=args.rollin_policy,
                rollin_rng=rollin_rng,
            )
            training_data_loader = build_training_data_loader()
            train_subset_loader = build_train_subset_loader()

        logging.info("Training...")
        transducer_.train()
        optimizer.zero_grad(set_to_none=True)
        with utils.Timer():
            train_loss = 0.
            margin_statistics = {
                "states": 0,
                "active": 0,
                "margin_sum": 0.,
                "margin_count": 0,
                "active_loss_sum": 0.,
            }
            focal_statistics = {
                "oracle_masses": [],
                "focal_weights": [],
            }
            contrastive_statistics = {
                "states": 0,
                "active": 0,
                "ranking_correct": 0,
                "margin_satisfied": 0,
                "decoder_top_states": 0,
                "decoder_top_optimal": 0,
                "decoder_top_finite": 0,
                "decoder_top_excluded": 0,
                "loss_sum": 0.,
                "gap_sum": 0.,
                "gap_count": 0,
                "negative_finite_expert": 0,
                "negative_expert_excluded": 0,
                "positive_count_sum": 0,
                "positive_count_max": 0,
                "negative_type_counts": collections.Counter(),
                "decoder_error_type_counts": collections.Counter(),
                "delete_gap_sum": 0.,
                "delete_gap_count": 0,
                "delete_logit_advantage_sum": 0.,
                "delete_ranking_correct": 0,
                "delete_margin_satisfied": 0,
            }
            # rollin not implemented at the moment
            # rollin = rollin_schedule(epoch)
            j = 0
            batch_count = len(training_data_loader)
            for j, batch in enumerate(training_data_loader):
                losses = transducer_.training_step(encoded_input=batch.encoded_input,
                                                   encoded_features=batch.encoded_features,
                                                   action_history=batch.action_history,
                                                   output_history=batch.output_history,
                                                   alignment_history=batch.alignment_history,
                                                   expert_action_costs=batch.expert_action_costs,
                                                   optimal_actions_mask=batch.optimal_actions_mask,
                                                   valid_actions_mask=batch.valid_actions_mask)
                if transducer_.last_margin_statistics is not None:
                    for key, value in transducer_.last_margin_statistics.items():
                        margin_statistics[key] += value
                if transducer_.last_focal_statistics is not None:
                    focal_statistics["oracle_masses"].append(
                        transducer_.last_focal_statistics["oracle_masses"])
                    focal_statistics["focal_weights"].append(
                        transducer_.last_focal_statistics["focal_weights"])
                if transducer_.last_contrastive_statistics is not None:
                    for key, value in transducer_.last_contrastive_statistics.items():
                        if key in {"negative_type_counts", "decoder_error_type_counts"}:
                            contrastive_statistics[key].update(value)
                        elif key == "positive_count_max":
                            contrastive_statistics[key] = max(
                                contrastive_statistics[key], value)
                        else:
                            contrastive_statistics[key] += value
                train_loss += torch.mean(losses.squeeze(dim=0)).item()  # mean per batch
                scale = accumulation_loss_scale(j, batch_count, args.grad_accumulation)
                reduced_loss = reduce_loss(losses) / scale
                reduced_loss.backward()
                if should_step(j, batch_count, args.grad_accumulation):
                    optimizer.step()
                    if scheduler is not None and scheduler.type == 'step':
                        lrs_before = optimizer_learning_rates(optimizer)
                        scheduler.step()
                        log_learning_rate_change(lrs_before, optimizer, args.scheduler)
                    optimizer.zero_grad(set_to_none=True)
                if j > 0 and j % 100 == 0:
                    logging.info("\t\t...%d batches", j)
            logging.info("\t\t...%d batches", j + 1)

        # avg. loss per sample
        avg_loss = train_loss / len(training_data_loader)
        logging.info("Average train loss: %.4f.", avg_loss)
        if args.expert_loss == "margin":
            log_margin_statistics(margin_statistics)
        if args.expert_loss == "focal_marginal":
            log_focal_statistics(focal_statistics)
        if args.expert_loss == "contrastive":
            log_contrastive_statistics(contrastive_statistics)

        transducer_.eval()
        with torch.no_grad():
            logging.info("Evaluating on training data subset...")
            with utils.Timer():
                train_accuracy = decode(transducer_, train_subset_loader).string_accuracy

            if train_accuracy > best_train_accuracy:
                best_train_accuracy = train_accuracy

            patience += 1

            logging.info("Evaluating on development data...")
            with utils.Timer():
                decoding_output = decode(transducer_, development_data_loader)
                dev_accuracy = decoding_output.string_accuracy
                dev_symbol_accuracy = decoding_output.symbol_accuracy
                avg_dev_loss = decoding_output.loss

        if scheduler is not None and scheduler.type == 'metric':
            lrs_before = optimizer_learning_rates(optimizer)
            scheduler.step(dev_accuracy)
            log_learning_rate_change(lrs_before, optimizer, args.scheduler)
            lrs_after = optimizer_learning_rates(optimizer)
            if args.reload_best_on_lr_reduction and \
                    learning_rates_changed(lrs_before, lrs_after):
                optimizer, scheduler = reload_best_model_and_reset_optimizer(
                    transducer_,
                    args,
                    best_model_path,
                    lrs_after,
                )

        selection_key = model_selection_key(dev_accuracy, dev_symbol_accuracy)
        if selection_key > best_selection_key:
            best_selection_key = selection_key
            best_dev_accuracy = dev_accuracy
            best_dev_symbol_accuracy = dev_symbol_accuracy
            best_epoch = epoch
            patience = 0
            logging.info(
                "Found best dev selection: string accuracy %.4f, symbol accuracy %.4f.",
                best_dev_accuracy,
                best_dev_symbol_accuracy,
            )
            torch.save(transducer_.state_dict(), best_model_path)
            write_checkpoint_metadata(
                best_model_metadata_path,
                args,
                epoch,
                dev_accuracy,
                dev_symbol_accuracy,
                train_accuracy,
            )
            logging.info("Saved new best model to %s.", best_model_path)

        logging.info(
            f"Epoch {epoch} / {args.epochs - 1}: train loss: {avg_loss:.4f} "
            f"dev loss: {avg_dev_loss:.4f} train string acc: {train_accuracy:.4f} "
            f"dev string acc: {dev_accuracy:.4f} dev symbol acc: {dev_symbol_accuracy:.4f} "
            f"best train string acc: {best_train_accuracy:.4f} "
            f"best dev string acc: {best_dev_accuracy:.4f} "
            f"best dev symbol acc: {best_dev_symbol_accuracy:.4f} "
            f"best epoch: {best_epoch} "
            f"patience: {patience} / {max_patience}"
        )

        log_line = (
            f"{epoch}\t{avg_loss:.4f}\t{train_accuracy:.4f}\t"
            f"{dev_accuracy:.4f}\t{dev_symbol_accuracy:.4f}\n"
        )
        with open(train_log_path, "a") as a:
            a.write(log_line)

        if should_stop_for_patience(patience, max_patience):
            logging.info("Out of patience after %d epochs.", epoch + 1)
            train_progress_bar.finish()
            break

        train_progress_bar.update(epoch)

    logging.info("Finished training.")

    if args.epochs < 1:
        raise ValueError("At least one epoch is required to produce a model checkpoint.")
    if not os.path.exists(best_model_path):
        raise RuntimeError(f"No model checkpoint was written to {best_model_path}.")

    transducer_ = load_transducer_for_device(
        vocabulary_,
        expert,
        args,
        best_model_path,
        args.device,
    )
    beam_transducer = transducer_
    if args.beam_width > 0 and torch.device(args.device).type != "cpu":
        logging.info(
            "Using CPU copy of best model for beam search decoding "
            "because beam search is slow on accelerator backends."
        )
        beam_transducer = load_transducer_for_device(
            vocabulary_,
            expert,
            args,
            best_model_path,
            "cpu",
        )

    transducer_.eval()
    with torch.no_grad():
        evaluations = [(development_data_loader, "dev")]
        if args.test is not None:
            evaluations.append((test_data_loader, "test"))
        for data, dataset_name in evaluations:
            if args.beam_width > 0:
                logging.info("Evaluating best model on %s data using beam search "
                             "(beam width %d)...", dataset_name, args.beam_width)
                with utils.Timer():
                    beam_decoding = decode(beam_transducer, data, args.beam_width)
                utils.write_results(beam_decoding.string_accuracy,
                                    beam_decoding.predictions, args.output,
                                    args.nfd, dataset_name, args.beam_width,
                                    symbol_accuracy=beam_decoding.symbol_accuracy,
                                    dargs=vars(args))
            logging.info("Evaluating best model on %s data using greedy decoding"
                         , dataset_name)
            with utils.Timer():
                greedy_decoding = decode(transducer_, data)
            utils.write_results(greedy_decoding.string_accuracy,
                                greedy_decoding.predictions, args.output,
                                args.nfd, dataset_name,
                                symbol_accuracy=greedy_decoding.symbol_accuracy,
                                dargs=vars(args))


def cli_main():
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train a g2p neural transducer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--pytorch-seed", type=int,
                        help="Random seed used by PyTorch.")
    parser.add_argument("--train", type=str,
                        help="Path to train set data. Only required if --precomputed-train and --vocabulary is not"
                             " provided.")
    parser.add_argument("--precomputed-train", type=str,
                        help="Path to precomputed train set data. "
                             "If provided, --vocabulary option must be provided, as well.")
    parser.add_argument("--save-precomputed-train", action="store_true", default=False,
                        help="Store the precomputed training set (i.e., containing the expert's information needed"
                             " for training). Can be used to speed up the training process for large datasets.")
    parser.add_argument("--vocabulary", type=str,
                        help="Path to the vocabulary. "
                             "If provided, --precomputed-train must be provided, as well.")
    parser.add_argument("--dev", type=str, required=True,
                        help="Path to development set data.")
    parser.add_argument("--test", type=str,
                        help="Path to development set data.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--nfd", action="store_true", default=False,
                        help="Train on NFD-normalized data. Write out in NFC.")
    parser.add_argument("--source-separator", default=utils.Tokenizer.NONE_VALUE,
                        help="Literal source-token separator. Use 'none' for character tokens.")
    parser.add_argument("--target-separator", default=utils.Tokenizer.NONE_VALUE,
                        help="Literal target-token separator. Use 'none' for character tokens.")
    parser.add_argument("--char-dim", type=int, default=100,
                        help="Character peak_embedding dimension.")
    parser.add_argument("--feat-dim", type=int, default=None,
                        help="Feature embedding dimension, if any."
                             " The data is assumed to be in UniMorph format.")
    parser.add_argument("--action-dim", type=int, default=100,
                        help="Action peak_embedding dimension.")
    parser.add_argument("--output-feedback-dim", type=int, default=0,
                        help="Previous emitted-output embedding dimension. A value of 0 disables output feedback.")
    parser.add_argument("--expert-temperature", type=float, default=0.,
                        help="Soft oracle temperature. A value of 0 uses the hard set-valued oracle loss.")
    parser.add_argument("--expert-loss",
                        choices=[
                            "marginal", "focal_marginal", "normalized_ce",
                            "margin", "contrastive",
                        ],
                        default="marginal",
                        help="Expert training objective. marginal uses hard set-valued oracle mass; "
                             "focal_marginal focalizes hard oracle mass; normalized_ce matches the "
                             "normalized soft expert distribution; margin uses fixed-margin logit ranking; "
                             "contrastive ranks the expert-optimal set above non-optimal expert actions.")
    parser.add_argument("--focal-gamma", type=float, default=1.0,
                        help="Focusing parameter for --expert-loss=focal_marginal. gamma=0 recovers ordinary marginal training.")
    parser.add_argument("--focal-start", type=int, default=0,
                        help="First epoch using focal marginal loss. Earlier epochs use gamma=0.")
    parser.add_argument("--focal-ramp", type=int, default=0,
                        help="Number of epochs over which focal gamma is linearly ramped to --focal-gamma. A value <= 0 uses the target gamma immediately at --focal-start.")
    parser.add_argument("--expert-margin", type=float, default=1.0,
                        help="Logit margin for --expert-loss=margin or contrastive.")
    parser.add_argument("--contrastive-negative",
                        choices=["hard", "all"],
                        default="hard",
                        help="Negative set for --expert-loss=contrastive. hard uses the highest-logit non-optimal action; all aggregates all finite non-optimal actions.")
    parser.add_argument("--critic-model-action", action="store_true",
                        help="Score and report the model's greedy action when it is not already scored by the expert. "
                             "This is diagnostic-only unless --critic-augment-model-action is also set.")
    parser.add_argument("--critic-augment-model-action", action="store_true",
                        help="Also add the scored greedy model action to the SED expert cost distribution. "
                             "Requires --critic-model-action.")
    parser.add_argument("--rollin-prob", type=float, default=0.0,
                        help="Probability of following the current model during periodic imitation-learning "
                             "trajectory refresh. A value of 0 disables roll-in refreshes.")
    parser.add_argument("--rollin-start", type=int, default=5,
                        help="First epoch at which to refresh trajectories with model roll-in.")
    parser.add_argument("--rollin-refresh", type=int, default=5,
                        help="Number of epochs between model roll-in trajectory refreshes.")
    parser.add_argument("--rollin-policy", choices=["greedy"], default="greedy",
                        help="Policy used when a trajectory refresh follows the current model.")
    parser.add_argument("--rollin-seed", type=int,
                        help="Random seed for model/expert roll-in choices. Defaults to --pytorch-seed, or 1.")
    parser.add_argument("--enc-type", type=str, default='lstm',
                        choices=ENCODER_MAPPING.keys(),
                        help="Type of used encoder.")
    parser.add_argument("--dec-hidden-dim", type=int, default=200,
                        help="Decoder LSTM state dimension.")
    parser.add_argument("--dec-layers", type=int, default=1,
                        help="Number of decoder LSTM layers.")
    parser.add_argument("--beam-width", type=int, default=4,
                        help="Beam width for beam search decoding. A value < 1 will disable beam search decoding.")
    # parser.add_argument("--k", type=int, default=1,
    #                     help="k for inverse sigmoid rollin schedule.")
    parser.add_argument("--patience", type=int, default=12,
                        help="Maximal patience for early stopping.")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Maximal number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Batch size for training.")
    parser.add_argument("--eval-batch-size", type=int,
                        help="Batch size for evaluation. Will be set to training batch size (--batch-size) if not"
                             " specified.")
    parser.add_argument("--loss-reduction", type=str, default="mean", choices=["sum", "mean"],
                        help="How the loss is reduced during training.")
    parser.add_argument("--grad-accumulation", type=int, default=1,
                        help="Gradient accumulation.")
    parser.add_argument("--train-subset-eval-size", type=int, default=5,
                        help="Percentage of training data used to evaluate training accuracy every epoch ("
                             "randomly sampled).")
    parser.add_argument("--optimizer", type=str, default="adadelta",
                        choices=OPTIMIZER_MAPPING.keys(),
                        help="Optimizer used in training.")
    parser.add_argument("--scheduler", type=str,
                        choices=LR_SCHEDULER_MAPPING.keys(),
                        help="Scheduler used in training.")
    parser.add_argument("--reload-best-on-lr-reduction",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="When reduce_on_plateau lowers the learning rate, reload best.model and reset optimizer state at the reduced learning rate.")
    parser.add_argument("--sed-em-iterations", type=int, default=10,
                        help="SED EM iterations.")
    # Project default: keep damped EM for existing training behavior. This is a
    # stabilized variant, not the paper-pure Ristad-Yianilos estimator; use
    # --sed-em-mode strict for the paper-faithful update.
    parser.add_argument("--sed-em-mode", choices=["strict", "damped"],
                        default="damped",
                        help="SED EM estimator. strict is paper-faithful; damped interpolates with previous parameters.")
    parser.add_argument("--sed-em-damping", type=float, default=0.9,
                        help="Weight of the strict EM estimate when --sed-em-mode=damped. Must satisfy 0 < x <= 1.")
    parser.add_argument("--sed-params", type=str,
                        help="Path to learned SED parameters.")
    parser.add_argument("--device", type=str, default='cpu',
                        help="Device to run training on.")

    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--enc-type", type=str, default='lstm',
                                    choices=ENCODER_MAPPING.keys())
    preliminary_parser.add_argument("--optimizer", type=str, default="adadelta",
                                    choices=OPTIMIZER_MAPPING.keys())
    preliminary_parser.add_argument("--scheduler", type=str,
                                    choices=LR_SCHEDULER_MAPPING.keys())
    preliminary_args, _ = preliminary_parser.parse_known_args()

    # encoder-specific configs
    encoder_group = parser.add_argument_group("Encoder specific configuration")
    ENCODER_MAPPING[preliminary_args.enc_type].add_args(encoder_group)

    # optimizer-specific configs
    optimizer_group = parser.add_argument_group("Optimizer specific configuration")
    OPTIMIZER_MAPPING[preliminary_args.optimizer].add_args(optimizer_group)

    # scheduler-specific configs
    if preliminary_args.scheduler is not None:
        scheduler_group = parser.add_argument_group("LR scheduler specific configuration")
        LR_SCHEDULER_MAPPING[preliminary_args.scheduler].add_args(scheduler_group)

    args = parser.parse_args()
    if args.rollin_prob < 0 or args.rollin_prob > 1:
        parser.error("--rollin-prob must satisfy 0 <= p <= 1.")
    if args.rollin_start < 0:
        parser.error("--rollin-start must be nonnegative.")
    if args.rollin_refresh < 1:
        parser.error("--rollin-refresh must be at least 1.")
    if args.focal_gamma < 0:
        parser.error("--focal-gamma must be nonnegative.")
    if args.focal_start < 0:
        parser.error("--focal-start must be nonnegative.")
    if args.expert_loss == "focal_marginal" and args.expert_temperature != 0:
        parser.error("--expert-loss=focal_marginal requires --expert-temperature=0.")
    if args.expert_loss == "contrastive" and args.expert_temperature != 0:
        parser.error("--expert-loss=contrastive requires --expert-temperature=0.")
    if args.critic_augment_model_action and not args.critic_model_action:
        parser.error("--critic-augment-model-action requires --critic-model-action.")
    if args.reload_best_on_lr_reduction and args.scheduler != "reduce_on_plateau":
        parser.error("--reload-best-on-lr-reduction requires --scheduler=reduce_on_plateau.")

    # custom logic for handling mutually inclusive/exclusive set of options
    # --> train, precomputed_train and vocabulary
    # train is required
    if args.train is None and \
            (args.precomputed_train is None and args.vocabulary is None):
        parser.error("--train is required if --precomputed-train and --vocabulary is not provided.")
    # precomputed_train and vocabulary is required
    elif args.train is None and \
            (args.precomputed_train is None or args.vocabulary is None):
        parser.error("--precomputed_train and --vocabulary must both be specified, if one of them is provided "
                     "(mutually inclusive).")
    # precomputed_train and vocabulary not allowed
    elif args.train is not None and \
            args.precomputed_train is not None and args.vocabulary is not None:
        parser.error("If --train is specified, --precomputed-train and --vocabulary should not be provided.")

    main(args)


if __name__ == "__main__":
    cli_main()
