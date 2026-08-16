"""Defines a neural transducer."""
from typing import Any, Dict, List, Optional, Tuple, Union
import dataclasses
import functools
import heapq
import argparse
import logging
import collections

import torch
import numpy as np

from trans import optimal_expert
from trans import utils
from trans import vocabulary
from trans.actions import ConditionalCopy, ConditionalDel, ConditionalIns, \
    ConditionalSub, Copy, Del, Edit, EndOfSequence, GenerativeEdit, \
    BeginOfSequence, Ins, Sub
from trans.vocabulary import BEGIN_WORD, COPY, DELETE, END_WORD, PAD, \
    FeatureVocabularies
from trans import ENCODER_MAPPING


MAX_ACTION_SEQ_LEN = 150


@functools.total_ordering
@dataclasses.dataclass
class Output:
    """Decoded output.

    For greedy decoding, ``log_p`` is normalized over generated actions,
    including END_WORD and excluding BEGIN_WORD.
    """
    action_history: List[Any]
    output: Union[str, List[str]]
    log_p: float
    losses: Optional[torch.tensor] = None

    def __lt__(self, other):
        return self.log_p < other.log_p

    def __eq__(self, other):
        return self.log_p == other.log_p


@dataclasses.dataclass
class Hypothesis:
    action_history: torch.tensor
    alignment: torch.tensor
    decoder: Tuple[torch.tensor, torch.tensor]
    negative_log_p: torch.tensor
    output: List[str]
    previous_output: Optional[torch.tensor] = None


@dataclasses.dataclass(frozen=True)
class ActionEffect:
    output_symbol: Optional[Any]
    alignment_delta: int
    stop: bool


@functools.total_ordering
@dataclasses.dataclass
class Expansion:
    action: Any
    decoder: Tuple[torch.tensor, torch.tensor]
    from_hypothesis: Hypothesis
    negative_log_p: float

    def __lt__(self, other):
        return self.negative_log_p < other.negative_log_p

    def __eq__(self, other):
        return self.negative_log_p == other.negative_log_p


class Transducer(torch.nn.Module):
    def __init__(self, vocab: vocabulary.Vocabularies,
                 expert: optimal_expert.Expert, args: argparse.Namespace):

        super().__init__()
        self.device = torch.device(args.device)

        self.vocab = vocab
        self.optimal_expert = expert
        self.expert_temperature = getattr(args, "expert_temperature", 0.)
        if self.expert_temperature < 0:
            raise ValueError("expert_temperature must be nonnegative.")
        self.expert_loss = getattr(args, "expert_loss", "marginal")
        if self.expert_loss not in {
                "marginal", "focal_marginal", "normalized_ce", "margin",
                "contrastive"}:
            raise ValueError(
                "expert_loss must be one of: marginal, focal_marginal, "
                "normalized_ce, margin, contrastive.")
        self.focal_gamma = getattr(args, "focal_gamma", 1.0)
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be nonnegative.")
        self.expert_margin = getattr(args, "expert_margin", 1.0)
        if self.expert_margin < 0:
            raise ValueError("expert_margin must be nonnegative.")
        self.contrastive_negative = getattr(args, "contrastive_negative", "hard")
        if self.contrastive_negative not in {"hard", "all"}:
            raise ValueError("contrastive_negative must be one of: hard, all.")
        self.last_margin_statistics = None
        self.last_focal_statistics = None
        self.last_contrastive_statistics = None
        self.source_tokenizer = utils.Tokenizer.from_cli(
            getattr(args, "source_separator", getattr(vocab, "source_separator", None)))
        self.target_tokenizer = utils.Tokenizer.from_cli(
            getattr(args, "target_separator", getattr(vocab, "target_separator", None)))

        self.number_characters = len(vocab.characters)
        self.number_actions = len(vocab.actions)
        self.substitutions = self.vocab.substitutions
        self.inserts = self.vocab.insertions

        self.dec_layers = args.dec_layers
        self.dec_hidden_dim = args.dec_hidden_dim

        # encoder
        self.char_lookup = torch.nn.Embedding(
            num_embeddings=self.number_characters,
            embedding_dim=args.char_dim,
            device=self.device,
            padding_idx=PAD
        )
        if args.enc_type == 'transformer':
            torch.nn.init.normal_(self.char_lookup.weight, mean=0, std=args.char_dim ** -0.5)
            torch.nn.init.constant_(self.char_lookup.weight[PAD], 0)

        self.enc = ENCODER_MAPPING[args.enc_type](args)

        decoder_input_dim = self.enc.output_size + args.action_dim
        self.output_feedback_dim = getattr(args, "output_feedback_dim", 0)
        if self.output_feedback_dim < 0:
            raise ValueError("output_feedback_dim must be nonnegative.")
        if self.output_feedback_dim > 0:
            self.output_lookup = torch.nn.Embedding(
                num_embeddings=len(vocab.target_symbols),
                embedding_dim=self.output_feedback_dim,
                device=self.device,
                padding_idx=PAD,
            )
            decoder_input_dim += self.output_feedback_dim
        else:
            self.output_lookup = None

        # feature encoder if required
        if isinstance(vocab, FeatureVocabularies):
            self.has_features = True
            self.number_features = len(vocab.features)
            self.feat_lookup = torch.nn.Embedding(
                num_embeddings=self.number_features,
                embedding_dim=args.feat_dim,
                device=self.device,
                padding_idx=PAD,
            )
            decoder_input_dim += args.feat_dim
        else:
            self.has_features = False
            self.number_features = None
            self.feat_lookup = None

        # decoder
        self.act_lookup = torch.nn.Embedding(
            num_embeddings=self.number_actions,
            embedding_dim=args.action_dim,
            device=self.device,
            padding_idx=PAD
        )

        self.dec = torch.nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=args.dec_hidden_dim,
            num_layers=args.dec_layers,
            device=self.device,
        )

        self._h0_c0 = None

        # classifier
        self.W = torch.nn.Linear(
            in_features=args.dec_hidden_dim,
            out_features=self.number_actions,
            device=self.device,

        )

        # maps action index to alignment update
        alignment_update = [0] * self.number_actions
        for i, action in enumerate(self.vocab.actions.i2w):
            if isinstance(action,
                          (ConditionalCopy, ConditionalDel, ConditionalSub)):
                alignment_update[i] = 1
        self.alignment_update = torch.tensor(alignment_update, device=self.device)

        self.register_buffer(
            "valid_actions_exhausted",
            self.compute_valid_actions(1),
        )
        self.register_buffer(
            "valid_actions_available",
            self.compute_valid_actions(2),
        )

    @property
    def h0_c0(self):
        return self._h0_c0

    @h0_c0.setter
    def h0_c0(self, batch_size):
        if not self._h0_c0 or \
                (batch_size and self._h0_c0[0].size(1) != batch_size):
            self._h0_c0 = (
                torch.zeros((self.dec_layers, batch_size, self.dec_hidden_dim), device=self.device),
                torch.zeros((self.dec_layers, batch_size, self.dec_hidden_dim), device=self.device),
            )

    def input_embedding(self, input_: torch.tensor, is_training: bool) -> torch.tensor:
        """Returns a list of character embeddings for the input.

        Args:
            input_: The encoded input string(s).
            is_training: is_training: Bool indicating whether model is in training or not. If True, UNK words are represented as average trained embeddings.

        Returns:
            The corresponding embeddings.
            """
        emb = self.char_lookup(input_)

        if not is_training:

            unk_indices = input_ >= self.number_characters
            if unk_indices.sum().item() > 0:
                # UNK is the average of trained embeddings (excluding UNK)
                ids_tensor = torch.tensor(
                    range(1, self.number_characters),
                    dtype=torch.int,
                    device=self.device,
                )
                unk = self.char_lookup(ids_tensor).mean(dim=0)
                emb[unk_indices] = unk

        return torch.transpose(emb, 0, 1)

    def feature_embedding(self, features: Optional[torch.Tensor], is_training: bool = False) -> Optional[torch.Tensor]:
        """Computes an embedding of the all input features."""
        if not self.has_features:
            return None

        emb = self.feat_lookup(features)  # (batch_size x features x feat_dim)

        if not is_training:

            unk_indices = features >= self.number_features
            if unk_indices.sum().item() > 0:
                # UNK is the average of trained embeddings (excluding UNK)
                ids_tensor = torch.tensor(
                    range(4, self.number_features),
                    dtype=torch.int,
                    device=self.device,
                )
                unk = self.feat_lookup(ids_tensor).mean(dim=0)
                emb[unk_indices] = unk

        return emb.sum(dim=1).unsqueeze(dim=0)  # (1 x batch_size x feat_dim)

    def compute_valid_actions(
            self,
            length_encoder_suffix: int,
            device: Optional[Union[str, torch.device]] = None) -> torch.tensor:
        """Computes the valid actions for a given encoder suffix as a boolean mask.

        Args:
            length_encoder_suffix: The length of the encoder suffix.

        Returns:
            The boolean mask for the given length."""
        if device is None:
            device = self.device
        valid_actions = torch.full((self.number_actions,), False,
                                   dtype=torch.bool, device=device)
        valid_actions[END_WORD] = True
        valid_actions[self.inserts] = True
        if length_encoder_suffix > 1:
            valid_actions[[COPY, DELETE]] = True
            valid_actions[self.substitutions] = True
        return valid_actions

    def valid_actions_for_suffixes(self, suffix_lengths: torch.tensor) -> torch.tensor:
        has_input = suffix_lengths.reshape(-1).to(self.device) > 1
        return torch.where(
            has_input.unsqueeze(dim=1),
            self.valid_actions_available.unsqueeze(dim=0),
            self.valid_actions_exhausted.unsqueeze(dim=0),
        ).unsqueeze(dim=0)

    @staticmethod
    def trim_encoded_action_history(action_history: torch.tensor) -> List[List[int]]:
        trimmed = []
        for seq in action_history.squeeze(dim=0).cpu().tolist():
            if END_WORD in seq:
                trimmed.append(seq[1:seq.index(END_WORD) + 1])
            else:
                trimmed.append(seq[1:])
        return trimmed

    @staticmethod
    def sample(log_probs: np.array) -> int:
        """Samples an action from a log-probability distribution."""
        dist = np.exp(log_probs)
        rand = np.random.rand()
        for action, p in enumerate(dist):
            rand -= p
            if rand <= 0:
                break
        return action

    @staticmethod
    def remap_actions(action_scores: Dict[Any, float]) -> Dict[Any, float]:
        """Maps a generative oracle's edit to their conditional counterparts."""
        remapped_action_scores = dict()
        for action, score in action_scores.items():
            if isinstance(action, GenerativeEdit):
                remapped_action = action.conditional_counterpart()
            elif isinstance(action, Edit):
                remapped_action = action
            else:
                raise ValueError(f"Unknown action: {action, score}.\n"
                                 f"action_scores: {action_scores}")
            remapped_action_scores[remapped_action] = score
        return remapped_action_scores

    def encode_known_action(self, action: Any, context: str) -> int:
        try:
            action_id = self.vocab.encode_unseen_action(action)
        except KeyError as error:
            raise RuntimeError(
                f"{context}: expert action is absent from action vocabulary: "
                f"action={action!r}."
            ) from error
        if action_id < 0 or action_id >= self.number_actions:
            raise RuntimeError(
                f"{context}: expert action id is outside action vocabulary: "
                f"action={action!r}, encoded_action={action_id}, "
                f"num_actions={self.number_actions}."
            )
        return action_id

    def expert_rollout(self, input_: str, target: str, alignment: int,
                       prediction: List[str]) -> List[int]:
        """Rolls out with optimal expert policy.

        Args:
            input_: Input string (x).
            target: Target prediction (t).
            alignment: Position of control in the input string.
            prediction: The current prediction so far (y).

        Returns:
            List of optimal actions as integer codes."""
        action_scores = self.expert_action_scores(
            input_, target, alignment, prediction)

        optimal_value = min(action_scores.values())
        return [
            self.encode_known_action(action, "expert_rollout")
            for action, value in action_scores.items()
            if value == optimal_value
        ]

    def expert_action_scores(self, input_: str, target: str, alignment: int,
                             prediction: List[str]) -> Dict[Any, float]:
        raw_action_scores = self.optimal_expert.score(
            input_, target, alignment, prediction)
        return self.remap_actions(raw_action_scores)

    def expert_action_from_id(self, input_: Union[str, List[str]],
                              action_id: int,
                              alignment: Union[int, torch.tensor]) -> Edit:
        if torch.is_tensor(alignment):
            alignment = alignment.item()
        action = self.vocab.decode_action(action_id)
        if isinstance(action, (ConditionalCopy, ConditionalDel, ConditionalIns,
                               ConditionalSub, EndOfSequence)):
            return action
        raise ValueError(f"Cannot convert decoder action to expert action: {action}.")

    def concrete_expert_action_from_id(self, input_: Union[str, List[str]],
                                       action_id: int,
                                       alignment: Union[int, torch.tensor]) -> Edit:
        if torch.is_tensor(alignment):
            alignment = alignment.item()
        action = self.vocab.decode_action(action_id)
        if isinstance(action, ConditionalCopy):
            if alignment >= len(input_):
                raise ValueError("Cannot copy after input is exhausted.")
            return Copy(input_[alignment], input_[alignment])
        if isinstance(action, ConditionalDel):
            if alignment >= len(input_):
                raise ValueError("Cannot delete after input is exhausted.")
            return Del(input_[alignment])
        if isinstance(action, ConditionalSub):
            if alignment >= len(input_):
                raise ValueError("Cannot substitute after input is exhausted.")
            return Sub(input_[alignment], action.new)
        if isinstance(action, ConditionalIns):
            return Ins(action.new)
        if isinstance(action, EndOfSequence):
            return action
        raise ValueError(f"Cannot convert decoder action to expert action: {action}.")

    def action_effect(self, input_: Union[str, List[str]],
                      alignment: Union[int, torch.tensor],
                      action: Union[int, Edit]) -> ActionEffect:
        if torch.is_tensor(action):
            action = action.item()
        if isinstance(action, int):
            action = self.vocab.decode_action(action)
        if torch.is_tensor(alignment):
            alignment = alignment.item()
        if isinstance(action, ConditionalCopy):
            if alignment >= len(input_):
                raise ValueError("Cannot copy after input is exhausted.")
            return ActionEffect(input_[alignment], 1, False)
        if isinstance(action, ConditionalDel):
            if alignment >= len(input_):
                raise ValueError("Cannot delete after input is exhausted.")
            return ActionEffect(None, 1, False)
        if isinstance(action, ConditionalIns):
            return ActionEffect(action.new, 0, False)
        if isinstance(action, ConditionalSub):
            if alignment >= len(input_):
                raise ValueError("Cannot substitute after input is exhausted.")
            return ActionEffect(action.new, 1, False)
        if isinstance(action, EndOfSequence):
            return ActionEffect(None, 0, True)
        if isinstance(action, BeginOfSequence):
            return ActionEffect(None, 0, False)
        raise ValueError(f"Unknown action: {action}.")

    def expert_score_action(self, input_: Union[str, List[str]],
                            target: Union[str, List[str]],
                            alignment: int,
                            prediction: List[str],
                            action_id: int) -> float:
        effect = self.action_effect(input_, alignment, action_id)
        action = self.concrete_expert_action_from_id(input_, action_id, alignment)
        successor_prediction = list(prediction)
        if effect.output_symbol is not None:
            successor_prediction.append(effect.output_symbol)
        return (
            self.optimal_expert.aligner.action_cost(action) +
            self.optimal_expert.score_state(
                input_,
                target,
                alignment + effect.alignment_delta,
                successor_prediction,
            )
        )

    def expert_score_decoder_action(self, input_: Union[str, List[str]],
                                    target: Union[str, List[str]],
                                    alignment: int,
                                    prediction: List[str],
                                    action_id: int):
        effect = self.action_effect(input_, alignment, action_id)
        successor_prediction = list(prediction)
        if effect.output_symbol is not None:
            successor_prediction.append(effect.output_symbol)
        if effect.stop:
            terminal_cost = optimal_expert.levenshtein_distance(
                successor_prediction,
                target,
            )[-1, -1]
            return self.optimal_expert.decoder_state_score(
                prefix_cost=float(terminal_cost),
                continuation_cost=0.,
                total=float(terminal_cost),
                target_prefix_index=len(target),
            )
        return self.optimal_expert.score_decoder_state(
            input_,
            target,
            alignment + effect.alignment_delta,
            successor_prediction,
        )

    def expert_score_action_direct(self, input_: Union[str, List[str]],
                                   target: Union[str, List[str]],
                                   alignment: int,
                                   prediction: List[str],
                                   action_id: int) -> float:
        action = self.concrete_expert_action_from_id(input_, action_id, alignment)
        return self.optimal_expert.score_action(
            input_,
            target,
            alignment,
            prediction,
            action,
        )

    def encode_expert_action_costs(
            self,
            action_scores: Dict[Any, float],
            device: Optional[Union[str, torch.device]] = None) -> torch.tensor:
        if device is None:
            device = self.device
        costs = torch.full(
            (self.number_actions,),
            float("inf"),
            dtype=torch.float,
            device=device,
        )
        for action, cost in action_scores.items():
            action_id = self.encode_known_action(action, "encode_expert_action_costs")
            costs[action_id] = cost
        return costs

    def mark_as_invalid(self, logits: torch.tensor,
                        valid_actions_mask: torch.tensor) -> torch.tensor:
        """Mark all logits for all non-valid actions as such (i.e., they are set to -infinity).

        Args:
            logits: The logits.
            valid_actions_mask: A boolean mask indicating all valid actions for all tokens in the batch.
        Returns:
            The 'corrected' logits."""
        log_validity = torch.full(
            logits.size(),
            -np.inf,
            device=self.device,
        )  # All actions invalid by default.
        log_validity[valid_actions_mask] = 0.
        return logits + log_validity

    def log_softmax(self, logits: torch.tensor,
                    valid_actions_mask: torch.tensor) -> torch.tensor:
        """Applies the log_softmax function to all valid actions.
        Args:
            logits: The logits.
            valid_actions_mask: A boolean mask indicating all valid actions for all tokens in the batch.

        Returns:
            The logits after marking non-valid actions as such and applying the log_softmax function."""
        logits_valid = self.mark_as_invalid(logits, valid_actions_mask)
        return torch.nn.functional.log_softmax(logits_valid, dim=2)

    def validate_index_tensor(self, name: str, values: torch.tensor,
                              upper_bound: int) -> None:
        if values.numel() == 0:
            return
        if torch.any(values < 0).item() or torch.any(values >= upper_bound).item():
            values_cpu = values.detach().cpu()
            raise RuntimeError(
                f"{name} contains ids outside [0, {upper_bound}): "
                f"min={values_cpu.min().item()}, max={values_cpu.max().item()}, "
                f"shape={tuple(values_cpu.shape)}."
            )

    def log_sum_softmax_loss(self, logits: torch.tensor,
                             optimal_actions_mask: torch.tensor,
                             valid_actions_mask: torch.tensor) -> torch.tensor:
        """Compute log loss similar to Riezler et al 2000.

        Args:
            logits: The logits for which the loss is computed.
            optimal_actions_mask: A boolean mask indicating the optimal action for all tokens in the batch.
            valid_actions_mask: A boolean mask indicating all valid actions for all tokens in the batch.

        Returns:
            The computed loss."""
        logits_valid = self.mark_as_invalid(logits, valid_actions_mask)
        # padding can be inferred from optimal actions
        # --> if mask only consists of False values
        paddings = ~torch.any(optimal_actions_mask, dim=2)
        logits_valid[paddings] = -np.inf
        logits_optimal = logits_valid.clone()
        valid_optimal_actions = valid_actions_mask & optimal_actions_mask
        logits_optimal[~valid_optimal_actions] = -np.inf

        log_sum_selected_terms = torch.logsumexp(
            logits_optimal,
            dim=2,
        )

        normalization_term = torch.logsumexp(
            logits_valid,
            dim=2,
        )

        if paddings.sum() > 0:
            log_sum_selected_terms =\
                torch.where(~paddings, log_sum_selected_terms, torch.tensor(0., device=self.device))
            normalization_term =\
                torch.where(~paddings, normalization_term, torch.tensor(0., device=self.device))

        return log_sum_selected_terms - normalization_term

    def focal_marginal_loss(self, logits: torch.tensor,
                            optimal_actions_mask: torch.tensor,
                            valid_actions_mask: torch.tensor,
                            gamma: float) -> torch.tensor:
        """Focalized hard set-valued oracle loss.

        For every non-padding state:
            p* = sum_{a in A*(s)} p(a | s)
            L = -(1 - p*)**gamma * log(p*)

        gamma=0 exactly recovers ordinary marginal training.
        """
        if gamma < 0:
            raise ValueError("gamma must be nonnegative.")
        log_p_optimal = self.log_sum_softmax_loss(
            logits,
            optimal_actions_mask,
            valid_actions_mask,
        )
        p_optimal = torch.exp(log_p_optimal)
        focal_weight = (1.0 - p_optimal).clamp(min=0.0).pow(gamma)
        losses = -focal_weight * log_p_optimal
        paddings = ~torch.any(optimal_actions_mask, dim=2)
        losses = torch.where(
            ~paddings,
            losses,
            torch.zeros_like(losses),
        )
        if not torch.isfinite(losses[~paddings]).all():
            raise FloatingPointError("Focal marginal loss produced non-finite values.")
        valid_p_optimal = p_optimal[~paddings].detach().cpu()
        valid_focal_weight = focal_weight[~paddings].detach().cpu()
        self.last_focal_statistics = {
            "states": int(valid_p_optimal.numel()),
            "oracle_masses": valid_p_optimal,
            "focal_weights": valid_focal_weight,
        }
        return losses

    def soft_oracle_loss(self, logits: torch.tensor,
                         expert_action_costs: torch.tensor,
                         valid_actions_mask: torch.tensor,
                         temperature: float) -> torch.tensor:
        """Cost-sensitive oracle log mass.

        This computes log sum_a p(a) * exp(-(C(a)-C*) / temperature) over
        mechanically valid actions. The hard oracle loss is the zero-temperature
        limiting case and remains the default path.
        """
        if temperature <= 0:
            raise ValueError("Soft oracle temperature must be positive.")
        paddings = ~torch.any(valid_actions_mask, dim=2)
        safe_valid_actions_mask = valid_actions_mask.clone()
        safe_valid_actions_mask[paddings, 0] = True
        log_probs = self.log_softmax(logits, safe_valid_actions_mask)
        valid_costs = expert_action_costs.clone()
        valid_costs[~safe_valid_actions_mask] = float("inf")
        valid_costs[paddings, 0] = 0.
        best_costs = torch.min(valid_costs, dim=2, keepdim=True).values
        cost_gaps = valid_costs - best_costs
        weighted_terms = log_probs - cost_gaps / temperature
        weighted_terms[~safe_valid_actions_mask] = -float("inf")
        log_weighted_mass = torch.logsumexp(weighted_terms, dim=2)
        log_weighted_mass = torch.where(
            ~paddings,
            log_weighted_mass,
            torch.tensor(0., device=self.device),
        )
        if not torch.isfinite(log_weighted_mass[~paddings]).all():
            raise FloatingPointError("Soft oracle loss produced non-finite values.")
        return log_weighted_mass

    def normalized_soft_expert_loss(self, logits: torch.tensor,
                                    expert_action_costs: torch.tensor,
                                    valid_actions_mask: torch.tensor,
                                    temperature: float) -> torch.tensor:
        """Expected log-probability under a normalized soft expert distribution.

        The expert distribution is defined over actions with finite expert costs.
        Decoder-valid but expert-unscored actions remain in the model
        normalization, but receive no expert target mass.
        """
        if temperature <= 0:
            raise ValueError("Soft expert temperature must be positive.")
        finite_expert_mask = torch.isfinite(expert_action_costs)
        paddings = ~torch.any(finite_expert_mask, dim=2)
        safe_valid_actions_mask = valid_actions_mask.clone()
        safe_valid_actions_mask[paddings, 0] = True
        log_probs = self.log_softmax(logits, safe_valid_actions_mask)

        safe_costs = expert_action_costs.clone()
        safe_costs[~finite_expert_mask] = float("inf")
        safe_costs[paddings, 0] = 0.
        best_costs = torch.min(safe_costs, dim=2, keepdim=True).values
        cost_gaps = safe_costs - best_costs
        expert_logits = -cost_gaps / temperature
        expert_logits[~finite_expert_mask] = -float("inf")
        expert_logits[paddings, 0] = 0.
        expert_log_probs = torch.nn.functional.log_softmax(expert_logits, dim=2)
        expert_probs = torch.exp(expert_log_probs).masked_fill(
            ~finite_expert_mask,
            0.,
        )
        safe_log_probs = log_probs.masked_fill(~finite_expert_mask, 0.)
        expected_log_prob = torch.sum(expert_probs * safe_log_probs, dim=2)
        expected_log_prob = torch.where(
            ~paddings,
            expected_log_prob,
            torch.zeros_like(expected_log_prob),
        )
        if not torch.isfinite(expected_log_prob[~paddings]).all():
            raise FloatingPointError("Normalized soft expert loss produced non-finite values.")
        return expected_log_prob

    def margin_expert_loss(self, logits: torch.tensor,
                           optimal_actions_mask: torch.tensor,
                           valid_actions_mask: torch.tensor,
                           margin: float) -> torch.tensor:
        """Fixed-margin hinge loss over expert-optimal vs decoder-valid actions."""
        paddings = ~torch.any(optimal_actions_mask, dim=2)
        oracle_mask = valid_actions_mask & optimal_actions_mask
        nonoracle_mask = valid_actions_mask & ~oracle_mask
        oracle_mask = oracle_mask.clone()
        nonoracle_mask = nonoracle_mask.clone()
        # MPS max backward cannot handle reductions where every action was
        # masked to -inf: its internal argmax can become -1 and fail in a
        # scatter kernel. Padding rows and rows without a competitor are
        # semantically zero-loss, so give them one finite dummy action before
        # reducing and mask them out after the hinge computation.
        no_oracle = ~torch.any(oracle_mask, dim=2)
        no_nonoracle = ~torch.any(nonoracle_mask, dim=2)
        oracle_mask[no_oracle, 0] = True
        nonoracle_mask[no_nonoracle, 0] = True

        oracle_best = logits.masked_fill(~oracle_mask, -torch.inf).max(dim=2).values
        nonoracle_best = logits.masked_fill(~nonoracle_mask, -torch.inf).max(dim=2).values
        hinge = torch.relu(margin + nonoracle_best - oracle_best)
        zero_loss = paddings | no_oracle | no_nonoracle
        hinge = torch.where(~zero_loss, hinge, torch.zeros_like(hinge))
        if not torch.isfinite(hinge[~paddings]).all():
            raise FloatingPointError("Margin expert loss produced non-finite values.")

        margins = oracle_best - nonoracle_best
        valid_states = ~zero_loss
        finite_margins = margins[valid_states & torch.isfinite(margins)]
        active = hinge[valid_states] > 0
        active_losses = hinge[valid_states][active]
        self.last_margin_statistics = {
            "states": int(valid_states.sum().item()),
            "active": int(active.sum().item()),
            "margin_sum": float(finite_margins.sum().item()) if finite_margins.numel() > 0 else 0.,
            "margin_count": int(finite_margins.numel()),
            "active_loss_sum": float(active_losses.sum().item()) if active_losses.numel() > 0 else 0.,
        }
        return hinge

    def action_type_name(self, action_id: int) -> str:
        action = self.vocab.decode_action(action_id)
        if isinstance(action, ConditionalCopy):
            return "COPY"
        if isinstance(action, ConditionalSub):
            return "SUB"
        if isinstance(action, ConditionalIns):
            return "INS"
        if isinstance(action, ConditionalDel):
            return "DEL"
        if isinstance(action, EndOfSequence):
            return "EOS"
        return action.__class__.__name__.upper()

    def contrastive_expert_loss(self, logits: torch.tensor,
                                expert_action_costs: torch.tensor,
                                valid_actions_mask: torch.tensor,
                                margin: float,
                                negative_mode: str = "hard",
                                cost_epsilon: float = 1e-6) -> torch.tensor:
        """Pairwise ranking loss between optimal expert set and nonoptimal actions."""
        if negative_mode not in {"hard", "all"}:
            raise ValueError("negative_mode must be one of: hard, all.")
        finite_expert_mask = torch.isfinite(expert_action_costs)
        paddings = ~torch.any(finite_expert_mask, dim=2)
        safe_costs = expert_action_costs.clone()
        safe_costs[~finite_expert_mask] = float("inf")
        safe_costs[paddings, 0] = 0.
        best_costs = torch.min(safe_costs, dim=2, keepdim=True).values
        positive_mask = (
            finite_expert_mask &
            (torch.abs(expert_action_costs - best_costs) <= cost_epsilon)
        )
        # The decoder competes over every mechanically valid action, so the
        # contrastive hard negative must come from the decoder-valid space, not
        # only from actions to which the expert assigned a finite cost.
        negative_mask = valid_actions_mask & ~positive_mask

        no_positive = ~torch.any(positive_mask, dim=2)
        no_negative = ~torch.any(negative_mask, dim=2)
        safe_positive_mask = positive_mask.clone()
        safe_negative_mask = negative_mask.clone()
        safe_positive_mask[no_positive, 0] = True
        safe_negative_mask[no_negative, 0] = True

        positive_score = torch.logsumexp(
            logits.masked_fill(~safe_positive_mask, -torch.inf),
            dim=2,
        )
        negative_logits = logits.masked_fill(~safe_negative_mask, -torch.inf)
        if negative_mode == "hard":
            negative_score, negative_ids = negative_logits.max(dim=2)
        else:
            negative_score = torch.logsumexp(negative_logits, dim=2)
            negative_ids = negative_logits.max(dim=2).indices
        negative_ids = torch.where(
            no_negative,
            torch.zeros_like(negative_ids),
            negative_ids,
        )

        losses = torch.nn.functional.softplus(
            negative_score - positive_score + margin)
        zero_loss = paddings | no_positive | no_negative
        losses = torch.where(~zero_loss, losses, torch.zeros_like(losses))
        if not torch.isfinite(losses[~paddings]).all():
            raise FloatingPointError("Contrastive expert loss produced non-finite values.")

        valid_states = ~zero_loss
        positive_counts = positive_mask.sum(dim=2)
        valid_positive_counts = positive_counts[valid_states]
        ranking_correct = positive_score[valid_states] > negative_score[valid_states]
        margin_satisfied = (
            positive_score[valid_states] >=
            negative_score[valid_states] + margin
        )
        active = losses[valid_states] > 0
        negative_costs = torch.gather(
            expert_action_costs,
            dim=2,
            index=negative_ids.unsqueeze(dim=2),
        ).squeeze(dim=2)
        negative_has_expert_cost = torch.isfinite(negative_costs)
        negative_gaps = negative_costs - best_costs.squeeze(dim=2)
        valid_gap_states = valid_states & negative_has_expert_cost
        valid_gaps = negative_gaps[valid_gap_states]
        decoder_top_logits = logits.masked_fill(~valid_actions_mask, -torch.inf)
        decoder_top_ids = decoder_top_logits.max(dim=2).indices
        decoder_top_valid_states = torch.any(valid_actions_mask, dim=2)
        decoder_top_ids = torch.where(
            decoder_top_valid_states,
            decoder_top_ids,
            torch.zeros_like(decoder_top_ids),
        )
        decoder_top_is_optimal = torch.gather(
            positive_mask,
            dim=2,
            index=decoder_top_ids.unsqueeze(dim=2),
        ).squeeze(dim=2)
        decoder_top_is_finite = torch.gather(
            finite_expert_mask,
            dim=2,
            index=decoder_top_ids.unsqueeze(dim=2),
        ).squeeze(dim=2)
        decoder_top_nonoptimal = decoder_top_valid_states & ~decoder_top_is_optimal
        negative_type_counts = collections.Counter()
        decoder_error_type_counts = collections.Counter()
        delete_gap_sum = 0.
        delete_gap_count = 0
        delete_logit_advantage_sum = 0.
        delete_ranking_correct = 0
        delete_margin_satisfied = 0
        for action_id, gap, logit_advantage, is_correct, is_satisfied in zip(
                negative_ids[valid_states].detach().cpu().tolist(),
                negative_gaps[valid_states].detach().cpu().tolist(),
                (negative_score[valid_states] -
                 positive_score[valid_states]).detach().cpu().tolist(),
                ranking_correct.detach().cpu().tolist(),
                margin_satisfied.detach().cpu().tolist()):
            action_type = self.action_type_name(action_id)
            negative_type_counts[action_type] += 1
            if action_type == "DEL" and np.isfinite(gap):
                delete_gap_sum += gap
                delete_gap_count += 1
                delete_logit_advantage_sum += logit_advantage
                delete_ranking_correct += int(is_correct)
                delete_margin_satisfied += int(is_satisfied)
        for action_id in decoder_top_ids[decoder_top_nonoptimal].detach().cpu().tolist():
            decoder_error_type_counts[self.action_type_name(action_id)] += 1
        self.last_contrastive_statistics = {
            "states": int(valid_states.sum().item()),
            "active": int(active.sum().item()),
            "ranking_correct": int(ranking_correct.sum().item()),
            "margin_satisfied": int(margin_satisfied.sum().item()),
            "decoder_top_states": int(decoder_top_valid_states.sum().item()),
            "decoder_top_optimal": int(
                decoder_top_is_optimal[decoder_top_valid_states].sum().item()),
            "decoder_top_finite": int(
                decoder_top_is_finite[decoder_top_valid_states].sum().item()),
            "decoder_top_excluded": int(
                (~decoder_top_is_finite[decoder_top_valid_states]).sum().item()),
            "loss_sum": float(losses[valid_states].sum().item()) if valid_states.any() else 0.,
            "gap_sum": float(valid_gaps.sum().item()) if valid_gaps.numel() else 0.,
            "gap_count": int(valid_gaps.numel()),
            "negative_finite_expert": int(
                negative_has_expert_cost[valid_states].sum().item()),
            "negative_expert_excluded": int(
                (~negative_has_expert_cost[valid_states]).sum().item()),
            "positive_count_sum": int(valid_positive_counts.sum().item()),
            "positive_count_max": int(valid_positive_counts.max().item()) if valid_positive_counts.numel() else 0,
            "negative_type_counts": negative_type_counts,
            "decoder_error_type_counts": decoder_error_type_counts,
            "delete_gap_sum": delete_gap_sum,
            "delete_gap_count": delete_gap_count,
            "delete_logit_advantage_sum": delete_logit_advantage_sum,
            "delete_ranking_correct": delete_ranking_correct,
            "delete_margin_satisfied": delete_margin_satisfied,
        }
        return losses

    def encoder_step(self, encoded_input: torch.tensor, is_training: bool = False) -> torch.tensor:
        """Runs the encoder.

        Args:
            encoded_input: Encoded input character codes.
            is_training: Bool indicating whether model is in training or not.

        Returns:
            Encoder output."""
        input_emb = self.input_embedding(encoded_input, is_training)

        # encoder input: L x B x E
        if isinstance(self.enc, ENCODER_MAPPING['transformer']):
            bidirectional_emb = self.enc(input_emb,
                                         src_key_padding_mask=(encoded_input == PAD))
        else:
            bidirectional_emb, _ = self.enc(input_emb)

        return bidirectional_emb[1:]  # drop BEGIN_WORD

    def decoder_step(self, encoder_output: torch.tensor,
                     feature_embedding: Optional[torch.tensor],
                     decoder_cell_state: torch.tensor,
                     alignment: torch.tensor,
                     action_history: torch.tensor,
                     output_history: Optional[torch.tensor] = None) -> torch.tensor:
        """Runs the decoder.

        Args:
            encoder_output: The encoder output.
            feature_embedding: Optional feature embedding, the same for all decoder steps.
            decoder_cell_state: The initial decoder cell state.
            alignment: The alignment for all sequences in the batch. This tensor is of shape (L x B) x 1.
            action_history: The action history.

        Returns:
            Decoder output."""
        # build decoder input
        batch_size = encoder_output.size(1)
        decoder_steps = len(alignment) // batch_size
        safe_alignment = alignment.clamp(min=0, max=encoder_output.size(0) - 1)
        self.validate_index_tensor(
            "decoder alignment",
            safe_alignment,
            encoder_output.size(0),
        )
        alignment_by_batch = safe_alignment.view(batch_size, decoder_steps).transpose(0, 1)
        gather_index = alignment_by_batch.unsqueeze(dim=2).expand(
            -1,
            -1,
            encoder_output.size(2),
        )
        input_char_embedding = torch.gather(
            encoder_output,
            dim=0,
            index=gather_index,
        )
        safe_action_history = action_history.masked_fill(action_history < 0, PAD)
        self.validate_index_tensor(
            "decoder action history",
            safe_action_history,
            self.number_actions,
        )
        previous_action_embedding = self.act_lookup(safe_action_history)
        if self.device.type == "mps":
            torch.mps.synchronize()

        decoder_inputs = [input_char_embedding, previous_action_embedding]
        if self.output_lookup is not None:
            if output_history is None:
                raise ValueError("Output feedback requires output_history.")
            safe_output_history = output_history.masked_fill(output_history < 0, PAD)
            self.validate_index_tensor(
                "decoder output history",
                safe_output_history,
                len(self.vocab.target_symbols),
            )
            decoder_inputs.append(self.output_lookup(safe_output_history))
            if self.device.type == "mps":
                torch.mps.synchronize()
        if self.has_features:
            # Repeats the feature embedding along the decoder steps dimension.
            number_of_decoder_steps = previous_action_embedding.shape[0]
            broadcast_feature_embedding = feature_embedding.\
                repeat((number_of_decoder_steps, 1, 1))
            decoder_inputs.append(broadcast_feature_embedding)

        decoder_input = torch.cat(decoder_inputs, dim=2)

        return self.dec(decoder_input, decoder_cell_state)

    def calculate_actions(self, decoder_output: torch.tensor, valid_actions_mask: torch.tensor)\
            -> Tuple[torch.tensor, torch.tensor]:
        """Calculates the optimal actions (by choosing the max arguments) and log probabilites given the decoder
        output and valid actions for this step.

        Args:
            decoder_output: The output of the decoder.
            valid_actions_mask: A boolean mask indicating all valid actions for all tokens in the batch.

        Returns:
            tuple: A tuple containing:

                actions: The actions.
                log_probabilites: The log probabilites of all actions.
        """
        logits = self.W(decoder_output)
        log_probs = self.log_softmax(logits, valid_actions_mask)
        actions = torch.argmax(log_probs, dim=2)

        return actions, log_probs

    def training_step(self, encoded_input: torch.tensor,
                      encoded_features: Optional[torch.tensor],
                      action_history: torch.tensor,
                      output_history: Optional[torch.tensor],
                      alignment_history: torch.tensor,
                      expert_action_costs: Optional[torch.tensor],
                      optimal_actions_mask: torch.tensor,
                      valid_actions_mask: torch.tensor,
                      ) -> torch.tensor:
        """Run a training step and return the respective loss for all sequences in the batch.

        Args:
            encoded_input: Encoded input character codes.
            encoded_features: Optional encoded features.
            action_history: The action history for all sequences. During training this is based on the optimal actions (from the expert).
            alignment_history: The alignment history for all sequences. During training this is based on the optimal alignment (from the expert).
            optimal_actions_mask: A boolean mask indicating the optimal action for all tokens in the batch.
            valid_actions_mask: A boolean mask indicating all valid actions for all tokens in the batch.

        Returns:
            The loss for sequences in the batch. The loss is calculated on sequence-level, i.e., for each sequence
            a single gradient is produced."""
        self.last_margin_statistics = None
        self.last_focal_statistics = None
        self.last_contrastive_statistics = None
        batch_size = encoded_input.size()[0]

        # adjust initial decoder states if batch_size has changed
        self.h0_c0 = batch_size

        # run encoder
        bidirectional_emb = self.encoder_step(encoded_input, True)

        # compute feature embedding
        feature_emb = self.feature_embedding(encoded_features, True)

        # run decoder & classifier
        decoder_output, _ = self.decoder_step(
            bidirectional_emb, feature_emb, self.h0_c0,
            alignment_history, action_history, output_history)
        logits = self.W(decoder_output)

        # compute losses
        # the loss for each seq in the batch is divided by the nr of non-padding elements
        # --> loss per seq = avg. loss per token in seq
        true_action_lengths = action_history.size(0) - (action_history == PAD).sum(dim=0)
        if self.expert_loss == "margin":
            losses = self.margin_expert_loss(
                logits,
                optimal_actions_mask,
                valid_actions_mask,
                self.expert_margin,
            )
            losses = losses.sum(dim=0) / true_action_lengths
            return losses
        if self.expert_loss == "focal_marginal":
            losses = self.focal_marginal_loss(
                logits,
                optimal_actions_mask,
                valid_actions_mask,
                self.focal_gamma,
            )
            losses = losses.sum(dim=0) / true_action_lengths
            return losses
        if self.expert_loss == "contrastive":
            if expert_action_costs is None:
                raise ValueError("Contrastive expert loss requires expert_action_costs.")
            losses = self.contrastive_expert_loss(
                logits,
                expert_action_costs,
                valid_actions_mask,
                self.expert_margin,
                self.contrastive_negative,
            )
            losses = losses.sum(dim=0) / true_action_lengths
            return losses
        if self.expert_temperature > 0:
            if expert_action_costs is None:
                raise ValueError("Soft oracle loss requires expert_action_costs.")
            if self.expert_loss == "normalized_ce":
                losses = self.normalized_soft_expert_loss(
                    logits,
                    expert_action_costs,
                    valid_actions_mask,
                    self.expert_temperature,
                )
            else:
                losses = self.soft_oracle_loss(
                    logits,
                    expert_action_costs,
                    valid_actions_mask,
                    self.expert_temperature,
                )
        else:
            losses = self.log_sum_softmax_loss(logits, optimal_actions_mask, valid_actions_mask)
        losses = -losses.sum(dim=0) / true_action_lengths

        return losses

    def transduce(self, input_: List[List[str]], encoded_input: torch.tensor,
                  encoded_features: Optional[torch.tensor]) -> Output:
        """Runs the transducer for greedy decoding.

        Args:
            input_: Input string.
            encoded_input: Tensor with integer character codes with dimensions (B x L x E).
            encoded_features: Optional tensor integer feature codes (padded and batched).

        Returns:
            An Output object holding the decoded input."""
        batch_size = encoded_input.size()[0]

        # adjust initial decoder states if batch_size has changed
        self.h0_c0 = batch_size

        # initialize state variables
        alignment = torch.full((batch_size,), 0, device=self.device)
        action_history = torch.tensor([[[BEGIN_WORD]] * batch_size],
                                      device=self.device, dtype=torch.int)
        previous_output = torch.tensor(
            [[self.vocab.encode_output_symbol(vocabulary.BOS_OUTPUT)] * batch_size],
            device=self.device, dtype=torch.long)
        log_p = torch.full((1, batch_size), 0.0, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        action_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        true_input_lengths = torch.tensor(
            # +1 because end word is not included in input
            [len(i) + 1 for i in input_], device=self.device)

        # run encoder
        bidirectional_emb = self.encoder_step(encoded_input)

        # compute feature embedding
        feature_emb = self.feature_embedding(encoded_features)

        # initial cell state for decoder
        decoder = self.h0_c0

        while not torch.all(finished) and action_history.size(2) <= MAX_ACTION_SEQ_LEN:
            active = ~finished
            valid_actions_mask = self.valid_actions_for_suffixes(true_input_lengths - alignment)

            # run decoder
            decoder_output, decoder = self.decoder_step(
                bidirectional_emb, feature_emb, decoder,
                alignment, action_history[:, :, -1],
                previous_output if self.output_lookup is not None else None)

            # get actions
            actions, log_probs = self.calculate_actions(decoder_output, valid_actions_mask)
            action_ids = actions.squeeze(dim=0)

            # update states
            selected_log_probs = log_probs[
                0,
                torch.arange(batch_size, device=self.device),
                action_ids,
            ]
            log_p[0, active] += selected_log_probs[active]
            action_lengths[active] += 1
            action_history = torch.cat(
                (action_history, actions.unsqueeze(dim=2)),
                dim=2
            )
            if self.output_lookup is not None:
                next_output = torch.tensor([[
                    self.output_symbol_id_for_action(
                        input_[i],
                        action_ids[i].item(),
                        alignment[i],
                    )
                    for i in range(batch_size)
                ]], device=self.device, dtype=torch.long)
                previous_output = torch.where(active.unsqueeze(dim=0), next_output, previous_output)
            alignment = alignment + self.alignment_update[action_ids] * active
            finished = finished | (active & (action_ids == END_WORD))

        # adjust log_p
        # --> return the generated-action avg. of all seqs in the batch
        log_p = torch.mean(log_p.squeeze(dim=0) / action_lengths).item()

        # trim action history
        # --> first element is not considered (begin-of-sequence-token)
        # --> and only token up to the first end-of-sequence-token (as encoded integer output, including it)
        action_history = self.trim_encoded_action_history(action_history)

        return Output(action_history, self.decode_encoded_output(input_, action_history),
                      log_p, None)

    def decode_encoded_output(self, input_: List[List[str]], encoded_output: List[List[int]]) -> List[str]:
        """Decode a list of encoded output sequences given their string input.

        Args:
            input_: Input string.
            encoded_output: Holds the encoded integers output (--> encoded actions) corresponding to the input sequences.

        Returns:
            A list of the decoded strings."""
        output = []
        for i, seq in enumerate(encoded_output):
            decoded_seq = []
            alignment = 0
            for a in seq:
                char_, alignment, _ = self.decode_single_action(input_[i], a, alignment)
                if char_ != "":
                    decoded_seq.append(char_)
            output.append(self.target_tokenizer.untokenize(decoded_seq))

        return output

    def output_symbol_for_action(self, input_: Union[str, List[str]],
                                 action: Union[int, Edit],
                                 alignment: Union[int, torch.tensor]) -> Any:
        if torch.is_tensor(action):
            action = action.item()
        if isinstance(action, int):
            action = self.vocab.decode_action(action)
        if isinstance(action, BeginOfSequence):
            return vocabulary.BOS_OUTPUT
        effect = self.action_effect(input_, alignment, action)
        if effect.output_symbol is None:
            return vocabulary.NO_OUTPUT
        return effect.output_symbol

    def output_symbol_id_for_action(self, input_: Union[str, List[str]],
                                    action: Union[int, Edit],
                                    alignment: Union[int, torch.tensor]) -> int:
        return self.vocab.encode_output_symbol(
            self.output_symbol_for_action(input_, action, alignment))

    def decode_single_action(self, input_: Union[str, List[str]], action: Union[int, Edit],
                             alignment: Union[int, torch.tensor]) -> Tuple[str, int, bool]:
        """Decodes a single char, given the corresponding input string, action and alignment.

        Args:
            input_: The input string.
            action: The action, may be encoded or not.
            alignment: Position of control in the input string.

        Returns:
            tuple: A tuple containing:
                char_: The decoded char.
                alignment: The updated alignment.
                stop: A bool indicating whether the end of sequence is reached.
            """
        effect = self.action_effect(input_, alignment, action)
        char_ = "" if effect.output_symbol is None else effect.output_symbol
        return char_, alignment + effect.alignment_delta, effect.stop

    def beam_search_decode(self, input_: str, encoded_input: torch.tensor,
                           encoded_features: Optional[torch.tensor],
                           beam_width: int) -> List[Output]:
        """Runs the transducer with beam search.

        Args:
            input_: Input string.
            encoded_input: List of integer character codes.
            encoded_features: Optional tensor of feature codes.
            beam_width: Width of the beam search.

        Returns:
            A list holding the output of the best search paths.
        """
        # adjust initial decoder states if batch_size has changed
        self.h0_c0 = 1  # nothing else possible at the moment

        # run encoder
        bidirectional_emb = self.encoder_step(encoded_input)

        # compute feature embedding
        feature_emb = self.feature_embedding(encoded_features)

        input_length = len(input_) + 1  # +1 because of begin-of-seq-token

        beam: List[Hypothesis] = [
            Hypothesis(action_history=torch.tensor([[BEGIN_WORD]], device=self.device),
                       alignment=torch.tensor([0], device=self.device),
                       decoder=self.h0_c0,
                       negative_log_p=torch.tensor(0., device=self.device),
                       output=[],
                       previous_output=torch.tensor(
                           [[self.vocab.encode_output_symbol(vocabulary.BOS_OUTPUT)]],
                           device=self.device, dtype=torch.long))]

        search_beam_width = beam_width
        num_outputs_needed = beam_width
        hypothesis_length = 0
        complete_hypotheses = []
        n_decoder_calls = 0
        n_expansions = 0
        max_active_beam = 0
        beam_sizes = []

        while beam and len(complete_hypotheses) < num_outputs_needed \
                and hypothesis_length <= MAX_ACTION_SEQ_LEN:

            beam_sizes.append(len(beam))
            max_active_beam = max(max_active_beam, len(beam))

            expansions: List[Expansion] = []

            for hypothesis in beam:
                n_decoder_calls += 1

                length_encoder_suffix = max(input_length - hypothesis.alignment, torch.tensor([0], device=self.device))
                valid_actions_mask = self.valid_actions_for_suffixes(length_encoder_suffix)
                # decoder
                decoder_output, decoder = self.decoder_step(bidirectional_emb,
                                                            feature_emb,
                                                            hypothesis.decoder,
                                                            hypothesis.alignment,
                                                            hypothesis.action_history[-1].unsqueeze(dim=0),
                                                            hypothesis.previous_output
                                                            if self.output_lookup is not None else None)
                logits = self.W(decoder_output)
                log_probs = self.log_softmax(logits, valid_actions_mask)

                for action in torch.arange(0, valid_actions_mask.size(2), device=self.device):
                    if not valid_actions_mask[0, 0, action]:
                        continue
                    n_expansions += 1
                    log_p = hypothesis.negative_log_p - \
                            log_probs[0, 0, action]  # min heap, so minus

                    heapq.heappush(expansions,
                                   Expansion(action.reshape(1, -1), decoder,
                                             hypothesis, log_p))

            beam: List[Hypothesis] = []

            for _ in range(min(search_beam_width, len(expansions))):

                expansion: Expansion = heapq.heappop(expansions)
                from_hypothesis = expansion.from_hypothesis
                action = expansion.action
                action_history = from_hypothesis.action_history
                action_history = torch.cat(
                    (action_history, action)
                )
                output = list(from_hypothesis.output)

                # execute the action to update the transducer state
                action = self.vocab.decode_action(action.item())

                if isinstance(action, EndOfSequence):
                    # 1. COMPLETE HYPOTHESIS, REDUCE BEAM
                    complete_hypothesis = Output(
                        action_history=action_history.squeeze(dim=1).cpu().tolist()[1:],
                        output=self.target_tokenizer.untokenize(output),
                        log_p=-expansion.negative_log_p.item())  # undo min heap minus

                    complete_hypotheses.append(complete_hypothesis)
                else:
                    # 2. EXECUTE ACTION AND ADD FULL HYPOTHESIS TO NEW BEAM
                    alignment = from_hypothesis.alignment.clone()

                    char_, alignment, _ = self.decode_single_action(input_, action, alignment)
                    if char_ != "":
                        output.append(char_)
                    previous_output = from_hypothesis.previous_output
                    if self.output_lookup is not None:
                        previous_output = torch.tensor(
                            [[self.output_symbol_id_for_action(
                                input_,
                                action,
                                from_hypothesis.alignment,
                            )]],
                            device=self.device,
                            dtype=torch.long,
                        )

                    hypothesis = Hypothesis(
                        action_history=action_history,
                        alignment=alignment,
                        decoder=expansion.decoder,
                        negative_log_p=expansion.negative_log_p,
                        output=output,
                        previous_output=previous_output)

                    beam.append(hypothesis)

            hypothesis_length += 1

        if not complete_hypotheses:
            # nothing found because the model is very bad
            for hypothesis in beam:

                complete_hypothesis = Output(
                    action_history=hypothesis.action_history.squeeze(dim=1).cpu().tolist()[1:],
                    output=self.target_tokenizer.untokenize(hypothesis.output),
                    log_p=-hypothesis.negative_log_p.item())  # undo min heap minus

                complete_hypotheses.append(complete_hypothesis)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            mean_active = sum(beam_sizes) / len(beam_sizes) if beam_sizes else 0.
            logging.debug(
                "Beam stats: requested_width=%d max_active=%d "
                "decoder_calls=%d expansions=%d steps=%d completed=%d "
                "mean_active=%.2f",
                beam_width,
                max_active_beam,
                n_decoder_calls,
                n_expansions,
                hypothesis_length,
                len(complete_hypotheses),
                mean_active,
            )

        complete_hypotheses.sort(reverse=True)
        return complete_hypotheses[:num_outputs_needed]
