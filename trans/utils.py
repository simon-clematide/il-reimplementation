"""Utility functions and classes."""
from typing import Any, Dict, Iterable, List, Optional, TextIO, Union
import dataclasses
import logging
import os
import time
import re
import unicodedata
import torch
import pickle
from trans.vocabulary import PAD


class Tokenizer:
    """Tokenize text as characters or by a literal separator."""

    NONE_VALUE = "none"

    def __init__(self, separator: Optional[str] = None):
        self.separator = separator

    @classmethod
    def from_cli(cls, separator: Optional[str]) -> "Tokenizer":
        if separator in (None, cls.NONE_VALUE):
            return cls(None)
        return cls(separator)

    def tokenize(self, text: str) -> List[str]:
        if self.separator is None:
            return list(text)
        tokens = text.split(self.separator)
        if any(token == "" for token in tokens):
            raise ValueError(
                f"Empty token in {text!r} with separator {self.separator!r}.")
        return tokens

    def untokenize(self, tokens: Iterable[str]) -> str:
        if self.separator is None:
            return "".join(tokens)
        return self.separator.join(tokens)


@dataclasses.dataclass
class Sample:
    input: str
    target: Optional[str]
    encoded_input: Optional[torch.tensor] = None
    action_history: Optional[torch.tensor] = None
    output_history: Optional[torch.tensor] = None
    alignment_history: Optional[torch.tensor] = None
    expert_action_costs: Optional[torch.tensor] = None
    optimal_actions_mask: Optional[torch.tensor] = None
    valid_actions_mask: Optional[torch.tensor] = None
    features: Optional[str] = None
    encoded_features: Optional[torch.tensor] = None

    _tensor_attrs = (
        "encoded_input",
        "action_history",
        "output_history",
        "alignment_history",
        "expert_action_costs",
        "optimal_actions_mask",
        "valid_actions_mask",
        "encoded_features",
    )

    def to(self, device: str = 'cpu') -> "Sample":
        for attr in self._tensor_attrs:
            attr_val = getattr(self, attr)
            if torch.is_tensor(attr_val):
                setattr(self, attr, attr_val.to(device))
        return self


@dataclasses.dataclass
class TrainingBatch:
    encoded_input: torch.tensor
    action_history: torch.tensor
    output_history: Optional[torch.tensor]
    alignment_history: torch.tensor
    expert_action_costs: Optional[torch.tensor]
    optimal_actions_mask: torch.tensor
    valid_actions_mask: torch.tensor
    encoded_features: Optional[torch.tensor] = None


@dataclasses.dataclass
class EvalBatch:
    input: List[str]
    target: Optional[List[str]]
    encoded_input: torch.tensor
    features: Optional[List[str]] = None
    encoded_features: Optional[torch.tensor] = None


class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples: Optional[List[Sample]] = None):
        self.samples = samples if samples else []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, id_) -> Sample:
        return self.samples[id_]

    def add_samples(self, samples: Union[List[Sample], Sample]):
        if isinstance(samples, list):
            self.samples.extend(samples)
        else:
            self.samples.append(samples)

    def get_data_loader(self, is_training: bool = False, device: str = 'cpu', **kwargs):
        if 'collate_fn' not in kwargs:
            if 'pad_index' not in kwargs:
                pad_index = PAD

            target_device = torch.device(device)

            def cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.to("cpu")

            def to_target_device(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if tensor is None:
                    return None
                return tensor.to(target_device)

            def pad_sequence_on_cpu(
                    sequences: List[torch.Tensor],
                    **pad_kwargs) -> torch.Tensor:
                return torch.nn.utils.rnn.pad_sequence(
                    [cpu_tensor(s) for s in sequences],
                    **pad_kwargs,
                )

            def nonnegative_cpu_tensor(tensor: torch.Tensor,
                                       replacement: int) -> torch.Tensor:
                tensor = cpu_tensor(tensor)
                return tensor.masked_fill(tensor < 0, replacement)

            def batch_encode_features(batch: List[Sample]):
                """Creates a padded batch of encoded features."""
                if batch[0].encoded_features is None:
                    encoded_features = None
                else:
                    features_ = [list(cpu_tensor(s.encoded_features)) for s in batch]
                    max_len = len(max(features_, key=len))
                    pad = torch.tensor(pad_index)
                    padded_features = [
                        f + [pad] * (max_len - len(f))
                        for f in features_
                    ]
                    encoded_features = torch.tensor(padded_features)
                return to_target_device(encoded_features)

            if is_training:
                def collate(batch: List[Sample]):
                    return TrainingBatch(
                        to_target_device(pad_sequence_on_cpu(
                            [s.encoded_input for s in batch],
                            batch_first=True,
                            padding_value=pad_index)),
                        to_target_device(pad_sequence_on_cpu(
                            [nonnegative_cpu_tensor(s.action_history, PAD) for s in batch],
                            padding_value=PAD)),
                        to_target_device(pad_sequence_on_cpu(
                            [nonnegative_cpu_tensor(s.output_history, PAD) for s in batch],
                            padding_value=PAD,
                        )) if batch[0].output_history is not None else None,
                        to_target_device(pad_sequence_on_cpu(
                            [nonnegative_cpu_tensor(s.alignment_history, 0) for s in batch],
                            batch_first=True,
                            padding_value=0).view(-1)),
                        to_target_device(pad_sequence_on_cpu(
                            [s.expert_action_costs for s in batch],
                            padding_value=float("inf"),
                        )) if batch[0].expert_action_costs is not None else None,
                        to_target_device(pad_sequence_on_cpu(
                            [s.optimal_actions_mask for s in batch],
                            padding_value=False)),
                        to_target_device(pad_sequence_on_cpu(
                            [s.valid_actions_mask for s in batch],
                            padding_value=False)),
                        encoded_features=batch_encode_features(batch),
                    )
            else:
                def collate(batch: List[Sample]):
                    return EvalBatch([s.input for s in batch],
                                     [s.target for s in batch],
                                     to_target_device(pad_sequence_on_cpu(
                                         [s.encoded_input for s in batch],
                                         batch_first=True,
                                         padding_value=pad_index)),
                                     features=[s.features for s in batch],
                                     encoded_features=batch_encode_features(batch),
                                     )

            kwargs['collate_fn'] = collate

        return torch.utils.data.DataLoader(self, **kwargs)

    def to(self, device: str = 'cpu'):
        for s in self.samples:
            s.to(device)

    def persist(self, filename: str):
        with open(filename, mode="wb") as w:
            pickle.dump(self.samples, w)
        logging.info("Wrote precomputed training data to %s.", filename)

    @classmethod
    def from_pickle(cls, path2pkl: str, device: str = 'cpu'):
        logging.info("Loading precomputed training data from file: %s", path2pkl)
        with open(path2pkl, "rb") as w:
            params: List[Sample] = pickle.load(w)

        instance = cls(params)
        instance.to(device)
        return instance


@dataclasses.dataclass
class DecodingOutput:
    string_accuracy: float
    symbol_accuracy: float
    loss: float
    predictions: List[str]

    @property
    def accuracy(self) -> float:
        return self.string_accuracy


class OpenNormalize:

    def __init__(self, filename: str, normalize: bool, mode: str = "rt"):
        self.filename = filename
        self.file: Optional[TextIO] = None
        mode_pattern = re.compile(r"[arw]t?$")
        if not mode_pattern.match(mode):
            raise ValueError(f"Unexpected mode {mode_pattern.pattern}: {mode}.")
        self.mode = mode
        if normalize:
            form = "NFD" if self.mode.startswith("r") else "NFC"
            self.normalize = lambda line: unicodedata.normalize(form, line)
        else:
            self.normalize = lambda line: line

    def __enter__(self):
        self.file = open(self.filename, mode=self.mode, encoding="utf8")
        return self

    def __iter__(self):
        for line in self.file:
            yield self.normalize(line)

    def write(self, line: str):
        if not isinstance(line, str):
            raise ValueError(
                f"Line is not a unicode string ({type(line)}): {line}")
        return self.file.write(self.normalize(line))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


def write_results(string_accuracy: float, predictions: List[str], output: str,
                  normalize: bool, dataset_name: str, beam_width: int = 1,
                  symbol_accuracy: Optional[float] = None,
                  decoding_name: Optional[str] = None,
                  dargs: Dict[str, Any] = None):
    logging.info("%s set string accuracy: %.4f.", dataset_name.title(), string_accuracy)
    if symbol_accuracy is not None:
        logging.info("%s set symbol accuracy: %.4f.", dataset_name.title(), symbol_accuracy)

    if decoding_name is None:
        decoding_name = "greedy" if beam_width == 1 else f"beam{beam_width}"

    eval_file = os.path.join(output, f"{dataset_name}_{decoding_name}.eval")

    with open(eval_file, mode="w") as w:
        if dargs is not None:
            for key, value in dargs.items():
                w.write(f"{key}: {value}\n")
        if symbol_accuracy is not None:
            w.write(f"{dataset_name} symbol accuracy: {symbol_accuracy:.4f}\n")
        w.write(f"{dataset_name} string accuracy: {string_accuracy:.4f}\n")

    predictions_tsv = os.path.join(
        output, f"{dataset_name}_{decoding_name}.predictions")

    with OpenNormalize(predictions_tsv, normalize, mode="w") as w:
        w.write("\n".join(predictions))


class Timer:

    def __init__(self):
        self.time = None

    def __enter__(self):
        self.time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.info("\t...finished in %.3f sec.", time.time() - self.time)
