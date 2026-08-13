"""Optimizer classes and lr scheduler used in training."""

import argparse
import inspect
from trans import register_component

import torch


@register_component('adam', 'optimizer')
class Adam(torch.optim.Adam):
    """Adam optimizer."""
    def __init__(self, params, args: argparse.Namespace):
        super().__init__(
            params,
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--lr", type=float, default=0.001,
                            help="Learning rate.")
        parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999),
                            help="Adam beta coefficients.")
        parser.add_argument("--eps", type=float, default=1e-08,
                            help="Adam epsilon.")
        parser.add_argument("--weight-decay", type=float, default=0,
                            help="Weight decay.")
        parser.add_argument("--amsgrad", action=argparse.BooleanOptionalAction, default=False,
                            help="Use the AMSGrad Adam variant.")


@register_component('adamw', 'optimizer')
class AdamW(torch.optim.AdamW):
    """AdamW optimizer."""
    def __init__(self, params, args: argparse.Namespace):
        super().__init__(
            params,
            lr=args.lr,
            betas=args.betas,
            eps=args.opt_eps,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--lr", type=float, default=0.001,
                            help="Learning rate.")
        parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999),
                            help="AdamW beta coefficients.")
        parser.add_argument("--opt-eps", type=float, default=1e-08,
                            help="AdamW epsilon.")
        parser.add_argument("--weight-decay", type=float, default=0,
                            help="Weight decay.")
        parser.add_argument("--amsgrad", action=argparse.BooleanOptionalAction, default=False,
                            help="Use the AMSGrad AdamW variant.")


@register_component('adadelta', 'optimizer')
class Adadelta(torch.optim.Adadelta):
    """Adadelta optimizer."""
    def __init__(self, params, args: argparse.Namespace):
        super().__init__(
            params,
            lr=args.lr,
            rho=args.rho,
            eps=args.opt_eps,
            weight_decay=args.weight_decay
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--lr", type=float, default=1.0,
                            help="Learning rate.")
        parser.add_argument("--rho", type=float, default=0.9,
                            help="Adadelta coefficient for running average of squared gradients.")
        parser.add_argument("--opt-eps", type=float, default=1e-06,
                            help="Adadelta epsilon.")
        parser.add_argument("--weight-decay", type=float, default=0.,
                            help="Weight decay.")


@register_component('inv_sr', 'lr_scheduler')
class WarmupInverseSquareRootSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup and then inverse square root decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Inverse square root decreases learning rate from 1. to 0. over remaining steps.
    """

    def __init__(self, optimizer: torch.optim, args: argparse.Namespace):
        self.warmup_steps = args.warmup_steps
        self.decay_factor = args.warmup_steps**0.5
        self.type = 'step'
        super().__init__(
            optimizer, self.lr_lambda, last_epoch=args.last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return self.decay_factor * step**-0.5

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--warmup-steps", type=int, default=20,
                            help="Number of linear warmup steps.")
        parser.add_argument("--last-epoch", type=int, default=-1,
                            help="Index of the last epoch for scheduler state.")


@register_component('reduce_on_plateau', 'lr_scheduler')
class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """Scheduler for reducing learning rate on plateau."""
    def __init__(self, optimizer: torch.optim, args: argparse.Namespace):
        self.type = 'metric'
        kwargs = {
            "optimizer": optimizer,
            "mode": 'max',
            "factor": args.factor,
            "patience": args.lrs_patience,
            "threshold": args.threshold,
            "threshold_mode": args.threshold_mode,
            "cooldown": args.cooldown,
            "min_lr": args.min_lr,
            "eps": args.lrs_eps,
            "verbose": args.verbose,
        }
        supported = inspect.signature(
            torch.optim.lr_scheduler.ReduceLROnPlateau.__init__
        ).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in supported}
        super().__init__(**kwargs)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--factor", type=float, default=0.5,
                            help="Factor by which to reduce the learning rate.")
        parser.add_argument("--lrs-patience", type=int, default=2,
                            help="Number of unimproved epochs before reducing learning rate.")
        parser.add_argument("--threshold", type=float, default=1e-4,
                            help="Threshold for measuring metric improvement.")
        parser.add_argument("--threshold-mode", type=str, default='rel',
                            help="Threshold mode for plateau detection.")
        parser.add_argument("--cooldown", type=int, default=0,
                            help="Cooldown epochs after reducing learning rate.")
        parser.add_argument("--min-lr", type=float, default=0.,
                            help="Lower bound for learning rate.")
        parser.add_argument("--lrs-eps", type=float, default=1e-8,
                            help="Minimum learning-rate change.")
        parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False,
                            help="Print scheduler messages when supported by installed PyTorch.")
