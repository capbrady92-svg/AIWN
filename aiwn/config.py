"""
Shared configuration utilities — device resolution and GPU info printing.
Kept separate so experiments and run.py can both import without circular deps.
"""

import argparse
import torch


def resolve_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def print_device_info(device: torch.device):
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        L2 = (getattr(props, 'l2_cache_size', None) or
              getattr(props, 'L2_cache_size', 0)) / 1e6
        print(f"GPU   : {props.name}  "
              f"VRAM: {props.total_memory/1e9:.1f}GB  "
              f"L2: {L2:.0f}MB")
    from aiwn.layers.indexed_linear import TRITON_OK
    print(f"Triton: {'available ✓' if TRITON_OK else 'unavailable — eager fallback'}")


def add_common_args(parser: argparse.ArgumentParser):
    """Args shared across all experiments."""
    parser.add_argument('--device', default='auto',
                        help='cuda | cpu | auto (default: auto)')
    parser.add_argument('--out_dir', default='.',
                        help='Directory for output files (default: cwd)')