"""Shared utilities for benchmark scripts: CLI args, config matrix, labels."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sbw.prf import seeding_scheme_lookup

MIN_CONTEXT_WIDTH = {
    "minskipgram_prf": 2,
    "simple_skip_prf": 2,
    "anchored_skipgram_prf": 2,
    "anchored_minhash_prf": 2,
}


def add_benchmark_args(parser):
    """Add common watermark sweep arguments to an argparse parser."""
    parser.add_argument("--schemes", type=str, default="simple_1",
                        help="Comma-separated seeding schemes (default: simple_1)")
    parser.add_argument("--context-widths", type=str, default=None,
                        help="Comma-separated context widths; overrides scheme defaults via ff- syntax")
    parser.add_argument("--gammas", type=str, default="0.5",
                        help="Comma-separated gamma values (default: 0.5)")
    parser.add_argument("--deltas", type=str, default="2.0",
                        help="Comma-separated delta values (default: 2.0)")
    parser.add_argument("--hash-keys", type=str, default="15485863",
                        help="Comma-separated hash keys (default: 15485863)")


def parse_benchmark_args(args):
    """Parse comma-separated CLI args and build config matrix."""
    schemes = [s.strip() for s in args.schemes.split(",")]
    context_widths = [int(x) for x in args.context_widths.split(",")] if args.context_widths else None
    gammas = [float(x) for x in args.gammas.split(",")]
    deltas = [float(x) for x in args.deltas.split(",")]
    hash_keys = [int(x) for x in args.hash_keys.split(",")]
    return build_config_matrix(schemes, context_widths, gammas, deltas, hash_keys)


def build_config_matrix(schemes, context_widths, gammas, deltas, hash_keys):
    """Build list of watermark configs from parameter lists.

    Returns list of dicts with keys: seeding_scheme, gamma, delta, hash_key, label
    """
    resolved_schemes = []

    for scheme in schemes:
        if context_widths is None:
            resolved_schemes.append(scheme)
        else:
            prf_type, _, self_salt = seeding_scheme_lookup(scheme)
            for cw in context_widths:
                min_cw = MIN_CONTEXT_WIDTH.get(prf_type, 1)
                if cw < min_cw:
                    print(f"WARNING: Skipping {prf_type} with context_width={cw} (minimum: {min_cw})")
                    continue
                ff = f"ff-{prf_type}-{cw}-{self_salt}"
                resolved_schemes.append(ff)

    configs = []
    for scheme in resolved_schemes:
        for gamma in gammas:
            for delta in deltas:
                for hash_key in hash_keys:
                    cfg = {
                        "seeding_scheme": scheme,
                        "gamma": gamma,
                        "delta": delta,
                        "hash_key": hash_key,
                    }
                    cfg["label"] = config_label(cfg)
                    configs.append(cfg)
    return configs


def config_label(cfg):
    """Generate a short human-readable label, omitting default values."""
    scheme = cfg["seeding_scheme"]

    # For freeform, extract a readable name
    if scheme.startswith("ff-"):
        parts = scheme.split("-")
        prf_short = parts[1].replace("_prf", "")
        label = f"{prf_short}_cw{parts[2]}"
        if parts[3] == "True":
            label += "_ss"
    else:
        label = scheme

    suffixes = []
    if cfg["gamma"] != 0.5:
        suffixes.append(f"g{cfg['gamma']}")
    if cfg["delta"] != 2.0:
        suffixes.append(f"d{cfg['delta']}")
    if cfg["hash_key"] != 15485863:
        suffixes.append(f"h{cfg['hash_key']}")

    if suffixes:
        label += "_" + "_".join(suffixes)
    return label
