#!/usr/bin/env python
# -*- coding: ascii -*-

import argparse
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import sys
import tempfile
import torch
from tqdm import tqdm
from typing import Any
from utils import build_template, default_template, fit_values, get_value, scale
import warnings
import zipfile

__version__ = "1.4.0"


def main(args: dict[str, Any]) -> None:
    if len(args["files"]) == 1:
        print(args["files"][0])
        return
    if args["template"]:
        templates = np.load("templates.npz")
        template = build_template(
            list(zip(templates["x"], templates["y"][args["template"]]))
        )
    elif args["xy"]:
        template = build_template(args["xy"])
    else:
        template = default_template
    values = dict()
    with tempfile.TemporaryDirectory() as tempdir:
        if zipfile.is_zipfile(os.path.dirname(__file__)):
            encoder = torch.load(
                zipfile.ZipFile(os.path.dirname(__file__)).extract(
                    "album_feature_encoder.pt", path=tempdir
                )
            )
        else:
            encoder = torch.load("album_feature_encoder.pt")
    for i, filename in enumerate(
        tqdm(args["files"], ascii=True, desc="Extracting Narrative Essence")
    ):
        values[filename] = get_value(filename, encoder)
    min_value = min(values.values())
    max_value = max(values.values())
    for k, v in values.items():
        values[k] = scale(v, min_value, max_value, 0, 1)
    playlist, _ = fit_values(values, template=template)
    if args["outfile"]:
        with open(args["outfile"], "w") as outfile:
            for song in playlist:
                print(song, file=outfile)
    else:
        for song in playlist:
            print(song)
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    plt.subplots(1, 1, figsize=(6, 6))
    x = np.linspace(0, 1, num=1000)
    plt.plot(
        [scale(i, 0, 1, 1, len(playlist)) for i in x],
        [scale(template(i), 0, 1, min_value, max_value) for i in x],
    )
    plt.plot(
        np.arange(len(playlist)) + 1,
        [scale(values[song], 0, 1, min_value, max_value) for song in playlist],
        "o-",
    )
    plt.xticks(np.arange(len(playlist)) + 1, playlist, rotation=270)
    plt.ylabel("Normalized Narrative Essence", labelpad=5)
    plt.tight_layout()
    plt.show()


def parse_args(args: list[str] = sys.argv[1:]) -> dict[str, Any]:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="embeds a story into a music playlist by sorting "
        "the playlist so that the order of the music follows a "
        "narrative arc",
        prog="sdistil",
    )
    parser.add_argument(
        "files",
        help="individual audio files that make up the playlist",
        nargs="+",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="file to write fitting song ordering to",
    )
    parser.add_argument(
        "-t",
        "--template",
        help="index of the the narrative template curves to use for "
        "fitting the playlist; cannot be used in conjunction with --xy",
        type=int,
    )
    parser.add_argument(
        "--xy",
        help="comma-separated list of (x, y) tuples that form the "
        "narrative template curve to use for fitting the playlist; "
        "all values are assumed to be in the range [0, 1] with no "
        "duplicate x values; cannot be used in conjunction with -t or "
        "--template",
        type=str,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {}".format(__version__),
    )
    args = vars(parser.parse_args(args))
    for filename in args["files"]:
        assert os.path.isfile(filename)
    if args["outfile"]:
        assert not os.path.isfile(args["outfile"])
    if args["template"]:
        assert 0 <= args["template"] <= 21
    assert not (args["template"] and args["xy"])
    if args["xy"]:
        try:
            original = copy(args["xy"])
            args["xy"] = args["xy"].split("),(")
            assert len(args["xy"]) > 0
            for i in range(len(args["xy"])):
                xy = args["xy"][i].strip("(").strip(")")
                xy = [v.strip() for v in xy.split(",")]
                assert len(xy) == 2
                x = float(xy[0])
                assert 0 <= x <= 1
                y = float(xy[1])
                assert 0 <= y <= 1
                args["xy"][i] = (x, y)
            assert len(args["xy"]) == len(set([v[0] for v in args["xy"]]))
            args["xy"].sort(key=lambda v: v[0])
        except (AssertionError, ValueError):
            raise ValueError(
                'expected xy to be something like "(0.1,0.5),(0.4,0.2),(0.7,0.8)" but '
                'instead got "{}"'.format(original)
            )
    return args


if __name__ == "__main__":
    args = parse_args()
    warnings.filterwarnings("ignore")  # because we're in production
    main(args)
