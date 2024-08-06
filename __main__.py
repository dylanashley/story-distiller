#!/usr/bin/env python
# -*- coding: ascii -*-

from album_extractor import LSTMAudioFeatureEncoder, Mean
from copy import copy
from features import compute_features, get_duration
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple
import argparse
import collections
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import scipy.interpolate
import seaborn as sns
import sys
import tempfile
import torch
import warnings
import zipfile

__version__ = "1.3.0"


def build_template(xy: List[Tuple[float, float]]) -> Callable[[float], float]:
    """Returns a callable narrative template curve."""
    x, y = zip(*xy)
    min_y = min(y)
    max_y = max(y)
    y = list(y)
    if min_y != max_y:
        for i in range(len(y)):
            y[i] = scale(y[i], min_y, max_y, 0, 1)
    if len(x) == 1:
        kind = "nearest"
    elif len(x) == 2:
        kind = "linear"
    elif len(x) == 3:
        kind = "quadratic"
    else:
        kind = "cubic"
    return scipy.interpolate.interp1d(x, y, kind=kind)


def default_template(time: float, order: int = 2) -> float:
    """Returns the value of a narrative template curve for a time in the
    range [0, 1].
    """
    assert 0 <= time <= 1
    assert order in [1, 2]
    if order == 1:
        if time <= 0.2:
            return 1 / 2 + 5 / 4 * time
        elif time <= 0.5:
            return 5 / 4 - 5 / 2 * time
        elif time <= 0.8:
            return -5 / 3 + 10 / 3 * time
        else:
            return 2 - 5 / 4 * time
    else:
        if time <= 0.2:
            return 1 / 2 + 5 / 2 * time - 25 / 4 * time**2
        elif time <= 0.3:
            return -1 / 4 + 10 * time - 25 * time**2
        elif time <= 0.5:
            return 25 / 8 - 25 / 2 * time + 25 / 2 * time**2
        elif time <= 0.65:
            return 50 / 9 - 200 / 9 * time + 200 / 9 * time**2
        elif time <= 0.8:
            return -119 / 9 + 320 / 9 * time - 200 / 9 * time**2
        else:
            return -3 + 10 * time - 25 / 4 * time**2


def fit_values(
    values: Dict[str, float],
    template: Callable[[float], float] = default_template,
) -> List[str]:
    """Fits a set of values to a narrative template curve."""
    values = collections.OrderedDict(values)
    filenames = list(values.keys())
    rv = [-1 for _ in range(len(values))]
    distances = np.zeros((len(values), len(values)), dtype=float)
    for i, y in enumerate(values.values()):
        for j in range(len(values)):
            x = scale(j, 0, len(values), 0, 1)
            distances[i, j] = np.square(y - template(x))

    # binary search to find smallest deviation matching
    candidates = np.sort(distances.flatten())
    min_idx = 0
    max_idx = len(candidates) - 1
    while min_idx != max_idx:
        pivot = min_idx + (max_idx - min_idx) // 2
        edges = [
            (filenames[i], j)
            for i in range(len(values))
            for j in range(len(values))
            if distances[i, j] <= candidates[pivot]
        ]
        graph = nx.Graph(edges)
        try:
            matching = nx.bipartite.maximum_matching(graph)
            if all([filename in matching for filename in filenames]):
                max_idx = pivot
            else:
                min_idx = pivot + 1
        except nx.AmbiguousSolution:
            min_idx = pivot + 1

    # now minimize the average devation as well
    edges = [
        (filenames[i], j, {"weight": distances[i, j]})
        for i in range(len(values))
        for j in range(len(values))
        if distances[i, j] <= candidates[max_idx]
    ]
    graph = nx.Graph(edges)
    matching = nx.bipartite.minimum_weight_full_matching(graph)
    return [matching[i] for i in range(len(filenames))]


def get_value(filename: str, encoder: torch.nn.Module) -> float:
    """Returns the narrative essence for an audio file."""
    a = torch.from_numpy(compute_features(filename).to_numpy()).float()
    durations_mean = 257.6875
    durations_std = 191.8393
    duration = get_duration(filename)
    duration = (duration - 257.6875) / 191.8393
    duration = torch.FloatTensor([duration])
    features = torch.cat([a, duration.repeat(7)]).unsqueeze(0)
    value = encoder(features)
    return value.item()


def main(args: Dict[str, Any]) -> None:
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
    playlist = fit_values(values, template=template)
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


def parse_args(args: List[str] = sys.argv[1:]) -> Dict[str, Any]:
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


def scale(
    value: float, start_min: float, start_max: float, end_min: float, end_max: float
) -> float:
    """Returns the result of scaling value from the range
    [start_min, start_max] to [end_min, end_max].
    """
    return end_min + (end_max - end_min) * (value - start_min) / (start_max - start_min)


if __name__ == "__main__":
    args = parse_args()
    warnings.filterwarnings("ignore")  # because we're in production
    main(args)
