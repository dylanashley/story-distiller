#!/usr/bin/env python
# -*- coding: ascii -*-

from pathlib import Path
from sklearn.decomposition import PCA
from types import SimpleNamespace
import json
import lzma
import numpy as np
import os
import pickle
import random
import torch
import urllib


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(int(seed))
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


class AudioFeatureDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mode="train",
        normalize_features=True,
        include_learned_feature=False,
    ):
        super().__init__()
        self.data_dir = os.path.join(Path(__file__).parent, "data")
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.include_learned_feature = include_learned_feature
        self.dataset = self.assemble_dataset()

        self.audio_features_mean = self.dataset["audio_features"].mean(dim=0)
        self.audio_features_std = self.dataset["audio_features"].std(dim=0)
        self.audio_features_std[self.audio_features_std == 0.0] = 1.0

        self.durations_mean = self.dataset["durations"].float().mean(dim=0)
        self.durations_std = self.dataset["durations"].float().std(dim=0)

        if normalize_features:
            self.dataset["audio_features"] = (
                self.dataset["audio_features"] - self.audio_features_mean
            ) / self.audio_features_std
            self.dataset["durations"] = (
                self.dataset["durations"].float() - self.durations_mean
            ) / self.durations_std

        album_indices = self.dataset["album_indices"]
        split = self.dataset["split"]

        if mode == "validation":
            split_mode = 1
        elif mode == "test":
            split_mode = 2
        else:
            split_mode = 0

        album_in_dataset = split[album_indices] == split_mode
        if mode == "full":
            album_in_dataset = torch.ones_like(album_in_dataset)
        self.album_indices = album_indices[album_in_dataset]
        self.album_lengths = self.dataset["album_lengths"][album_in_dataset]
        self.album_ids = self.dataset["album_ids"][album_in_dataset]

        self.audio_features = self.dataset["audio_features"]
        self.durations = self.dataset["durations"]
        self.track_numbers = self.dataset["track_numbers"]

        self.pca = PCA(n_components=1)
        self.pca.fit(
            torch.cat([self.audio_features, self.durations.unsqueeze(1)], dim=1)
        )
        torch.save(torch.Tensor(self.pca.components_.T), "pca_audio_features_matrix.pt")
        self.pca_matrix = torch.load("pca_audio_features_matrix.pt", weights_only=False)

        transform_dict = {
            "duration_mean": self.durations_mean,
            "duration_std": self.durations_std,
            "audio_features_mean": self.audio_features_mean,
            "audio_features_std": self.audio_features_std,
            "pca_matrix": self.pca_matrix,
        }
        torch.save(transform_dict, "audio_transform_dict.pt")

    def __getitem__(self, item):
        album_idx = self.album_indices[item]
        l = self.album_lengths[item]
        a = self.audio_features[album_idx : album_idx + l]
        d = self.durations[album_idx : album_idx + l]
        n = self.track_numbers[album_idx : album_idx + l]
        pca_features = torch.matmul(
            torch.cat([a, d.unsqueeze(1)], dim=1), self.pca_matrix
        )

        features = torch.cat([a, d.unsqueeze(1).repeat(1, 7)], dim=1)

        if self.include_learned_feature:
            album_id = self.album_ids[item].item()
            learned_features = self.learned_feature_dict[album_id]
            return features, n, learned_features.unsqueeze(1)
        else:
            return features, n

    def __len__(self):
        return len(self.album_indices)

    def assemble_dataset(self):
        loaded_data = []
        for i in range(1, 5):
            with lzma.open(f"data/fma_albums_{i}.xz") as infile:
                d = pickle.load(infile)
            loaded_data += d

        filtered_dataset_list = []
        for album in loaded_data:
            track_numbers = [t["track number"] for t in album]
            if track_numbers[0] == 0:
                track_numbers = [n + 1 for n in track_numbers]
                for t in album:
                    t["track number"] += 1
            duplicates = [x for x in track_numbers if track_numbers.count(x) > 1]
            if len(duplicates) > 0:
                continue
            num_album_tracks = album[0]["album tracks"]
            if len(track_numbers) != num_album_tracks:
                continue
            if track_numbers[-1] != num_album_tracks:
                continue
            filtered_dataset_list.append(album)

        audio_features = []
        durations = []
        track_numbers = []
        split = []
        album_indices = []
        album_lengths = []
        album_ids = []

        for album in filtered_dataset_list:
            album_indices.append(len(durations))
            album_lengths.append(len(album))
            for track in album:
                audio_features.append(torch.Tensor(track["audio features"]))
                durations.append(track["track duration"])
                track_numbers.append(track["track number"])
                split.append(track["set split"])
            album_ids.append(track["album id"])

        audio_features = torch.stack(audio_features, dim=0)
        durations = torch.LongTensor(durations)
        track_numbers = torch.LongTensor(track_numbers)
        split_dict = {"training": 0, "validation": 1, "test": 2}
        split = torch.Tensor([split_dict[s] for s in split]).long()
        album_indices = torch.LongTensor(album_indices)
        album_lengths = torch.LongTensor(album_lengths)
        album_ids = torch.LongTensor(album_ids)

        dataset_list = {
            "audio_features": audio_features,
            "durations": durations,
            "split": split,
            "track_numbers": track_numbers,
            "album_indices": album_indices,
            "album_lengths": album_lengths,
            "album_ids": album_ids,
        }

        return dataset_list


class AudioFeatureDatasetEchonest(AudioFeatureDataset):
    def __init__(
        self,
        mode="train",
        normalize_features=True,
        include_learned_feature=False,
    ):

        super().__init__(
            mode=mode,
            normalize_features=normalize_features,
            include_learned_feature=include_learned_feature,
        )

        self.echonest_features_mean = self.dataset["echonest_features"].mean(dim=0)
        self.echonest_features_std = self.dataset["echonest_features"].std(dim=0)
        self.echonest_features_std[self.echonest_features_std == 0.0] = 1.0

        if normalize_features:
            self.dataset["echonest_features"] = (
                self.dataset["echonest_features"] - self.echonest_features_mean
            ) / self.echonest_features_std

        self.echonest_features = self.dataset["echonest_features"]

        self.echonest_pca = PCA(n_components=1)
        self.echonest_pca.fit(
            torch.cat([self.echonest_features, self.durations.unsqueeze(1)], dim=1)
        )

        transform_dict = {
            "echonest_features_mean": self.echonest_features_mean,
            "echonest_features_std": self.echonest_features_std,
            "echonest_pca": self.echonest_pca.components_.T,
        }
        torch.save(transform_dict, "audio_echonest_transform_dict.pt")

    def __getitem__(self, item):
        album_idx = self.album_indices[item]
        l = self.album_lengths[item]
        a = self.audio_features[album_idx : album_idx + l]
        d = self.durations[album_idx : album_idx + l]
        n = self.track_numbers[album_idx : album_idx + l]
        e = self.echonest_features[album_idx : album_idx + l]

        echonest_features_pca = torch.from_numpy(
            self.echonest_pca.transform(torch.cat([e, d.unsqueeze(1)], dim=1))
        ).float()
        audio_features_pca = torch.matmul(
            torch.cat([a, d.unsqueeze(1)], dim=1), self.pca_matrix
        )

        features = torch.cat(
            [e, d.unsqueeze(1), echonest_features_pca, audio_features_pca], dim=1
        )

        if self.include_learned_feature:
            album_id = self.album_ids[item].item()
            learned_features = self.learned_feature_dict[album_id]
            return features, n, learned_features.unsqueeze(1)
        else:
            return features, n

    def assemble_dataset(self):
        with lzma.open(f"data/fma_albums_echonest_subset.xz") as infile:
            loaded_data = pickle.load(infile)

        if self.include_learned_feature:
            self.learned_feature_dict = {}
            json_path = os.path.join(self.data_dir, "fma_albums_learned_feature.json")
            with open(json_path, "rb") as f:
                albums_with_learned_features = json.load(f)
            for album in albums_with_learned_features:
                learned_features = torch.Tensor(
                    [t["learned scalar feature"] for t in album]
                )
                self.learned_feature_dict[album[0]["album id"]] = learned_features

        filtered_dataset_list = []
        for album in loaded_data:
            track_numbers = [t["track number"] for t in album]
            if track_numbers[0] == 0:
                track_numbers = [n + 1 for n in track_numbers]
                for t in album:
                    t["track number"] += 1
            duplicates = [x for x in track_numbers if track_numbers.count(x) > 1]
            if len(duplicates) > 0:
                continue
            if album[0]["album id"] == 284:
                # for some reason this does not exist in the full dataset
                continue
            num_album_tracks = album[0]["album tracks"]
            if len(track_numbers) != num_album_tracks:
                continue
            if track_numbers[-1] != num_album_tracks:
                continue
            filtered_dataset_list.append(album)

        audio_features = []
        echonest_features = []
        durations = []
        track_numbers = []
        split = []
        album_indices = []
        album_lengths = []
        album_ids = []

        for album in filtered_dataset_list:
            album_indices.append(len(durations))
            album_lengths.append(len(album))
            for track in album:
                audio_features.append(torch.Tensor(track["audio features"]))
                echonest_features.append(
                    torch.Tensor(
                        [
                            track["acousticness"],
                            track["danceability"],
                            track["energy"],
                            track["instrumentalness"],
                            track["liveness"],
                            track["speechiness"],
                            track["tempo"],
                            track["valence"],
                        ]
                    )
                )
                durations.append(track["track duration"])
                track_numbers.append(track["track number"])
                split.append(track["set split"])
            album_ids.append(track["album id"])

        audio_features = torch.stack(audio_features, dim=0)
        echonest_features = torch.stack(echonest_features, dim=0)
        durations = torch.LongTensor(durations)
        track_numbers = torch.LongTensor(track_numbers)
        split_dict = {"training": 0, "validation": 1, "test": 2}
        split = torch.Tensor([split_dict[s] for s in split]).long()
        album_indices = torch.LongTensor(album_indices)
        album_lengths = torch.LongTensor(album_lengths)
        album_ids = torch.LongTensor(album_ids)

        dataset_dict = {
            "audio_features": audio_features,
            "echonest_features": echonest_features,
            "durations": durations,
            "split": split,
            "track_numbers": track_numbers,
            "album_indices": album_indices,
            "album_lengths": album_lengths,
            "album_ids": album_ids,
        }
        return dataset_dict

        torch.save(dataset_dict, self.dataset_file)
        return torch.load(self.dataset_file)


def collate_album_features_to_packed_seqs(album_feature_list):
    # return:
    #     [track_features: PackedSequence, durations: PackedSequence,
    #      track_numbers: PackedSequence, sequence_length: LongTensor]
    n = len(album_feature_list[0])
    feature_list = [
        [album_feature_list[j][k] for j in range(len(album_feature_list))]
        for k in range(n)
    ]
    sequence_lengths = torch.LongTensor([s.shape[0] for s in feature_list[0]])
    packed_sequences = [
        torch.nn.utils.rnn.pack_sequence(seqs, enforce_sorted=False)
        for seqs in feature_list
    ]
    return packed_sequences + [sequence_lengths]


class LSTMAudioFeatureEncoder(torch.nn.Module):
    def __init__(
        self,
        num_in_features,
        hidden_size,
        num_layers,
        bidirectional=True,
        num_out_features=1,
        dropout=0.1,
    ):
        super().__init__()
        self.num_in_features = num_in_features
        self.lstm = torch.nn.LSTM(
            input_size=num_in_features,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        d = 2 if bidirectional else 1
        self.h0 = torch.nn.Parameter(torch.randn(d * num_layers, 1, hidden_size))
        self.c0 = torch.nn.Parameter(torch.randn(d * num_layers, 1, hidden_size))
        self.output_transformation = torch.nn.Linear(
            hidden_size * d * num_layers, num_out_features
        )

    def forward(self, x):
        # x: batch_size x seq_length x num_in_features
        # or batch_size x seq_length * num_in_features
        batch_size = x.shape[0]
        if len(x.shape) == 2:
            x = x.view(batch_size, -1, self.num_in_features)

        output, (h_n, c_n) = self.lstm(
            x, (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1))
        )
        encoding = h_n.transpose(0, 1).reshape(batch_size, -1)
        x = self.output_transformation(encoding)
        x = torch.sigmoid(x)
        return x


class Mean(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x.mean(dim=-1)[..., None]


class OrderingLSTMEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bidirectional=True,
        dropout=0.0,
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            bidirectional=True if bidirectional else False,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
        )
        d = 2 if bidirectional else 1
        self.input_projection = torch.nn.Linear(
            in_features=input_size, out_features=hidden_size
        )
        self.output_projection = torch.nn.Linear(
            in_features=d * hidden_size * num_layers, out_features=1
        )
        self.start_embedding = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.end_embedding = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.h0 = torch.nn.Parameter(torch.randn(d * num_layers, 1, hidden_size))
        self.c0 = torch.nn.Parameter(torch.randn(d * num_layers, 1, hidden_size))

    def forward(self, padded_seqs, seq_lengths):
        # x: padded_seq_length x batch_size x num_features
        # seq_lengths: batch_size
        batch_size = padded_seqs.shape[1]
        padded_seqs = self.input_projection(padded_seqs)
        padded_seqs = torch.nn.functional.pad(padded_seqs, [0, 0, 0, 0, 0, 1])

        # add start of sequence
        padded_seqs = torch.cat(
            [self.start_embedding.repeat(1, batch_size, 1), padded_seqs], dim=0
        )
        # add end_of sequence
        padded_seqs[seq_lengths, torch.arange(batch_size).to(padded_seqs.device)] = (
            self.end_embedding.repeat(1, batch_size, 1)
        )

        narrative_features_packed_list = torch.nn.utils.rnn.pack_padded_sequence(
            padded_seqs,
            lengths=seq_lengths + 2,
            enforce_sorted=False,
            batch_first=False,
        )

        output, (h_n, c_n) = self.lstm(
            narrative_features_packed_list,
            (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1)),
        )
        encoding = h_n.transpose(0, 1).reshape(batch_size, -1)
        score = self.output_projection(encoding)
        return score, encoding


def train_model(args):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.seed >= 0:
        seed_everything(args.seed)
    num_negative_examples = args.num_negative_examples

    use_learned_feature = (
        args.available_feature_to_use == "learned" and args.use_available_feature
    )
    feature_list = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "speechiness",
        "tempo",
        "valence",
        "duration",
        "echonest_features_pca",
        "fma_audio_features_pca",
        "learned",
    ]
    feature_index = feature_list.index(args.available_feature_to_use)

    dataset_class = (
        AudioFeatureDatasetEchonest if args.small_dataset else AudioFeatureDataset
    )

    training_set = dataset_class(
        mode="training", include_learned_feature=use_learned_feature
    )
    training_loader = torch.utils.data.DataLoader(
        dataset=training_set,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_album_features_to_packed_seqs,
    )

    validation_set = dataset_class(
        mode="validation", include_learned_feature=use_learned_feature
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_set,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_album_features_to_packed_seqs,
    )

    num_features = 9 if args.small_dataset else 525
    if args.use_available_feature:
        assert args.small_dataset
        assert args.feature_encoder_type == "mean"
        num_features = 1

    if args.feature_encoder_type == "mean":
        feature_encoder = Mean()
    elif args.feature_encoder_type == "lstm":
        feature_encoder = LSTMAudioFeatureEncoder(
            num_in_features=7,
            hidden_size=args.feature_encoder_hidden_units,
            num_layers=args.feature_encoder_num_layers,
            bidirectional=True,
            dropout=args.feature_encoder_dropout,
            num_out_features=args.num_encoding_features,
        )
    feature_encoder = feature_encoder.to(dev)

    ordering_encoder = OrderingLSTMEncoder(
        input_size=args.num_encoding_features,
        hidden_size=args.ordering_encoder_hidden_units,
        num_layers=args.ordering_encoder_num_layers,
        bidirectional=args.ordering_encoder_bidirectional,
        dropout=args.ordering_encoder_dropout,
    )
    ordering_encoder = ordering_encoder.to(dev)

    feature_encoder_optimizer = torch.optim.Adam(feature_encoder.parameters(), lr=1e-4)
    ordering_encoder_optimizer = torch.optim.Adam(
        ordering_encoder.parameters(),
        lr=1e-4,
        weight_decay=args.ordering_encoder_weight_decay,
    )

    def validate():
        seed = torch.randint(0, 1000, size=(1,))
        seed_everything(42)  # always use the same seed for validation
        feature_encoder.eval()
        ordering_encoder.eval()
        all_losses = []
        num_tracks = range(3, 21)
        losses_per_num_tracks = {i: [] for i in num_tracks}
        for batch in iter(validation_loader):
            if use_learned_feature:
                _, track_numbers, track_features, seq_lengths = [
                    i.to(dev) for i in batch
                ]
            else:
                track_features, track_numbers, seq_lengths = [i.to(dev) for i in batch]

            batch_size = seq_lengths.shape[0]

            padded_features, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                track_features
            )

            if args.use_available_feature and not use_learned_feature:
                padded_features = padded_features[:, :, feature_index].unsqueeze(2)

            narrative_features = feature_encoder(padded_features.view(-1, num_features))
            narrative_features = narrative_features.view(
                -1, batch_size, args.num_encoding_features
            )

            feature_mask = torch.arange(narrative_features.shape[0]).unsqueeze(
                1
            ) < seq_lengths.unsqueeze(0)
            feature_mask = feature_mask.unsqueeze(2).float().to(dev)
            valid_features = narrative_features * feature_mask

            # normalize features
            feature_mean = valid_features.sum(dim=0) / feature_mask.sum(dim=0)
            feature_var = ((valid_features - feature_mean.unsqueeze(0)) ** 2).sum(
                dim=0
            ) / feature_mask.sum(dim=0)
            feature_std = feature_var**0.5
            valid_features = (valid_features - feature_mean) / feature_std

            narrative_features = valid_features

            padded_track_number, _ = torch.nn.utils.rnn.pad_packed_sequence(
                track_numbers, padding_value=-1
            )
            r_batch = (
                torch.arange(batch_size)
                .unsqueeze(1)
                .repeat(1, num_negative_examples + 1)
                .flatten()
            )
            r_seq_lengths = (
                seq_lengths.unsqueeze(1).repeat(1, num_negative_examples + 1).flatten()
            )
            permutation = [
                torch.arange(l) if i % batch_size == 0 else torch.randperm(l)
                for i, l in enumerate(r_seq_lengths)
            ]
            padded_permutations = torch.nn.utils.rnn.pad_sequence(permutation).to(dev)
            shuffled_narrative_features = narrative_features[
                padded_permutations, r_batch.unsqueeze(0), :
            ]

            ordering_scores, ordering_encodings = ordering_encoder(
                shuffled_narrative_features, r_seq_lengths
            )
            ordering_scores = ordering_scores.view(
                batch_size, num_negative_examples + 1
            )
            targets = torch.zeros(batch_size, dtype=torch.long).to(dev) + 0

            loss = torch.nn.functional.cross_entropy(
                ordering_scores.detach(), targets, reduction="none"
            )
            for i, l in enumerate(loss):
                losses_per_num_tracks[seq_lengths[i].item()].append(l.cpu().detach())
            all_losses.append(loss.detach())
        mean_loss_per_num_tracks = {
            k: torch.stack(v).mean().item() if len(v) > 0 else 0.0
            for k, v in losses_per_num_tracks.items()
        }
        all_losses = torch.cat(all_losses)
        validation_loss = all_losses.mean()
        feature_encoder.train()
        ordering_encoder.train()
        seed_everything(seed)
        return validation_loss.item(), mean_loss_per_num_tracks

    step = 0
    lowest_validation_loss = float("inf")
    best_epoch = 0
    verbose = True
    for epoch in range(200):
        if verbose:
            print(f"epoch {epoch}")
        num_tracks = range(3, 21)
        losses_per_num_tracks = {i: [] for i in num_tracks}
        for batch in iter(training_loader):
            if use_learned_feature:
                _, track_numbers, track_features, seq_lengths = [
                    i.to(dev) for i in batch
                ]
            else:
                track_features, track_numbers, seq_lengths = [i.to(dev) for i in batch]

            batch_size = seq_lengths.shape[0]

            padded_features, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                track_features
            )

            if args.use_available_feature and not use_learned_feature:
                padded_features = padded_features[:, :, feature_index].unsqueeze(2)

            narrative_features = feature_encoder(padded_features.view(-1, num_features))
            narrative_features = narrative_features.view(
                -1, batch_size, args.num_encoding_features
            )  # padded_sequence_length x batch_size x 1

            feature_mask = torch.arange(narrative_features.shape[0]).unsqueeze(
                1
            ) < seq_lengths.unsqueeze(0)
            feature_mask = feature_mask.unsqueeze(2).float().to(dev)
            valid_features = narrative_features * feature_mask

            # normalize features
            feature_mean = valid_features.sum(dim=0) / feature_mask.sum(dim=0)
            feature_var = ((valid_features - feature_mean.unsqueeze(0)) ** 2).sum(
                dim=0
            ) / feature_mask.sum(dim=0)
            feature_std = feature_var**0.5
            valid_features = (valid_features - feature_mean) / feature_std

            narrative_features = valid_features

            padded_track_number, _ = torch.nn.utils.rnn.pad_packed_sequence(
                track_numbers, padding_value=-1
            )
            r_batch = (
                torch.arange(batch_size)
                .unsqueeze(1)
                .repeat(1, num_negative_examples + 1)
                .flatten()
            )
            r_seq_lengths = (
                seq_lengths.unsqueeze(1).repeat(1, num_negative_examples + 1).flatten()
            )
            permutation = [
                torch.arange(l) if i % batch_size == 0 else torch.randperm(l)
                for i, l in enumerate(r_seq_lengths)
            ]
            padded_permutations = torch.nn.utils.rnn.pad_sequence(permutation).to(dev)
            shuffled_narrative_features = narrative_features[
                padded_permutations, r_batch.unsqueeze(0), :
            ]

            ordering_scores, ordering_encodings = ordering_encoder(
                shuffled_narrative_features, r_seq_lengths
            )
            ordering_scores = ordering_scores.view(
                batch_size, num_negative_examples + 1
            )
            targets = torch.zeros(batch_size, dtype=torch.long).to(dev) + 0

            losses = torch.nn.functional.cross_entropy(
                ordering_scores, targets, reduction="none"
            )
            loss = losses.mean()
            feature_encoder_optimizer.zero_grad()
            ordering_encoder_optimizer.zero_grad()
            loss.backward()
            feature_encoder_optimizer.step()
            ordering_encoder_optimizer.step()

            for i, l in enumerate(losses):
                losses_per_num_tracks[seq_lengths[i].item()].append(l.cpu().detach())

            step += 1
        validation_loss, validation_mean_loss_per_num_tracks = validate()
        validation_mi_lower_bound = np.log(num_negative_examples + 1) - validation_loss

        training_mean_loss_per_num_tracks = {
            k: torch.stack(v).mean().item() if len(v) > 0 else 0.0
            for k, v in losses_per_num_tracks.items()
        }

        if validation_loss < lowest_validation_loss:
            lowest_validation_loss = validation_loss
            best_epoch = epoch

            if args.save_model:
                print("saving models")
                encoder_name = (
                    "feature_encoder_" + args.available_feature_to_use + ".pt"
                )
                torch.save(
                    feature_encoder.cpu(),
                    os.path.join(os.getcwd(), encoder_name),
                )
                if encoder_name == "feature_encoder_learned.pt":
                    torch.save(
                        feature_encoder,
                        os.path.join(os.getcwd(), "model.pt"),
                    )
                torch.save(
                    ordering_encoder.cpu(),
                    os.path.join(
                        os.getcwd(),
                        "ordering_encoder_" + args.available_feature_to_use + ".pt",
                    ),
                )
                feature_encoder.to(dev)
                ordering_encoder.to(dev)
        else:
            if epoch - best_epoch > args.patience_epochs:
                print(f"early stopping-best model after epoch {best_epoch}")
                break
        if verbose:
            print(f"validation loss: {validation_loss}")
            print(f"validation MI lower bound: {validation_mi_lower_bound}")

    print(
        f"Highest MI lower bound: {np.log(num_negative_examples + 1) - lowest_validation_loss} nats\n"
    )


if __name__ == "__main__":
    args = SimpleNamespace()
    args.seed = 123
    args.save_model = False
    args.small_dataset = False
    args.normalize_features = True
    args.use_available_feature = False
    args.available_feature_to_use = "learned"

    args.num_negative_examples = 31
    args.patience_epochs = 20
    args.feature_encoder_type = "lstm"
    args.feature_encoder_hidden_units = 128
    args.feature_encoder_num_layers = 2
    args.feature_encoder_dropout = 0.1
    args.num_encoding_features = 1

    args.ordering_encoder_hidden_units = 32
    args.ordering_encoder_num_layers = 2
    args.ordering_encoder_bidirectional = True
    args.ordering_encoder_dropout = 0.0
    args.ordering_encoder_weight_decay = 1e-5

    print("train narrative essence extractor")
    train_model(args)

    args.small_dataset = True
    args.use_available_feature = True
    args.feature_encoder_type = "mean"
    args.patience_epochs = 50

    feature_list = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "speechiness",
        "tempo",
        "valence",
        "duration",
        "echonest_features_pca",
        "fma_audio_features_pca",
        "learned",
    ]

    for feature in feature_list:
        print(f"estimate MI for the feature {feature}")
        args.available_feature_to_use = feature
        train_model(args)
