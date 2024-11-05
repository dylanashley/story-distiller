#!/usr/bin/env python
# -*- coding: ascii -*-

from album_extractor import Mean, OrderingLSTMEncoder, seed_everything
import argparse
import h5py
import numpy as np
from pathlib import Path
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb_logging",
        type=int,
        default=0,
        help="log with wandb if 1, if 0, log on the terminal",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="seed of the experiment",
    )
    parser.add_argument(
        "--save_model",
        type=int,
        default=0,
        help="whether to save the model",
    )
    parser.add_argument(
        "--min_sequence_length",
        type=int,
        default=10,
        help="minimum sequence length",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=30,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size",
    )
    parser.add_argument(
        "--num_negative_examples",
        type=int,
        default=31,
        help="number of negative examples",
    )
    parser.add_argument(
        "--patience_epochs",
        type=int,
        default=20,
        help="patience epochs",
    )
    parser.add_argument(
        "--feature_encoder_type",
        type=str,
        default="mlp",
        help="type of feature encoder",
    )
    parser.add_argument(
        "--feature_encoder_hidden_units",
        type=int,
        default=128,
        help="hidden units of feature encoder",
    )
    parser.add_argument(
        "--feature_encoder_num_layers",
        type=int,
        default=1,
        help="number of layers of feature encoder",
    )
    parser.add_argument(
        "--feature_encoder_dropout",
        type=float,
        default=0.0,
        help="dropout of feature encoder",
    )
    parser.add_argument(
        "--num_encoding_features",
        type=int,
        default=1,
        help="number of encoding features",
    )
    parser.add_argument(
        "--ordering_encoder_hidden_units",
        type=int,
        default=128,
        help="hidden units of ordering encoder",
    )
    parser.add_argument(
        "--ordering_encoder_num_layers",
        type=int,
        default=2,
        help="number of layers of ordering encoder",
    )
    parser.add_argument(
        "--ordering_encoder_bidirectional",
        type=int,
        default=1,
        help="whether ordering encoder is bidirectional",
    )
    parser.add_argument(
        "--ordering_encoder_dropout",
        type=float,
        default=0.0,
        help="dropout of ordering encoder",
    )
    parser.add_argument(
        "--ordering_encoder_weight_decay",
        type=float,
        default=0.00001,
        help="weight decay of ordering encoder",
    )
    return parser.parse_args()


class MADShotClipSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path,
        mode="train",
        min_sequence_length=10,
        max_sequence_length=30,
        discard_first_n_frames=20,
        discard_last_n_frames=10,
        return_metadata=False,
    ):
        super().__init__()

        files = list(Path(dataset_path).rglob("*.h5"))
        self.train_files = [f for f in files if hash(f) % 10 < 8]
        self.validation_files = [f for f in files if hash(f) % 10 >= 8]
        self.files = self.train_files if mode == "training" else self.validation_files

        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.discard_first_n_frames = discard_first_n_frames
        self.discard_last_n_frames = discard_last_n_frames
        self.return_metadata = return_metadata

    def __getitem__(self, item):
        # load h5 file
        f = self.files[item]
        clip = np.array(h5py.File(f, "r")["CLIP"])
        clip = torch.tensor(clip, dtype=torch.float32)
        clip = clip[self.discard_first_n_frames : -self.discard_last_n_frames]

        # divide into 20 equal parts
        # sample sequence_length
        sequence_length = torch.randint(
            self.min_sequence_length, self.max_sequence_length, (1,)
        ).item()
        num_frames = clip.shape[0]
        chunk_size = num_frames // sequence_length
        chunks = clip[: sequence_length * chunk_size].view(
            sequence_length, chunk_size, -1
        )
        # sample one frame per chunk
        indices = torch.randint(chunk_size, size=(sequence_length,))
        frames = chunks[torch.arange(sequence_length), indices]

        if self.return_metadata:
            absolute_indices = (
                torch.arange(sequence_length) * chunk_size
                + indices
                + self.discard_first_n_frames
            )
            file = str(f).split("/")[-1].split("_")[0]
            metadata = {
                "movie": file,
                "indices": (
                    absolute_indices + 1
                ).tolist(),  # add 1, because the filenames start at 1
                "max_index": num_frames,
            }
            return [frames], metadata
        return [frames]

    def __len__(self):
        return len(self.files)


def collate_movie_features_to_packed_seqs(movie_feature_list):
    # return: [track_features: PackedSequence, durations: PackedSequence,
    #          track_numbers: PackedSequence, sequence_length: LongTensor]
    n = len(movie_feature_list[0])
    feature_list = [
        [movie_feature_list[j][k] for j in range(len(movie_feature_list))]
        for k in range(n)
    ]
    sequence_lengths = torch.LongTensor([s.shape[0] for s in feature_list[0]])
    packed_sequences = [
        torch.nn.utils.rnn.pack_sequence(seqs, enforce_sorted=False)
        for seqs in feature_list
    ]
    return packed_sequences + [sequence_lengths]


class MLPClipEncoder(torch.nn.Module):
    def __init__(
        self,
        num_in_features,
        hidden_size,
        num_layers,
        num_out_features=1,
        dropout=0.1,
    ):
        super().__init__()
        self.num_in_features = num_in_features
        self.mlp = torch.nn.Sequential()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.mlp.add_module(
                    f"linear_{i}",
                    torch.nn.Linear(
                        in_features=num_in_features if i == 0 else hidden_size,
                        out_features=hidden_size,
                    ),
                )
                self.mlp.add_module(f"relu_{i}", torch.nn.ReLU())
                self.mlp.add_module(f"dropout_{i}", torch.nn.Dropout(p=dropout))
            else:
                self.mlp.add_module(
                    f"linear_{i}",
                    torch.nn.Linear(
                        in_features=num_in_features if i == 0 else hidden_size,
                        out_features=num_out_features,
                    ),
                )

    def forward(self, x):
        # x: batch_size x seq_length x num_in_features
        # or batch_size x seq_length * num_in_features
        batch_size = x.shape[0]
        if len(x.shape) == 2:
            x = x.view(batch_size, -1, self.num_in_features)

        x = self.mlp(x)
        x = torch.sigmoid(x)
        return x


def train_model(args):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    seed_everything(args.seed)
    num_negative_examples = args.num_negative_examples
    dataset_path = "data/clip_features"

    training_set = MADShotClipSequenceDataset(
        dataset_path=dataset_path,
        mode="training",
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
    )
    training_loader = torch.utils.data.DataLoader(
        dataset=training_set,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_movie_features_to_packed_seqs,
        num_workers=4,
    )

    validation_set = MADShotClipSequenceDataset(
        dataset_path=dataset_path,
        mode="validation",
        min_sequence_length=args.min_sequence_length,
        max_sequence_length=args.max_sequence_length,
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_set,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_movie_features_to_packed_seqs,
        num_workers=4,
    )

    num_features = 768

    if args.feature_encoder_type == "mean":
        feature_encoder = Mean()
    elif args.feature_encoder_type == "mlp":
        feature_encoder = MLPClipEncoder(
            num_in_features=num_features,
            hidden_size=args.feature_encoder_hidden_units,
            num_layers=args.feature_encoder_num_layers,
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
        # always use the same seed for validation
        seed_everything(42)
        feature_encoder.eval()
        ordering_encoder.eval()
        all_losses = []
        num_tracks = range(args.min_sequence_length, args.max_sequence_length + 1)
        losses_per_num_tracks = {i: [] for i in num_tracks}
        for batch in iter(validation_loader):
            frame_features, seq_lengths = [i.to(dev) for i in batch]

            batch_size = seq_lengths.shape[0]

            padded_features, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                frame_features
            )

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
        num_tracks = range(args.min_sequence_length, args.max_sequence_length + 1)
        losses_per_num_tracks = {i: [] for i in num_tracks}
        for batch in iter(training_loader):
            frame_features, seq_lengths = [i.to(dev) for i in batch]

            batch_size = seq_lengths.shape[0]

            padded_features, seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(
                frame_features
            )

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
                print("save models")
                encoder_name = "movie_feature_encoder.pt"
                torch.save(
                    feature_encoder.cpu(),
                    encoder_name,
                )
                torch.save(
                    ordering_encoder.cpu(),
                    "movie_ordering_encoder.pt",
                )
                feature_encoder.to(dev)
                ordering_encoder.to(dev)
        else:
            if epoch - best_epoch > args.patience_epochs:
                print(f"early stopping-best model after epoch {best_epoch}")
                break

        highest_mi_lower_bound = (
            np.log(num_negative_examples + 1) - lowest_validation_loss
        )
        print(
            {
                "epoch": epoch,
                "validation_loss": validation_loss,
                "validation_mi_lower_bound": validation_mi_lower_bound,
                "lowest_validation_loss": lowest_validation_loss,
                "highest_mi_lower_bound": highest_mi_lower_bound,
            }
        )

    print(
        f"Highest MI lower bound: {np.log(num_negative_examples + 1) - lowest_validation_loss} nats\n"
    )


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
