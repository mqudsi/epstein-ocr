#!/usr/bin/env -S uv.exe run
"""
Compact synthetic OCR:
    CNN -> BiLSTM -> CTC

Dependencies:
    torch >= 2.3
    numpy
    pillow
    skia-python  # needed for --renderer skia/mixed

Dataset:
    ocr_dataset_ctc/
        train/
            images/*.png
            labels.jsonl
        val/
            images/*.png
            labels.jsonl

labels.jsonl records:
    {"image": "images/train_000000.png", "text": "Abc123+/="}

For convenience, datasets made by the original YOLO generator can also be read:
    split/images/*
    split/labels/*.txt

In that case, text is reconstructed by sorting character labels by x center.
New JSONL datasets are preferable because they retain the exact source string.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    Sampler,
)


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
BLANK = 0
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(ALPHABET)}
ID_TO_CHAR = {i + 1: c for i, c in enumerate(ALPHABET)}

DEFAULT_FONTS = [
    "times.ttf",
    "./fonts/NimbusRomNo9L-Reg.otf",
    "./fonts/texgyretermes-regular.otf",
    "C:/Windows/Fonts/times.ttf",
]

DEFAULT_CFG = {
    "alphabet": ALPHABET,
    "input_height": 48,
    # Fixed minimum crop height in ORIGINAL pixels. This preserves the
    # approximate absolute size of short/lowercase/isolated glyphs.
    "window_height": 32,
    "margin": 4,
    "ink_threshold": 245,
    "max_width": 2048,
    "hidden_size": 128,
    "lstm_layers": 2,
    "dropout": 0.1,
}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


# ---------------------------------------------------------------------------
# Shared image preprocessing
# ---------------------------------------------------------------------------

def prepare_line(image, cfg):
    """
    Crop horizontal whitespace, place ink in a minimum-height native-pixel
    window, then resize to input_height while preserving aspect ratio.

    Returns a [1, H, W] float tensor: white=+1, black=-1.

    No silent clipping or width squeezing.
    """
    image = image.convert("L")
    arr = np.asarray(image)
    ys, xs = np.where(arr < cfg["ink_threshold"])

    if len(xs) == 0:
        raise ValueError("Image contains no foreground pixels")

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    margin = cfg["margin"]
    crop_w = (x1 - x0) + 2 * margin
    crop_h = max(
        cfg["window_height"],
        (y1 - y0) + 2 * margin,
    )

    canvas = Image.new("L", (crop_w, crop_h), 255)

    # Pasting the original image also retains real margin pixels.
    # Pixels outside the new canvas are cropped by PIL.
    paste_x = margin - x0
    paste_y = (crop_h - (y1 - y0)) // 2 - y0
    canvas.paste(image, (paste_x, paste_y))

    out_h = cfg["input_height"]
    out_w = max(4, int(round(crop_w * out_h / crop_h)))

    if out_w > cfg["max_width"]:
        raise ValueError(
            f"Prepared line width {out_w} exceeds max_width="
            f"{cfg['max_width']}. Split the line or deliberately increase "
            "the configured limit; the script will not truncate it."
        )

    canvas = canvas.resize(
        (out_w, out_h),
        Image.Resampling.BILINEAR,
    )

    pixels = np.array(canvas, dtype=np.float32, copy=True)
    pixels = pixels / 127.5 - 1.0
    return torch.from_numpy(pixels).unsqueeze(0)


def extract_document_lines(image, threshold=245, line_gap=2):
    """
    Row-projection segmenter, retaining the practical approach from the
    original program. Returns PIL images in reading order.

    line_gap is the maximum number of blank rows allowed inside one line.
    Segmentation uses a thresholded mask; recognition uses original pixels.
    """
    gray = image.convert("L")
    arr = np.asarray(gray)
    active_rows = np.flatnonzero(np.any(arr < threshold, axis=1))

    if len(active_rows) == 0:
        return []

    groups = []
    start = previous = int(active_rows[0])

    for row in active_rows[1:]:
        row = int(row)
        if row - previous > line_gap + 1:
            groups.append((start, previous + 1))
            start = row
        previous = row
    groups.append((start, previous + 1))

    # Horizontal cropping and white vertical margins are supplied by
    # prepare_line(), independently for each returned line.
    return [
        gray.crop((0, top, gray.width, bottom))
        for top, bottom in groups
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LineDataset(Dataset):
    def __init__(self, root, split, cfg, limit=None):
        self.cfg = cfg
        self.items = []

        split_dir = Path(root) / split
        manifest = split_dir / "labels.jsonl"

        if manifest.exists():
            with manifest.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    self.items.append((
                        split_dir / row["image"],
                        row["text"],
                    ))
        else:
            # Compatibility with the original YOLO dataset.
            image_dir = split_dir / "images"
            label_dir = split_dir / "labels"

            if image_dir.exists() and label_dir.exists():
                for path in sorted(image_dir.iterdir()):
                    if path.suffix.lower() not in {
                        ".png", ".jpg", ".jpeg", ".bmp"
                    }:
                        continue

                    label_path = label_dir / (path.stem + ".txt")
                    if not label_path.exists():
                        raise FileNotFoundError(label_path)

                    chars = []
                    with label_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            fields = line.split()
                            if not fields:
                                continue
                            class_id = int(fields[0])
                            x_center = float(fields[1])
                            chars.append((x_center, ALPHABET[class_id]))

                    text = "".join(c for _, c in sorted(chars))
                    self.items.append((path, text))

                if self.items:
                    eprint(
                        f"{split}: reconstructing text from legacy YOLO labels"
                    )

        if limit is not None:
            self.items = self.items[:limit]

        if not self.items:
            raise RuntimeError(
                f"No samples found for {split!r} under {root}. "
                "Generate a dataset first."
            )

        for path, text in self.items:
            if not text:
                raise ValueError(f"Empty target: {path}")
            unknown = set(text) - set(ALPHABET)
            if unknown:
                raise ValueError(f"Unknown characters {unknown}: {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, text = self.items[index]
        with Image.open(path) as image:
            tensor = prepare_line(image, self.cfg)
        return tensor, text, str(path)

class VerticalShiftDataset(Dataset):
    """
    Apply small vertical translations AFTER prepare_line().

    Training only. Validation and prediction remain unchanged.
    """

    def __init__(self, dataset, max_shift=2):
        self.dataset = dataset
        self.max_shift = max_shift

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, text, path = self.dataset[index]

        shift = random.randint(
            -self.max_shift,
            self.max_shift,
        )

        if shift == 0:
            return image, text, path

        # Avoid introducing clipped glyphs. The image is normalized:
        # white = +1, black = -1.
        if shift > 0:
            discarded = image[:, -shift:, :]
        else:
            discarded = image[:, :-shift, :]

        if torch.any(discarded < 0.99):
            return image, text, path

        shifted = torch.ones_like(image)

        if shift > 0:
            shifted[:, shift:, :] = image[:, :-shift, :]
        else:
            shifted[:, :shift, :] = image[:, -shift:, :]

        return shifted, text, path

class RegressionDataset(Dataset):
    """
    Directory structure:

        regressions/
            case_001/
                line.png
                actual.txt
            case_002/
                line.png
                actual.txt

    Images must be raw/non-normalized single-line crops.
    """

    def __init__(self, root, cfg):
        self.cfg = cfg
        self.items = []

        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"Regression directory does not exist: {root}"
            )

        for case_dir in sorted(root.iterdir()):
            if not case_dir.is_dir():
                continue

            image_path = case_dir / "line.png"
            text_path = case_dir / "actual.txt"

            if not image_path.is_file() or not text_path.is_file():
                raise RuntimeError(
                    f"Regression case {case_dir} must contain "
                    "line.png and actual.txt"
                )

            # Accept a UTF-8 BOM and trailing line endings, but do not
            # silently remove spaces or other unexpected characters.
            text = text_path.read_text(
                encoding="utf-8-sig"
            ).rstrip("\r\n")

            if not text:
                raise ValueError(
                    f"Empty regression transcription: {text_path}"
                )

            unknown = set(text) - set(ALPHABET)
            if unknown:
                raise ValueError(
                    f"Unexpected characters {unknown!r} in {text_path}. "
                    "Each actual.txt must contain exactly one "
                    "transcription, without spaces or internal newlines."
                )

            self.items.append((image_path, text))

        if not self.items:
            raise RuntimeError(
                f"No regression cases found under {root}"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path, text = self.items[index]

        with Image.open(image_path) as image:
            tensor = prepare_line(image, self.cfg)

        return tensor, text, str(image_path)


class MixedRegressionSampler(Sampler):
    """
    For ConcatDataset([synthetic_dataset, regression_dataset]):

      * Include each synthetic example exactly once per epoch.
      * Add random regression examples, sampled with replacement.
      * Shuffle the entire combined epoch.

    Uses the same saved torch.Generator as the training DataLoader.
    """

    def __init__(
        self,
        synthetic_count,
        regression_count,
        regression_fraction,
        generator,
    ):
        if synthetic_count < 1 or regression_count < 1:
            raise ValueError("Both datasets must be nonempty")

        if not 0.0 < regression_fraction < 1.0:
            raise ValueError(
                "regression_fraction must be greater than 0 and less than 1"
            )

        self.synthetic_count = synthetic_count
        self.regression_count = regression_count
        self.generator = generator

        self.regression_draws = max(
            1,
            round(
                synthetic_count
                * regression_fraction
                / (1.0 - regression_fraction)
            ),
        )

    def __len__(self):
        return self.synthetic_count + self.regression_draws

    def __iter__(self):
        synthetic_indices = torch.arange(
            self.synthetic_count,
            dtype=torch.long,
        )

        regression_indices = (
            torch.randint(
                low=0,
                high=self.regression_count,
                size=(self.regression_draws,),
                generator=self.generator,
            )
            + self.synthetic_count
        )

        indices = torch.cat((
            synthetic_indices,
            regression_indices,
        ))

        order = torch.randperm(
            len(indices),
            generator=self.generator,
        )

        return iter(indices[order].tolist())


def collate_lines(samples):
    images, texts, paths = zip(*samples)
    widths = torch.tensor(
        [image.shape[-1] for image in images],
        dtype=torch.long,
    )

    batch = torch.ones(
        len(images),
        1,
        images[0].shape[1],
        int(widths.max()) + 32,
        dtype=torch.float32,
    )

    for i, image in enumerate(images):
        batch[i, :, :, :image.shape[-1]] = image

    target_lengths = torch.tensor(
        [len(text) for text in texts],
        dtype=torch.long,
    )

    # CNN horizontal stride is exactly 2, using floor division.
    input_lengths = widths // 2

    for text, available, path in zip(texts, input_lengths.tolist(), paths):
        repeats = sum(a == b for a, b in zip(text, text[1:]))
        required = len(text) + repeats
        if available < required:
            raise ValueError(
                f"Insufficient CTC time steps: {path}: "
                f"{available} available, at least {required} required. "
                "Increase input_height or reduce horizontal downsampling."
            )

    targets = torch.tensor(
        [CHAR_TO_ID[c] for text in texts for c in text],
        dtype=torch.long,
    )

    return (
        batch,
        widths,
        targets,
        target_lengths,
        list(texts),
        list(paths),
    )


def make_loader(
    dataset,
    args,
    shuffle=False,
    generator=None,
    sampler=None,
):
    return DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=args.device_obj.type == "cuda",
        collate_fn=collate_lines,
        generator=generator,
        persistent_workers=args.workers > 0,
    )

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CRNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        height = cfg["input_height"]
        if height < 16 or height % 16:
            raise ValueError("input_height must be a positive multiple of 16")

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # Horizontal stride: 2.
        # Vertical stride: 16.
        self.cnn = nn.Sequential(
            block(1, 32),
            nn.MaxPool2d((2, 2)),
            block(32, 64),
            nn.MaxPool2d((2, 1)),
            block(64, 128),
            nn.MaxPool2d((2, 1)),
            block(128, 192),
            nn.MaxPool2d((2, 1)),
        )

        feature_size = 192 * (height // 16)

        self.rnn = nn.LSTM(
            input_size=feature_size,
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["lstm_layers"],
            dropout=(
                cfg["dropout"] if cfg["lstm_layers"] > 1 else 0.0
            ),
            bidirectional=True,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            2 * cfg["hidden_size"],
            len(ALPHABET) + 1,
        )

    def forward(self, images, widths):
        features = self.cnn(images)              # B, C, H, T
        b, c, h, t = features.shape

        features = (
            features.permute(0, 3, 1, 2)
            .contiguous()
            .view(b, t, c * h)
        )

        lengths = widths.cpu() // 2

        # Packing prevents right-padding from changing the backward LSTM's
        # interpretation of the real line.
        packed = pack_padded_sequence(
            features,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed, _ = self.rnn(packed)
        sequence, _ = pad_packed_sequence(
            packed,
            batch_first=True,
            total_length=t,
        )

        logits = self.classifier(sequence)      # B, T, classes
        return logits, lengths


def greedy_decode(logits, lengths):
    ids = logits.argmax(dim=-1).cpu().tolist()
    results = []

    for sequence, length in zip(ids, lengths.tolist()):
        previous = BLANK
        chars = []

        for token in sequence[:length]:
            if token != BLANK and token != previous:
                chars.append(ID_TO_CHAR[token])
            # A blank resets repetition collapsing.
            previous = token

        results.append("".join(chars))

    return results


def ctc_loss(logits, lengths, targets, target_lengths, criterion, device):
    # Compute log_softmax and CTC in float32 even when CNN/LSTM use AMP.
    log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
    return criterion(
        log_probs,
        targets.to(device, non_blocking=True),
        lengths.cpu(),
        target_lengths.cpu(),
    )


# ---------------------------------------------------------------------------
# Transcription metrics
# ---------------------------------------------------------------------------

def edit_counts(reference, hypothesis):
    """
    Levenshtein alignment returning substitutions, deletions, insertions.
    There may be multiple equally optimal alignments; ties are deterministic.
    """

    # Fast path, useful after initial training has yielded results
    if reference == hypothesis:
        return 0, 0, 0

    # Tuple: total edits, substitutions, deletions, insertions
    previous = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]

    for i, ref_char in enumerate(reference, 1):
        current = [(i, 0, i, 0)]

        for j, hyp_char in enumerate(hypothesis, 1):
            diagonal = previous[j - 1]

            if ref_char == hyp_char:
                substitution = diagonal
            else:
                substitution = (
                    diagonal[0] + 1,
                    diagonal[1] + 1,
                    diagonal[2],
                    diagonal[3],
                )

            above = previous[j]
            deletion = (
                above[0] + 1,
                above[1],
                above[2] + 1,
                above[3],
            )

            left = current[j - 1]
            insertion = (
                left[0] + 1,
                left[1],
                left[2],
                left[3] + 1,
            )

            current.append(min(
                (substitution, deletion, insertion),
                key=lambda item: item[0],
            ))

        previous = current

    _, substitutions, deletions, insertions = previous[-1]
    return substitutions, deletions, insertions


@torch.inference_mode()
def evaluate(model, loader, device, amp, errors_path=None):
    model.eval()
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=False)

    total_loss = 0.0
    total_lines = 0
    total_chars = 0
    exact = 0
    substitutions = deletions = insertions = 0

    errors_file = None
    if errors_path:
        errors_path = Path(errors_path)
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        errors_file = errors_path.open("w", encoding="utf-8")

    try:
        for images, widths, targets, target_lengths, texts, paths in loader:
            images = images.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                enabled=amp,
            ):
                logits, lengths = model(images, widths)

            loss = ctc_loss(
                logits, lengths, targets, target_lengths, criterion, device
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite validation CTC loss. "
                    "Check labels, time-step counts, and numeric stability."
                )

            predictions = greedy_decode(logits, lengths)

            total_loss += float(loss) * len(texts)
            total_lines += len(texts)

            for truth, prediction, path in zip(texts, predictions, paths):
                if truth == prediction:
                    s = d = i = 0
                else:
                    s, d, i = edit_counts(truth, prediction)
                substitutions += s
                deletions += d
                insertions += i
                total_chars += len(truth)
                exact += truth == prediction

                if errors_file and truth != prediction:
                    errors_file.write(json.dumps({
                        "image": path,
                        "truth": truth,
                        "prediction": prediction,
                        "substitutions": s,
                        "deletions": d,
                        "insertions": i,
                    }) + "\n")
    finally:
        if errors_file:
            errors_file.close()

    return {
        "loss": total_loss / total_lines,
        "cer": (
            substitutions + deletions + insertions
        ) / max(1, total_chars),
        "exact": exact / total_lines,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "lines": total_lines,
        "characters": total_chars,
    }


def print_metrics(metrics):
    eprint(
        f"CTC loss={metrics['loss']:.6f}  "
        f"CER={100 * metrics['cer']:.5f}%  "
        f"exact lines={100 * metrics['exact']:.3f}%  "
        f"S/D/I={metrics['substitutions']}/"
        f"{metrics['deletions']}/{metrics['insertions']}"
    )


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------

def load_snapshot(path):
    # Only load snapshots you trust: weights_only=False permits pickle.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("format") != "compact_ctc_v1":
        raise ValueError(f"Not a compatible CTC snapshot: {path}")
    if checkpoint["cfg"]["alphabet"] != ALPHABET:
        raise ValueError("Snapshot alphabet does not match this script")
    return checkpoint


def capture_rng(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        ),
        "loader": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    cuda_state = state.get("cuda")
    if (
        cuda_state is not None
        and torch.cuda.is_available()
        and len(cuda_state) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(cuda_state)

    generator.set_state(state["loader"])


def save_snapshot(path, checkpoint):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    if args.resume and args.model:
        raise ValueError("Use --resume OR --model, not both")

    checkpoint = None
    source = args.resume or args.model
    if source:
        checkpoint = load_snapshot(source)

    cfg = dict(checkpoint["cfg"] if checkpoint else DEFAULT_CFG)
    model = CRNN(cfg).to(args.device_obj)

    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)

    if args.resume:
        settings = dict(checkpoint["train_settings"])
        if args.lr is not None or args.weight_decay is not None:
            raise ValueError(
                "--resume preserves optimizer settings. "
                "Use --model for fine-tuning with new settings."
            )
        total_epochs = (
            args.epochs if args.epochs is not None
            else settings["schedule_epochs"]
        )
        out = (
            Path(args.out) if args.out
            else Path(args.resume).resolve().parent
        )
    else:
        total_epochs = args.epochs if args.epochs is not None else 100
        settings = {
            "base_lr": (
                args.lr if args.lr is not None
                else (1e-4 if args.model else 1e-3)
            ),
            "weight_decay": (
                args.weight_decay
                if args.weight_decay is not None else 1e-4
            ),
            "schedule_epochs": total_epochs,
        }
        out = (
            Path(args.out) if args.out
            else Path("runs") / "ctc" / time.strftime("%Y%m%d-%H%M%S")
        )

    out.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["base_lr"],
        weight_decay=settings["weight_decay"],
    )

    amp = args.device_obj.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch = 1
    best_key = (float("inf"), 0.0, float("inf"))

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if amp and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1

        best_key = tuple(checkpoint["best_key"])
        # Migrate snapshots created before the validation-loss tie-breaker.
        if len(best_key) == 2:
            best_key = (*best_key, float("inf"))

        restore_rng(checkpoint["rng"], generator)

        if total_epochs > settings["schedule_epochs"]:
            eprint(
                "Extending beyond the original schedule: learning rate "
                "will remain at its minimum after the original horizon."
            )

    if start_epoch > total_epochs:
        raise ValueError(
            f"Snapshot finished epoch {start_epoch - 1}; "
            f"requested final epoch is {total_epochs}."
        )

    synthetic_train_data = LineDataset(
        args.data,
        "train",
        cfg,
        limit=args.limit_train,
    )

    val_data = LineDataset(
        args.data,
        "val",
        cfg,
        limit=args.limit_val,
    )

    regression_data = None
    regression_loader = None
    train_sampler = None
    train_data = synthetic_train_data

    if args.regressions:
        regression_data = RegressionDataset(
            args.regressions,
            cfg,
        )

        train_data = ConcatDataset([
            synthetic_train_data,
            regression_data,
        ])

        train_sampler = MixedRegressionSampler(
            synthetic_count=len(synthetic_train_data),
            regression_count=len(regression_data),
            regression_fraction=args.regression_fraction,
            generator=generator,
        )

        regression_loader = make_loader(
            regression_data,
            args,
            shuffle=False,
        )

        total_exposures = len(train_sampler)
        fraction = train_sampler.regression_draws / total_exposures

        eprint(
            f"Regression mixing: {len(regression_data)} unique cases; "
            f"{train_sampler.regression_draws} regression draws + "
            f"{len(synthetic_train_data)} synthetic examples per epoch "
            f"({fraction:.1%} regression exposure)."
        )

    train_data = VerticalShiftDataset(
        train_data,
        max_shift=2,
    )

    train_loader = make_loader(
        train_data,
        args,
        shuffle=True,
        generator=generator,
        sampler=train_sampler,
    )

    val_loader = make_loader(
        val_data,
        args,
    )
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=False)

    parameter_count = sum(p.numel() for p in model.parameters())
    eprint(
        f"Parameters: {parameter_count:,}; device={args.device_obj}; "
        f"AMP={amp}; train={len(train_data)}; val={len(val_data)}"
    )
    eprint(f"Snapshots/results: {out}")

    csv_path = out / "results.csv"

    if csv_path.exists() and not args.resume:
        raise FileExistsError(
            f"{csv_path} already exists. Choose a new --out directory."
        )

    fields = [
        "epoch", "seconds", "lr", "train_loss", "val_loss",
        "cer", "exact_line_accuracy",
        "substitutions", "deletions", "insertions",
    ]

    # Branching from an older snapshot in the same directory should not
    # retain CSV rows from epochs that have now been rolled back.
    if args.resume and csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            old_rows = [
                row for row in csv.DictReader(f)
                if int(row["epoch"]) < start_epoch
            ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(old_rows)

    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    for epoch in range(start_epoch, total_epochs + 1):
        began = time.perf_counter()

        # Explicit cosine schedule. Its original horizon is saved so
        # resuming does not silently restart or redefine the schedule.
        horizon = settings["schedule_epochs"]
        progress = min(
            (epoch - 1) / max(1, horizon - 1),
            1.0,
        )
        lr_factor = 0.05 + 0.95 * (
            0.5 * (1.0 + math.cos(math.pi * progress))
        )
        lr = settings["base_lr"] * lr_factor
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()

        # Preserve the checkpoint's running normalization statistics.
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

        accumulated_loss = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, 1):
            images, widths, targets, target_lengths, texts, _ = batch
            images = images.to(args.device_obj, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=args.device_obj.type,
                enabled=amp,
            ):
                logits, lengths = model(images, widths)

            loss = ctc_loss(
                logits, lengths, targets, target_lengths,
                criterion, args.device_obj,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training CTC loss. Try --no-amp, "
                    "check input lengths, or reduce --lr."
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            accumulated_loss += float(loss.detach()) * len(texts)
            seen += len(texts)

            if step % args.log_every == 0:
                eprint(
                    f"epoch {epoch}/{total_epochs} "
                    f"batch {step}/{len(train_loader)} "
                    f"loss={accumulated_loss / seen:.5f}"
                )

        metrics = evaluate(
            model, val_loader, args.device_obj, amp
        )

        regression_metrics = None
        if regression_loader is not None:
            regression_metrics = evaluate(
                model,
                regression_loader,
                args.device_obj,
                amp,
                errors_path=out / "regression_errors.jsonl",
            )

        train_loss = accumulated_loss / seen

        # key = (metrics["cer"], -metrics["exact"])
        # Use validation loss as a tie-breaker for tied transcription scores
        key = (
            metrics["cer"],
            -metrics["exact"],
            metrics["loss"],
        )
        improved = key < best_key
        if improved:
            best_key = key

        snapshot = {
            "format": "compact_ctc_v1",
            "cfg": cfg,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "train_settings": settings,
            "best_key": best_key,
            "metrics": metrics,
            "rng": capture_rng(generator),
            "regression_metrics": regression_metrics,
        }

        save_snapshot(out / "last.pt", snapshot)
        if improved:
            save_snapshot(out / "best.pt", snapshot)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_snapshot(
                out / f"epoch_{epoch:04d}.pt",
                snapshot,
            )

        elapsed = time.perf_counter() - began

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writerow({
                "epoch": epoch,
                "seconds": round(elapsed, 3),
                "lr": lr,
                "train_loss": train_loss,
                "val_loss": metrics["loss"],
                "cer": metrics["cer"],
                "exact_line_accuracy": metrics["exact"],
                "substitutions": metrics["substitutions"],
                "deletions": metrics["deletions"],
                "insertions": metrics["insertions"],
            })

        eprint(
            f"Epoch {epoch}/{total_epochs}: "
            f"{elapsed:.1f}s; train_loss={train_loss:.6f}; "
            f"lr={lr:.3g}"
            + ("; NEW BEST" if improved else "")
        )

        eprint("Synthetic validation:")
        print_metrics(metrics)

        if regression_metrics is not None:
            eprint("Known regressions (included in training):")
            print_metrics(regression_metrics)

# ---------------------------------------------------------------------------
# Validation and document prediction
# ---------------------------------------------------------------------------

def load_inference_model(args):
    if not args.model:
        raise ValueError("--model is required")

    checkpoint = load_snapshot(args.model)
    cfg = checkpoint["cfg"]
    model = CRNN(cfg).to(args.device_obj)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    return model, cfg


def validate(args):
    model, cfg = load_inference_model(args)

    if args.split == "regressions":
        if not args.regressions:
            raise ValueError(
                "--split regressions requires --regressions PATH"
            )

        dataset = RegressionDataset(
            args.regressions,
            cfg,
        )
    else:
        dataset = LineDataset(
            args.data,
            args.split,
            cfg,
            limit=args.limit_val,
        )

    loader = make_loader(dataset, args)
    amp = args.device_obj.type == "cuda" and not args.no_amp

    metrics = evaluate(
        model,
        loader,
        args.device_obj,
        amp,
        args.errors,
    )
    print_metrics(metrics)

@torch.inference_mode()
def recognize_lines(model, lines, cfg, args):
    results = []
    amp = args.device_obj.type == "cuda" and not args.no_amp

    for start in range(0, len(lines), args.batch):
        chunk = lines[start:start + args.batch]
        tensors = [prepare_line(line, cfg) for line in chunk]
        widths = torch.tensor(
            [tensor.shape[-1] for tensor in tensors],
            dtype=torch.long,
        )

        images = torch.ones(
            len(tensors),
            1,
            cfg["input_height"],
            int(widths.max()) + 32,
        )

        for index, tensor in enumerate(tensors):
            images[index, :, :, :tensor.shape[-1]] = tensor

        images = images.to(args.device_obj)

        with torch.autocast(
            device_type=args.device_obj.type,
            enabled=amp,
        ):
            logits, lengths = model(images, widths)

        results.extend(greedy_decode(logits, lengths))

    return results

class LineLengthDiagnostic:
    def __init__(self, expected=76):
        self.expected = expected
        self.pending = None
        self.cumulative = 0
        self.this_file = 0
        self.last_fname = None

    def feed(self, filename, line_number, text):
        if not text.strip():
            return

        current = (str(filename), line_number, len(text))

        if self.last_fname != filename:
            self.cumulative += self.this_file
            self.this_file = 0
            self.last_fname = filename
        self.this_file = line_number

        if self.pending is not None:
            previous = self.pending
            self.pending = None

            combined = previous[2] + current[2]

            if combined == self.expected:
                return  # Both lines form one valid pair.

            eprint(
                f"WARNING: {previous[0]}:{previous[1]}: (overall line {self.cumulative + self.this_file - 1}) "
                f"{previous[2]} characters; neither this line alone "
                f"nor together with the next nonempty line has "
                f"{self.expected} characters. "
                f"Next: {current[0]}:{current[1]} "
                f"({current[2]} characters; combined={combined})."
            )

            # The previous line failed. Evaluate the current line afresh,
            # so one failure does not automatically invalidate later lines.

        if current[2] == self.expected:
            return

        if current[2] > self.expected:
            eprint(
                f"WARNING: {current[0]}:{current[1]}: (overall line {self.cumulative + self.this_file}) "
                f"{current[2]} characters, exceeding "
                f"the expected {self.expected}."
            )
            return

        self.pending = current

    def finish(self):
        if self.pending is None:
            return

        filename, line_number, length = self.pending
        eprint(
            f"WARNING: {filename}:{line_number}: "
            f"{length} characters; expected {self.expected}, "
            "and no following nonempty line remains to form a pair."
        )
        self.pending = None

def predict(args):
    if not args.files:
        raise ValueError("Provide one or more image files")

    model, cfg = load_inference_model(args)

    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    length_diagnostic = LineLengthDiagnostic(expected=76)

    for filename in args.files:
        eprint(f"Processing: {filename}")

        with Image.open(filename) as source:
            image = source.convert("L")

        if args.single_line:
            arr = np.asarray(image)
            lines = (
                [image]
                if np.any(arr < cfg["ink_threshold"])
                else []
            )
        else:
            lines = extract_document_lines(
                image,
                threshold=cfg["ink_threshold"],
                line_gap=args.line_gap,
            )

        for index, line in enumerate(lines):
            if line.height > 24:
                eprint(
                    f"WARNING: line {index} is {line.height}px tall; "
                    "possible merged lines or unexpected source scale."
                )

        if debug_dir:
            stem = Path(filename).stem

            for index, line in enumerate(lines):
                prefix = debug_dir / f"{stem}_line_{index:04d}"

                # Before cropping, padding, or resizing for the network.
                line.save(str(prefix) + "_raw.png")

                tensor = prepare_line(line, cfg)
                pixels = (
                    (tensor.squeeze(0).numpy() + 1.0) * 127.5
                ).clip(0, 255).astype(np.uint8)

                Image.fromarray(pixels).save(
                    str(prefix) + "_normalized.png"
                )

                eprint(
                    f"line {index:03d}: "
                    f"raw={line.width}x{line.height}, "
                    f"network={tensor.shape[-1]}x{tensor.shape[-2]}"
                )

        text = recognize_lines(model, lines, cfg, args)

        for line_number, line_text in enumerate(text, start=1):
            length_diagnostic.feed(
                filename,
                line_number,
                line_text,
            )

        try:
            print("\n".join(text))
        except BrokenPipeError:
            return

    # Outside the file loop: only the final input file ends the stream.
    length_diagnostic.finish()


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------

def add_repeated_run(text, probability=0.35):
    """
    Overwrite part of a line with a repeated-character run.
    Keeps the original line length unchanged.
    """
    if len(text) < 2 or random.random() >= probability:
        return text

    run_length = random.choices(
        population=[2, 3, 4, 5, 6, 7, 8],
        weights=[50, 25, 12, 5, 4, 2, 2],
        k=1,
    )[0]
    run_length = min(run_length, len(text))

    char = random.choice(ALPHABET)
    start = random.randint(0, len(text) - run_length)

    return (
        text[:start]
        + char * run_length
        + text[start + run_length:]
    )

def random_text():
    hard = "iilljjttr3cfe"

    p = random.random()
    if p < 0.03:
        length = 1
    elif p < 0.35:
        length = random.randint(2, 19)
    else:
        length = random.randint(20, 80)

    if random.random() < 0.55:
        hard_count = int(length * 0.75)
        chars = (
            random.choices(hard, k=hard_count)
            + random.choices(ALPHABET, k=length - hard_count)
        )
        random.shuffle(chars)
        text = "".join(chars)
    else:
        text = "".join(random.choices(ALPHABET, k=length))

    return add_repeated_run(text)

def random_base64():
    kind = random.randrange(3)
    size = random.randint(64, 512)

    if kind == 0:
        payload = bytes(random.getrandbits(8) for _ in range(size))
    elif kind == 1:
        payload = bytes(
            0 if random.random() < 0.8 else random.getrandbits(8)
            for _ in range(size)
        )
    else:
        start = random.randrange(256)
        payload = bytes((start + i) % 256 for i in range(size))

    return base64.b64encode(payload).decode("ascii")

class SyntheticRenderer:
    """
    Render using explicit font files with either PIL or Skia.

    In mixed mode:
        P(PIL)  = 0.5
        P(Skia) = 0.5

    Within either renderer, each loaded font has equal probability.

    Skia typefaces are loaded from the actual font files, rather than from
    a single system font family.
    """

    def __init__(self, args):
        self.args = args
        self.fonts = []
        self.skia = None

        if args.renderer in {"skia", "mixed"}:
            try:
                import skia
            except ImportError as exc:
                raise RuntimeError(
                    "Install skia-python or use --renderer pil"
                ) from exc

            self.skia = skia

        candidates = args.font or DEFAULT_FONTS
        explicit_fonts = bool(args.font)
        seen_paths = set()

        for requested_path in candidates:
            # Pillow also resolves names such as "times.ttf" through its
            # normal platform-specific font lookup.
            try:
                pil_font = ImageFont.truetype(
                    requested_path,
                    args.font_size,
                )
            except OSError as exc:
                if explicit_fonts:
                    raise RuntimeError(
                        f"Could not load requested font: {requested_path}"
                    ) from exc

                eprint(
                    f"Skipping unavailable default font: {requested_path}"
                )
                continue

            resolved_path = Path(
                os.fsdecode(pil_font.path)
            ).resolve()

            if not resolved_path.is_file():
                raise RuntimeError(
                    "Font did not resolve to a readable file: "
                    f"{resolved_path}"
                )

            # normcase handles case-insensitive Windows paths.
            path_key = os.path.normcase(str(resolved_path))

            if path_key in seen_paths:
                continue

            family, style = pil_font.getname()
            skia_font = None
            skia_family = None

            if self.skia is not None:
                typeface = self.skia.Typeface.MakeFromFile(
                    str(resolved_path)
                )

                if typeface is None:
                    raise RuntimeError(
                        "Skia could not load font file: "
                        f"{resolved_path}"
                    )

                skia_font = self.skia.Font(
                    typeface,
                    args.font_size,
                )
                skia_font.setSubpixel(True)
                skia_font.setEdging(
                    self.skia.Font.Edging.kSubpixelAntiAlias
                )

                skia_family = typeface.getFamilyName()

            self.fonts.append({
                "path": str(resolved_path),
                "family": family,
                "style": style,
                "pil": pil_font,
                "skia": skia_font,
            })
            seen_paths.add(path_key)

            eprint(
                f"Loaded font {len(self.fonts)}: "
                f"{family} / {style}\n"
                f"  {resolved_path}"
            )

            if skia_family is not None:
                eprint(f"  Skia family: {skia_family}")

        if not self.fonts:
            raise RuntimeError(
                "No font files loaded. Supply one or more --font PATH "
                "options. Explicit font files are now required for both "
                "the PIL and Skia rendering paths."
            )

        if args.renderer == "mixed":
            eprint(
                f"Generator: {len(self.fonts)} fonts × 2 renderers; "
                "equal renderer/font sampling."
            )
        else:
            eprint(
                f"Generator: {len(self.fonts)} fonts with "
                f"{args.renderer}; equal font sampling."
            )

    def _position(self, text, advance, width, height, is_skia):
        """
        Preserve the original layout variation while explicitly sampling
        fractional x/y phases.

        Bound the optional leftward shift so it cannot by itself move the
        text advance outside the canvas.
        """
        if advance > width - 8:
            raise ValueError(
                f"Generated string advance {advance:.2f}px exceeds "
                f"the {width}px canvas with margins. "
                "Reduce text length/font size or increase canvas width."
            )

        x = (width - advance) / 2

        if is_skia:
            # Skia drawString() uses a baseline origin.
            y = (height + self.args.font_size / 2) / 2
        else:
            # Preserve the existing PIL text-origin convention.
            y = (height - self.args.font_size) / 2

        y += random.randint(-8, 8)

        if len(text) < 70:
            max_left_shift = min(
                60,
                max(0, int(math.floor(x - 4))),
            )
            x -= random.randint(0, max_left_shift)

        # Explicitly sample the actual fractional coordinate.
        # PIL may quantize some phases; Skia has subpixel positioning on.
        x = math.floor(x) + random.random()
        y = math.floor(y) + random.random()

        return x, y

    def render(self, text):
        engine = self.args.renderer

        if engine == "mixed":
            engine = random.choice(("pil", "skia"))

        # The same font pool is used by both engines.
        entry = random.choice(self.fonts)

        width, height = 800, 64
        shade = random.randint(0, 100)

        if engine == "skia":
            skia = self.skia
            font = entry["skia"]

            glyphs = font.textToGlyphs(text)
            advance = sum(font.getWidths(glyphs))

            x, baseline = self._position(
                text,
                advance,
                width,
                height,
                is_skia=True,
            )

            surface = skia.Surface(width, height)
            canvas = surface.getCanvas()
            canvas.clear(skia.ColorWHITE)

            paint = skia.Paint(
                AntiAlias=True,
                Color=skia.Color(shade, shade, shade),
            )
            canvas.drawString(
                text,
                x,
                baseline,
                font,
                paint,
            )

            pixels = surface.makeImageSnapshot().toarray(
                colorType=skia.kRGB_888x_ColorType
            )[:, :, :3]

            return Image.fromarray(pixels).convert("RGB")

        font = entry["pil"]
        advance = font.getlength(text)

        x, y = self._position(
            text,
            advance,
            width,
            height,
            is_skia=False,
        )

        image = Image.new(
            "RGB",
            (width, height),
            "white",
        )
        draw = ImageDraw.Draw(image)
        draw.text(
            (x, y),
            text,
            font=font,
            fill=(shade, shade, shade),
        )

        return image


def generate(args):
    renderer = SyntheticRenderer(args)

    for split, count in (
        ("train", args.train_count),
        ("val", args.val_count),
    ):
        split_dir = Path(args.data) / split
        image_dir = split_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        manifest = split_dir / "labels.jsonl"
        eprint(f"Generating {count} {split} samples")

        with manifest.open("w", encoding="utf-8") as f:
            for index in range(count):
                # text = random_text()
                text = random_text()
                image = renderer.render(text)

                filename = f"{split}_{index:06d}.png"
                image.save(image_dir / filename)

                f.write(json.dumps({
                    "image": f"images/{filename}",
                    "text": text,
                }) + "\n")

                if (index + 1) % 1000 == 0:
                    eprint(f"  {index + 1}/{count}")

        eprint(f"Wrote {manifest}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compact CNN-BiLSTM-CTC OCR"
    )

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate", action="store_true")
    action.add_argument("--train", action="store_true")
    action.add_argument("--validate", action="store_true")
    action.add_argument("--predict", action="store_true")
    action.add_argument("--sample", type=str)

    parser.add_argument("--data", default="ocr_dataset_ctc")
    parser.add_argument("--model", help="Weights for inference or fine-tuning")
    parser.add_argument("--resume", help="Snapshot for full training resume")
    parser.add_argument("--out", help="Training output directory")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)

    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--split", default="val")
    parser.add_argument("--errors", help="Validation failures JSONL")

    parser.add_argument("--single-line", action="store_true")
    parser.add_argument("--line-gap", type=int, default=0)
    parser.add_argument("--debug-dir")

    parser.add_argument(
        "--renderer",
        choices=["skia", "pil", "mixed"],
        default="mixed",
    )
    parser.add_argument("--font", action="append")
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--train-count", type=int, default=7500)
    parser.add_argument("--val-count", type=int, default=1500)

    parser.add_argument("files", nargs="*")

    parser.add_argument(
        "--regressions",
        help=(
            "Directory containing regression case subdirectories, "
            "each with line.png and actual.txt"
        ),
    )
    parser.add_argument(
        "--regression-fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction of training exposures assigned to regression "
            "examples when --regressions is supplied; default: 0.10"
        ),
    )

    args = parser.parse_args()

    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    if args.resume and not args.train:
        parser.error("--resume is only valid with --train")
    if not 0.0 < args.regression_fraction < 1.0:
        parser.error(
            "--regression-fraction must be greater than 0 and less than 1"
        )

    return args


def main():
    args = parse_args()

    torch.set_num_threads(max(1, args.threads))
    seed_everything(args.seed)
    args.device_obj = choose_device(args.device)

    if args.generate:
        generate(args)
    elif args.sample is not None:
        renderer = SyntheticRenderer(args)
        renderer.render(args.sample).save("sample.png")
        eprint("Wrote sample.png")
    elif args.train:
        train(args)
    elif args.validate:
        validate(args)
    elif args.predict:
        predict(args)

def _start_windows_parent_watchdog():
    if sys.platform != "win32":
        return

    import ctypes
    import threading
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    TH32CS_SNAPPROCESS = 0x00000002
    SYNCHRONIZE = 0x00100000
    INFINITE = 0xFFFFFFFF

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == ctypes.c_void_p(-1).value:
        return

    parents = {}
    names = {}
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        if kernel32.Process32FirstW(snap, ctypes.byref(entry)):
            while True:
                parents[entry.th32ProcessID] = entry.th32ParentProcessID
                names[entry.th32ProcessID] = entry.szExeFile.lower()
                if not kernel32.Process32NextW(snap, ctypes.byref(entry)):
                    break
    finally:
        kernel32.CloseHandle(snap)

    # Watch our immediate parent (.venv python stub) and uv.exe if present.
    # Stop before wslhost.exe/wsl.exe, which may outlive individual commands.
    handles = []
    pid = os.getpid()
    for _ in range(4):
        ppid = parents.get(pid)
        if not ppid or ppid == pid or ppid in parents and names.get(ppid) in {"wslhost.exe", "wsl.exe"}:
            break
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, ppid)
        if handle:
            handles.append(handle)
        if names.get(ppid) == "uv.exe":
            break
        pid = ppid

    if not handles:
        return

    def watch():
        arr = (wintypes.HANDLE * len(handles))(*handles)
        kernel32.WaitForMultipleObjects(len(handles), arr, False, INFINITE)
        os._exit(130)

    threading.Thread(target=watch, daemon=True).start()

if __name__ == "__main__":
    try:
        _start_windows_parent_watchdog()
        main()
    except BrokenPipeError:
        # redirect stdout to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(0)
    except KeyboardInterrupt:
        eprint("\nInterrupted.")
        sys.exit(130)
