#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn

try:
    import librosa
except ImportError as exc:
    raise SystemExit(
        "librosa is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def compute_log_mel(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)


def load_audio(audio_path: Path, sample_rate: int) -> Tuple[np.ndarray, int]:
    waveform, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    return waveform, sr


def mel_to_tensor(mel_db: np.ndarray, target_frames: int) -> torch.Tensor:
    mean = mel_db.mean()
    std = mel_db.std() + 1e-8
    mel_db = (mel_db - mean) / std
    n_mels_dim, time = mel_db.shape
    if time < target_frames:
        pad = target_frames - time
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant")
    elif time > target_frames:
        mel_db = mel_db[:, :target_frames]
    return torch.from_numpy(mel_db).float().unsqueeze(0).unsqueeze(0)


def load_label_map(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        class_to_idx: Dict[str, int] = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict instrument from audio using a trained CNN. "
            "Pass one or more files and/or directories; directories are scanned recursively."
        )
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help=(
            "Audio inputs: files or directories. Directories are searched recursively for "
            "common audio extensions."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "trained_models" / "best_model.pt",
        help="Path to trained model weights (.pt)",
    )
    parser.add_argument(
        "--label_map",
        type=Path,
        default=Path(__file__).resolve().parent / "trained_models" / "label_map.json",
        help="Path to label map JSON produced during training",
    )
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--fmax", type=float, default=8000.0)
    parser.add_argument("--target_frames", type=int, default=512)
    parser.add_argument(
        "--segment_seconds",
        type=float,
        default=0.0,
        help=(
            "If > 0, split audio into overlapping segments of this many seconds and "
            "aggregate predictions across segments."
        ),
    )
    parser.add_argument(
        "--segment_hop_seconds",
        type=float,
        default=None,
        help=(
            "Hop between segments in seconds. Defaults to half of --segment_seconds."
        ),
    )
    parser.add_argument(
        "--multi_threshold",
        type=float,
        default=0.35,
        help="Probability threshold for multi-label output when not using --multi_topk.",
    )
    parser.add_argument(
        "--multi_topk",
        type=int,
        default=0,
        help=(
            "If > 0, return top-K classes regardless of threshold. If 0, use thresholding."
        ),
    )
    parser.add_argument(
        "--csv_out",
        type=Path,
        default=None,
        help="Optional path to write a CSV summary of predictions.",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".wav,.mp3,.flac,.m4a,.ogg",
        help="Comma-separated audio extensions to include when scanning directories.",
    )
    return parser.parse_args()


def gather_audio_files(inputs: List[Path], extensions: List[str]) -> List[Path]:
    files: List[Path] = []
    exts = tuple(ext.strip().lower() for ext in extensions if ext.strip())
    for inp in inputs:
        if inp.is_file():
            if inp.suffix.lower() in exts:
                files.append(inp)
        elif inp.is_dir():
            for p in inp.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    files.append(p)
        else:
            # non-existent inputs will be handled by empty list result
            pass
    return sorted(set(files))


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.label_map.exists():
        raise SystemExit(f"Label map not found: {args.label_map}")

    _, idx_to_class = load_label_map(args.label_map)
    num_classes = len(idx_to_class)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SmallCNN(num_classes=num_classes).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()

    extensions = [e for e in args.exts.split(",") if e]
    audio_files = gather_audio_files(args.inputs, extensions)
    if not audio_files:
        raise SystemExit(
            "No audio files found from the provided inputs. Pass valid file(s) or directory(ies), "
            "or adjust --exts."
        )

    rows = []
    for audio_path in audio_files:
        try:
            waveform, sr = load_audio(audio_path, args.sample_rate)
            fmax_value = None if args.fmax is not None and args.fmax <= 0 else args.fmax

            if args.segment_seconds and args.segment_seconds > 0.0:
                # Compute mel once for whole audio, then window across frames
                full_mel = compute_log_mel(
                    waveform=waveform,
                    sample_rate=sr,
                    n_mels=args.n_mels,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    fmin=args.fmin,
                    fmax=fmax_value,
                )
                frames_per_seg = max(1, int(round(args.segment_seconds * sr / args.hop_length)))
                hop_frames = int(round((args.segment_hop_seconds or (args.segment_seconds / 2.0)) * sr / args.hop_length))
                if hop_frames <= 0:
                    hop_frames = max(1, frames_per_seg // 2)

                probs_accum = []
                start = 0
                T = full_mel.shape[1]
                while start < T:
                    seg = full_mel[:, start : start + frames_per_seg]
                    x = mel_to_tensor(seg, target_frames=frames_per_seg).to(device)
                    with torch.no_grad():
                        logits = model(x)
                        seg_probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    probs_accum.append(seg_probs)
                    if start + frames_per_seg >= T:
                        break
                    start += hop_frames
                probs = np.mean(np.stack(probs_accum, axis=0), axis=0)
            else:
                # Single pass on whole audio mel
                mel_db = compute_log_mel(
                    waveform=waveform,
                    sample_rate=sr,
                    n_mels=args.n_mels,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    fmin=args.fmin,
                    fmax=fmax_value,
                )
                x = mel_to_tensor(mel_db, target_frames=args.target_frames).to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            # Multi-label decision
            selected: List[int]
            if args.multi_topk and args.multi_topk > 0:
                selected = list(np.argsort(-probs)[: args.multi_topk])
            else:
                selected = [i for i, p in enumerate(probs) if p >= args.multi_threshold]
                if not selected:
                    selected = [int(np.argmax(probs))]

            labels = [idx_to_class[int(i)] for i in selected]
            printable = ", ".join(f"{lbl} ({probs[int(i)]:.3f})" for i, lbl in zip(selected, labels))
            print(f"{audio_path}: {printable}")

            row = {"path": str(audio_path), "prediction": ";".join(labels)}
            for i in range(len(probs)):
                row[idx_to_class[i]] = float(probs[i])
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"ERR - {audio_path}: {exc}")

    if args.csv_out and rows:
        # Ensure consistent column order: path, prediction, then classes
        class_cols = [idx_to_class[i] for i in range(num_classes)]
        fieldnames = ["path", "prediction", *class_cols]
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                # fill missing class columns with 0.0 if any
                for c in class_cols:
                    if c not in r:
                        r[c] = 0.0
                writer.writerow(r)
        print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()


