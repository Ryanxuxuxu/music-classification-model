import argparse
import os
import json
import pickle
import glob
from typing import List, Tuple, Optional

import numpy as np
import torch
import librosa

from .cnn_model import create_model


# Mel parameters (match the actual dataset configuration)
SR = 24000
N_FFT = 2048
HOP_LENGTH = 1024
WINDOW = "hann"
CENTER = True
PAD_MODE = "constant"
POWER = 2.0
N_MELS = 128  # Updated to match the actual dataset
MEL_FMIN = 30
MEL_FMAX = 12000


def compute_melspec(y: np.ndarray) -> np.ndarray:
    """Compute mel spectrogram with Harmonix params. Returns (n_mels, T)."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=WINDOW,
        center=CENTER,
        pad_mode=PAD_MODE,
        power=POWER,
        n_mels=N_MELS,
        fmin=MEL_FMIN,
        fmax=MEL_FMAX,
    )
    return mel


def save_audio_feature(mel: np.ndarray, mp3_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(mp3_path))[0]
    out_path = os.path.join(out_dir, f"{base}-mel.npy")
    np.save(out_path, mel)
    return out_path


def save_info_json(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "info.json")
    out_dict = {
        "librosa_version": librosa.__version__,
        "numpy_version": np.__version__,
        "SR": SR,
        "N_FFT": N_FFT,
        "HOP_LENGTH": HOP_LENGTH,
        "WINDOW": WINDOW,
        "CENTER": CENTER,
        "PAD_MODE": PAD_MODE,
        "POWER": POWER,
        "N_MELS": N_MELS,
        "MEL_FMIN": MEL_FMIN,
        "MEL_FMAX": MEL_FMAX,
    }
    with open(out_json, "w") as f:
        json.dump(out_dict, f, indent=4)


def sliding_windows(features_TxM: np.ndarray, window_size: int, stride: int) -> List[Tuple[int, int, np.ndarray]]:
    """
    Create sliding windows over time-axis features (T, M).
    Returns list of (start_frame, end_frame, window_features_TxM).
    """
    T = features_TxM.shape[0]
    windows = []
    for start in range(0, max(T - window_size + 1, 0), stride):
        end = start + window_size
        if end <= T:
            windows.append((start, end, features_TxM[start:end]))
    # If no windows (short audio), pad last window to size
    if not windows and T > 0:
        pad_amount = max(0, window_size - T)
        pad_feat = np.pad(features_TxM, ((0, pad_amount), (0, 0)), mode="edge")
        windows.append((0, window_size, pad_feat[:window_size]))
    return windows


def frames_to_time(frame_idx: int) -> float:
    return frame_idx * HOP_LENGTH / SR


def postprocess_to_segments(pred_ids: List[int], starts: List[int], ends: List[int], id_to_label: dict) -> List[Tuple[float, str]]:
    """
    Collapse consecutive identical class IDs into (time, label) change points.
    Returns list of (time_in_seconds, label) like Harmonix segments files.
    """
    if not pred_ids:
        return []
    segments = []
    current_id = pred_ids[0]
    current_start = starts[0]
    for i in range(1, len(pred_ids)):
        if pred_ids[i] != current_id:
            segments.append((frames_to_time(current_start), id_to_label.get(current_id, str(current_id))))
            current_id = pred_ids[i]
            current_start = starts[i]
    # Last segment
    segments.append((frames_to_time(current_start), id_to_label.get(current_id, str(current_id))))
    return segments


def save_segments_txt(segments: List[Tuple[float, str]], mp3_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(mp3_path))[0]
    out_path = os.path.join(out_dir, f"{base}.txt")
    with open(out_path, "w") as f:
        for t, label in segments:
            f.write(f"{t:.6f} {label}\n")
    return out_path


def _merge_checkpoint_chunks(chunk_files: List[str], device: str = None) -> dict:
    """
    合并拆分后的checkpoint文件。
    
    Args:
        chunk_files: 分片文件路径列表（已排序）
        device: 加载设备
    
    Returns:
        合并后的checkpoint字典
    """
    print(f"检测到拆分模型，正在合并 {len(chunk_files)} 个分片文件...")
    
    # 读取所有分片
    chunks_data = []
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
            chunks_data.append(chunk_data)
    
    # 验证分片完整性
    total_chunks = chunks_data[0]['total_chunks']
    total_size = chunks_data[0]['total_size']
    
    if len(chunks_data) != total_chunks:
        raise ValueError(f"分片数量不匹配: 期望 {total_chunks}, 实际 {len(chunks_data)}")
    
    # 合并所有分片数据
    checkpoint_bytes = b''.join([chunk['data'] for chunk in chunks_data])
    
    if len(checkpoint_bytes) != total_size:
        raise ValueError(f"合并后大小不匹配: 期望 {total_size}, 实际 {len(checkpoint_bytes)}")
    
    # 反序列化checkpoint
    checkpoint = pickle.loads(checkpoint_bytes)
    
    # 如果指定了device，将tensor移到对应设备
    if device is not None and 'model_state_dict' in checkpoint:
        checkpoint['model_state_dict'] = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in checkpoint['model_state_dict'].items()
        }
    
    print("合并完成！")
    return checkpoint


def load_checkpoint_model(checkpoint_path: str, device: str = None):
    """
    加载checkpoint模型，自动检测并支持拆分后的模型文件。
    
    如果checkpoint_path指向的文件不存在，会自动查找对应的分片文件（.part*.pth）。
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # 检查是否是拆分后的模型文件
    if os.path.exists(checkpoint_path):
        # 尝试作为普通checkpoint加载
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception:
            # 如果加载失败，可能是分片文件，尝试查找其他分片
            checkpoint = None
    else:
        checkpoint = None
    
    # 如果普通加载失败或文件不存在，尝试作为分片文件处理
    if checkpoint is None:
        # 检查是否存在分片文件
        base_dir = os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else '.'
        base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        
        # 查找所有匹配的分片文件
        pattern = os.path.join(base_dir, f"{base_name}.part*.pth")
        chunk_files = sorted(glob.glob(pattern))
        
        if chunk_files:
            # 找到分片文件，合并加载
            checkpoint = _merge_checkpoint_chunks(chunk_files, device)
        else:
            # 如果既不是普通文件也不是分片文件，抛出错误
            raise FileNotFoundError(
                f"Checkpoint文件不存在: {checkpoint_path}\n"
                f"也未找到分片文件: {pattern}"
            )
    
    input_shape = tuple(checkpoint.get('input_shape')) if isinstance(checkpoint.get('input_shape'), (list, tuple)) else checkpoint.get('input_shape')
    num_classes = int(checkpoint.get('num_classes'))
    label_mapping = checkpoint.get('label_mapping')

    model = create_model(input_shape, num_classes, model_type='classification')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    # Build id_to_label mapping
    id_to_label = None
    if isinstance(label_mapping, dict):
        # mapping likely {label: id} or {id: label}; normalize to id->label
        # Heuristic: if any key is str and value is int, it's label->id
        if label_mapping and all(isinstance(k, str) and isinstance(v, int) for k, v in label_mapping.items()):
            inv = {v: k for k, v in label_mapping.items()}
            id_to_label = inv
        else:
            id_to_label = {int(k): v for k, v in label_mapping.items()}
    else:
        # Fallback to indices
        id_to_label = {i: str(i) for i in range(num_classes)}

    return model, input_shape, id_to_label, device


def run_inference(mp3_path: str, checkpoint_path: str, audio_features_dir: str, segments_dir: str, stride: int = 64):
    # 1) Load audio and compute mel
    y, _ = librosa.load(mp3_path, sr=SR)
    mel = compute_melspec(y)  # (n_mels, T)

    # 2) Save mel and info.json
    save_audio_feature(mel, mp3_path, audio_features_dir)
    # Only write info.json if absent to avoid overwriting dataset's
    info_json_path = os.path.join(audio_features_dir, "info.json")
    if not os.path.exists(info_json_path):
        save_info_json(audio_features_dir)

    # 3) Prepare model and data windows
    model, input_shape, id_to_label, device = load_checkpoint_model(checkpoint_path)
    seq_len, n_mels = input_shape

    # Align axes: training expects (sequence_length, n_mels)
    feats_TxM = mel.T  # (T, n_mels)
    if feats_TxM.shape[1] != n_mels:
        raise ValueError(f"Model expects n_mels={n_mels}, but computed {feats_TxM.shape[1]}.")

    windows = sliding_windows(feats_TxM, window_size=seq_len, stride=stride)
    if not windows:
        raise ValueError("No windows produced from features. Check audio length and parameters.")

    # 4) Batch inference
    batch = np.stack([w[2] for w in windows], axis=0).astype(np.float32)  # (N, seq_len, n_mels)
    tensor = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        logits = model(tensor)  # (N, num_classes)
        pred_ids = logits.argmax(dim=1).cpu().numpy().tolist()

    starts = [w[0] for w in windows]
    ends = [w[1] for w in windows]

    print(starts)
    print(ends)
    print(pred_ids)

    # 5) Collapse to segments and save txt
    segments = postprocess_to_segments(pred_ids, starts, ends, id_to_label)
    save_segments_txt(segments, mp3_path, segments_dir)


def main():
    parser = argparse.ArgumentParser(description="Infer segments on an MP3 and write Harmonix-style outputs")
    parser.add_argument("--mp3", required=True, help="Path to input MP3 file (can be any MP3 file from anywhere on your system)")
    parser.add_argument("--checkpoint", default="../models/best_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--audio_features_dir", default="../audio_features", help="Output dir for mel features")
    parser.add_argument("--segments_dir", default="../segments", help="Output dir for segments txt")
    parser.add_argument("--stride", type=int, default=64, help="Sliding window stride (frames)")
    
    args = parser.parse_args()
    
    # Validate MP3 file exists
    if not os.path.exists(args.mp3):
        print(f"Error: MP3 file not found: {args.mp3}")
        print("Please check the file path and try again.")
        return
    
    # Validate it's actually an MP3 file
    if not args.mp3.lower().endswith('.mp3'):
        print(f"Warning: File '{args.mp3}' doesn't have .mp3 extension")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            return

    run_inference(
        mp3_path=args.mp3,
        checkpoint_path=args.checkpoint,
        audio_features_dir=args.audio_features_dir,
        segments_dir=args.segments_dir,
        stride=args.stride,
    )

    print("Inference complete.")


if __name__ == "__main__":
    main()


