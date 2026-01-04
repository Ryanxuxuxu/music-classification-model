import os
import numpy as np
from typing import Any

import torch
import librosa

from genre.predict_genre import predict_genre_from_audio
from instrument.predict_instrument import load_label_map, SmallCNN, compute_log_mel, mel_to_tensor
from structure.src.infer_on_mp3 import load_checkpoint_model, sliding_windows, postprocess_to_segments

def music_prediction(audio_path):
  metainfo: dict[str, dict[str, Any]] = {}
  music_prediction = MusicPrediction(audio_path)
  print("Running genre prediction...")
  metainfo["genre"] = music_prediction.predict_genre()
  print("Running instrument prediction...")
  metainfo["instrument"] = music_prediction.predict_instrument()
  print("Running structure prediction...") 
  metainfo["structure"] = music_prediction.predict_structure()
  print("Running bpm prediction...")
  metainfo["bpm"] = music_prediction.output_bpm()
  print("All predictions completed.")
  return metainfo

class MusicPrediction():
  def __init__(self, audio_path):
    self.audio_path = audio_path

  def predict_genre(self):
    model_path = 'genre/genre_classifier_model.pth'
    genre, confidence = predict_genre_from_audio(self.audio_path, model_path)
    if genre:
      return {"result": genre, "confidence": round(confidence, 3)}
    else:
      return {"result": "None", "confidence": "None"}

  def predict_instrument(self):
    args = {
      "model": "instrument/trained_models/best_model.pt",
      "label_map": "instrument/trained_models/label_map.json",
      "sample_rate": 22050,
      "n_mels": 128,
      "n_fft": 2048,
      "hop_length": 512,
      "fmin": 20.0,
      "fmax": 8000.0,
      "target_frames": 512,
      "multi_topk": 3,
      # "multi_threshold": 0.35,
    }

    _, idx_to_class = load_label_map(args["label_map"])
    num_classes = len(idx_to_class)
    device = torch.device("cpu")
    model = SmallCNN(num_classes=num_classes).to(device)
    state = torch.load(args["model"], map_location=device)
    model.load_state_dict(state)
    model.eval()

    waveform, sr = librosa.load(self.audio_path, sr=args["sample_rate"], mono=True)
    fmax_value = None if args["fmax"] is not None and args["fmax"] <= 0 else args["fmax"]
    mel_db = compute_log_mel(
      waveform=waveform,
      sample_rate=sr,
      n_mels=args["n_mels"],
      n_fft=args["n_fft"],
      hop_length=args["hop_length"],
      fmin=args["fmin"],
      fmax=fmax_value,
    )
    x = mel_to_tensor(mel_db, target_frames=args["target_frames"]).to(device)
    with torch.no_grad():
      logits = model(x)
      probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    selected: list[int]
    if args["multi_topk"] and args["multi_topk"] > 0:
      selected = list(np.argsort(-probs)[: args["multi_topk"]])
    else:
      selected = [i for i, p in enumerate(probs) if p >= args["multi_threshold"]]
      if not selected:
        selected = [int(np.argmax(probs))]
    labels = [idx_to_class[int(i)] for i in selected]
    confidence = [str(np.round(float(probs[i]), 3)) for i in selected]
    instrument = [{"instrument": labels[i], "confidence": confidence[i]} for i in range(len(labels))]

    return instrument

  def predict_structure(self):
    args = {
      "SR": 24000,
      "N_FFT": 2048,
      "HOP_LENGTH": 1024,
      "WINDOW": "hann",
      "CENTER": True,
      "PAD_MODE": "constant",
      "POWER": 2.0,
      "N_MELS": 128,
      "MEL_FMIN": 30,
      "MEL_FMAX": 12000,
      "checkpoint_path": "structure/models/best_model.pth",
      "stride": 64,
    }

    y, _ = librosa.load(self.audio_path, sr=args["SR"])
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=args["SR"],
        n_fft=args["N_FFT"],
        hop_length=args["HOP_LENGTH"],
        window=args["WINDOW"],
        center=args["CENTER"],
        pad_mode=args["PAD_MODE"],
        power=args["POWER"],
        n_mels=args["N_MELS"],
        fmin=args["MEL_FMIN"],
        fmax=args["MEL_FMAX"],
    )

    model, input_shape, id_to_label, device = load_checkpoint_model(args["checkpoint_path"])
    seq_len, n_mels = input_shape

    feats_TxM = mel.T  # (T, n_mels)

    windows = sliding_windows(feats_TxM, window_size=seq_len, stride=args["stride"])
    if not windows:
        raise ValueError("No windows produced from features. Check audio length and parameters.")

    batch = np.stack([w[2] for w in windows], axis=0).astype(np.float32)  # (N, seq_len, n_mels)
    tensor = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        logits = model(tensor)  # (N, num_classes)
        pred_ids = logits.argmax(dim=1).cpu().numpy().tolist()

    starts = [w[0] for w in windows]
    ends = [w[1] for w in windows]

    output = postprocess_to_segments(pred_ids, starts, ends, id_to_label)
    segments = [{"time": round(float(item[0]), 3), "label": item[1]} for item in output]
    return segments

  def output_bpm(self):
    waveform, sr = librosa.load(self.audio_path, sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=waveform, sr=sr)
    bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    
    return {"result": round(bpm, 2)}


if __name__ == "__main__":
  audio_path =  "./blues.00000.wav"
  metainfo = music_prediction(audio_path)
  print(metainfo)