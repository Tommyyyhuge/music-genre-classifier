"""Local inference: mp3 file -> mel spectrogram -> genre prediction."""
import torch
import numpy as np
import librosa
from PIL import Image
from torchvision import transforms

from src.training.config import GENRE_NAMES
from src.models.ast_model import build_ast_model


def _get_logits(output):
    return output.logits if hasattr(output, "logits") else output


class GenrePredictor:
    """Load a trained AST model and predict genre from mp3 files."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = build_ast_model(
            "MIT/ast-finetuned-audioset-10-10-0.4593",
            num_classes=16,
            device=self.device,
        )
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def predict_segment(self, audio_segment: np.ndarray, sr: int = 22050) -> tuple:
        """Predict genre from an 11.4s audio segment.

        Returns (genre_name, confidence, top3_list).
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=11025
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_min, mel_max = mel_db.min(), mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
        mel_img = Image.fromarray((mel_norm * 255).astype(np.uint8))

        x = self.transform(mel_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(x)
            logits = _get_logits(output)
            probs = torch.softmax(logits, dim=1)[0]

        top3_idx = probs.argsort(descending=True)[:3].cpu().numpy()
        top3 = [(GENRE_NAMES[i], probs[i].item()) for i in top3_idx]

        return GENRE_NAMES[top3_idx[0]], probs[top3_idx[0]].item(), top3

    def predict(self, audio_path: str, top_k: int = 3) -> list:
        """Predict genre for a full mp3 file.

        Slides an 11.4s window with 50% overlap, averages logits.
        Returns top-K predictions with confidence scores.
        """
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        window_samples = int(11.4 * sr)
        hop_samples = window_samples // 2

        all_logits = []
        for start in range(0, len(y) - window_samples + 1, hop_samples):
            segment = y[start : start + window_samples]
            mel_spec = librosa.feature.melspectrogram(
                y=segment, sr=sr, n_mels=128, n_fft=2048, hop_length=512, fmax=11025
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_min, mel_max = mel_db.min(), mel_db.max()
            mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)
            mel_img = Image.fromarray((mel_norm * 255).astype(np.uint8))

            x = self.transform(mel_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(x)
                logits = _get_logits(output).cpu()
            all_logits.append(logits)

        if not all_logits:
            return self._predict_short(y, sr, top_k)

        avg_logits = torch.stack(all_logits).mean(dim=0)[0]
        probs = torch.softmax(avg_logits, dim=0)

        top_k_idx = probs.argsort(descending=True)[:top_k].cpu().numpy()
        return [(GENRE_NAMES[i], round(probs[i].item(), 4)) for i in top_k_idx]

    def _predict_short(self, y: np.ndarray, sr: int, top_k: int) -> list:
        """Handle audio shorter than 11.4s by zero-padding."""
        target_len = int(11.4 * sr)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        return self.predict_segment(y, sr)[2]


def predict(audio_path: str, checkpoint_path: str = "./checkpoints/best_model.pt",
            device: str = "cpu") -> list:
    """Convenience function for quick inference."""
    predictor = GenrePredictor(checkpoint_path, device)
    return predictor.predict(audio_path)
