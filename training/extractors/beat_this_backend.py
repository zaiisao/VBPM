"""Beat This! extractor backend for end-to-end CHART training.

Uses the vendored Beat This! beat tracker (extractors/beat_this/) as the
acoustic frontend in place of WaveBeat. Two frame-rate modes are supported:

  --extractor_fps_mode native    Run the entire pipeline at Beat This' native
                                 50 fps (sample_rate=22050, hop=441).

  --extractor_fps_mode resample  Resample Beat This' 50 fps logits up to
                                 WaveBeat's 86.13 fps (sample_rate=22050,
                                 hop=256) so SVT, target alignment, and the
                                 fixed 512-frame crop window stay byte-equiv
                                 with the WaveBeat backend.

The dataset is reused from the WaveBeat vendor (DownbeatDataset), as it
already produces ``(audio, [beat, downbeat]_target)`` tuples and exposes the
``audio_sample_rate`` / ``target_factor`` attributes that
AudioPhaseBridgeDataset reads via getattr.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from training.dataset import (
    AudioPhaseBridgeDataset,
    MultiSourceAudioDataset,
    discover_wavebeat_dataset_specs,
)


_BEAT_THIS_CHECKPOINT_URL_BASE = (
    "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"
)


# Beat This native: sr=22050, hop=441 -> 50.0 fps.
# WaveBeat native:  sr=22050, hop=256 -> 86.13 fps.
_FPS_MODE_CONFIG: dict[str, dict[str, int]] = {
    "native": {"target_factor": 441},
    "resample": {"target_factor": 256},
}


def _import_beat_this(beat_this_root: str) -> tuple[type, type]:
    root = Path(beat_this_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"beat_this_root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from beat_this.model.beat_tracker import BeatThis  # type: ignore[import-not-found]
    from beat_this.model.loss import ShiftTolerantBCELoss  # type: ignore[import-not-found]

    return BeatThis, ShiftTolerantBCELoss


def _import_wavebeat_dataset(wavebeat_root: str) -> type:
    root = Path(wavebeat_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"wavebeat_root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from wavebeat.data import DownbeatDataset  # type: ignore[import-not-found]
    return DownbeatDataset


def _center_crop_last_dim(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-1] == length:
        return x
    if x.shape[-1] < length:
        raise ValueError(
            f"Cannot crop tensor of length {x.shape[-1]} to larger length {length}"
        )
    start = (x.shape[-1] - length) // 2
    return x[..., start : start + length]


class _LogMelSpectBatched(torch.nn.Module):
    """Batched re-implementation of beat_this.preprocessing.LogMelSpect.

    The upstream module hardcodes ``.T`` on the mel output, which only works
    on unbatched (T, n_mels) input. This version takes ``[B, N]`` waveforms
    and returns ``[B, T, n_mels]`` with the same numerical recipe.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 441,
        f_min: float = 30.0,
        f_max: float = 11000.0,
        n_mels: int = 128,
        log_multiplier: float = 1000.0,
    ) -> None:
        super().__init__()
        self.spect = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            mel_scale="slaney",
            normalized="frame_length",
            power=1,
        )
        self.log_multiplier = log_multiplier

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.spect(audio)  # [B, n_mels, T]
        return torch.log1p(self.log_multiplier * mel.transpose(-1, -2))


class BeatThisBackend:
    name = "beat_this"

    def __init__(self) -> None:
        self._criterion: torch.nn.Module | None = None
        self._spect: _LogMelSpectBatched | None = None
        self._fps_mode: str = "native"

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--beat_this_root",
            type=str,
            default=None,
            help="Optional override for vendored Beat This package path "
                 "(default: extractors/beat_this).",
        )
        parser.add_argument(
            "--wavebeat_root",
            type=str,
            default=None,
            help="Path to vendored WaveBeat package; reused for DownbeatDataset "
                 "(default: extractors/wavebeat).",
        )
        parser.add_argument(
            "--extractor_fps_mode",
            choices=["native", "resample"],
            default="native",
            help="native: pipeline runs at Beat This' 50 fps. "
                 "resample: Beat This logits are linear-interp upsampled to "
                 "WaveBeat's 86.13 fps so the rest of the pipeline is unchanged.",
        )
        parser.add_argument(
            "--beat_this_checkpoint",
            type=str,
            default="final0",
            help="Beat This pretrained checkpoint shortname, URL, or local path. "
                 "Used as a fallback when --extractor_ckpt is unset. "
                 "Shortnames (e.g. 'final0') are downloaded via torch.hub.",
        )
        parser.add_argument(
            "--beat_this_loss_tolerance",
            type=int,
            default=3,
            help="ShiftTolerantBCELoss tolerance (frames). Beat This default: "
                 "3 frames @ 50 fps (~60 ms).",
        )

        # Dataset args (mirror the WaveBeat backend; reused via DownbeatDataset).
        parser.add_argument("--dataset_root", type=str, default=None)
        parser.add_argument(
            "--dataset_include",
            type=str,
            default="ballroom,beatles,gtzan,hains,rwc_popular",
        )
        parser.add_argument("--audio_dir", type=str, default=None)
        parser.add_argument("--annot_dir", type=str, default=None)
        parser.add_argument("--wavebeat_dataset", type=str, default="ballroom")
        parser.add_argument("--audio_sample_rate", type=int, default=22050)
        parser.add_argument(
            "--target_factor",
            type=int,
            default=None,
            help="Target frame-rate factor (sr/factor = fps). "
                 "Default: auto from --extractor_fps_mode (441 native, 256 resample).",
        )
        parser.add_argument(
            "--train_length",
            type=int,
            default=661500,
            help="Audio sample length per training example. "
                 "Default 661500 (~30 s @ 22050 Hz, ~1500 frames @ 50 fps, "
                 "matching Beat This' chunk_size).",
        )
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--examples_per_epoch", type=int, default=1000)
        parser.add_argument("--preload", action="store_true")
        parser.add_argument("--augment", action="store_true")
        parser.add_argument("--dry_run", action="store_true")

    # ------------------------------------------------------------------
    # Path resolution + arg defaults
    # ------------------------------------------------------------------

    def _resolve_extractor_root(self, args: argparse.Namespace) -> str:
        if args.beat_this_root is not None and str(args.beat_this_root).strip() != "":
            return str(args.beat_this_root)
        return str(Path("extractors") / self.name)

    def _resolve_wavebeat_root(self, args: argparse.Namespace) -> str:
        if args.wavebeat_root is not None and str(args.wavebeat_root).strip() != "":
            return str(args.wavebeat_root)
        return str(Path("extractors") / "wavebeat")

    def _resolve_args(self, args: argparse.Namespace) -> None:
        cfg = _FPS_MODE_CONFIG[args.extractor_fps_mode]
        if args.target_factor is None or args.target_factor <= 0:
            args.target_factor = cfg["target_factor"]
        # audio_sample_rate stays at the user-supplied value (default 22050).
        self._fps_mode = args.extractor_fps_mode

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _make_downbeat_dataset(
        self,
        args: argparse.Namespace,
        DownbeatDataset: type,
        spec_audio_dir: str,
        spec_annot_dir: str,
        spec_dataset: str,
        subset: str,
        augment: bool,
    ) -> Dataset:
        return DownbeatDataset(
            audio_dir=spec_audio_dir,
            annot_dir=spec_annot_dir,
            dataset=spec_dataset,
            audio_sample_rate=args.audio_sample_rate,
            target_factor=args.target_factor,
            subset=subset,
            length=args.train_length,
            preload=args.preload,
            augment=augment,
            examples_per_epoch=args.examples_per_epoch,
            half=False,
            dry_run=args.dry_run,
        )

    def _build_source_dataset(
        self,
        args: argparse.Namespace,
        DownbeatDataset: type,
        subset: str,
        augment: bool,
    ) -> Dataset | None:
        if args.dataset_root is not None:
            include_keys: set[str] | None
            if args.dataset_include.strip().lower() == "all":
                include_keys = None
            else:
                include_keys = {
                    item.strip().lower()
                    for item in args.dataset_include.split(",")
                    if item.strip()
                }

            specs = discover_wavebeat_dataset_specs(
                root_dir=args.dataset_root, include_keys=include_keys,
            )

            source_datasets: list[Dataset] = []
            source_keys: list[str] = []
            for spec in specs:
                try:
                    candidate = self._make_downbeat_dataset(
                        args=args,
                        DownbeatDataset=DownbeatDataset,
                        spec_audio_dir=str(spec.audio_dir),
                        spec_annot_dir=str(spec.annot_dir),
                        spec_dataset=spec.wavebeat_dataset,
                        subset=subset,
                        augment=augment,
                    )
                except Exception as exc:
                    print(f"Skipping dataset '{spec.key}': {exc}")
                    continue

                if len(getattr(candidate, "audio_files", [])) == 0:
                    print(f"Skipping dataset '{spec.key}': no {subset} audio files selected")
                    continue
                try:
                    _ = candidate[0]
                except Exception as exc:
                    print(f"Skipping dataset '{spec.key}': sample load failed ({exc})")
                    continue

                source_datasets.append(candidate)
                source_keys.append(spec.key)

            if not source_datasets:
                return None

            print(f"{subset.capitalize()} datasets: {', '.join(source_keys)}")
            return MultiSourceAudioDataset(
                source_datasets=source_datasets, source_keys=source_keys,
            )

        if args.audio_dir is None or args.annot_dir is None:
            return None
        return self._make_downbeat_dataset(
            args=args,
            DownbeatDataset=DownbeatDataset,
            spec_audio_dir=args.audio_dir,
            spec_annot_dir=args.annot_dir,
            spec_dataset=args.wavebeat_dataset,
            subset=subset,
            augment=augment,
        )

    def build_dataloader(self, args: argparse.Namespace) -> DataLoader:
        self._resolve_args(args)
        DownbeatDataset = _import_wavebeat_dataset(self._resolve_wavebeat_root(args))

        phases_dir = args.phases_dir
        if phases_dir is None and args.dataset_root is not None:
            phases_dir = args.dataset_root
        if phases_dir is None:
            raise ValueError(
                "Provide --dataset_root (preferred) or --phases_dir for mode=end2end"
            )

        source_dataset = self._build_source_dataset(
            args=args, DownbeatDataset=DownbeatDataset, subset="train", augment=args.augment,
        )
        if source_dataset is None:
            raise RuntimeError(
                "No usable training dataset. Check --dataset_root / --audio_dir / --annot_dir."
            )

        dataset = AudioPhaseBridgeDataset(
            source_dataset=source_dataset, phases_dir=phases_dir,
        )

        dist_rank = getattr(args, "dist_rank", 0)
        dist_world_size = getattr(args, "dist_world_size", 1)
        if dist_world_size > 1:
            sampler: DistributedSampler | None = DistributedSampler(
                dataset, num_replicas=dist_world_size, rank=dist_rank, shuffle=True,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def build_val_dataloader(self, args: argparse.Namespace) -> DataLoader | None:
        self._resolve_args(args)
        DownbeatDataset = _import_wavebeat_dataset(self._resolve_wavebeat_root(args))

        phases_dir = args.phases_dir
        if phases_dir is None and args.dataset_root is not None:
            phases_dir = args.dataset_root
        if phases_dir is None:
            return None

        source_dataset = self._build_source_dataset(
            args=args, DownbeatDataset=DownbeatDataset, subset="val", augment=False,
        )
        if source_dataset is None:
            return None

        dataset = AudioPhaseBridgeDataset(
            source_dataset=source_dataset, phases_dir=phases_dir,
        )

        # Val/test subsets emit variable-length audio (no cropping); keep batch_size=1.
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def build_model(self, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
        self._resolve_args(args)
        BeatThis, ShiftTolerantBCELoss = _import_beat_this(
            self._resolve_extractor_root(args)
        )

        self._criterion = ShiftTolerantBCELoss(
            pos_weight=1, tolerance=args.beat_this_loss_tolerance,
        ).to(device)
        self._spect = _LogMelSpectBatched(sample_rate=args.audio_sample_rate).to(device)

        return BeatThis().to(device)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        args: argparse.Namespace,
        device: torch.device,
    ) -> None:
        ckpt_path = args.extractor_ckpt
        if ckpt_path is None or str(ckpt_path).strip() == "":
            ckpt_path = args.beat_this_checkpoint
        if ckpt_path is None or str(ckpt_path).strip() == "":
            return

        ckpt = self._load_beat_this_checkpoint(ckpt_path, device)
        state_dict = ckpt.get("state_dict", ckpt)

        cleaned: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("model."):
                new_key = new_key[len("model.") :]
            cleaned[new_key] = value

        try:
            model.load_state_dict(cleaned, strict=True)
        except RuntimeError:
            model.load_state_dict(cleaned, strict=False)

    @staticmethod
    def _load_beat_this_checkpoint(
        ckpt_path: str, device: torch.device,
    ) -> dict[str, object]:
        path = Path(ckpt_path)
        if path.is_file():
            return torch.load(str(path), map_location=device, weights_only=False)

        if str(ckpt_path).startswith(("http://", "https://")):
            ckpt_url = str(ckpt_path)
            file_name: str | None = None
        else:
            # Treat as Beat This shortname (e.g. "final0").
            ckpt_url = f"{_BEAT_THIS_CHECKPOINT_URL_BASE}/{ckpt_path}.ckpt"
            file_name = f"beat_this-{ckpt_path}.ckpt"

        return torch.hub.load_state_dict_from_url(
            ckpt_url, file_name=file_name, map_location=device,
        )

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------

    def compute_loss_and_activations(
        self,
        model: torch.nn.Module,
        audio: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._spect is None or self._criterion is None:
            raise RuntimeError(
                "BeatThisBackend.build_model must be called before "
                "compute_loss_and_activations."
            )

        if audio.dim() == 3:
            audio = audio.squeeze(1)  # [B, 1, N] -> [B, N]
        elif audio.dim() != 2:
            raise ValueError(f"Expected audio shape [B, 1, N] or [B, N], got {tuple(audio.shape)}")

        # Lazy device sync (handles checkpoint reloads / device changes).
        spect_device = next(self._spect.buffers()).device
        if spect_device != audio.device:
            self._spect = self._spect.to(audio.device)

        spect = self._spect(audio)  # [B, T_native, 128]
        out = model(spect)
        beat_logits = out["beat"]          # [B, T_native]
        downbeat_logits = out["downbeat"]  # [B, T_native]

        if self._fps_mode == "resample":
            T_target = target.shape[-1]
            beat_logits = F.interpolate(
                beat_logits.unsqueeze(1),
                size=T_target,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            downbeat_logits = F.interpolate(
                downbeat_logits.unsqueeze(1),
                size=T_target,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            beat_target = target[:, 0, :].float()
            downbeat_target = target[:, 1, :].float()
        else:
            # Native 50 fps. Beat This' LogMelSpect (torchaudio MelSpectrogram
            # with center=True) emits one extra frame versus N//hop, so we
            # align the longer of (logits, target) down to the shorter.
            T_native = beat_logits.shape[-1]
            T_target_dim = target.shape[-1]
            T_aligned = min(T_native, T_target_dim)
            beat_logits = _center_crop_last_dim(beat_logits, T_aligned)
            downbeat_logits = _center_crop_last_dim(downbeat_logits, T_aligned)
            beat_target = _center_crop_last_dim(target[:, 0, :], T_aligned).float()
            downbeat_target = _center_crop_last_dim(target[:, 1, :], T_aligned).float()

        loss = self._criterion(beat_logits, beat_target) + self._criterion(
            downbeat_logits, downbeat_target,
        )

        beat_act = torch.sigmoid(beat_logits)
        db_act = torch.sigmoid(downbeat_logits)
        activations = torch.stack([beat_act, db_act], dim=-1)  # [B, T, 2]
        return loss, activations
