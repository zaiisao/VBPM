"""WaveBeat extractor backend for end-to-end CHART training."""

from __future__ import annotations

import argparse
import sys
import types
from collections import OrderedDict
from pathlib import Path


class _AnyAttrModule(types.ModuleType):
    """Stub module that returns a no-op callable for every attribute access.

    Used to satisfy pickle when loading PL 1.x checkpoints in a PL 2.x
    environment: old module paths no longer exist but their names are
    embedded in the checkpoint's pickle stream.
    """

    def __getattr__(self, name: str) -> object:
        def _stub(*args: object, **kwargs: object) -> None:
            pass
        _stub.__name__ = name
        _stub.__qualname__ = f"{self.__name__}.{name}"
        setattr(self, name, _stub)
        return _stub


def _register_pl1_compat_stubs() -> None:
    """Register catch-all stubs for pytorch_lightning 1.x modules missing in 2.x."""
    missing = [
        "pytorch_lightning.utilities.argparse_utils",
        "pytorch_lightning.utilities.parsing",
    ]
    for mod_name in missing:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _AnyAttrModule(mod_name)

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from training.dataset import (
    AudioPhaseBridgeDataset,
    MultiSourceAudioDataset,
    discover_wavebeat_dataset_specs,
)


def _import_wavebeat_components(wavebeat_root: str) -> tuple[type, type]:
    root = Path(wavebeat_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"wavebeat_root not found: {root}")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from wavebeat.data import DownbeatDataset  # type: ignore[import-not-found]
    from wavebeat.dstcn import dsTCNModel  # type: ignore[import-not-found]

    return DownbeatDataset, dsTCNModel


def _center_crop_last_dim(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[-1] == length:
        return x
    if x.shape[-1] < length:
        raise ValueError(f"Cannot crop tensor of length {x.shape[-1]} to larger length {length}")
    start = (x.shape[-1] - length) // 2
    end = start + length
    return x[..., start:end]


class WaveBeatBackend:
    name = "wavebeat"

    def __init__(self) -> None:
        self._criterion = torch.nn.BCEWithLogitsLoss()

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--wavebeat_root",
            type=str,
            default=None,
            help="Optional override for extractor package path (default: extractors/<extractor>)",
        )
        parser.add_argument(
            "--dataset_root",
            type=str,
            default=None,
            help=(
                "Root folder containing multiple dataset folders. "
                "When set, auto-discovers known datasets and overrides "
                "--audio_dir/--annot_dir/--wavebeat_dataset."
            ),
        )
        parser.add_argument(
            "--dataset_include",
            type=str,
            default="ballroom,beatles,gtzan,hains,rwc_popular",
            help=(
                "Comma-separated dataset keys to include when --dataset_root is set. "
                "Use 'all' for every discovered dataset. "
                "Supported keys: ballroom,beatles,beatles_old,gtzan,hains,rwc_popular"
            ),
        )
        parser.add_argument("--audio_dir", type=str, default=None, help="WaveBeat audio directory")
        parser.add_argument("--annot_dir", type=str, default=None, help="WaveBeat annotation directory")
        parser.add_argument("--wavebeat_dataset", type=str, default="ballroom", help="WaveBeat dataset name")
        # Defaults per WaveBeat README training command (NOT code defaults)
        parser.add_argument("--audio_sample_rate", type=int, default=22050)
        parser.add_argument("--target_factor", type=int, default=256)
        parser.add_argument("--train_length", type=int, default=2097152)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--examples_per_epoch", type=int, default=1000)
        parser.add_argument("--preload", action="store_true")
        parser.add_argument("--augment", action="store_true")
        parser.add_argument("--dry_run", action="store_true")

    def _resolve_extractor_root(self, args: argparse.Namespace) -> str:
        if args.wavebeat_root is not None and str(args.wavebeat_root).strip() != "":
            return str(args.wavebeat_root)
        return str(Path("extractors") / self.name)

    def build_dataloader(self, args: argparse.Namespace) -> DataLoader:
        phases_dir = args.phases_dir
        if phases_dir is None and args.dataset_root is not None:
            phases_dir = args.dataset_root

        if phases_dir is None:
            raise ValueError("Provide --dataset_root (preferred) or --phases_dir for mode=end2end")

        extractor_root = self._resolve_extractor_root(args)
        DownbeatDataset, _ = _import_wavebeat_components(extractor_root)

        if args.dataset_root is not None:
            include_keys: set[str] | None
            if args.dataset_include.strip().lower() == "all":
                include_keys = None
            else:
                include_keys = {
                    item.strip().lower() for item in args.dataset_include.split(",") if item.strip()
                }

            specs = discover_wavebeat_dataset_specs(
                root_dir=args.dataset_root,
                include_keys=include_keys,
            )
            if len(specs) == 0:
                raise RuntimeError(
                    "No known datasets were discovered from --dataset_root. "
                    "Check folder structure or --dataset_include."
                )

            source_datasets: list[Dataset] = []
            source_keys: list[str] = []
            for spec in specs:
                try:
                    source_dataset_candidate = DownbeatDataset(
                        audio_dir=str(spec.audio_dir),
                        annot_dir=str(spec.annot_dir),
                        dataset=spec.wavebeat_dataset,
                        audio_sample_rate=args.audio_sample_rate,
                        target_factor=args.target_factor,
                        subset="train",
                        length=args.train_length,
                        preload=args.preload,
                        augment=args.augment,
                        examples_per_epoch=args.examples_per_epoch,
                        half=False,
                        dry_run=args.dry_run,
                    )
                except Exception as exc:
                    print(f"Skipping dataset '{spec.key}': {exc}")
                    continue

                num_audio_files = len(getattr(source_dataset_candidate, "audio_files", []))
                if num_audio_files == 0:
                    print(f"Skipping dataset '{spec.key}': no train audio files selected")
                    continue

                try:
                    _ = source_dataset_candidate[0]
                except Exception as exc:
                    print(f"Skipping dataset '{spec.key}': sample load failed ({exc})")
                    continue

                source_datasets.append(source_dataset_candidate)
                source_keys.append(spec.key)

            if len(source_datasets) == 0:
                raise RuntimeError(
                    "No usable datasets remained after initialization. "
                    "Check annotation matching and dataset format consistency."
                )

            source_dataset: Dataset = MultiSourceAudioDataset(
                source_datasets=source_datasets,
                source_keys=source_keys,
            )
            print(f"Auto-discovered datasets: {', '.join(source_keys)}")
        else:
            if args.audio_dir is None or args.annot_dir is None:
                raise ValueError(
                    "--audio_dir and --annot_dir are required unless --dataset_root is provided"
                )

            source_dataset = DownbeatDataset(
                audio_dir=args.audio_dir,
                annot_dir=args.annot_dir,
                dataset=args.wavebeat_dataset,
                audio_sample_rate=args.audio_sample_rate,
                target_factor=args.target_factor,
                subset="train",
                length=args.train_length,
                preload=args.preload,
                augment=args.augment,
                examples_per_epoch=args.examples_per_epoch,
                half=False,
                dry_run=args.dry_run,
            )

        dataset = AudioPhaseBridgeDataset(
            source_dataset=source_dataset,
            phases_dir=phases_dir,
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
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

    def build_val_dataloader(self, args: argparse.Namespace) -> DataLoader | None:
        phases_dir = args.phases_dir
        if phases_dir is None and args.dataset_root is not None:
            phases_dir = args.dataset_root

        if phases_dir is None:
            return None

        extractor_root = self._resolve_extractor_root(args)
        DownbeatDataset, _ = _import_wavebeat_components(extractor_root)

        if args.dataset_root is not None:
            include_keys: set[str] | None
            if args.dataset_include.strip().lower() == "all":
                include_keys = None
            else:
                include_keys = {
                    item.strip().lower() for item in args.dataset_include.split(",") if item.strip()
                }

            specs = discover_wavebeat_dataset_specs(
                root_dir=args.dataset_root,
                include_keys=include_keys,
            )

            source_datasets: list[Dataset] = []
            source_keys: list[str] = []
            for spec in specs:
                try:
                    ds = DownbeatDataset(
                        audio_dir=str(spec.audio_dir),
                        annot_dir=str(spec.annot_dir),
                        dataset=spec.wavebeat_dataset,
                        audio_sample_rate=args.audio_sample_rate,
                        target_factor=args.target_factor,
                        subset="val",
                        length=args.train_length,
                        preload=args.preload,
                        augment=False,
                        examples_per_epoch=args.examples_per_epoch,
                        half=False,
                        dry_run=args.dry_run,
                    )
                except Exception:
                    continue

                if len(getattr(ds, "audio_files", [])) == 0:
                    continue
                try:
                    _ = ds[0]
                except Exception:
                    continue

                source_datasets.append(ds)
                source_keys.append(spec.key)

            if len(source_datasets) == 0:
                return None

            source_dataset: Dataset = MultiSourceAudioDataset(
                source_datasets=source_datasets,
                source_keys=source_keys,
            )
            print(f"Validation datasets: {', '.join(source_keys)}")
        else:
            if args.audio_dir is None or args.annot_dir is None:
                return None

            source_dataset = DownbeatDataset(
                audio_dir=args.audio_dir,
                annot_dir=args.annot_dir,
                dataset=args.wavebeat_dataset,
                audio_sample_rate=args.audio_sample_rate,
                target_factor=args.target_factor,
                subset="val",
                length=args.train_length,
                preload=args.preload,
                augment=False,
                examples_per_epoch=args.examples_per_epoch,
                half=False,
                dry_run=args.dry_run,
            )

        dataset = AudioPhaseBridgeDataset(
            source_dataset=source_dataset,
            phases_dir=phases_dir,
        )

        # Val/test subsets return variable-length audio (no cropping),
        # so batch_size must be 1 for collation to work.
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

    def build_model(self, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
        extractor_root = self._resolve_extractor_root(args)
        _, dsTCNModel = _import_wavebeat_components(extractor_root)

        # Use known WaveBeat defaults (checkpoint hparams are not read to avoid
        # pytorch_lightning version incompatibilities during deserialization)
        return dsTCNModel(
            ninputs=1,
            noutputs=2,
            nblocks=8,
            kernel_size=15,
            stride=2,
            dilation_growth=8,
            channel_growth=32,
            channel_width=32,
            stack_size=4,
        ).to(device)

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        args: argparse.Namespace,
        device: torch.device,
    ) -> None:
        if args.extractor_ckpt is None:
            return

        _register_pl1_compat_stubs()
        ckpt = torch.load(args.extractor_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            cleaned_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
            for key, value in state_dict.items():
                new_key = key
                if new_key.startswith("model."):
                    new_key = new_key[len("model.") :]
                cleaned_state_dict[new_key] = value

            model.load_state_dict(cleaned_state_dict, strict=False)

    def compute_loss_and_activations(
        self,
        model: torch.nn.Module,
        audio: torch.Tensor,
        target: torch.Tensor,
        frozen: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frozen:
            with torch.no_grad():
                logits = model(audio)
                activations = torch.sigmoid(logits).transpose(1, 2)
            zero = torch.zeros((), device=logits.device, dtype=logits.dtype)
            return zero, activations
        logits = model(audio)
        aligned_target = _center_crop_last_dim(target, logits.shape[-1])
        loss = self._criterion(logits, aligned_target)
        activations = torch.sigmoid(logits).transpose(1, 2)
        return loss, activations
