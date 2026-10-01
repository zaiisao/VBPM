"""VBPM's fixed Beat This annotation and spectrogram splits."""
from pathlib import Path

DATA_DIR = Path("/disk4/shared/beat_this/data")


def load_beat_this(fold: int):
    """Build Beat This's datasets for the given validation fold."""
    from beat_this.dataset.dataset import BeatDataModule

    data = BeatDataModule(data_dir=DATA_DIR, fold=fold, test_dataset="gtzan",
                          train_length=None, augmentations={})
    data.setup("fit")
    return data
