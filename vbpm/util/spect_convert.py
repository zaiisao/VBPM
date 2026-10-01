"""Beat This spectrogram -> MusicFM mel conversion (from beatFM)."""
import torch
import torch.nn.functional as F
import torchaudio.functional as AF


def upsample_time(x):
    """50 fps -> 100 fps by linear interpolation, last frame dropped as MusicFM does."""
    T = x.shape[-1]
    x = F.interpolate(x, size=2 * T - 1, mode="linear", align_corners=True)
    return x[..., :-1]


def bt_to_mag(spect):
    """Undo Beat This log1p(1000 * x): back to mel magnitude."""
    return torch.expm1(spect.float()) / 1000.0


def band_centers_hz(fb, sr, n_fft):
    """Centre frequency in Hz of each mel band."""
    w = fb / fb.sum(0, keepdim=True)
    bins = torch.arange(fb.shape[0], dtype=fb.dtype)
    return (bins[:, None] * w).sum(0) * sr / n_fft


def build_freq_map():
    """(Beat This band widths, Beat This -> MusicFM band power map)."""
    fb_bt = AF.melscale_fbanks(513, 30.0, 11000.0, 128, 22050, norm=None, mel_scale="slaney")
    fb_fm = AF.melscale_fbanks(1025, 0.0, 12000.0, 128, 24000, norm=None, mel_scale="htk")
    hz_bt = band_centers_hz(fb_bt, 22050, 1024)
    hz_fm = band_centers_hz(fb_fm, 24000, 2048)

    f = hz_fm.clamp(hz_bt[0], hz_bt[-1])
    k = torch.searchsorted(hz_bt, f).clamp(1, 127)
    a = (f - hz_bt[k - 1]) / (hz_bt[k] - hz_bt[k - 1])
    interp = torch.zeros(128, 128)
    j = torch.arange(128)
    interp[k - 1, j] = 1 - a
    interp[k, j] += a
    return fb_bt.sum(0), interp * fb_fm.sum(0)[None, :]


def mel_bt_to_fm(mag, width_bt, freq_map):
    """Beat This mel magnitude -> MusicFM mel power."""
    per_bin_power = (mag / width_bt[:, None]) ** 2
    return torch.einsum("bkt,kj->bjt", per_bin_power, freq_map)


def power_to_musicfm(power, stats, db_offset):
    """MusicFM mel power -> normalised MusicFM input."""
    db = 10.0 * torch.log10(power.clamp(min=1e-10)) + db_offset
    return (db - stats["melspec_2048_mean"]) / stats["melspec_2048_std"]


def level_offset(width_bt, freq_map, seconds: int = 30):
    """dB gap between the two pipelines, measured on seeded white noise."""
    from beat_this.preprocessing import LogMelSpect
    from musicfm.modules.features import MelSTFT

    noise = torch.randn(24000 * seconds, generator=torch.Generator().manual_seed(0))

    bt_mel = LogMelSpect()(AF.resample(noise, 24000, 22050))[None]
    bt_power = mel_bt_to_fm(bt_to_mag(bt_mel.transpose(1, 2)), width_bt, freq_map)
    fm_power = MelSTFT()(noise[None])
    frames = min(bt_power.shape[-1], fm_power.shape[-1])

    gap = (10.0 * torch.log10(fm_power[..., :frames].clamp(min=1e-10))
           - 10.0 * torch.log10(bt_power[..., :frames].clamp(min=1e-10)))

    return gap.median().item()
