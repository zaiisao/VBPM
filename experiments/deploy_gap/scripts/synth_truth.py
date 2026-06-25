"""TASK 3 — SYNTHETIC-TRUTH RECOVERY.

Generate clean click-track audio at KNOWN tempo / meter (m=4), render it through the REAL
LogMel frontend, and set beats = the planted beat frames. Train a fresh faithful model on a
small set of such clips and check whether inference recovers the planted beats/phase.

If the model cannot recover beat-F ~1.0 on clean, perfectly-periodic, known-truth audio, the
inference/optimization is broken independent of real-audio ambiguity. This is the cleanest
known-answer test of the whole machinery.

Recovery measured: teacher-forced posterior beat-F (subdivision), free-run beat-F, and the
circular correlation between the recovered posterior phase and the planted bar phase.
"""
import sys, math
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/home/sogang/jaehoon/CHART")
from faithful.model import BarPointerVAE
from faithful.elbo import strict_elbo, free_run
from faithful.data import FPS, N_MELS, LogMel
from faithful.distributions import TWO_PI
from faithful.evaluate import beats_from_barphase, f_measure

dev = "cuda"
SR = 22050
M = 4


def click_track(bpm, dur_s=12.0):
    n = int(dur_s * SR)
    audio = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    times = np.arange(0.0, dur_s, period)
    click_len = int(0.012 * SR)
    env = np.exp(-np.arange(click_len) / (0.003 * SR))
    for i, t in enumerate(times):
        s = int(t * SR)
        if s + click_len > n:
            break
        freq = 2000.0 if (i % M == 0) else 1000.0          # accent downbeats
        tone = np.sin(2 * math.pi * freq * np.arange(click_len) / SR) * env
        audio[s:s + click_len] += tone.astype(np.float32)
    return audio, times


def planted(times, T):
    """beat target frames + planted bar phase (global Phi=2pi per bar)."""
    bframe = np.round(times * FPS).astype(int)
    bframe = bframe[bframe < T]
    b = np.zeros(T, dtype=np.float32); b[bframe] = 1.0
    Phi_anchor = TWO_PI * (np.arange(len(times)) / M)       # global phase at each beat
    phi = np.interp(np.arange(T) / FPS, times, Phi_anchor,
                    left=0.0, right=Phi_anchor[-1] if len(Phi_anchor) else 0.0) % TWO_PI
    return b, phi


def circ_corr(a, b):
    a = a - np.angle(np.mean(np.exp(1j * a))); b = b - np.angle(np.mean(np.exp(1j * b)))
    sa, sb = np.sin(a), np.sin(b)
    return float(np.sum(sa * sb) / (math.sqrt(np.sum(sa**2) * np.sum(sb**2)) + 1e-9))


@torch.no_grad()
def tf_posterior(model, h, b):
    B, T, _ = h.shape
    pc = model.encode_prior(h); qc = model.encode_posterior(h, b)
    z0 = model.z0.unsqueeze(0).expand(B, -1)
    qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(torch.cat([qc[:, 0], z0], -1)))
    phi = qpm % TWO_PI; lt = qtm; meter = F.softmax(qm, -1); traj = [phi]
    for t in range(1, T):
        qm, qpm, qpk, qtm, qts = model.unpack(model.post_head(
            torch.cat([qc[:, t], model.z_features(meter, phi, lt)], -1)))
        phi = qpm % TWO_PI; lt = qtm; meter = F.softmax(qm, -1); traj.append(phi)
    return torch.stack(traj, 1)[0].cpu().numpy()


def main():
    logmel = LogMel().to(dev)
    bpms = [90, 110, 130, 150]
    clips = []
    for bpm in bpms:
        audio, times = click_track(bpm)
        h = logmel(torch.from_numpy(audio).to(dev).unsqueeze(0))
        T = min(h.shape[1], 250)
        b, phi_true = planted(times, T)
        clips.append((bpm, h[:, :T], torch.from_numpy(b).to(dev).unsqueeze(0), phi_true,
                      np.where(b > 0.5)[0] / FPS))
    print(f"clips: {bpms}  (m={M})", flush=True)
    torch.manual_seed(0)
    model = BarPointerVAE(h_dim=N_MELS, hidden=64, num_meters=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    steps = 800
    for s in range(1, steps + 1):
        bpm, h, b, phi_true, ref = clips[s % len(clips)]
        opt.zero_grad()
        temp = max(0.3, 1.0 - 0.7 * s / steps)
        loss, info = strict_elbo(model, h, b, temperature=temp)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if s % 200 == 0 or s == 1:
            model.eval()
            tf, fr, cc = [], [], []
            for bpm, h, b, phi_true, ref in clips:
                T = h.shape[1]
                phi = tf_posterior(model, h, b)
                tf.append(f_measure(ref, beats_from_barphase(phi, M, FPS)))
                cc.append(circ_corr(phi, phi_true))
                frp = free_run(model, h, temperature=0.3)["phase_mu"][0, :T].cpu().numpy()
                fr.append(f_measure(ref, beats_from_barphase(frp, M, FPS)))
            print(f"  s{s} recon={info['recon']:.1f} TF-beatF={np.mean(tf):.3f} "
                  f"freerun-beatF={np.mean(fr):.3f} phase-circcorr={np.mean(cc):.3f}", flush=True)
            model.train()
    print(f"FINAL: TF-beatF={np.mean(tf):.3f} freerun-beatF={np.mean(fr):.3f} circcorr={np.mean(cc):.3f}")


if __name__ == "__main__":
    main()
