# Faithful strict-ELBO bar-pointer VAE

A from-scratch, production-scale implementation of the variational bar-pointer model in
*ELBO for DBN* (Jaehoon Ahn), trained **end-to-end from random weights** on real audio.
It is the runnable counterpart of [`notebooks/ELBO_for_DBN.ipynb`](../notebooks/ELBO_for_DBN.ipynb):
the notebook is the line-by-line reference on a toy sequence; this package is the same
model on the real datasets, with no pretrained frontend and no score-chasing additions.

This exists so that **every deviation in the production `models/svt_core.py` can be named
and justified against a clean reference.** If a behaviour is not in this package, it is not
in the paper.

## Why this is "end-to-end from random weights"

The observation `h` fed to the model is a **fixed log-mel spectrogram** (`faithful/data.py`,
22.05 kHz / hop 256 → 86.13 fps / 128 mels) — the same kind of low-level input Beat This and
WaveBeat themselves ingest. It is **not** a pretrained frontend's activations. The only
learned parameters anywhere in the pipeline are the VAE's own, initialised randomly. This is
the regime we argued is the honest test: there is no frozen discriminative head for the latent
to be redundant against; the model must learn the structure itself.

## The objective is the strict ELBO

```
L = Σ_t BCE(b_t, σ(decoder(z_t, h)))  +  Σ_t [ KL_meter + KL_phase + KL_tempo ]
```

with **β = 1 from step 0** and a single MC sample over the sampled `z_{t-1}` (Algorithm 1).

| Paper element | Implementation |
|---|---|
| latent `z_t = [m_t, φ_t, φ̇_t]` (§2) | sampled in `elbo.strict_elbo` |
| phase prior mean `μ^p_φ = φ_{t-1} + φ̇_{t-1}` (§5.2) | `elbo.py` (`p_phi_mu`), **no audio correction** |
| tempo prior mean `μ^p_τ = log φ̇_{t-1}` (§5.3) | `elbo.py` (`p_tau_mu`), log-space random walk |
| meter prior `f^m_ψ(m_{t-1}, φ_t, φ_{t-1}, h)` (§5.1) | `model.meter_prior_logp`, evaluated **after** sampling φ_t |
| von Mises Best–Fisher sampler + implicit reparam (Alg. 2) | `distributions.VonMisesSample` |
| Gumbel-Softmax meter (§5.1) | `distributions.gumbel_softmax` |
| Log-Normal tempo reparam (§5.3) | `μ + σ·ε` in `elbo.py` |
| closed-form KLs (§5.1–5.3) | `distributions.kl_{categorical,von_mises,log_normal}` |
| Bernoulli decoder `σ(NN_θ(z_t, h))` (§5.4) | `model.decode` — **reads h** |
| posterior reads `b_{1:T}, ẑ_{t-1}, h` (Alg. 1 line 15) | `post_head([post_ctx[t], z_prev_feat])` |

## What is deliberately ABSENT (and why each is a bandage)

None of these appear here. Each is something the production code added to push the score:

- **free-bits / KL floor** — changes the objective; hides collapse instead of reporting it.
- **KL annealing (β warmup)** — a looser bound; not the strict ELBO.
- **latent supervision** (meter / phase / bar-phase / tempo-density / τ_bar) — injects label
  gradients the generative model is supposed to discover itself.
- **audio-driven correction of the prior MEAN** (`phase_corr`, `tempo_corr`) — lets audio
  *replace* the bar-pointer dynamics rather than the dynamics doing the work.
- **tempo clamps** (`LOG_TEMPO_MIN/MAX`) — masks the random-walk divergence.
- **BCE `pos_weight`** — reweights the Bernoulli likelihood.
- **scheduled sampling** — an exposure-bias curriculum, not in the ELBO.
- **extra latents** (bar-phase φ^bar, global-tempo τ_bar), **delta-VAE**, **DVBF** — added structure.

The one schedule kept — Gumbel-Softmax temperature annealing 1.0 → 0.3 — is the relaxation
temperature used in the reference notebook; it does not alter the ELBO.

## The experiment

`decoder` reads `h`, so the teacher-forced reconstruction and the decoder read-out can look
healthy even if the latent is dead. The honest signal is the **phase-wrap read-out** (pure
latent dynamics) and the **per-latent KL trajectory**. Expected result: the strict ELBO from
random init exhibits **posterior collapse** — KL terms decay toward ~0 and phase-wrap F sinks
to the metronome floor while the decoder read-out rides the audio. That is the point: it
demonstrates the collapse is a property of the objective, not of any frozen frontend.

## Run

```bash
CH=/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python
$CH -m faithful.train \
  --data_root /home/sogang/mnt/db_1/jaehoon/beat-tracking/labeled_data \
  --datasets ballroom,beatles,hains,rwc_popular \
  --frames 256 --batch_size 16 --steps 1500 --eval_every 150 \
  --max_eval_songs 12 --out runs/strict_elbo
```

Outputs `runs/strict_elbo/metrics.jsonl` (per-step KL breakdown + periodic eval), `best.pt`,
`final.pt`. The `--latent_only` flag toggles the documented deviation (decoder without `h`).
