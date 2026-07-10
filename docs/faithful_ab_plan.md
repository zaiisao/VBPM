# Faithful VBPM — A/B campaign plan

Run **everything** as single-flag deviations off one fixed faithful baseline, so every effect is
attributable. Stop drifting back to the bulky synthesis config (`filter`+`autocorr`+`sawtooth` at once).

## The faithful baseline (`is_default_vbpm`, all divergences OFF)

| flag | value | note |
|---|---|---|
| `context_arch` | `gru` | simpler; A/B'd ≥ transformer |
| `tempo_family` | `laplace` | adopted per decision — data-faithful to the heavy-tailed increment law; score-neutral vs Gaussian (0.567 vs 0.618, noise), so the reason is fidelity, not metric |
| `divergence_phase_update` | `free` | codebase's faithful default: posterior reads phase from audio, prior = vM(advance) |
| `phase_family` | `wrapped_cauchy` | adopted per decision (parallel to Laplace tempo) — data-faithful to the heavy-tailed beat-microtiming residual; closed-form KL (no Bessel), score-neutral, so the reason is fidelity, not metric. von Mises is the light-tailed ablation arm |
| `divergence_tempo_source` | `latent` | tempo is a latent RW, not autocorr |
| `divergence_sawtooth_weight` | `0.0` | no phase-grounding aux |
| `divergence_meter` | `latent` | K×K transition, paper eq. (4) |
| `divergence_meter_sup_weight` | `0.0` | meter unsupervised |
| `divergence_readout_meter` | `config` | — |
| all `*_free_bits` | `0.0` | — |

**Known behavior of the strict baseline:** beats ~0.55 (from the audio-read `free` phase), **downbeats ~0.00**
(phase frozen for bar structure). So step 0 is establishing a *working* baseline.

### Step 0 — pick the working baseline (the only "both" run)
Run **strict** vs **strict + free-bits** (`divergence_meter_free_bits` / phase free-bits > 0).
Free-bits is the *faithful* anti-collapse fix (un-freezes phase coverage 0.08→0.4). Expect it to recover
usable downbeats. The winner becomes the **working baseline B** all arms branch from.
Report: beat F, downbeat F, **φ̇ CV** (phase smoothness), posterior KL per latent (collapse check).

## Single-flag arms (each flips exactly one thing off B)

| # | arm | flag change | question it answers |
|---|---|---|---|
| 1 | Gaussian tempo (ablation) | `tempo_family=gaussian` | confirm Laplace is score-neutral, not a regression, on clean ground |
| 2 | Deterministic dynamics phase | `divergence_phase_update=integrator_det` | can the *dynamics-driven* phase (clean ramp) match the audio-read `free`? |
| 2b | vM dynamics phase | `divergence_phase_update=integrator` | isolates whether von-Mises phase noise helps vs 2 |
| 3 | Meter as per-song class | `meter_sup_weight>0` + `meter_sup_songlevel` + `meter_sup_scale_frames` + `readout_meter=inferred` | meter capability, measured on clean ground |
| 4 | Sawtooth grounding | `divergence_sawtooth_weight=0.5` | isolate the aux's effect on φ̇ and downbeats |
| 5 | Filter phase | `divergence_phase_update=filter` | **isolate the filter** — suspected source of jumpy φ̇ (CV~0.9) |
| 6 | Autocorr tempo | `divergence_tempo_source=autocorr` | isolate the computed-tempo crutch |
| 7 | Transformer context | `context_arch=transformer` | re-confirm it's ≤ GRU |
| 8 | von Mises phase (ablation) | `phase_family=von_mises` | confirm wrapped Cauchy is score-neutral, not a regression, on clean ground (the light-tailed lineage law). NOTE on fit: WC does **not** win by raw NLL (0.156 vs 0.148, quantization-confounded) — von Mises is 2000× too thin in the far tail but WC over-corrects. WC is adopted as a robustness/fidelity swap (heavy-tailed like the data), not a metric win; this arm checks beat F does not drop. Reverting also restores the implicit-reparameterization concentration gradient (vM) vs WC's higher-variance pathwise one. |

## Key combined conditions (the questions single flags can't answer)

- **Does correct meter CONVERT on a clean phase?** = arm 2 (`integrator_det`) **+** arm 3 (meter).
  This is the clean re-test of "correct meter → better beats," free of the filter confound. Watch
  **non-4/4 beat F (M=inferred vs M=4)** and **φ̇ CV**.
- **Is the filter the cause of jumpy φ̇?** = compare φ̇ CV of arm 2 (`integrator_det`) vs arm 5 (`filter`).

## Metrics to report for every run
beat F, downbeat F, **φ̇ CV** (per meter subset 4/4 vs non-4/4), per-latent posterior KL (collapse),
and for meter arms: non-4/4 recall + non-4/4 beat F at M=inferred vs M=4.

## Open decisions to settle as we go
- **Phase baseline:** `free` (faithful VAE, audio-read) vs `integrator_det` (dynamics-driven). Baseline uses
  `free` per the codebase; arm 2 tests whether the dynamics can carry it.
- **Meter structure:** K×K (baseline) vs the decided per-song-K collapse (currently emulated by
  `meter_sup_songlevel`; structural collapse is a later refactor).
- **Laplace:** adopted as the baseline tempo law (decision), for fidelity to the measured increment law
  despite being score-neutral; the Gaussian is kept only as an ablation arm.
- **Wrapped Cauchy:** adopted as the baseline phase law (same decision, same rationale) — heavy-tailed to
  match real beat-microtiming residuals, score-neutral; von Mises kept only as an ablation arm. Cost of the
  swap: WC's pathwise concentration gradient is higher-variance than vM's implicit-reparam gradient (the
  concentration's low-variance signal comes from the closed-form KL, and the μ gradient is bounded).
