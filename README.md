# VBPM — the rungs ladder

Beat & downbeat tracking on frozen frontend activations, built as a ladder of models where each
rung changes EXACTLY ONE thing relative to the rung below it. The goal: by the time the top rung
(a learned, audio-conditioned model) beats the bottom rung (the published madmom baseline), every
point of improvement is attributable to a named change.

(The previous VBPM incarnation — the bar-pointer DVAE and its ablation flags — is archived in
`archive_2026-07-14/` and in git history at `43ecf34`, along with the old docs/, notebooks/ and
experiments/.)

## The ladder

| rung | model | factors | status |
|------|-------|---------|--------|
| R0 | madmom's bar-pointer DBN, exactly as Beat This / Beat Transformer use it | hand-set | done |
| R1 | the same model on OUR engine (torch, differentiable) | hand-set | done, **certified ≡ R0** |
| R2 | same, but the factors are learned by maximizing the exact forward log-likelihood | learned scalars/tables | next |
| R3 | transitions conditioned on audio per frame | learned, audio-conditioned | — |
| R4 | neural emission + transition (Neural HMM) | learned networks | — |

## Layout

```
tracker.py               # by-name registries (frontends + bar-pointer models) + Tracker glue, above both packages
track.py                 # CLI inference: python track.py song.wav [--config configs/track.yaml]
configs/
  track.yaml             # the tracker composition (frontend + bar-pointer model + their kwargs)
frontends/
  __init__.py            # Frontend interface only (selection/pairing lives in tracker.py)
  beat_this.py           # wraps the OFFICIAL beat_this.inference.Audio2Frames (one script per frontend)
rungs/
  base.py                # the Rung contract: predict() -> events, coercion, Böck decorrelation
  r0_madmom_dbn.py       # Baseline A: the official madmom DBN + the standard decorrelation
  r1_2016_dbn.py         # the same model rebuilt on our engine (the certificate rung)
  deployment.py          # model-independent deployment lessons (threshold crop), off by default
  bar_pointer/           # the shared R1-R4 chassis (rungs change ONLY how factors are produced)
    state_space.py       # Krebs 2015 bar-pointer state space (interval i owns i states)
    structured_dp.py     # THE ENGINE: exact forward + Viterbi, O(K + M*V^2)/frame, GPU, autograd
    inference.py         # the readable dense reference the engine is certified against
    readout.py           # MAP state path -> beat/downbeat times (shared by all rungs)
data/
  songs.py               # live song catalog: official annotations + 8-fold splits + local audio
tests/
  test_inference.py      # dense DP vs hmmlearn AND torch-struct (independent oracles)
  test_structured_dp.py  # structured engine vs dense DP + compact emission + gradient checks
```

## The certificate chain

Nothing here is trusted by eye; every layer is machine-checked against something independent:

1. `rungs/bar_pointer/inference.py` (readable, textbook) ≡ **hmmlearn** ≡ **torch-struct** (LL to ~1e-14, paths exact)
2. `rungs/bar_pointer/structured_dp.py` (the engine) ≡ the dense reference (same model written out as a matrix)
3. R1 on the engine ≡ **madmom**: identical Viterbi path AND identical path score, 25/25 val songs —
   including {3,4} meter selection (25/25 same choice)
4. R1 with madmom's shipped deployment options (`num_tempi=60, threshold=0.05, correct=True` — R1's
   defaults) ≡ R0 as shipped: **event-identical output**, 25/25 songs. The bare model (certificate
   configuration, what rung comparisons use) is the opt-out `num_tempi=None, threshold=0, correct=False`.

Point 4 means the R0-vs-R1 F difference under defaults (~0.02) is entirely madmom's three deployment
conveniences (fade-crop, peak-snap, tempo grid), each measured, none of them the model.

## Running

```bash
PYTHONPATH=. python tests/test_inference.py       # certify the dense reference vs two libraries
PYTHONPATH=. python tests/test_structured_dp.py   # certify the engine vs the dense reference
PYTHONPATH=. python rungs/r1_2016_dbn.py   # synthetic smoke test
```

Environment: needs torch + madmom (+ hmmlearn/torch-struct/mir_eval for tests). Known-good local
interpreter: `/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python` (madmom is a source checkout
at `~/jaehoon/madmom`, built for py3.10 — the repo's own `.venv` is py3.8 and cannot import it).

NO ACTIVATION CACHES (decision 2026-07-15): activations are computed live through `frontends/`,
so there is exactly one code path from audio to activations and live == eval by construction. The
old `cache/acts/*` records were produced by a second, retired pipeline (different chunking/padding;
activations correlate ~0.97 with the live path, predictions agree to mean |dF| 0.005 — measured, but
never certified). `data/songs.py` enumerates the data: 2,304 songs live (1,305 across
ballroom/beatles/hainsworth/hjdb with official 8-fold assignments + 999 GTZAN test-only);
run `python data/songs.py` for the coverage report, including which annotated datasets lack
local audio. Fold-honesty: evaluate song `s` with checkpoint `fold{s.fold}`; GTZAN (fold None) is
held out of every checkpoint. `fps` is a property of the frontend, never a constant.
