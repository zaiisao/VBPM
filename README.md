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
rungs/
  r0_madmom_dbn.py       # Baseline A: the official madmom DBN + the standard decorrelation
  r1_handcrafted_hmm.py  # the same model rebuilt on our engine (the certificate rung)
common/
  state_space.py         # Krebs 2015 bar-pointer state space (interval i owns i states)
  structured_dp.py       # THE ENGINE: exact forward + Viterbi, O(K + M*V^2)/frame, GPU, autograd
  inference.py           # the readable dense reference the engine is certified against
  readout.py             # MAP state path -> beat/downbeat times (shared by all rungs)
  deployment.py          # model-independent decode lessons (threshold crop), off by default
tests/
  test_inference.py      # dense DP vs hmmlearn AND torch-struct (independent oracles)
  test_structured_dp.py  # structured engine vs dense DP + compact emission + gradient checks
```

## The certificate chain

Nothing here is trusted by eye; every layer is machine-checked against something independent:

1. `common/inference.py` (readable, textbook) ≡ **hmmlearn** ≡ **torch-struct** (LL to ~1e-14, paths exact)
2. `common/structured_dp.py` (the engine) ≡ the dense reference (same model written out as a matrix)
3. R1 on the engine ≡ **madmom**: identical Viterbi path AND identical path score, 25/25 val songs —
   including {3,4} meter selection (25/25 same choice)
4. R1 with madmom's shipped decode options (`num_tempi=60, threshold=0.05, correct=True`) ≡ R0
   as shipped: **event-identical output**, 25/25 songs

Point 4 means the R0-vs-R1 F difference under defaults (~0.02) is entirely madmom's three decode
conveniences (fade-crop, peak-snap, tempo grid), each measured, none of them the model.

## Running

```bash
PYTHONPATH=. python tests/test_inference.py       # certify the dense reference vs two libraries
PYTHONPATH=. python tests/test_structured_dp.py   # certify the engine vs the dense reference
PYTHONPATH=. python rungs/r1_handcrafted_hmm.py   # synthetic smoke test
```

Environment: needs torch + madmom (+ hmmlearn/torch-struct/mir_eval for tests). Known-good local
interpreter: `/home/sogang/mnt/db_2/anaconda3/envs/chart/bin/python` (madmom is a source checkout
at `~/jaehoon/madmom`, built for py3.10 — the repo's own `.venv` is py3.8 and cannot import it).

Activations are read from `cache/acts/*` records (`act2` [T,2] + `fps` + fold-honest targets);
`fps` is a property of the cache record, never a constant.
