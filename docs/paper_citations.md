# VBPM paper — citation inventory

Organized by the role each reference plays. ⚠ = citation details to verify before camera-ready
(year/venue from memory; check exact titles/BibTeX).

## 1. The bar-pointer / DBN lineage (our generative model's ancestry)

- **Whiteley, Cemgil & Godsill (2006)**, "Bayesian modelling of temporal structure in musical audio," ISMIR — the bar-pointer model itself; emission depends on pointer *position*, tempo only parameterizes transitions (our §"faithful fix" leans on this reading).
- **Cemgil & Kappen (2003)** ⚠, "Monte Carlo methods for tempo tracking and rhythm quantization," JAIR — particle-filter inference in the lineage.
- **Hainsworth & Macleod (2004)**, "Particle filtering applied to musical tempo tracking," EURASIP JASP — PF beat tracking + the Hainsworth dataset.
- **Krebs, Böck & Widmer (2013)**, "Rhythmic pattern modeling for beat and downbeat tracking in musical audio," ISMIR — Ballroom downbeats, pattern states.
- **Krebs, Böck & Widmer (2015)**, "An efficient state-space model for joint tempo and meter tracking," ISMIR — the discretized DBN we compare against (the "grid" branch of predict-correct; our filter is the sampling branch).
- **Böck, Korzeniowski, Schlüter, Krebs & Widmer (2016)**, "madmom: a new Python audio and music signal processing library," ACM MM — reference DBN implementation (transition_lambda / observation_lambda analogs).
- **Böck & Davies (2020)**, "Deconstruct, analyse, reconstruct: how to improve tempo, beat, and downbeat estimation," ISMIR — TCN baseline + label-broadening convention.

## 2. Frontends and SOTA trackers (baselines / evidence source)

- **Foscarin, Schlüter & Widmer (2024)**, "Beat This! Accurate beat tracking without DBN postprocessing," (arXiv 2407.21658 / ISMIR) — our frozen frontend (final0), aug scheme we borrow as-is, and the "no post-processing" position we argue against on SMC.
- **Zhao, Xia & Wang (2022)**, "Beat Transformer: demixed beat and downbeat tracking with dilated self-attention," arXiv 2209.07140 / ISMIR — SOTA baseline.
- **Hung, Wang, Song, Lu & Won (2022)**, "Modeling beats and downbeats with a time-frequency transformer," ICASSP (SpecTNT) — SOTA baseline.
- **Cheng & Goto (2023)**, "Transformer-based beat tracking with low-resolution encoder and high-resolution decoder," ISMIR — baseline.
- **Steinmetz & Reiss (2021)** ⚠, "WaveBeat: end-to-end beat and downbeat tracking in the waveform domain," AES — earlier frontend in our history; modular-frontend demonstration.
- **Ru, Wang, Zhao, Wu, Yu, Jiang, Wang & Li (2025)**, "BeatFM: Improving beat tracking with pre-trained music foundation model," arXiv 2508.09790 — foundation-model frontend + DBN; our same-activations post-processor A/B target; SMC 8-fold numbers.
- **Li et al. (2023)**, "MERT: Acoustic music understanding model with large-scale self-supervised training," ICLR — our self-supervised frontend arm.
- **Won, Hung & Le (2024)**, "A foundation model for music informatics," ICASSP (MusicFM) — BeatFM's second backbone; context.
- **Heydari & Duan (2022)** ⚠, "Singing beat tracking with self-supervised front-end and linear transformers," arXiv 2208.14578 — SSL-features-for-beat precedent.
- **Desblancs, Lostanlen & Hennequin (2023)** ⚠, "Zero-note samba: self-supervised beat tracking," IEEE/ACM TASLP.
- **Chiu, Müller, Davies, Su et al. (2023)** ⚠, "Local periodicity-based beat tracking for expressive classical piano music," IEEE/ACM TASLP — expressive/rubato beat tracking context (SMC-adjacent).

## 3. VAE / dynamical-VAE foundations (our training objective)

- **Kingma & Welling (2014)**, "Auto-encoding variational Bayes," ICLR — ELBO, reparameterization.
- **Rezende, Mohamed & Wierstra (2014)**, "Stochastic backpropagation and approximate inference in deep generative models," ICML.
- **Sohn, Lee & Yan (2015)**, "Learning structured output representation using deep conditional generative models," NeurIPS — CVAE; our recognition/prior-network deployment framing (and the toy-ladder notation).
- **Chung et al. (2015)**, "A recurrent latent variable model for sequential data," NeurIPS (VRNN) — sequential VAE lineage.
- **Krishnan, Shalit & Sontag (2017)**, "Structured inference networks for nonlinear state space models," AAAI (DMM) — the model class VBPM instantiates; cited in the code per the refactor plan.
- **Girin et al. (2021)**, "Dynamical variational autoencoders: a comprehensive review," Foundations and Trends in ML — DVAE taxonomy.
- **Fraccaro, Kamronn, Paquet & Winther (2017)**, "A disentangled recognition and nonlinear dynamics model for unsupervised learning," NeurIPS (KVAE) — our Kalman-era proof-of-paradigm; exact-inference baseline.
- **Maddison et al. (2017)**, "Filtering variational objectives," NeurIPS (FIVO); **Le, Igl et al. (2018)**, "Auto-encoding sequential Monte Carlo," ICLR; **Naesseth et al. (2018)** ⚠ variational SMC — filtering-objective family (discussed, not adopted).

## 4. Posterior collapse, KL control, information routing (the side-channel mechanism)

- **Bowman et al. (2016)**, "Generating sentences from a continuous space," CoNLL — KL annealing.
- **Kingma et al. (2016)**, "Improved variational inference with inverse autoregressive flow," NeurIPS — free bits (appendix).
- **Razavi, van den Oord, Poole & Vinyals (2019)**, "Preventing posterior collapse with δ-VAEs," ICLR — committed rate (our fallback discussion).
- **Hafner et al. (2021)**, "Mastering Atari with discrete world models," ICLR (DreamerV2) — KL balancing; our prior-preserving free-bits gradient channel is its ELBO-exact cousin.
- **Hafner et al. (2019)**, "Learning latent dynamics for planning from pixels," ICML (PlaNet) — latent overshooting (toy probe 2).
- **Chen et al. (2017)**, "Variational lossy autoencoder," ICLR — decoder/latent information routing; the "cheapest channel wins" argument.
- **Alemi et al. (2018)**, "Fixing a broken ELBO," ICML — rate-distortion framing of KL prices (mechanism section).
- **Lin, Goyal, Girshick, He & Dollár (2017)**, "Focal loss for dense object detection," ICCV — the meter class-imbalance arm.

## 5. Directional statistics & distributions (the faithful latents)

- **Mardia & Jupp (2000)**, *Directional Statistics*, Wiley — wrapped Cauchy, von Mises fundamentals.
- **Kato & Jones (2013)** ⚠, wrapped-Cauchy family properties (check exact ref for the closed-form KL we use).
- **Best & Fisher (1979)**, "Efficient simulation of the von Mises distribution," JRSS C — vM sampler (and our sampler-bug war story).
- **Figurnov, Mohamed & Mnih (2018)**, "Implicit reparameterization gradients," NeurIPS — vM reparameterized gradients (torch.distributions path).

## 6. Sawtooth / phase-target provenance (what we do differently)

- **Oyama, Ishizuka & Yoshii (2021)** ⚠, per-beat K-class phase supervision (see data/targets.py comment for exact cite) — prior sawtooth-adjacent supervision.
- **Chen & Su (2022)** ⚠, triangular distance-to-beat label embedding (data/targets.py comment) — ours differs: unified bar-level circular regression on a *latent*, as a vM/WC *emission* with concentration κ.

## 7. Particle methods (deployment)

- **Doucet & Johansen (2009)**, "A tutorial on particle filtering and smoothing," Handbook of Nonlinear Filtering — bootstrap PF, ESS, resampling.
- **Douc & Cappé (2005)**, "Comparison of resampling schemes for particle filtering," ISPA — systematic resampling.
- **Doucet, de Freitas, Murphy & Russell (2000)**, "Rao-Blackwellised particle filtering for dynamic Bayesian networks," UAI — the exact-subfilter upgrade path.
- (Particle smoothing ref if adopted: **Godsill, Doucet & West (2004)**, "Monte Carlo smoothing for nonlinear time series," JASA ⚠.)

## 8. Datasets & evaluation

- **Gouyon et al. (2006)**, "An experimental comparison of audio tempo induction algorithms," IEEE TASLP — Ballroom.
- **Goto et al. (2002)**, "RWC music database: popular, classical and jazz music databases," ISMIR.
- **Davies, Degara & Plumbley (2009)**, "Evaluation methods for musical audio beat tracking algorithms," QMUL Tech. Rep. C4DM-TR-09-06 — F-measure conventions (±70 ms).
- **Harte (2010) / Davies et al.** ⚠ — Beatles beat/downbeat annotations (verify canonical cite).
- **Hainsworth (2004)** — dataset (see §1).
- **Nieto et al. (2019)**, "The Harmonix set: beats, downbeats, and functional segment annotations of western popular music," ISMIR.
- **Tzanetakis & Cook (2002)**, "Musical genre classification of audio signals," IEEE TSAP — GTZAN audio; **Marchand & Peeters (2015)** ⚠, GTZAN-Rhythm beat annotations.
- **Holzapfel, Davies, Zapata, Oliveira & Gouyon (2012)**, "Selective sampling for beat tracking evaluation," IEEE TASLP — SMC_MIREX.
- **Foscarin et al. (2020)** ⚠, "ASAP: a dataset of aligned scores and performances," ISMIR — if the ASAP arm ships.
- **Raffel et al. (2014)**, "mir_eval: a transparent implementation of common MIR metrics," ISMIR.
- **Karaosmanoğlu (2012)** ⚠, SymbTr — only if the SymbTr meter material appears.

## 9. Our own prior work

- **SMC Blind Spot paper** — arXiv 2605.12287 (user's own): DBN tempo-coverage blind spots; this paper is its constructive sequel (the untrained-control result — architecture, not learning, removes the blind spots — directly extends it).

## 10. Software / tools (footnotes or acknowledgments)

- madmom (§1), mir_eval (§8), PyTorch ⚠ (Paszke et al. 2019, NeurIPS), pedalboard (Spotify — the as-is tempo augmentation), torchaudio ⚠.

---
**Verification pass needed** on every ⚠ before submission; several live in code comments
(data/targets.py for §6) and in memory files rather than checked BibTeX.
