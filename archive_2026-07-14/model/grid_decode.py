"""Exact grid-Viterbi decode of the LEARNED bar-pointer model -- the apples-to-apples counterpart
to madmom's DBN. madmom discretizes the same (phase, tempo) bar-pointer state and runs Viterbi with
HAND-SET transition/observation probabilities; here we run the identical max-product recursion but
with OUR model's LEARNED phase concentration, tempo drift, and emission. So the comparison against
madmom becomes learned-dynamics vs hand-tuned-dynamics under the SAME (exact, bidirectional-optimal)
inference algorithm -- removing the causal/approximate handicap the particle filter imposes.

v1 scope (documented approximations, all tightenable):
  * meter fixed to `beats_per_bar` for the emission harmonic (4/4 default; non-4/4 is a separate
    axis, handled elsewhere by the meter latent -- NOT claimed here).
  * time-invariant transition scales: the per-frame prior concentration / tempo sigma are averaged
    over the song (the EMISSION stays exact per-frame; only the transition kernel is shared).
  * base spec transition (the learned g-correction is context/time-varying; off here).
Returns dict with map_phase [T], beat_activation [T], downbeat_activation [T] (Viterbi max-marginal
wrap indicators), mirroring particle_filter.run_particle_filter's read-outs.
"""
import math

import torch
import torch.nn.functional as F

from model.latents import TWO_PI, concentration_to_rho


@torch.no_grad()
def grid_viterbi_decode(model, features, observations, n_phase=360, n_tempo=48,
                        min_bpm=50.0, max_bpm=215.0, fps=22050.0 / 256.0,
                        observation_temperature=3.0, downbeat_evidence_weight=3.0,
                        beats_per_bar=4, phase_support=24,
                        phase_concentration_scale=50.0, tempo_sigma_scale=0.05,
                        tempo_prior_strength=0.5, tempo_prior_center_bpm=110.0,
                        tempo_prior_sigma_bpm=60.0):
    device = features.device
    obs = observations
    T = obs.shape[0]
    context = model.prior_context_projection(model.prior_encoder(features))[0]        # [T, H]

    # ---- grids ----
    phase_grid = torch.arange(n_phase, device=device).float() * TWO_PI / n_phase       # [P]
    # phase is BAR phase (one 2pi rotation per BAR), so the per-frame advance is the BEAT rate
    # divided by beats_per_bar -- the model's exp(log_tempo) is a bar-phase advance, not beat-rate.
    adv_min = TWO_PI * (min_bpm / 60.0) / fps / beats_per_bar
    adv_max = TWO_PI * (max_bpm / 60.0) / fps / beats_per_bar
    advance = torch.exp(torch.linspace(math.log(adv_min), math.log(adv_max),
                                       n_tempo, device=device))                        # [Tau] rad/frame
    advance_cells = advance * n_phase / TWO_PI                                         # [Tau]
    # tempo prior (what madmom's DBN has and we lacked -> the octave-error fix): a soft log-normal
    # over the tempo grid favouring common tempos, added per frame so the GLOBAL octave choice is
    # biased toward the central metrical level instead of half/double.
    beat_bpm = advance * beats_per_bar * fps / TWO_PI * 60.0                            # [Tau] beats/min
    tempo_log_prior = -0.5 * ((beat_bpm - tempo_prior_center_bpm) / tempo_prior_sigma_bpm) ** 2
    tempo_log_prior = tempo_prior_strength * (tempo_log_prior - torch.logsumexp(tempo_log_prior, 0))

    # ---- transition scales (time-invariant approx: song-mean) ----
    # match deployment sharpening: the raw prior transition is too diffuse for crisp Viterbi decode
    # (the particle filter sharpens the proposal by the same scales at deployment).
    concentration = (model.prior_phase_concentration(context).mean()
                     * phase_concentration_scale).clamp(min=1e-3)
    gamma = float(-torch.log(concentration_to_rho(concentration)))                     # wrapped-Cauchy scale
    log_tempo_grid = torch.log(advance)
    tempo_sigma = float((model.prior_tempo_sigma(context).mean()
                         * tempo_sigma_scale).clamp(min=1e-3))

    # phase transition kernel: log wrapped-Cauchy of displacement d - advance(tau), truncated to
    # +-phase_support cells. Shape [Tau, 2W+1] over displacement offsets.
    W = phase_support
    offsets = torch.arange(-W, W + 1, device=device).float()                          # [2W+1] cells
    disp_angle = offsets.unsqueeze(0) * (TWO_PI / n_phase) - advance.unsqueeze(1) % TWO_PI  # broadcast? do per-tau below
    # wrapped-Cauchy log pdf: log(sinh(g)) - log(cosh(g) - cos(theta)) - log(2pi)
    def wc_logpdf(theta):
        return (math.log(math.sinh(gamma)) - torch.log(math.cosh(gamma) - torch.cos(theta)) - math.log(TWO_PI))
    # displacement (in radians) between grid step `offset` and the predicted advance for each tau
    theta = offsets.unsqueeze(0) * (TWO_PI / n_phase) - (advance.unsqueeze(1))          # [Tau, 2W+1]
    phase_logk = wc_logpdf(theta)                                                      # [Tau, 2W+1]

    # tempo transition: Laplace on log-advance, banded over tempo index
    lt = log_tempo_grid
    tempo_logT = -(lt.unsqueeze(0) - lt.unsqueeze(1)).abs() / tempo_sigma - math.log(2 * tempo_sigma)
    tempo_logT = torch.log_softmax(tempo_logT, dim=1)                                  # [Tau, Tau] row-normalized

    # ---- emission grid (exact, per-frame): model event logits at each phase, fixed meter ----
    meter = torch.zeros(n_phase, model.num_meters, device=device)
    meter[:, min(beats_per_bar - 1, model.num_meters - 1)] = 1.0
    log_tempo_mid = log_tempo_grid[n_tempo // 2].expand(n_phase)
    lf = model.latent_feature_vector(meter, phase_grid, log_tempo_mid)
    emit_logits = model.event_decoder(model.decoder_input(lf))                         # [P, 2]

    # ---- Viterbi (max-product) over state [Tau, P] ----
    NEG = -1e9
    delta = torch.full((n_tempo, n_phase), NEG, device=device)
    delta[:] = math.log(1.0 / (n_tempo * n_phase))
    backptr_phase = torch.zeros((T, n_tempo, n_phase), dtype=torch.long, device=device)
    backptr_tempo = torch.zeros((T, n_tempo, n_phase), dtype=torch.long, device=device)

    def emission_at(t):
        # -temp * (BCE_beat + w*BCE_down) evaluated on the whole phase grid  [P]
        tgt_b = obs[t, 0].expand(n_phase); tgt_d = obs[t, 1].expand(n_phase)
        bce_b = F.binary_cross_entropy_with_logits(emit_logits[:, 0], tgt_b, reduction="none")
        bce_d = F.binary_cross_entropy_with_logits(emit_logits[:, 1], tgt_d, reduction="none")
        return -observation_temperature * (bce_b + downbeat_evidence_weight * bce_d)   # [P]

    delta = delta + emission_at(0).unsqueeze(0) + tempo_log_prior.unsqueeze(1)
    for t in range(1, T):
        # tempo step: for each target tempo, max over source tempo of delta + tempo_logT
        # delta [Tau, P]; tempo_logT [Tau_src, Tau_dst]
        after_tempo, tempo_arg = (delta.unsqueeze(2) + tempo_logT.unsqueeze(1)).max(dim=0)  # [Tau_dst? ] careful
        # delta.unsqueeze(2): [Tau_src, P, 1]; tempo_logT.unsqueeze(1): [Tau_src, 1, Tau_dst]
        # sum -> [Tau_src, P, Tau_dst]; max over Tau_src -> [P, Tau_dst]
        after_tempo = after_tempo.transpose(0, 1)          # [Tau_dst, P]
        tempo_arg = tempo_arg.transpose(0, 1)              # [Tau_dst, P]
        # phase step: for each tau, max over displacement offsets (source phase = dst - offset)
        # gather sources via roll: candidate for offset o = after_tempo shifted so that source phase
        # aligns to dst; value + phase_logk[tau, o]
        best = torch.full((n_tempo, n_phase), NEG, device=device)
        best_off = torch.zeros((n_tempo, n_phase), dtype=torch.long, device=device)
        for oi, o in enumerate(range(-W, W + 1)):
            src = torch.roll(after_tempo, shifts=o, dims=1)          # source phase = dst - o
            cand = src + phase_logk[:, oi].unsqueeze(1)              # [Tau, P]
            upd = cand > best
            best = torch.where(upd, cand, best)
            best_off = torch.where(upd, torch.full_like(best_off, oi), best_off)
        delta = best + emission_at(t).unsqueeze(0) + tempo_log_prior.unsqueeze(1)
        # store back-pointers: for each (tau,phase), source phase index and source tempo index
        src_phase = (torch.arange(n_phase, device=device).unsqueeze(0) - (best_off - W)) % n_phase
        backptr_phase[t] = src_phase
        backptr_tempo[t] = torch.gather(tempo_arg, 1, src_phase)

    # ---- backtrack ----
    map_phase_idx = torch.zeros(T, dtype=torch.long, device=device)
    map_tempo_idx = torch.zeros(T, dtype=torch.long, device=device)
    flat = delta.argmax()
    tau_i = int(flat // n_phase); ph_i = int(flat % n_phase)
    map_tempo_idx[T - 1] = tau_i; map_phase_idx[T - 1] = ph_i
    for t in range(T - 1, 0, -1):
        ph_prev = backptr_phase[t, tau_i, ph_i]
        tau_prev = backptr_tempo[t, tau_i, ph_i]
        ph_i = int(ph_prev); tau_i = int(tau_prev)
        map_phase_idx[t - 1] = ph_i; map_tempo_idx[t - 1] = tau_i

    map_phase = phase_grid[map_phase_idx]                                              # [T]
    # read-outs: beat where beats_per_bar*phase wraps; downbeat where phase wraps
    kphase = (beats_per_bar * map_phase) % TWO_PI
    beat_wrap = torch.zeros(T, device=device)
    down_wrap = torch.zeros(T, device=device)
    beat_wrap[1:] = ((kphase[1:] - kphase[:-1]) < -math.pi).float()
    down_wrap[1:] = ((map_phase[1:] - map_phase[:-1]) < -math.pi).float()
    return {"map_phase": map_phase.cpu().numpy(),
            "map_beats_per_bar": beats_per_bar,
            "beat_activation": beat_wrap.cpu().numpy(),
            "downbeat_activation": down_wrap.cpu().numpy()}


def grid_forward_loglik(model, features, observations, n_phase=180, n_tempo=32,
                        min_bpm=50.0, max_bpm=215.0, fps=22050.0 / 256.0,
                        observation_temperature=3.0, downbeat_evidence_weight=3.0,
                        beats_per_bar=4, phase_support=16,
                        phase_concentration_scale=50.0, tempo_sigma_scale=0.05,
                        tempo_prior_strength=0.5, tempo_prior_center_bpm=110.0,
                        tempo_prior_sigma_bpm=60.0):
    """DIFFERENTIABLE exact forward algorithm over the (phase,tempo) grid: returns [batch] log
    marginal likelihood log p(observations | model), summing over ALL latent paths (log-sum-exp,
    the sum-product twin of grid_viterbi's max-product). This is the VARIANCE-FREE version of the
    FIVO bound -- maximizing it trains the learned dynamics + emission to explain the test-time
    evidence exactly, with no particle noise. Reuses grid_viterbi's transition/emission but keeps
    everything in the autograd graph. Smaller default grid than the decoder (training cost)."""
    device = features.device
    B, T, _ = features.shape
    context = model.prior_context_projection(model.prior_encoder(features))            # [B, T, H]

    phase_grid = torch.arange(n_phase, device=device).float() * TWO_PI / n_phase
    adv_min = TWO_PI * (min_bpm / 60.0) / fps / beats_per_bar
    adv_max = TWO_PI * (max_bpm / 60.0) / fps / beats_per_bar
    advance = torch.exp(torch.linspace(math.log(adv_min), math.log(adv_max), n_tempo, device=device))
    beat_bpm = advance * beats_per_bar * fps / TWO_PI * 60.0
    tempo_log_prior = -0.5 * ((beat_bpm - tempo_prior_center_bpm) / tempo_prior_sigma_bpm) ** 2
    tempo_log_prior = tempo_prior_strength * (tempo_log_prior - torch.logsumexp(tempo_log_prior, 0))

    # learned transition scales (song-mean, per batch item), DIFFERENTIABLE
    conc = (model.prior_phase_concentration(context).mean(dim=1) * phase_concentration_scale).clamp(min=1e-3)  # [B]
    rho = concentration_to_rho(conc); gamma = -torch.log(rho)                            # [B]
    tempo_sigma = (model.prior_tempo_sigma(context).mean(dim=1) * tempo_sigma_scale).clamp(min=1e-3)  # [B]
    log_tempo_grid = torch.log(advance)

    W = phase_support
    offsets = torch.arange(-W, W + 1, device=device).float()                            # [2W+1]
    theta = offsets.view(1, 1, -1) - advance.view(1, -1, 1)                              # [1, Tau, 2W+1]
    # wrapped-Cauchy log pdf, per batch (gamma is [B])
    g = gamma.view(B, 1, 1)
    phase_logk = torch.log(torch.sinh(g)) - torch.log(torch.cosh(g) - torch.cos(theta)) - math.log(TWO_PI)  # [B,Tau,2W+1]

    ts = tempo_sigma.view(B, 1, 1)
    tempo_logT = -(log_tempo_grid.view(1, 1, -1) - log_tempo_grid.view(1, -1, 1)).abs() / ts - torch.log(2 * ts)
    tempo_logT = torch.log_softmax(tempo_logT, dim=2)                                    # [B, Tau_src, Tau_dst]

    # emission grid (differentiable), fixed meter, [B? no -> P,2] shared; but decoder is shared across B
    meter = torch.zeros(n_phase, model.num_meters, device=device)
    meter[:, min(beats_per_bar - 1, model.num_meters - 1)] = 1.0
    lt_mid = log_tempo_grid[n_tempo // 2].expand(n_phase)
    emit_logits = model.event_decoder(model.decoder_input(
        model.latent_feature_vector(meter, phase_grid, lt_mid)))                         # [P, 2]

    def emission_at(t):
        ob = observations[:, t, :]                                                       # [B, 2]
        # BCE per (batch, phase): -temp*(bce_beat + w*bce_down)
        lb = emit_logits[:, 0].view(1, n_phase); ld = emit_logits[:, 1].view(1, n_phase)
        pb = torch.sigmoid(lb); pd = torch.sigmoid(ld)
        bce_b = -(ob[:, 0:1] * torch.log(pb + 1e-8) + (1 - ob[:, 0:1]) * torch.log(1 - pb + 1e-8))
        bce_d = -(ob[:, 1:2] * torch.log(pd + 1e-8) + (1 - ob[:, 1:2]) * torch.log(1 - pd + 1e-8))
        return -observation_temperature * (bce_b + downbeat_evidence_weight * bce_d)     # [B, P]

    # log_alpha [B, Tau, P]
    log_alpha = (math.log(1.0 / (n_tempo * n_phase))
                 + emission_at(0).unsqueeze(1) + tempo_log_prior.view(1, n_tempo, 1))
    for t in range(1, T):
        # tempo step: logsumexp over source tempo
        after_tempo = torch.logsumexp(log_alpha.unsqueeze(3) + tempo_logT.unsqueeze(2), dim=1)  # [B,P,Tau_dst]? check
        # log_alpha[B,Tau_src,P]->unsqueeze(3)[B,Tau_src,P,1]; tempo_logT[B,Tau_src,Tau_dst]->unsqueeze(2)[B,Tau_src,1,Tau_dst]
        # sum -> [B,Tau_src,P,Tau_dst]; lse over Tau_src(dim1) -> [B,P,Tau_dst]
        after_tempo = after_tempo.permute(0, 2, 1)                                        # [B, Tau_dst, P]
        # phase step: circular logsumexp conv over offsets (source phase = dst - o)
        stack = []
        for oi, o in enumerate(range(-W, W + 1)):
            src = torch.roll(after_tempo, shifts=o, dims=2)                               # source phase = dst - o
            stack.append(src + phase_logk[:, :, oi].unsqueeze(2))                         # [B, Tau, P]
        log_alpha = torch.logsumexp(torch.stack(stack, dim=0), dim=0) + emission_at(t).unsqueeze(1)
    return torch.logsumexp(log_alpha.reshape(B, -1), dim=1)                               # [B]
