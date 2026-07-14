"""Bootstrap particle filter deployment for the VBPM -- the bar-pointer lineage's own inference
(Whiteley, Cemgil & Godsill 2006; Hainsworth & Macleod 2004), sampled rather than discretized
(madmom's DBN, Krebs/Boeck/Widmer 2015, is the grid branch of the same predict-correct recursion).

Defaults are the FIXED 2026-07-10 deployment (docs/emission_sidechannel_report.md sections 9-12):
physical proposal scales (the learned transition scales are inflated by the diagnosed side-channel
pressure), observation temperature 3, downbeat evidence up-weighted 3x (the only gauge-symmetry-
breaking evidence), and stratified bar-gauge lanes at birth (sharp proposals cannot slide a particle
across a beat, so every "which beat is beat 1" hypothesis must exist from frame 0 -- madmom
enumerates these exhaustively). All beats-per-bar factors come from each particle's own meter sample
(the soft latent) -- never a hardcoded 4.

Read-outs returned:
  * map_phase           [T]  highest-weight particle's phase trajectory
  * map_beats_per_bar        MAP particle's majority meter class (as beats per bar)
  * beat_activation     [T]  weighted fraction of particles whose BEAT phase wrapped into frame t
  * downbeat_activation [T]  same for full bar wraps (the Bayesian ensemble read-out; beats
                             same-evidence MAP by ~+0.13)
"""
import math

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Laplace

from model.latents import TWO_PI, sample_wrapped_cauchy


@torch.no_grad()
def run_particle_filter(model, features, observations, num_particles=800, ess_fraction=0.5,
                        observation_temperature=3.0, proposal_tempo_sigma_scale=0.01,
                        proposal_phase_concentration_scale=50.0, downbeat_evidence_weight=3.0,
                        stratified_gauge_init=True, observation_outlier_epsilon=0.0):
    """Filter ``observations`` [num_frames, 2] (frontend beat/downbeat probabilities in [0,1] --
    available at test time, so this needs no ground truth anywhere) through ``model``'s trained
    prior dynamics, scoring each particle with the trained z-only emission as soft Bernoulli
    evidence. Systematic resampling on low effective sample size."""
    from model.bar_pointer_vae import predicted_phase_mean

    device = features.device
    num_frames = features.shape[1]
    N = num_particles
    context = model.prior_context_projection(model.prior_encoder(features))[0]          # [T, hidden]
    # Proposal scales: near-constant tempo per particle, near-deterministic phase advance --
    # the evidence, not proposal noise, drives the trajectory.
    phase_concentration_all = model.prior_phase_concentration(context) * proposal_phase_concentration_scale
    sigma_all = model.prior_tempo_sigma(context) * proposal_tempo_sigma_scale            # [T]

    packed = model.initial_prior_head(context.mean(dim=0, keepdim=True))
    meter_logits, phase_mean, phase_concentration, log_tempo_mean, log_tempo_std = \
        model.unpack_distribution_parameters(packed)
    phase = sample_wrapped_cauchy(phase_mean[0].expand(N), phase_concentration[0])
    log_tempo = Laplace(log_tempo_mean[0], log_tempo_std[0]).sample((N,))
    meter = F.one_hot(Categorical(logits=meter_logits[0]).sample((N,)), model.num_meters).float()
    beats_per_class = torch.arange(1, model.num_meters + 1, device=device).float()
    if stratified_gauge_init:
        # Bar-gauge lanes from EACH PARTICLE'S OWN meter sample -- never a fixed 4.
        beats_per_bar_0 = (meter * beats_per_class).sum(-1)
        lane = torch.arange(N, device=device).float() % beats_per_bar_0
        phase = (phase + lane * (TWO_PI / beats_per_bar_0)) % TWO_PI
    meter_history = torch.zeros(N, model.num_meters, device=device)

    trajectory_phase = torch.empty(N, num_frames, device=device)
    trajectory_phase[:, 0] = phase
    beat_activation = torch.zeros(num_frames, device=device)
    downbeat_activation = torch.zeros(num_frames, device=device)

    channel_weights = torch.tensor([1.0, downbeat_evidence_weight], device=device)
    if observation_outlier_epsilon > 0.0:
        # Robust observation model: mix the frontend probabilities with a uniform outlier
        # component -- p' = (1-eps) p + eps/2 -- so a CONFIDENT-BUT-WRONG spike (p ~ 1 at a wrong
        # location) has bounded log-likelihood pull instead of dragging the ensemble off phase.
        # A spec choice of the observation model (the frontend is sometimes just wrong), not a
        # loss change; the direct counter to the blind-spot paper's miscalibrated-evidence regime.
        observations = ((1.0 - observation_outlier_epsilon) * observations
                        + observation_outlier_epsilon / 2.0)

    def emission_log_likelihood(t):
        logits = model.event_decoder(model.decoder_input(
            model.latent_feature_vector(meter, phase, log_tempo)))                       # [N, 2]
        bce = F.binary_cross_entropy_with_logits(
            logits, observations[t].expand(N, -1), reduction="none")
        # Downbeat channel up-weighted: it is the ONLY gauge-symmetry-breaking evidence (beat
        # evidence is beats-per-bar-fold symmetric), once per bar, and the learned bump is cautious.
        return -observation_temperature * (bce * channel_weights).sum(dim=-1)

    def systematic_resample(log_weights):
        weights = torch.softmax(log_weights, dim=0)
        positions = (torch.rand(1, device=device) + torch.arange(N, device=device)) / N
        # searchsorted may return N when float error leaves cumsum[-1] < positions[-1]; clamp.
        return torch.searchsorted(weights.cumsum(0).clamp(max=1.0), positions).clamp(max=N - 1)

    log_weights = emission_log_likelihood(0)
    for t in range(1, num_frames):
        previous_phase, previous_log_tempo, previous_meter = phase, log_tempo, meter
        delta_phase, delta_log_tempo = model.transition_mean_corrections(
            context[t].unsqueeze(0).expand(N, -1),
            model.latent_feature_vector(previous_meter, previous_phase, previous_log_tempo),
            previous_log_tempo)
        # State clamp: the unbounded tempo random walk explodes over full-length songs; bounding
        # the STATE keeps every downstream parameter finite.
        log_tempo = Laplace(previous_log_tempo + delta_log_tempo, sigma_all[t]).sample().clamp(-10.0, 3.0)
        predicted_phase = (predicted_phase_mean(previous_phase, previous_log_tempo) + delta_phase) % TWO_PI
        phase = sample_wrapped_cauchy(predicted_phase, phase_concentration_all[t])
        transition_logits = model.meter_transition_log_probabilities(
            previous_meter, phase, previous_phase, context[t].unsqueeze(0).expand(N, -1))
        meter = F.one_hot(Categorical(logits=transition_logits).sample(), model.num_meters).float()
        meter_history = meter_history + meter

        log_weights = log_weights + emission_log_likelihood(t)
        trajectory_phase[:, t] = phase

        # Bayesian wrap read-outs under the CURRENT filtering weights
        weights = torch.softmax(log_weights, dim=0)
        bar_wrapped = ((phase - previous_phase) < -math.pi).float()
        beats_per_bar_now = (meter * beats_per_class).sum(-1)          # per-particle, from the latent
        beat_step = (beats_per_bar_now * phase) % TWO_PI - (beats_per_bar_now * previous_phase) % TWO_PI
        beat_wrapped = (beat_step < -math.pi).float()
        downbeat_activation[t] = (weights * bar_wrapped).sum()
        beat_activation[t] = (weights * beat_wrapped).sum()

        if 1.0 / (weights.pow(2).sum() + 1e-12) < ess_fraction * N:
            ancestors = systematic_resample(log_weights)
            phase, log_tempo, meter = phase[ancestors], log_tempo[ancestors], meter[ancestors]
            trajectory_phase = trajectory_phase[ancestors]
            meter_history = meter_history[ancestors]
            log_weights = torch.zeros(N, device=device)

    map_index = int(log_weights.argmax())
    map_beats_per_bar = int(beats_per_class[meter_history[map_index].argmax()])
    return {"map_phase": trajectory_phase[map_index].cpu().numpy(),
            "map_beats_per_bar": map_beats_per_bar,   # MAP particle's majority meter (soft latent)
            "beat_activation": beat_activation.cpu().numpy(),
            "downbeat_activation": downbeat_activation.cpu().numpy()}


def fivo_bound(model, features, observations, num_particles=16, observation_temperature=3.0,
               downbeat_evidence_weight=3.0, gumbel_temperature=0.5):
    """Filtering Variational Objective (Maddison et al. 2017): the log of the bootstrap particle
    filter's marginal-likelihood estimate, DIFFERENTIABLE, batched over songs. Maximizing it trains
    the model + proposal so the FILTER's estimate is good -- i.e. it optimizes the exact deployment
    computation, closing the train/deploy gap that free bits / prior anchoring cannot reach.

    Resample-every-step bootstrap variant (proposal = the prior transition, so the incremental
    importance weight is the emission likelihood). Ancestor indices are detached at resampling
    (standard biased-but-workable FIVO gradient); the pathwise gradient flows through the
    reparameterized phase (wrapped-Cauchy) / tempo (Laplace) / soft-meter (gumbel) samples and the
    emission. Returns [batch] log-marginal-likelihood estimates (higher = better; loss = -mean).

    ``observations`` [batch, frames, 2] are the test-time frontend evidence (NOT ground truth).
    Use a SMALL num_particles in training (16-32); deployment still uses 800.
    """
    from model.bar_pointer_vae import predicted_phase_mean
    B, T, _ = features.shape
    N = num_particles
    device = features.device
    context = model.prior_context_projection(model.prior_encoder(features))          # [B, T, H]
    phase_conc_all = model.prior_phase_concentration(context)                          # [B, T]
    sigma_all = model.prior_tempo_sigma(context)                                       # [B, T]
    channel_weights = torch.tensor([1.0, downbeat_evidence_weight], device=device)
    beats_per_class = torch.arange(1, model.num_meters + 1, device=device).float()

    def emission_ll(t, phase, log_tempo, meter):
        logits = model.event_decoder(model.decoder_input(
            model.latent_feature_vector(meter, phase, log_tempo)))                     # [B, N, 2]
        obs_t = observations[:, t, :].unsqueeze(1).expand(B, N, 2)
        bce = F.binary_cross_entropy_with_logits(logits, obs_t, reduction="none")
        return -observation_temperature * (bce * channel_weights).sum(dim=-1)          # [B, N]

    def resample(log_w, tensors):
        # systematic resampling per batch row; indices detached (values keep their graph)
        w = torch.softmax(log_w, dim=1)
        pos = (torch.rand(B, 1, device=device) + torch.arange(N, device=device)) / N   # [B, N]
        idx = torch.searchsorted(w.cumsum(1).clamp(max=1.0), pos).clamp(max=N - 1)      # [B, N]
        out = []
        for x in tensors:
            if x.dim() == 3:
                out.append(torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])))
            else:
                out.append(torch.gather(x, 1, idx))
        return out

    packed = model.initial_prior_head(context.mean(dim=1))
    meter_logits, phase_mean, phase_conc0, log_tempo_mean, log_tempo_std = \
        model.unpack_distribution_parameters(packed)
    phase = sample_wrapped_cauchy(phase_mean.unsqueeze(1).expand(B, N), phase_conc0.unsqueeze(1))
    log_tempo = Laplace(log_tempo_mean.unsqueeze(1).expand(B, N),
                        log_tempo_std.unsqueeze(1)).rsample().clamp(-10.0, 3.0)
    meter = F.gumbel_softmax(meter_logits.unsqueeze(1).expand(B, N, -1), tau=gumbel_temperature, dim=-1)

    log_estimate = torch.zeros(B, device=device)
    log_w = emission_ll(0, phase, log_tempo, meter)
    log_estimate = log_estimate + torch.logsumexp(log_w, dim=1) - math.log(N)
    phase, log_tempo, meter = resample(log_w, [phase, log_tempo, meter])

    for t in range(1, T):
        prev_phase, prev_log_tempo, prev_meter = phase, log_tempo, meter
        ctx_t = context[:, t].unsqueeze(1).expand(B, N, -1)
        delta_phase, delta_log_tempo = model.transition_mean_corrections(
            ctx_t, model.latent_feature_vector(prev_meter, prev_phase, prev_log_tempo), prev_log_tempo)
        log_tempo = Laplace(prev_log_tempo + delta_log_tempo,
                            sigma_all[:, t].unsqueeze(1).expand(B, N)).rsample().clamp(-10.0, 3.0)
        predicted = (predicted_phase_mean(prev_phase, prev_log_tempo) + delta_phase) % TWO_PI
        phase = sample_wrapped_cauchy(predicted, phase_conc_all[:, t].unsqueeze(1).expand(B, N))
        trans_logits = model.meter_transition_log_probabilities(
            prev_meter.reshape(B * N, -1), phase.reshape(B * N), prev_phase.reshape(B * N),
            ctx_t.reshape(B * N, -1)).reshape(B, N, -1)
        meter = F.gumbel_softmax(trans_logits, tau=gumbel_temperature, dim=-1)
        log_w = emission_ll(t, phase, log_tempo, meter)
        log_estimate = log_estimate + torch.logsumexp(log_w, dim=1) - math.log(N)
        phase, log_tempo, meter = resample(log_w, [phase, log_tempo, meter])
    return log_estimate


def untrained_control_model(feature_dim=512, hidden_size=64, num_meters=4, seed=0,
                            transition_correction_scale=0.5, decoder_input_mode="parametric"):
    """The architecture-only control: an UNTRAINED model with the same construction, for quoting
    learned-vs-machinery deltas next to every filter number (e.g. clean GTZAN 2026-07-10: trained
    Bayes 0.868/0.754 vs untrained 0.615/0.548)."""
    from model.bar_pointer_vae import VariationalBarPointerModel
    torch.manual_seed(seed)
    model = VariationalBarPointerModel(
        feature_dim=feature_dim, hidden_size=hidden_size, num_meters=num_meters,
        transition_correction_scale=transition_correction_scale,
        decoder_input_mode=decoder_input_mode)
    model.eval()
    return model
