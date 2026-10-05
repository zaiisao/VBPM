"""Train the tutorial model on a small fixed batch of real MusicFM features.

The canonical model/loss/sampler files stay unchanged. This is a real-data
exercise with an explicit input adapter, not a repair of latent identification.
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from audit_reference import audit
from reference import vae_dbn as ref
from reference.train_logger import (
    TrainLogger, HealthMonitor, global_grad_norm, tensor_stats)
from debug import gradient_routes
from data import (
    FPS, FRAMES, CONTEXT_FRAMES, SAMPLE_RATE, OUTPUT_DIR, STORE, annotation_reference, rasterize, sha256)
from features import prepare
from model import MusicFMCVAEDBN

HERE = Path(__file__).resolve().parent
OUT = OUTPUT_DIR
STEPS, LR, ALPHA, BETA, TAU = 200, .003, .7, 1., .5
SNAPSHOTS = (0, 1, 20, 50, 100, 200)
CHECKS = []


def check(name, passed, **details):
    CHECKS.append(dict(name=name, passed=bool(passed), **details))
    print('CHECK', name, bool(passed), details, flush=True)
    if not passed:
        raise RuntimeError(name)


def save_report(report):
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def data_checks(data, manifest):
    ids = [clip['song_id'] for split in data.values() for clip in split['clips']]
    check('Eight distinct songs with disjoint training and validation splits',
          len(ids) == len(set(ids)) == 8)
    check('Native waveform extraction; 12 layers, no temporal pooling',
          manifest['native_waveform'] and manifest['layers'] == list(range(1, 13))
          and manifest['excluded_state'] == 0 and manifest['feature_fps'] == 25
          and all(entry['exact_layer_concatenation'] for entry in manifest['audits']))
    check('Feature cache checksum matches extraction manifest',
          sha256(OUT / 'features.pt') == manifest['cache_sha256'])
    check('Every target and encoder input covers 30 seconds with 750 native 25 Hz frames',
          FRAMES / FPS >= 30 and CONTEXT_FRAMES >= FRAMES
          and manifest['frames'] == FRAMES and manifest['context_frames'] == CONTEXT_FRAMES
          and all(entry['waveform_shape'] == [1, SAMPLE_RATE * CONTEXT_FRAMES / FPS]
                  and entry['feature_shape'] == [FRAMES, 12288] for entry in manifest['audits']))
    for split, batch in data.items():
        check(split + f': four clips with finite frozen [4,{FRAMES},12288] features',
              tuple(batch['x'].shape) == (4, FRAMES, 12288)
              and batch['x'].dtype == torch.float32 and not batch['x'].requires_grad
              and bool(torch.isfinite(batch['x']).all()))
        check(split + ': categorical targets contain all three classes',
              tuple(batch['b'].shape) == (4, FRAMES)
              and batch['b'].dtype == torch.int64
              and torch.equal(torch.unique(batch['b']), torch.arange(3)))
        check(split + ': two clips each in 3/4 and 4/4',
              Counter(clip['meter'] for clip in batch['clips']) == {3: 2, 4: 2})
        for i, clip in enumerate(batch['clips']):
            annotations = np.loadtxt(clip['annotation_path'])
            db = annotations[annotations[:, 1] == 1, 0]
            times = (clip['target_start_frame'] + np.arange(FRAMES)) / FPS
            phase, velocity, meters, valid = annotation_reference(annotations, times)
            check(clip['song_id'] + ': full native crop and masked edge references',
                  clip['target_seconds'] >= 30 and clip['context_start_seconds'] >= 0
                  and clip['context_start_seconds'] + clip['context_seconds'] <= clip['duration_seconds']
                  and np.array_equal(valid, batch['reference_valid'][i].numpy())
                  and np.array_equal(phase, batch['phase_reference'][i].numpy())
                  and np.array_equal(velocity, batch['velocity_reference'][i].numpy())
                  and np.array_equal(meters, batch['meter_reference'][i].numpy()),
                  target_seconds=clip['target_seconds'], reference_valid_frames=int(valid.sum()))
            expected, events = rasterize(annotations[:, 0], db, clip['target_start_frame'])
            errors = []
            for group in events.values():
                errors.extend(abs(group['indices'] / FPS - group['times']))
            check(clip['song_id'] + ': annotation rasterization and 40 ms timebase',
                  np.array_equal(expected, batch['b'][i].numpy())
                  and bool((batch['b'][i, clip['downbeat_indices']] == 2).all())
                  and max(errors, default=0) <= .0200001,
                  max_annotation_rounding_seconds=float(max(errors, default=0)))
            check(clip['song_id'] + ': native audio and annotation checksums unchanged',
                  sha256(clip['audio_path']) == clip['audio_sha256']
                  and sha256(clip['annotation_path']) == clip['annotation_sha256'])
    check('Validation remains official fold 7; training excludes it',
          all(c['fold'] == 7 for c in data['heldout']['clips'])
          and all(c['fold'] != 7 for c in data['training']['clips']))


def preflight(data):
    with torch.random.fork_rng(devices=[]):
        check('Original PDF standalone von Mises sampler debug test', ref.test_vonmises_reparam())
        torch.manual_seed(0)
        adapted = MusicFMCVAEDBN().eval()
        ordinary = ref.VAEDBN(x_dim=64).eval()
        old_state = {key: value for key, value in adapted.state_dict().items()
                     if not key.startswith('input_projection.')}
        ordinary.load_state_dict(old_state)
        raw, labels = data['x'][:2, :16], data['b'][:2, :16]
        projected = adapted.project_features(raw)
        check('Adapter produces finite 64-dimensional covariates',
              tuple(projected.shape) == (2, 16, 64) and bool(torch.isfinite(projected).all()))
        torch.manual_seed(456)
        expected = ref.hybrid_loss(ordinary, projected, labels)
        torch.manual_seed(456)
        actual = ref.hybrid_loss(adapted, raw, labels)
        check('Wrapped model exactly equals original tutorial on projected inputs',
              all(torch.equal(a, b) for a, b in zip(expected, actual)))
        check('Transition, emission, posterior inference and prediction methods are inherited',
              type(adapted).rollout is ref.VAEDBN.rollout
              and type(adapted).phase_params is ref.VAEDBN.phase_params
              and type(adapted).feats is ref.VAEDBN.feats
              and adapted.Delta == 1. and adapted.emit[0].in_features == 70
              and adapted.emit[-1].out_features == 3 and adapted.R == 3)
        inputs = raw.clone().requires_grad_()
        h = adapted.backbone_feats(inputs)
        gradient = torch.autograd.grad(h[:, 0].square().sum(), inputs)[0]
        check('Future MusicFM feature frames reach the prior backbone',
              float(gradient[:, 1:].norm()) > 0)
        routing = gradient_routes(adapted, dict(x=raw, b=labels))
        check('Finite gradients reach both velocity heads, phase heads and emission',
              all(math.isfinite(v) for item in routing.values() for v in item.values())
              and routing['post_velocity']['posterior_recon'] > 0
              and routing['prior_velocity']['prior_recon'] > 0
              and routing['prior_kappa']['prior_recon'] > 0
              and routing['post_phase']['posterior_recon'] > 0
              and routing['emission']['posterior_recon'] > 0)
        (OUT / 'gradient_routes.json').write_text(json.dumps(routing, indent=2) + '\n')


def circular_mae(delta):
    return float(torch.atan2(delta.sin(), delta.cos()).abs().mean() * 180 / math.pi)


@torch.no_grad()
def posterior_summary(model, data):
    path, parameters = ref.encode_path(model, data['x'], data['b'])
    phase = torch.stack([entry['mu_phi'] for entry in parameters], 1)
    velocity = torch.stack([entry['vbar'] for entry in parameters], 1)
    kappa = torch.stack([entry['kappa'] for entry in parameters], 1)
    meter_probability = torch.stack([entry['rho'] for entry in parameters], 1)
    h = model.backbone_feats(data['x'])
    features = torch.stack([torch.cat((model.feats(pp, vv, F.one_hot(mm, model.R).float()), h[:, t]), -1)
                            for t, (pp, vv, mm) in enumerate(path)], 1)
    posterior_logits = model.emit(features)
    reference = data['phase_reference']
    valid = data['reference_valid']
    aligned = []
    for sign in (1, -1):
        delta = (sign * phase - reference)[valid]
        offset = torch.atan2(delta.sin().mean(), delta.cos().mean())
        aligned.append(dict(sign=sign, offset_deg=float(offset * 180 / math.pi),
                            MAE_deg=circular_mae(delta - offset)))
    metrics = dict(annotation_reference_phase_MAE_deg=circular_mae((phase - reference)[valid]),
        physical_reference_scored_frames=int(valid.sum()),
        physical_reference_total_frames=valid.numel(),
        best_global_origin_direction_reference_MAE_deg=min(r['MAE_deg'] for r in aligned),
        evaluation_alignment_candidates=aligned,
        phase_MAE_at_annotated_downbeats_deg=circular_mae(phase[data['b'] == 2]),
        bar_average_velocity_reference_RMSE_radians_per_frame=float(
            (velocity - data['velocity_reference'])[valid].square().mean().sqrt()),
        inferred_velocity_mean_radians_per_frame=float(velocity.mean()),
        reference_velocity_mean_radians_per_frame=float(data['velocity_reference'][valid].mean()),
        negative_velocity_mean_fraction=float((velocity < 0).float().mean()),
        q_kappa_mean=float(kappa.mean()),
        posterior_mode_label_NLL_per_frame=float(F.cross_entropy(posterior_logits.transpose(1, 2), data['b'])),
        intended_meter_class_accuracy=float(((meter_probability.argmax(-1) + 2) == data['meter_reference'])[valid].float().mean()))
    diagnostics = dict(phase=phase, velocity=velocity, kappa=kappa, meter_probability=meter_probability,
                       posterior_probability=posterior_logits.softmax(-1))
    return metrics, diagnostics, features


def classification(probability, labels):
    prediction = probability.argmax(-1)
    confusion = torch.bincount((labels * 3 + prediction).flatten(), minlength=9).reshape(3, 3).float()
    tp = confusion.diag()
    precision = tp / confusion.sum(0).clamp_min(1)
    recall = tp / confusion.sum(1).clamp_min(1)
    f1 = 2 * tp / (confusion.sum(0) + confusion.sum(1)).clamp_min(1)
    return dict(accuracy=float((prediction == labels).float().mean()),
        balanced_accuracy=float(recall.mean()), macro_F1=float(f1.mean()),
        NLL_per_frame=float(-probability.gather(-1, labels[..., None]).clamp_min(1e-12).log().mean()),
        confusion_matrix_true_rows_predicted_columns=confusion.int().tolist(),
        per_class={name: dict(precision=float(precision[i]), recall=float(recall[i]), F1=float(f1[i]),
                             true_frames=int(confusion[i].sum()), predicted_frames=int(confusion[:, i].sum()))
                   for i, name in enumerate(('non_beat', 'ordinary_beat', 'downbeat'))})


def event_scores(probability, clips):
    output = {}
    for name, values, truth_name in (('beat', probability[..., 1:].sum(-1), 'beat_times'),
                                      ('downbeat', probability[..., 2], 'downbeat_times')):
        true_count = predicted_count = matched = 0
        events = []
        for curve, clip in zip(values.numpy(), clips):
            indices, _ = find_peaks(np.pad(curve, (1, 1), constant_values=-1), height=.5,
                                    distance=3)
            predicted = (indices - 1) / FPS
            truth = np.array(clip[truth_name])
            # Maximum one-to-one matches within tolerance on sorted time lists.
            i = j = matches = 0
            while i < len(predicted) and j < len(truth):
                if predicted[i] < truth[j] - .07:
                    i += 1
                elif truth[j] < predicted[i] - .07:
                    j += 1
                else:
                    matches += 1
                    i += 1
                    j += 1
            true_count += len(truth); predicted_count += len(predicted); matched += matches
            events.append(dict(song_id=clip['song_id'], predicted_times=predicted.tolist(),
                               annotated_times=truth.tolist(), matched=matches))
        output[name] = dict(F1=2 * matched / max(1, true_count + predicted_count),
            precision=matched / max(1, predicted_count), recall=matched / max(1, true_count),
            true_events=true_count, predicted_events=predicted_count, matched=matched,
            probability_threshold=.5, tolerance_seconds=.07, minimum_peak_spacing_seconds=.12,
            clips=events)
    return output


@torch.no_grad()
def evaluate(model, data, seed):
    metrics, diagnostic, features = posterior_summary(model, data)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4000 + seed)
        _, probability = ref.predict_labels(model, data['x'], N=64)
        # Frozen probes preserve the inferred path and change only decoder inputs.
        order = torch.randperm(FRAMES, generator=torch.Generator().manual_seed(909))
        nll = lambda value: float(F.cross_entropy(model.emit(value).transpose(1, 2), data['b']))
        shuffled = features.clone(); shuffled[..., :6] = features[:, order, :6]
        metrics['joint_latents_shuffled_NLL_per_frame'] = nll(shuffled)
        shuffled = features.clone(); shuffled[..., 6:] = features[:, order, 6:]
        metrics['decoder_context_shuffled_NLL_per_frame'] = nll(shuffled)
        # Same random numbers: audio-only prediction must never call q(b,x).
        saved_context = model.context
        def forbidden(*args, **kwargs):
            raise RuntimeError('Prior prediction accessed the posterior context')
        model.context = forbidden
        torch.manual_seed(5678)
        _, first = ref.predict_labels(model, data['x'][:, :8], N=2)
        torch.manual_seed(5678)
        _, second = ref.predict_labels(model, data['x'][:, :8], N=2)
        model.context = saved_context
        check('Prior label prediction has no annotation/posterior-context route', torch.equal(first, second))
    diagnostic['prior_probability'] = probability
    metrics['posterior_labels'] = classification(diagnostic['posterior_probability'], data['b'])
    metrics['prior_labels'] = classification(probability, data['b'])
    metrics['prior_events'] = event_scores(probability, data['clips'])
    metrics['majority_nonbeat_accuracy'] = float((data['b'] == 0).float().mean())
    return metrics, diagnostic


def train(data, heldout, seed):
    torch.manual_seed(seed)
    model = MusicFMCVAEDBN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    folder = OUT / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(str(folder / 'training'))
    health = HealthMonitor(kappa_key='kappa_q/mean', sigma_key='sigma_q/mean')
    names = dict(input_projection='input_projection', backbone='backbone', backbone_projection='hb',
        label_embedding='b_emb', encoder='encoder', posterior_context='hc', posterior_trunk='post',
        posterior_phase='post_phase', posterior_velocity='post_vel', posterior_meter='post_meter',
        prior_trunk='pri', prior_phase_concentration='pri_kappa', prior_velocity='pri_vel',
        prior_meter='pri_meter', emission='emit')
    groups = {label: getattr(model, attribute) for label, attribute in names.items()}
    warning_counts, observations = Counter(), []

    def observe(step):
        # Only posterior-mode inference; no training RNG is consumed.
        before = torch.random.get_rng_state()
        item = dict(step=step)
        for split, batch in (('training', data), ('heldout', heldout)):
            metrics, _, _ = posterior_summary(model, batch)
            item[split] = {key: metrics[key] for key in (
                'annotation_reference_phase_MAE_deg', 'q_kappa_mean',
                'bar_average_velocity_reference_RMSE_radians_per_frame',
                'posterior_mode_label_NLL_per_frame')}
        if not torch.equal(before, torch.random.get_rng_state()):
            raise RuntimeError('Inference observer consumed training randomness')
        observations.append(item)
        torch.save(dict(state=model.state_dict(), step=step, seed=seed, projected_dim=64),
                   folder / f'step_{step:03d}.pt')

    observe(0)
    for step in range(1, STEPS + 1):
        diag = {}
        loss, recon_q, kl, recon_p = ref.hybrid_loss(model, data['x'], data['b'],
                                                   alpha=ALPHA, beta=BETA, tau=TAU, diag=diag)
        if not all(torch.isfinite(t).all() for t in (loss, recon_q, kl, recon_p)):
            raise RuntimeError(f'Seed {seed}, step {step}: nonfinite loss')
        optimizer.zero_grad(); loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError(f'Seed {seed}, step {step}: nonfinite gradient')
        metrics = dict(loss=float(loss.detach()), recon=float(recon_q.detach()),
            kl=float(kl.detach()), gsnn_recon=float(recon_p.detach()),
            kl_phase=float(diag['kl_phase']), kl_vel=float(diag['kl_vel']), kl_meter=float(diag['kl_meter']),
            alpha=ALPHA, beta=BETA, tau=TAU)
        before = {name: [p.detach().clone() for p in module.parameters()] for name, module in groups.items()}
        for name, module in groups.items():
            metrics['grad/' + name] = global_grad_norm(module.parameters())
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError(f'Seed {seed}, step {step}: nonfinite parameter')
        for name, module in groups.items():
            parameters = list(module.parameters())
            change = math.sqrt(sum(float((p.detach() - old).square().sum()) for p, old in zip(parameters, before[name])))
            size = math.sqrt(sum(float(old.square().sum()) for old in before[name]))
            metrics['update_over_param/' + name] = change / max(size, 1e-12)
        metrics.update(tensor_stats('kappa_q', diag['kappa_q']))
        metrics.update(tensor_stats('sigma_q', diag['sigma_q']))
        metrics.update(tensor_stats('meter_entropy', diag['meter_entropy']))
        warnings = health.check(metrics)
        warning_counts.update(message.split(' -> ')[0].split(' (')[0] for message in warnings)
        logger.log(step, metrics, warnings)
        if step in SNAPSHOTS:
            observe(step)
        if step == 1 or step % 20 == 0:
            print('TRAIN', seed, step, 'loss', round(metrics['loss'], 3),
                  'recon', round(metrics['recon'], 3), 'kappa', metrics['kappa_q/mean'], flush=True)
    logger.close()
    torch.save(dict(state=model.state_dict(), optimizer=optimizer.state_dict(), seed=seed,
                    steps=STEPS, projected_dim=64, feature_manifest_sha256=sha256(OUT / 'feature_manifest.json')),
               folder / 'model.pt')
    rows = list(csv.DictReader((folder / 'training.csv').open()))
    check(f'Seed {seed}: 200 finite training rows', len(rows) == STEPS
          and all(math.isfinite(float(v)) for row in rows for v in row.values()))
    check(f'Seed {seed}: every individual model module receives gradients', all(
        max(float(row['grad/' + name]) for row in rows) > 0 for name in groups))
    check(f'Seed {seed}: fixed-batch hybrid loss decreases', float(rows[-1]['loss']) < float(rows[0]['loss']),
          initial=float(rows[0]['loss']), final=float(rows[-1]['loss']))
    (folder / 'observations.json').write_text(json.dumps(observations, indent=2) + '\n')
    return model.eval(), dict(rows=len(rows), initial_loss=float(rows[0]['loss']),
        final_loss=float(rows[-1]['loss']), warning_counts=dict(warning_counts), observations=observations)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract-device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--data-store', type=Path, default=STORE)
    parser.add_argument('--musicfm-weights', type=Path, default=ROOT / 'assets/musicfm/pretrained_msd.pt')
    parser.add_argument('--musicfm-stats', type=Path, default=ROOT / 'assets/musicfm/msd_stats.json')
    parser.add_argument('--tutorial-pdf', type=Path, help='Optional original PDF; otherwise audit archived code extractions')
    parser.add_argument('--extract-only', action='store_true')
    parser.add_argument('--rebuild-cache', action='store_true')
    args = parser.parse_args()
    OUT = args.output.resolve()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    fidelity = audit(OUT / 'reference_fidelity', args.tutorial_pdf)
    check('Canonical tutorial computations match the supplied PDF or archived extraction', fidelity['passed'])
    if args.rebuild_cache or not (OUT / 'features.pt').exists():
        data, manifest = prepare(args.extract_device, weights=args.musicfm_weights,
                                 statistics=args.musicfm_stats, store=args.data_store, output=OUT)
    else:
        manifest = json.loads((OUT / 'feature_manifest.json').read_text())
        check('Cached sequence lengths match the current 30-second configuration',
              manifest['frames'] == FRAMES and manifest['context_frames'] == CONTEXT_FRAMES)
        check('Cached feature bytes match their manifest before loading',
              sha256(OUT / 'features.pt') == manifest['cache_sha256'])
        data = torch.load(OUT / 'features.pt', weights_only=True)
    data_checks(data, manifest)
    if args.extract_only:
        (OUT / 'extraction_checks.json').write_text(json.dumps(CHECKS, indent=2) + '\n')
        print('DONE: extracted and checked', OUT / 'features.pt', flush=True)
        return
    preflight(data['training'])
    reference_paths = [ROOT / 'reference' / name
                       for name in ('vae_dbn.py', 'train_logger.py', 'plot_logs.py')]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in reference_paths}
    report = dict(settings=dict(seeds=[0, 1], steps=STEPS, learning_rate=LR,
        alpha=ALPHA, beta=BETA, tau=TAU, kl_warmup=False, appendix_defaults_retained=True,
        training_batch=4, heldout_batch=4, frames=FRAMES, fps=FPS, seconds_per_clip=FRAMES / FPS,
        encoder_audio_context_seconds=CONTEXT_FRAMES / FPS, feature_layers=list(range(1, 13)),
        concatenated_channels=12288, projected_channels=64,
        layer_normalization='per-frame, per-layer, no affine; no fitted validation statistics',
        velocity_units='radians per 25 Hz feature frame; Delta=1 frame, equivalent to .04 seconds',
        inferred_meter_mapping=[2, 3, 4], real_data_meters=[3, 4], prior_prediction_paths=64,
        training_device='cpu', musicfm_frozen=True, native_audio=True,
        observation_classes=['non_beat', 'ordinary_beat', 'downbeat'],
        scope='small fixed-batch real-data exercise; not a generalization benchmark or decoder repair',
        annotation_references='linear phase between downbeats and bar-average velocity; evaluation only; unbracketed edge frames masked',
        training_frames_per_clip=FRAMES, training_sequence_truncated=False),
        feature_manifest_sha256=sha256(OUT / 'feature_manifest.json'), source_hashes=source_hashes,
        fidelity=fidelity, checks=CHECKS, seeds={})
    save_report(report)
    for seed in (0, 1):
        model, health = train(data['training'], data['heldout'], seed)
        evaluations = {}
        for split, batch in data.items():
            print('EVALUATE', seed, split, '64 prior paths', flush=True)
            evaluations[split], diagnostic = evaluate(model, batch, seed)
            torch.save(dict(reference=batch, inferred=diagnostic), OUT / f'seed_{seed}' / f'{split}_diagnostics.pt')
        report['seeds'][str(seed)] = dict(training_health=health, evaluation=evaluations)
        save_report(report)
        print('RESULT', seed, json.dumps({split: dict(prior_accuracy=result['prior_labels']['accuracy'],
            beat_F1=result['prior_events']['beat']['F1'], downbeat_F1=result['prior_events']['downbeat']['F1'],
            reference_phase_MAE=result['annotation_reference_phase_MAE_deg'])
            for split, result in evaluations.items()}), flush=True)
    check('Canonical reference sources unchanged throughout experiment', all(
        sha256(ROOT / name) == expected for name, expected in source_hashes.items()))
    report['experiment_source_hashes'] = {name: sha256(HERE / name)
                                        for name in ('data.py', 'features.py', 'model.py', 'run.py', 'plot.py', 'debug.py', 'audit_reference.py')}
    save_report(report)
    from plot import render
    render(OUT)
    print('DONE', OUT / 'report.json', flush=True)


if __name__ == '__main__':
    main()
