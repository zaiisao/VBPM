"""Audit the unchanged PDF reference against its own synthetic latent ground truth."""

import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference'))
import vae_dbn as reference


@torch.no_grad()
def ground_truth():
    """Capture synth's existing local latent tensors without altering its computations."""
    truth = {}

    def capture(frame, event, value):
        if event == 'return' and frame.f_code is reference.synth.__code__:
            truth.update({name: frame.f_locals[name].clone() for name in ('phi', 'v', 'm')})

    previous_profile = sys.getprofile()
    sys.setprofile(capture)
    try:
        x, labels = reference.synth()
    finally:
        sys.setprofile(previous_profile)
    return x.to(reference.DEV), labels.to(reference.DEV), truth


def latent_metrics(phase, velocity, meter, truth):
    """Report absolute and rotation-aligned phase error, velocity error, and meter switching."""
    true_phase = truth['phi'].to(phase.device)
    true_velocity = truth['v'].to(velocity.device)
    true_meter = truth['m'].to(meter.device)[:, None].expand_as(true_phase)
    difference = phase - true_phase
    difference = torch.atan2(difference.sin(), difference.cos())
    rotation = torch.atan2(difference.sin().mean(), difference.cos().mean())
    aligned = difference - rotation
    aligned = torch.atan2(aligned.sin(), aligned.cos())
    agreement = (meter == true_meter).float().mean()
    best = max(float((torch.tensor(p, device=meter.device)[meter] == true_meter).float().mean())
               for p in itertools.permutations(range(3)))
    return dict(
        phase_mae_rad=float(difference.abs().mean()),
        phase_mae_after_global_rotation_rad=float(aligned.abs().mean()),
        phase_alignment_resultant=float(
            torch.hypot(difference.sin().mean(), difference.cos().mean())
        ),
        velocity_rmse_rad_per_step=float((velocity - true_velocity).square().mean().sqrt()),
        mean_velocity=float(velocity.mean()), reference_mean_velocity=float(true_velocity.mean()),
        negative_velocity_fraction=float((velocity < 0).float().mean()),
        meter_accuracy=float(agreement), meter_accuracy_best_label_permutation=best,
        meter_change_fraction=float((meter[..., 1:] != meter[..., :-1]).float().mean()),
    )


@torch.no_grad()
def main():
    """Run the reference's own prior and posterior inference, retaining emission inputs."""
    directory = Path('runs/pdf_reference')
    model = reference.load_model(str(directory / 'vae_dbn.pt'))
    x, labels, truth = ground_truth()
    emitted_inputs = []
    hook = model.emit.register_forward_pre_hook(
        lambda module, args: emitted_inputs.append(args[0].detach().clone())
    )
    try:
        predictions, probabilities = reference.predict_labels(model, x, N=64)
    finally:
        hook.remove()
    features = torch.stack(emitted_inputs).reshape(64, x.shape[1], x.shape[0], -1)
    features = features.permute(0, 2, 1, 3)
    phase = torch.atan2(features[..., 1], features[..., 0])
    velocity = features[..., 2]
    meter = features[..., 3:6].argmax(-1)
    paths = dict(prior=dict(phase=phase, velocity=velocity, meter=meter))
    results = {'prior': latent_metrics(phase, velocity, meter, truth)}
    results['prior']['label_accuracy'] = float((predictions == labels).float().mean())
    fixed = features.clone()
    fixed[..., :6] = torch.tensor([1., 0., 0., 1/3, 1/3, 1/3], device=x.device)
    audio_only = model.emit(fixed).softmax(-1).mean(0)
    no_audio = features.clone()
    no_audio[..., 6:] = 0
    latent_only = model.emit(no_audio).softmax(-1).mean(0)
    results['emission_controls'] = dict(
        fixed_latent_accuracy=float((audio_only.argmax(-1) == labels).float().mean()),
        zero_audio_accuracy=float((latent_only.argmax(-1) == labels).float().mean()),
        mean_absolute_probability_change_fixed_latents=float(
            (audio_only - probabilities).abs().mean()
        ),
        majority_class_accuracy=float(
            labels.flatten().bincount(minlength=3).max() / labels.numel()
        ),
    )
    for sample in (False, True):
        path, parameters = reference.encode_path(model, x, labels, sample=sample)
        phase = torch.stack([state[0] for state in path], dim=1)
        velocity = torch.stack([state[1] for state in path], dim=1)
        meter = torch.stack([state[2] for state in path], dim=1)
        name = 'posterior_sample' if sample else 'posterior_mode'
        paths[name] = dict(phase=phase, velocity=velocity, meter=meter)
        results[name] = latent_metrics(phase, velocity, meter, truth)
        results[name]['mean_concentration'] = float(
            torch.stack([p['kappa'] for p in parameters]).mean()
        )
        results[name]['mean_velocity_std'] = float(torch.stack([p['s'] for p in parameters]).mean())
    results['scope'] = 'Same synthetic training batch; original PDF source unchanged'
    (directory / 'latent_audit.json').write_text(json.dumps(results, indent=2) + '\n')
    torch.save(dict(truth=truth, paths=paths, probabilities=probabilities, labels=labels),
               directory / 'latent_audit.pt')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
