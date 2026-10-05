"""Training health, classification and physical-coordinate diagnostics."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from data import FPS, OUTPUT_DIR

HERE = Path(__file__).resolve().parent
OUT = OUTPUT_DIR


def render(output=OUTPUT_DIR):
    OUT = Path(output)
    report = json.loads((OUT / 'report.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for seed, color in ((0, '#2171b5'), (1, '#d95f0e')):
        rows = list(csv.DictReader((OUT / f'seed_{seed}' / 'training.csv').open()))
        steps = [int(row['step']) for row in rows]
        axes[0, 0].plot(steps, [float(row['loss']) for row in rows], color=color, label=f'Seed {seed}')
        for key, style in (('kl_phase', '-'), ('kl_vel', '--'), ('kl_meter', ':')):
            axes[0, 1].plot(steps, [float(row[key]) for row in rows], color=color, linestyle=style,
                            label=f'{key}, seed {seed}')
        axes[1, 0].plot(steps, [float(row['kappa_q/mean']) for row in rows], color=color, label=f'Seed {seed}')
        for split, style in (('training', '-'), ('heldout', '--')):
            history = report['seeds'][str(seed)]['training_health']['observations']
            axes[1, 1].plot([h['step'] for h in history],
                [h[split]['annotation_reference_phase_MAE_deg'] for h in history],
                color=color, linestyle=style, label=f'{split}, seed {seed}')
    axes[0, 0].set_title('Fixed-batch hybrid loss')
    axes[0, 1].set_title('Closed-form KL terms')
    axes[1, 0].set_title('Posterior phase concentration')
    axes[1, 1].set_title('Phase error against linear bar reference (degrees)')
    for ax in axes.flat:
        ax.set_xlabel('Adam update')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('MusicFM layers 1–12; native audio, 25 fps; original tutorial loss and dynamics')
    fig.savefig(OUT / 'training.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    labels = ['Training', 'Held-out']
    for seed, color in ((0, '#2171b5'), (1, '#d95f0e')):
        for j, split in enumerate(('training', 'heldout')):
            metrics = report['seeds'][str(seed)]['evaluation'][split]
            for ax, value in zip(axes, (metrics['prior_labels']['accuracy'],
                                       metrics['prior_events']['beat']['F1'],
                                       metrics['prior_events']['downbeat']['F1'])):
                ax.bar(j + (seed - .5) * .3, value, width=.28, color=color,
                       label=f'Seed {seed}' if j == 0 else None)
            axes[0].hlines(metrics['majority_nonbeat_accuracy'], j-.35, j+.35,
                           color='black', linestyle=':', linewidth=1)
    for ax, title in zip(axes, ('Frame accuracy; dotted = always non-beat',
                               'Beat event F1, 70 ms tolerance', 'Downbeat event F1, 70 ms tolerance')):
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 1], labels); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    fig.suptitle('Audio-only prior prediction, 64 paths; 4 clips per split, not a generalization benchmark')
    fig.savefig(OUT / 'label_scores.png', dpi=160); plt.close(fig)

    for seed in (0, 1):
        saved = torch.load(OUT / f'seed_{seed}' / 'heldout_diagnostics.pt', weights_only=True)
        data, inferred = saved['reference'], saved['inferred']
        selected = [next(i for i, clip in enumerate(data['clips']) if clip['meter'] == meter) for meter in (3, 4)]
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
        time = np.arange(data['x'].shape[1]) / FPS
        for col, i in enumerate(selected):
            clip = data['clips'][i]
            probability = inferred['prior_probability'][i].numpy()
            axes[0, col].plot(time, probability[:, 1:].sum(-1), label='Prior beat probability', color='#2171b5')
            axes[0, col].plot(time, probability[:, 2], label='Prior downbeat probability', color='#d95f0e')
            for t in clip['beat_times']:
                axes[0, col].axvline(t, color='gray', alpha=.3, linewidth=.8)
            for t in clip['downbeat_times']:
                axes[0, col].axvline(t, color='black', alpha=.8, linewidth=1)
            axes[0, col].set_title(clip['song_id'] + f" ({clip['meter']}/4)", fontsize=8)
            axes[0, col].set_ylim(-.02, 1.02); axes[0, col].legend(fontsize=7)
            valid = data['reference_valid'][i].numpy()
            truth = np.full(len(time), np.nan)
            truth[valid] = np.unwrap(data['phase_reference'][i].numpy()[valid]) / (2 * np.pi)
            predicted = np.unwrap(np.mod(inferred['phase'][i].numpy(), 2*np.pi)) / (2*np.pi)
            first = np.flatnonzero(valid)[0]
            predicted += np.round(truth[first] - predicted[first])
            axes[1, col].plot(time, truth, color='black', label='Linear bar reference from downbeats')
            axes[1, col].plot(time, predicted, color='#d95f0e', label='Posterior mean phase')
            axes[1, col].set_ylabel('Unwrapped phase (bar turns)'); axes[1, col].legend(fontsize=7)
            velocity_reference = data['velocity_reference'][i].numpy().copy() * FPS / (2*np.pi)
            velocity_reference[~valid] = np.nan
            axes[2, col].plot(time, velocity_reference,
                              color='black', label='Annotated bar-average rate')
            axes[2, col].plot(time, inferred['velocity'][i].numpy() * FPS / (2*np.pi),
                              color='#d95f0e', label='Posterior velocity mean')
            axes[2, col].set_ylabel('Bars per second'); axes[2, col].legend(fontsize=7)
        for ax in axes.flat:
            ax.set_xlabel(f"Seconds within the {report['settings']['seconds_per_clip']:g} s target clip"); ax.grid(alpha=.2)
        fig.suptitle(f'Held-out diagnostics, seed {seed}. Vertical lines: beats (gray), downbeats (black).\n'
                     'Phase/rate references are annotation-derived, not observed continuous latent truth.')
        fig.savefig(OUT / f'heldout_seed_{seed}.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    render()
