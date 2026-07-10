# Dataset Acquisition Pipeline (2026)

This repository now includes a reproducible acquisition entrypoint:

- scripts/datasets/acquire.sh
- scripts/datasets/the_session_dump.py

## Quick start

Dry-run only (safe):

```bash
./scripts/datasets/acquire.sh --datasets fma,maestro,groove --mode metadata
```

Run actual downloads:

```bash
./scripts/datasets/acquire.sh --datasets fma,maestro,groove --mode metadata --execute
```

Full mode (very large):

```bash
./scripts/datasets/acquire.sh --datasets all --mode full --execute
```

Default storage root:

- /disk1/jaehoon/dataset_store

In this workspace, dataset_store is a symlink to that location.

## Automated vs manual-intervention status

Fully automatable in this pipeline:

- fma
- mtg_jamendo
- maestro
- groove
- lmd
- pianocore (requires huggingface-cli in full mode)
- periscope (requires huggingface-cli in full mode)
- symbtr
- the_session (JSON API harvesting)

Likely manual intervention needed:

- sources requiring login or click-through acceptance pages
- anti-bot-protected portals (for example CHARM pages)
- datasets requiring explicit institution-level permission

Partial / hybrid:

- GiantMIDI-Piano: automation possible, but source availability and disclaimer flow can require human handling.
- AudioSet-style and web-index datasets: metadata handling is automated, but large-scale media retrieval needs legal/policy confirmation.
- WJazzD and Mazurka-related resources: ingestion is straightforward once files are obtained, but obtaining files may require manual access.

## Notes

- Default mode is metadata to avoid accidental multi-terabyte transfers.
- Commands are resumable where curl supports HTTP range requests.
- Always validate license and terms before redistribution.
