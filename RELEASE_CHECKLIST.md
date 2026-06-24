# Public Release Checklist

Use this before making the GitHub repository public.

## Paper Metadata

- [ ] Replace placeholder authors in `CITATION.cff`.
- [ ] Add paper DOI, arXiv URL, or accepted manuscript URL.
- [ ] Add final BibTeX citation to `README.md`.
- [ ] Add final venue and publication year.

## Code

- [ ] Migrate cleaned model code into `src/coduar/models/`.
- [ ] Migrate cleaned dataset code into `src/coduar/data/`.
- [ ] Migrate training and evaluation code into package modules.
- [ ] Migrate Assembly Copilot code into `src/assembly_copilot/`.
- [ ] Remove private absolute paths from all scripts and configs.
- [ ] Replace placeholder command wrappers with real package entry points.
- [ ] Add import, config, and tiny inference smoke tests.

## Data and Models

- [ ] Fill final class counts in `configs/train_coduar.example.yaml`.
- [ ] Add final label maps under `data/label_maps/`.
- [ ] Document dataset access and license in `docs/dataset.md`.
- [ ] Confirm README/configs clearly state that checkpoints are not released.
- [ ] Confirm `.gitignore` excludes local checkpoint files.

## Demo

- [ ] Decide where to host `demo_720.mp4`.
- [ ] Add a public demo link to `README.md`.
- [ ] Confirm no private footage or sensitive data is included.
- [ ] Connect `scripts/run_assembly_copilot_demo.sh` to the cleaned demo code.

## Results

- [ ] Fill final metrics in `docs/reproducing_results.md`.
- [ ] Add exact hardware, seed, split, and training schedule.
- [ ] Add scripts for reproducing paper tables or figures where possible.
