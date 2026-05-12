# Data Layout

The repository does not include raw datasets.

Use the following structure locally:

```text
data/
├── raw/          # downloaded UAV datasets or private videos
├── processed/    # extracted trajectories and intermediate features
└── results/      # generated output videos and metrics
```

`data/raw`, `data/processed`, and `data/results` are ignored by Git to avoid committing private or large experimental files.

The publication demo video is stored separately under:

```text
assets/results/topoekf_demo.mp4
```
