# GitHub Release Plan

Use this sequence when publishing the clean project to a new private repository.

## Recommended Commit Design

1. `chore: initialize TopoEKF research repository`
   - Add package metadata, requirements, gitignore, base directory structure.

2. `feat: add detection and tracking core`
   - Add YOLO detector, detection result model, EKF core, track lifecycle, and data association modules.

3. `feat: add topology-aware anomaly pipeline`
   - Add persistent homology, persistence image features, topology feedback, and Isolation Forest anomaly detector.

4. `feat: add video runner and visualization`
   - Add `scripts/run_tracking.py`, status-aware bounding boxes, trajectories, and annotated output generation.

5. `test: add unit and integration coverage`
   - Add EKF, association, topology, anomaly, visualizer, and pipeline tests.

6. `docs: add publication-ready README and methodology`
   - Add README, methodology notes, citation draft, and demo assets.

## Before Public Release

- Replace placeholder author profile links.
- Add final article metadata to `CITATION.cff`.
- Decide on a license and add `LICENSE`.
- Review `assets/results/topoekf_demo.mp4` for publication suitability.
- Confirm no private dataset, raw unpublished material, or institutional-only content is included.

## Suggested Repository Settings

- Start as a private repository.
- Enable GitHub Actions after the first push.
- Protect the default branch before public release.
- Use releases for tagged versions such as `v0.1.0-private-preview` and `v1.0.0-paper-release`.
