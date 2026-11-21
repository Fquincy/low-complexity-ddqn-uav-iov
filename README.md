# Low-complexity DDQN for UAV-assisted IoV offloading

Research source, declared comparisons and audited development evidence toward
an efficient DDQN offloading controller. **The intended combined low-complexity
contribution remains unproven.** Negative and mixed results are retained.

Start here:
- [Current project status](LOW_COMPLEXITY_PROJECT_STATUS.json)
- [Latest saved-trace diagnosis](VALUE_GRADIENT_ISOLATION_SAVED_TRACE_DIAGNOSIS_2026-11-21.txt)
- [Latest completed comparison](VALUE_GRADIENT_ISOLATION_DEVELOPMENT_RESULTS_REVIEW_2026-11-21.txt)
- [Earlier family-level contribution decision](LOW_COMPLEXITY_CONSOLIDATED_CONTRIBUTION_DECISION_2026-11-19.txt)
- [Setup and portable packaging checks](SETUP.md)
- [Experiment scope](SCOPE.md)
- [What this publication includes](PUBLICATION_SCOPE.md)

The last completed 67,120-slot comparison passed its declared relative value
criterion and failed its policy criteria. An audit pass establishes evidence
integrity, not policy superiority or novelty. The next documented gate is a
prospective controller-design decision brief, not automatic retraining.

The active physical model uses vehicle-local, UAV and RSU processing and UAV
movement. Cloud and charging stations are excluded; UAV energy constraints
remain. Security/trust is deferred contribution two.

The parallel [lightweight experiment](https://github.com/Fquincy/lightweight-ddqn-uav-iov)
is an independent approach to the same first-contribution objective. Neither
repository establishes a new thesis contribution merely by its name.

`Source/chapter3_iov_env_v2.py` and its adjacent v1 helper are packaged locally.
Historical runners/guards are preserved byte-for-byte; some are host-bound and
need the original dependencies/raw data to rerun. No new research launch is
authorized by cloning the repository. The minimal packaging check is runnable
with NumPy and is explicitly not a controller-training experiment.

This is a research source and compact-evidence export, **not the entire raw
archive**. More than 100 GB of private local run/validation artifacts are indexed
in [the artifact catalog](provenance/ARTIFACT_CATALOG.json). Full traces and
checkpoint binaries are not hosted here yet. Included original manifests record
their hashes where available. Complete historical audit reproduction requires
those artifacts; no public download is promised by this initial upload.
