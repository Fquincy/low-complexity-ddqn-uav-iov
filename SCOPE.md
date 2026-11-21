# Experiment scope

The active physical problem contains vehicle-local processing, UAV1, UAV2, RSU1
and RSU2. Both UAVs also have discrete movement choices. Cloud servers, charging
stations, cloud routes, recharge actions/bonuses and charger features are
excluded. UAV initial battery, propulsion/computation consumption and energy
feasibility remain in scope. A reset initial condition is not an in-episode
charging transition. Security/trust is a separate deferred contribution.

The common simulator copies are byte-identical to the source snapshot identified
in `provenance/COMMON_ENVIRONMENT_MANIFEST.json`. The v2 simulator imports helpers
from its adjacent v1 module; v1 is not the selected active timeline. Provisional
physical assumptions and compressed partial observations remain documented
limitations. A copied model is not a real-world deployment validation.

Legacy scripts under `reference/historical_legacy/`, where present, use an older
cloud/charging model. They are quarantined historical examples, not active
training commands. Port the learner into new files against the corrected
simulator before claiming it matches this scope. Existing incompatible weights
cannot simply be resumed after changing state/action definitions.
