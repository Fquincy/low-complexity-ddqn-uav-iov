LOW COMPLEXITY PROJECT
Created: 2026-11-01

Purpose
Develop and test a genuinely lower-cost action-value controller for the
unchanged UAV-IoV edge-computing simulator. The target remains local vehicles,
two UAVs and two RSUs. Cloud, charging stations and security/trust extensions
are excluded from this project.

The earlier Chapter 3 files, failed pilots and action-sensitivity evidence are
preserved under October 31. This folder is a separate development branch. A
successful result is not assumed.

Required claim gate
A candidate must show a predeclared reduction in a chosen complete-slot cost
metric and return/deadline performance no worse than the frozen additive DDQN
and the declared heuristics on fresh workloads. Parameter count alone is not
enough. Every failed candidate remains in the record.

First work item
Freeze the architecture and run a small mechanics/cost gate. Do not start a
long training run before the candidate completes this gate.

Recommended candidate
Single-pass interaction-aware DDQN with shared trunk, parallel component
heads and a bounded top-k joint reranker:

- one physical encoder pass per physical state;
- 32 parallel action heads for the 30 destinations and two UAV movements;
- top-k candidates per component, with k=2 initially;
- a compact joint scorer using the encoded state, pooled action embeddings and
  a low-rank interaction term;
- bounded beam width B=8 for complete-slot decoding;
- no 32-step environment interaction and no 32 separate Python stage updates.

This design is intended to preserve some joint-action information while
controlling inference cost. It is a proposal, not a proven solution. The
reranker may still be too expensive or may fail to learn useful interactions.

Frozen comparisons

1. Existing additive DDQN baseline.
2. Existing conditional sequential model as a diagnostic cost reference.
3. New single-pass interaction-aware candidate.
4. all-local, nearest-RSU and UAV-selective heuristics.

The simulator, reward equations, task generation, seeds, deadline definition,
evaluation metrics and physical action layout stay fixed. Only the candidate
learner changes.

Gate sequence

1. Implement the candidate in new files only.
2. Unit-test action mapping, top-k/beam decoding, joint-score dimensions,
   terminal handling and gradient flow.
3. Benchmark complete-slot selection on common observations, including context,
   candidate generation, reranking and decoding. Exclude environment stepping
   and file I/O. Record parameter bytes, replay bytes and wall time.
4. Run a small paired training smoke test with new seeds.
5. Review smoke results before a bounded multi-seed development pilot.
6. Use fresh workloads for the final development comparison; no favorable
   checkpoint selection.

Stopping rules

Stop if the candidate is slower without a measured quality benefit, if it does
not represent the intended interaction, or if the quality margin is not met.
Do not tune reward weights to create an apparent gain. Do not claim a thesis
contribution from the architecture until the full comparison passes.

Files copied into Source are working references only. They are not edited in
this branch until a new design file and tests are created.
