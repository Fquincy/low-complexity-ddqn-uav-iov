# Setup and validation

Use Python 3.11 or a compatible supported version in a fresh virtual environment.
Do not copy a virtual environment or use the owner's temporary C:/E: paths.

```text
python -m venv .venv
# Activate .venv using your operating system's command.
python -m pip install -r requirements-simulator.txt
python tools/check_environment.py
python tools/verify_export.py
```

The environment check executes a tiny engineering-only simulator fixture, with
no neural model, optimizer, teacher or reserved research seeds. It verifies
determinism, observation/action layout, finite rewards and excluded cloud/charging
mechanisms. It is not a research comparison or evidence of controller quality.

`requirements-simulator.txt` is a compatibility range for this minimal check,
not a recreation of all historical numerical environments. Record the resolved
versions and lock a new environment before future research. Controller references
also require PyTorch; install an appropriate official CPU/CUDA build for your
machine, then use `requirements-research-reference.txt` if necessary. The old
compiled/export paths additionally use packages such as Numba, llvmlite, ONNX,
ONNX Runtime, matplotlib and pandas. They are historical, host-bound paths:
do not treat their full suites or original audit reruns as portable until their
dependencies, backend and excluded raw artifacts have been restored and checked.

This publication does not implement or authorize a new lightweight training
runner. Follow the design, mechanics, protocol and explicit-launch gates.
The old research modules retain their original guards and source hashes.
