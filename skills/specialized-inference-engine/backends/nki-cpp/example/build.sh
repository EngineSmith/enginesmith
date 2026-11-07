#!/usr/bin/env bash
# Build-time only (Python/NKI/neuronx-cc). Produces neff/*.neff, then builds the native runner.
set -euo pipefail
cd "$(dirname "$0")"
V=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0
export PATH=$V/bin:$PATH                 # nki shells out to neuronx-cc
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export LNC=2                             # host runs logical-neuroncore-config 2
# NKI's documented route compiles each kernel by running it once: point NEURON_RT_VISIBLE_CORES at a free core.
export NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-$(( ${ENGINESMITH_NEURON_CORE:-0} + 2 ))}
python build_neff.py "$@"
make nrt_gemv nrt_inplace
