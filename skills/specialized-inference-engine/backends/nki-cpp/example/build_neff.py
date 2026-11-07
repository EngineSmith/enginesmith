"""Build time only: compile NKI kernels to standalone NEFFs through NKI's documented route.

NKI_ARTIFACTS_DIR is set before nki is imported; calling a @nki.jit kernel with numpy arrays then compiles it, runs it
once on a NeuronCore (NEURON_RT_VISIBLE_CORES must name a free one) and leaves kernel.neff in that directory. Each
kernel is built in its own process so that each gets its own artifacts directory.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def build_one(name: str, outdir: str = "neff") -> str:
    """Build kernel `name` (a function of kernels_main.py) in a fresh process; returns the NEFF path."""
    art = os.path.abspath(f"build/{name}")
    shutil.rmtree(art, ignore_errors=True)
    os.makedirs(art)
    env = {**os.environ, "NKI_ARTIFACTS_DIR": art, "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2"}
    subprocess.run([sys.executable, os.path.join(HERE, "kernels_main.py"), name], check=True, env=env)
    os.makedirs(outdir, exist_ok=True)
    dst = f"{outdir}/{name}.neff"
    shutil.copy(os.path.join(art, "kernel.neff"), dst)
    print("wrote", dst)
    return dst


if __name__ == "__main__":
    for name in sys.argv[1:] or ["tiny", "gemv_swz4", "rmsnorm_gemv", "inplace"]:
        build_one(name)
