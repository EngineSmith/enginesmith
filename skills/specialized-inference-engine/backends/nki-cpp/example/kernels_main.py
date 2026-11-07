"""Run one kernel of kernels.py once with test inputs (NKI_ARTIFACTS_DIR, set by build_neff.py, keeps its NEFF)."""
import os
import sys

assert os.environ.get("NKI_ARTIFACTS_DIR"), "run through build_neff.py"
import ml_dtypes  # noqa: E402
import numpy as np  # noqa: E402

import kernels  # noqa: E402

LNC = int(os.environ.get("LNC", "2"))
bf16 = ml_dtypes.bfloat16
K = N = 4096
rng = np.random.default_rng(0)
x = rng.standard_normal((1, K)).astype(bf16)
wT = (rng.standard_normal((K, N)) * 0.02).astype(bf16)


def swizzle(w, nprog):
    kk, nn = w.shape
    return np.ascontiguousarray(w.reshape(kk // 128, 128, nprog, nn // nprog).transpose(2, 1, 0, 3)).reshape(kk, nn)


name = sys.argv[1]
if name == "tiny":
    a = np.arange(512, dtype=np.float32).reshape(128, 4)
    print("tiny ok:", np.array_equal(kernels.tiny_copy[LNC](a), a))
elif name == "gemv_swz4":
    y = kernels.gemv_swz4[LNC](x, swizzle(wT, LNC)).astype(np.float32)
    ref = x.astype(np.float32) @ wT.astype(np.float32)
    print("gemv_swz4 max abs err", np.abs(y - ref).max(), "ref max", np.abs(ref).max())
elif name == "rmsnorm_gemv":
    g = (1.0 + 0.1 * rng.standard_normal((1, K))).astype(bf16)
    xf = x.astype(np.float32)
    ref = (xf / np.sqrt((xf * xf).mean() + 1e-6) * g.astype(np.float32)) @ wT.astype(np.float32)
    y = kernels.rmsnorm_gemv[LNC](x, g, wT).astype(np.float32)
    print("rmsnorm_gemv max abs err", np.abs(y - ref).max(), "ref max", np.abs(ref).max())
elif name == "inplace":
    cache = np.zeros((8, 256), np.float32)
    row = np.arange(256, dtype=np.float32)[None, :]
    print("inplace out ok:", np.array_equal(kernels.inplace_write[LNC](cache, row), 2 * row))
else:
    raise SystemExit(f"unknown kernel {name}")
