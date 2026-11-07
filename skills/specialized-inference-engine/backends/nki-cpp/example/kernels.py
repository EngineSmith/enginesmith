import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def tiny_copy(a):
    """Tiny kernel for measuring fixed nrt_execute overhead: out = a (128x4 fp32)."""
    t = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def gemv(x, wT):
    """y[1, N] = x[1, K] @ wT[K, N]   (i.e. y = W x with W [N, K], stored transposed).

    x is used as the stationary operand (M=1), wT tiles [128, 512] stream as moving.
    Under LNC=2 the N dimension is split between the two physical cores.
    """
    _, K = x.shape
    K_, N = wT.shape
    assert K == K_
    TK = 128
    TN = 512
    nprog = nl.num_programs()
    NPER = N // nprog
    assert NPER % TN == 0 and K % TK == 0
    pid = nl.program_id(0) if nprog > 1 else 0
    n0 = pid * NPER

    y = nl.ndarray((1, N), dtype=x.dtype, buffer=nl.shared_hbm)

    # x as [K/128 tiles] of [128, 1] in SBUF: xT[p, kt] = x[0, kt*128 + p]
    KT = K // TK
    xT = nl.ndarray((TK, KT), dtype=x.dtype, buffer=nl.sbuf)
    for kt in range(KT):
        nisa.dma_copy(dst=xT[0:TK, kt:kt + 1], src=x.reshape((K, 1))[kt * TK:(kt + 1) * TK, 0:1])

    for nt in range(NPER // TN):
        acc = nl.ndarray((1, TN), dtype=nl.float32, buffer=nl.psum)
        for kt in range(KT):
            wt = nl.ndarray((TK, TN), dtype=wT.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wt, src=wT[kt * TK:(kt + 1) * TK, n0 + nt * TN:n0 + (nt + 1) * TN])
            nisa.nc_matmul(dst=acc, stationary=xT[0:TK, kt:kt + 1], moving=wt)
        ysb = nl.ndarray((1, TN), dtype=y.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ysb, src=acc)
        nisa.dma_copy(dst=y[0:1, n0 + nt * TN:n0 + (nt + 1) * TN], src=ysb)
    return y


def _load_x_cols(x, K, TK):
    """x [1, K] (HBM) -> SBUF [128, K/128] with xT[p, kt] = x[kt*128 + p]; one DMA."""
    KT = K // TK
    xT = nl.ndarray((TK, KT), dtype=x.dtype, buffer=nl.sbuf)
    for kt in range(KT):
        nisa.dma_copy(dst=xT[0:TK, kt:kt + 1], src=x.reshape((K, 1))[kt * TK:(kt + 1) * TK, 0:1])
    return xT


@nki.jit
def gemv_mov(x, wT):
    """Same math as gemv, but weights streamed in big [128, NPER] slabs (one DMA per k-tile,
    4 KB contiguous per partition under LNC2), x stationary, 512-wide moving matmuls."""
    _, K = x.shape
    _, N = wT.shape
    TK, TN = 128, 512
    nprog = nl.num_programs()
    NPER = N // nprog
    pid = nl.program_id(0) if nprog > 1 else 0
    n0 = pid * NPER
    KT, NT = K // TK, NPER // TN
    y = nl.ndarray((1, N), dtype=x.dtype, buffer=nl.shared_hbm)
    xT = _load_x_cols(x, K, TK)
    accs = []
    for nt in range(NT):
        accs.append(nl.ndarray((1, TN), dtype=nl.float32, buffer=nl.psum))
    for kt in range(KT):
        wt = nl.ndarray((TK, NPER), dtype=wT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=wT[kt * TK:(kt + 1) * TK, n0:n0 + NPER])
        for nt in range(NT):
            nisa.nc_matmul(dst=accs[nt], stationary=xT[0:TK, kt:kt + 1], moving=wt[0:TK, nt * TN:(nt + 1) * TN])
    ysb = nl.ndarray((1, NPER), dtype=y.dtype, buffer=nl.sbuf)
    for nt in range(NT):
        nisa.tensor_copy(dst=ysb[0:1, nt * TN:(nt + 1) * TN], src=accs[nt])
    nisa.dma_copy(dst=y[0:1, n0:n0 + NPER], src=ysb)
    return y


@nki.jit
def gemv_stat(x, wT):
    """Weights as the stationary operand ([128 k, 128 n] tiles), x as a 1-column moving operand.
    Output lands as [128 (n within tile), NT] in PSUM."""
    _, K = x.shape
    _, N = wT.shape
    TK, TNs = 128, 128
    nprog = nl.num_programs()
    NPER = N // nprog
    pid = nl.program_id(0) if nprog > 1 else 0
    n0 = pid * NPER
    KT, NT = K // TK, NPER // TNs
    y = nl.ndarray((1, N), dtype=x.dtype, buffer=nl.shared_hbm)
    xT = _load_x_cols(x, K, TK)
    acc = nl.ndarray((TNs, NT), dtype=nl.float32, buffer=nl.psum)
    for kt in range(KT):
        wt = nl.ndarray((TK, NPER), dtype=wT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=wT[kt * TK:(kt + 1) * TK, n0:n0 + NPER])
        for nt in range(NT):
            nisa.nc_matmul(dst=acc[0:TNs, nt:nt + 1], stationary=wt[0:TK, nt * TNs:(nt + 1) * TNs], moving=xT[0:TK, kt:kt + 1])
    ysb = nl.ndarray((TNs, NT), dtype=y.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ysb, src=acc)
    for nt in range(NT):
        nisa.dma_copy(dst=y.reshape((N, 1))[n0 + nt * TNs:n0 + (nt + 1) * TNs, 0:1], src=ysb[0:TNs, nt:nt + 1])
    return y


def _gemv_slab(x, wT, TNL, alt_engines, resident):
    """x stationary; weights streamed as [128, TNL] slabs. alt_engines: alternate HWDGE issue engine
    (sync/scalar) across slabs; resident: one big SBUF buffer for the core's whole weight slice."""
    _, K = x.shape
    _, N = wT.shape
    TK, TN = 128, 512
    nprog = nl.num_programs()
    NPER = N // nprog
    pid = nl.program_id(0) if nprog > 1 else 0
    n0 = pid * NPER
    KT, NL, NT = K // TK, NPER // TNL, TNL // TN
    y = nl.ndarray((1, N), dtype=x.dtype, buffer=nl.shared_hbm)
    xT = _load_x_cols(x, K, TK)
    accs = []
    for j in range(NL * NT):
        accs.append(nl.ndarray((1, TN), dtype=nl.float32, buffer=nl.psum))
    if resident:
        wall = nl.ndarray((TK, KT * NPER), dtype=wT.dtype, buffer=nl.sbuf)
    i = 0
    for kt in range(KT):
        for l in range(NL):
            if resident:
                wt = wall[0:TK, (kt * NL + l) * TNL:(kt * NL + l + 1) * TNL]
            else:
                wt = nl.ndarray((TK, TNL), dtype=wT.dtype, buffer=nl.sbuf)
            src = wT[kt * TK:(kt + 1) * TK, n0 + l * TNL:n0 + (l + 1) * TNL]
            if alt_engines:
                eng = nisa.engine.sync if i % 2 == 0 else nisa.engine.scalar
                nisa.dma_copy(dst=wt, src=src, dge_mode=nisa.dge_mode.hwdge, engine=eng)
            else:
                nisa.dma_copy(dst=wt, src=src)
            i += 1
            for t in range(NT):
                nisa.nc_matmul(dst=accs[l * NT + t], stationary=xT[0:TK, kt:kt + 1], moving=wt[0:TK, t * TN:(t + 1) * TN])
    ysb = nl.ndarray((1, NPER), dtype=y.dtype, buffer=nl.sbuf)
    for j in range(NL * NT):
        nisa.tensor_copy(dst=ysb[0:1, j * TN:(j + 1) * TN], src=accs[j])
    nisa.dma_copy(dst=y[0:1, n0:n0 + NPER], src=ysb)
    return y


@nki.jit
def gemv_t1024(x, wT):
    return _gemv_slab(x, wT, 1024, False, False)


@nki.jit
def gemv_t512(x, wT):
    return _gemv_slab(x, wT, 512, False, False)


@nki.jit
def gemv_t1024_hw(x, wT):
    return _gemv_slab(x, wT, 1024, True, False)


@nki.jit
def gemv_res(x, wT):
    return _gemv_slab(x, wT, 1024, False, True)


@nki.jit
def gemv_res_hw(x, wT):
    return _gemv_slab(x, wT, 1024, True, True)

def _probe(x, wT, do_dma, do_mm, TNL):
    _, K = x.shape
    _, N = wT.shape
    TK, TN = 128, 512
    nprog = nl.num_programs()
    NPER = N // nprog
    pid = nl.program_id(0) if nprog > 1 else 0
    n0 = pid * NPER
    KT, NL, NT = K // TK, NPER // TNL, TNL // TN
    y = nl.ndarray((1, N), dtype=x.dtype, buffer=nl.shared_hbm)
    xT = _load_x_cols(x, K, TK)
    accs = []
    for j in range(NL * NT):
        accs.append(nl.ndarray((1, TN), dtype=nl.float32, buffer=nl.psum))
    wall = nl.ndarray((TK, KT * NPER), dtype=wT.dtype, buffer=nl.sbuf)
    if not do_dma:
        nisa.memset(dst=wall, value=0.0)
    for kt in range(KT):
        for l in range(NL):
            wt = wall[0:TK, (kt * NL + l) * TNL:(kt * NL + l + 1) * TNL]
            if do_dma:
                nisa.dma_copy(dst=wt, src=wT[kt * TK:(kt + 1) * TK, n0 + l * TNL:n0 + (l + 1) * TNL])
            if do_mm:
                for t in range(NT):
                    nisa.nc_matmul(dst=accs[l * NT + t], stationary=xT[0:TK, kt:kt + 1], moving=wt[0:TK, t * TN:(t + 1) * TN])
    ysb = nl.ndarray((1, NPER), dtype=y.dtype, buffer=nl.sbuf)
    if do_mm:
        for j in range(NL * NT):
            nisa.tensor_copy(dst=ysb[0:1, j * TN:(j + 1) * TN], src=accs[j])
    else:
        nisa.tensor_copy(dst=ysb[0:1, 0:NPER], src=wall[0:1, 0:NPER])
    nisa.dma_copy(dst=y[0:1, n0:n0 + NPER], src=ysb)
    return y


@nki.jit
def probe_dma(x, wT):
    return _probe(x, wT, True, False, 2048)


@nki.jit
def probe_mm(x, wT):
    return _probe(x, wT, False, True, 2048)


@nki.jit
def gemv_t2048(x, wT):
    return _probe(x, wT, True, True, 2048)


@nki.jit
def rmsnorm_gemv(x, g, wT):
    """y[1, N] = (rmsnorm(x) * g) @ wT, eps = 1e-6.
    Computed as rstd(x) * ((x * g) @ wT): the scalar rstd is applied while evacuating PSUM."""
    _, K = x.shape
    _, N = wT.shape
    TK, TN, TNL = 128, 512, 2048
    nprog = nl.num_programs()
    NPER = N // nprog
    pid = nl.program_id(0) if nprog > 1 else 0
    n0 = pid * NPER
    KT, NL, NT = K // TK, NPER // TNL, TNL // TN
    y = nl.ndarray((1, N), dtype=x.dtype, buffer=nl.shared_hbm)

    # sum of squares on one partition (ScalarE square + free-dim add-reduce), then rstd
    xr = nl.ndarray((1, K), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xr, src=x)
    sq = nl.ndarray((1, K), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation_reduce(dst=sq, op=nl.square, data=xr, reduce_op=nl.add, reduce_res=ss)
    epsb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epsb, value=1e-6)
    rstd = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, bias=epsb, scale=1.0 / K)

    # x*g in the [128, K/128] stationary layout
    xT = _load_x_cols(x, K, TK)
    gT = _load_x_cols(g, K, TK)
    xgT = nl.ndarray((TK, KT), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=xgT, data1=xT, data2=gT, op=nl.multiply)

    accs = []
    for j in range(NL * NT):
        accs.append(nl.ndarray((1, TN), dtype=nl.float32, buffer=nl.psum))
    wall = nl.ndarray((TK, KT * NPER), dtype=wT.dtype, buffer=nl.sbuf)
    for kt in range(KT):
        for l in range(NL):
            wt = wall[0:TK, (kt * NL + l) * TNL:(kt * NL + l + 1) * TNL]
            nisa.dma_copy(dst=wt, src=wT[kt * TK:(kt + 1) * TK, n0 + l * TNL:n0 + (l + 1) * TNL])
            for t in range(NT):
                nisa.nc_matmul(dst=accs[l * NT + t], stationary=xgT[0:TK, kt:kt + 1], moving=wt[0:TK, t * TN:(t + 1) * TN])
    ysb = nl.ndarray((1, NPER), dtype=y.dtype, buffer=nl.sbuf)
    for j in range(NL * NT):
        nisa.activation(dst=ysb[0:1, j * TN:(j + 1) * TN], op=nl.copy, data=accs[j], scale=rstd)
    nisa.dma_copy(dst=y[0:1, n0:n0 + NPER], src=ysb)
    return y


@nki.jit
def inplace_write(cache, row):
    """Writes row [1, D] into cache[3, :] IN PLACE (cache is an input); returns row * 2 (tiny output)."""
    t = nl.ndarray(row.shape, dtype=row.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=row)
    nisa.dma_copy(dst=cache[3:4, 0:row.shape[1]], src=t)
    t2 = nl.ndarray(row.shape, dtype=row.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=t2, data=t, op0=nl.multiply, operand0=2.0)
    out = nl.ndarray(row.shape, dtype=row.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=t2)
    return out


def _gemv_ct(x, wT, CT, TNL):
    """gemv with TensorE column tiling: each M=1 matmul occupies one of 128/CT column tiles so
    up to 128/CT matmuls (with different moving tiles) can be in flight concurrently."""
    _, K = x.shape
    _, N = wT.shape
    TK, TN = 128, 512
    nprog = nl.num_programs()
    NPER = N // nprog
    pid = nl.program_id(0) if nprog > 1 else 0
    n0 = pid * NPER
    KT, NL, NT = K // TK, NPER // TNL, TNL // TN
    NCT = 128 // CT
    y = nl.ndarray((1, N), dtype=x.dtype, buffer=nl.shared_hbm)
    xT = _load_x_cols(x, K, TK)
    accs = []
    for j in range(NL * NT):
        accs.append(nl.ndarray((1, TN), dtype=nl.float32, buffer=nl.psum))
    wall = nl.ndarray((TK, KT * NPER), dtype=wT.dtype, buffer=nl.sbuf)
    for kt in range(KT):
        for l in range(NL):
            wt = wall[0:TK, (kt * NL + l) * TNL:(kt * NL + l + 1) * TNL]
            nisa.dma_copy(dst=wt, src=wT[kt * TK:(kt + 1) * TK, n0 + l * TNL:n0 + (l + 1) * TNL])
            for t in range(NT):
                j = l * NT + t
                nisa.nc_matmul(dst=accs[j], stationary=xT[0:TK, kt:kt + 1], moving=wt[0:TK, t * TN:(t + 1) * TN],
                               tile_position=(0, (j % NCT) * CT), tile_size=(128, CT))
    ysb = nl.ndarray((1, NPER), dtype=y.dtype, buffer=nl.sbuf)
    for j in range(NL * NT):
        nisa.tensor_copy(dst=ysb[0:1, j * TN:(j + 1) * TN], src=accs[j])
    nisa.dma_copy(dst=y[0:1, n0:n0 + NPER], src=ysb)
    return y


@nki.jit
def gemv_ct32(x, wT):
    return _gemv_ct(x, wT, 32, 2048)


@nki.jit
def gemv_ct64(x, wT):
    return _gemv_ct(x, wT, 64, 2048)


def _load_x_cols1(x, K, TK):
    """x [1, K] -> SBUF [128, K/128], xT[p, kt] = x[kt*128 + p], as ONE strided DMA."""
    KT = K // TK
    xT = nl.ndarray((TK, KT), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xT, src=x.ap(pattern=[[1, TK], [TK, KT]], offset=0))
    return xT


def _gemv_dge(x, wT, TNL, mode, one_dma_x):
    _, K = x.shape
    _, N = wT.shape
    TK, TN = 128, 512
    nprog = nl.num_programs()
    NPER = N // nprog
    pid = nl.program_id(0) if nprog > 1 else 0
    n0 = pid * NPER
    KT, NL, NT = K // TK, NPER // TNL, TNL // TN
    y = nl.ndarray((1, N), dtype=x.dtype, buffer=nl.shared_hbm)
    if one_dma_x:
        xT = _load_x_cols1(x, K, TK)
    else:
        xT = _load_x_cols(x, K, TK)
    accs = []
    for j in range(NL * NT):
        accs.append(nl.ndarray((1, TN), dtype=nl.float32, buffer=nl.psum))
    wall = nl.ndarray((TK, KT * NPER), dtype=wT.dtype, buffer=nl.sbuf)
    for kt in range(KT):
        for l in range(NL):
            wt = wall[0:TK, (kt * NL + l) * TNL:(kt * NL + l + 1) * TNL]
            src = wT[kt * TK:(kt + 1) * TK, n0 + l * TNL:n0 + (l + 1) * TNL]
            if mode == 0:
                nisa.dma_copy(dst=wt, src=src, dge_mode=nisa.dge_mode.none)
            elif mode == 1:
                nisa.dma_copy(dst=wt, src=src, dge_mode=nisa.dge_mode.hwdge)
            else:
                nisa.dma_copy(dst=wt, src=src)
            for t in range(NT):
                nisa.nc_matmul(dst=accs[l * NT + t], stationary=xT[0:TK, kt:kt + 1], moving=wt[0:TK, t * TN:(t + 1) * TN])
    ysb = nl.ndarray((1, NPER), dtype=y.dtype, buffer=nl.sbuf)
    for j in range(NL * NT):
        nisa.tensor_copy(dst=ysb[0:1, j * TN:(j + 1) * TN], src=accs[j])
    nisa.dma_copy(dst=y[0:1, n0:n0 + NPER], src=ysb)
    return y


@nki.jit
def gemv_nodge(x, wT):
    return _gemv_dge(x, wT, 2048, 0, False)


@nki.jit
def gemv_hwdge(x, wT):
    return _gemv_dge(x, wT, 2048, 1, False)


@nki.jit
def gemv_nodge_x1(x, wT):
    return _gemv_dge(x, wT, 2048, 0, True)


@nki.jit
def gemv_x1(x, wT):
    return _gemv_dge(x, wT, 2048, 2, True)


def _gemv_swz(x, wS, CH):
    """wS is the weight pre-swizzled at build time so each core / partition reads one contiguous run:
    wS (flat) = [nprog, 128 (p), KT, NPER] with element (c, p, kt, n) = wT[kt*128 + p, c*NPER + n]."""
    _, K = x.shape
    _, N = wS.shape
    TK, TN = 128, 512
    nprog = nl.num_programs()
    NPER = N // nprog
    pid = nl.program_id(0) if nprog > 1 else 0
    n0 = pid * NPER
    KT, NT = K // TK, NPER // TN
    y = nl.ndarray((1, N), dtype=x.dtype, buffer=nl.shared_hbm)
    xT = _load_x_cols1(x, K, TK)
    accs = []
    for j in range(NT):
        accs.append(nl.ndarray((1, TN), dtype=nl.float32, buffer=nl.psum))
    wall = nl.ndarray((TK, KT * NPER), dtype=wS.dtype, buffer=nl.sbuf)
    base = pid * (TK * KT * NPER)
    for c in range(KT // CH):
        dst = wall[0:TK, c * CH * NPER:(c + 1) * CH * NPER]
        nisa.dma_copy(dst=dst, src=wS.ap(pattern=[[KT * NPER, TK], [1, CH * NPER]], offset=base + c * CH * NPER))
        for kk in range(CH):
            kt = c * CH + kk
            for t in range(NT):
                nisa.nc_matmul(dst=accs[t], stationary=xT[0:TK, kt:kt + 1],
                               moving=wall[0:TK, kt * NPER + t * TN:kt * NPER + (t + 1) * TN])
    ysb = nl.ndarray((1, NPER), dtype=y.dtype, buffer=nl.sbuf)
    for j in range(NT):
        nisa.tensor_copy(dst=ysb[0:1, j * TN:(j + 1) * TN], src=accs[j])
    nisa.dma_copy(dst=y[0:1, n0:n0 + NPER], src=ysb)
    return y


@nki.jit
def gemv_swz1(x, wS):
    return _gemv_swz(x, wS, 1)


@nki.jit
def gemv_swz4(x, wS):
    return _gemv_swz(x, wS, 4)
