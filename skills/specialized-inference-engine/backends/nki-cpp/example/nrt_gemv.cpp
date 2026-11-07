// Native NEFF runner: links only libnrt + libc/libstdc++.
// Usage: nrt_gemv <gemv.neff> <tiny.neff> [iters]
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_async.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

#define CK(x) do { NRT_STATUS s_ = (x); if (s_ != NRT_SUCCESS) { \
  fprintf(stderr, "%s:%d %s -> %d (%s)\n", __FILE__, __LINE__, #x, (int)s_, nrt_get_status_as_str(s_)); exit(1);} } while (0)

using clk = std::chrono::steady_clock;
static double us_since(clk::time_point t0) { return std::chrono::duration<double, std::micro>(clk::now() - t0).count(); }

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

static std::vector<char> slurp(const char* p) {
  std::ifstream f(p, std::ios::binary); if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
  return std::vector<char>((std::istreambuf_iterator<char>(f)), {});
}

struct Model {
  nrt_model_t* m = nullptr;
  std::vector<nrt_tensor_info_t> ins, outs;
  nrt_tensor_info_array_t* info = nullptr;
};

static Model load(const char* path, int vnc) {
  Model M; auto bytes = slurp(path);
  auto t0 = clk::now();
  CK(nrt_load(bytes.data(), bytes.size(), vnc, 1, &M.m));
  printf("[%s] nrt_load %.1f ms\n", path, us_since(t0) / 1e3);
  CK(nrt_get_model_tensor_info(M.m, &M.info));
  for (uint64_t i = 0; i < M.info->tensor_count; i++) {
    auto& t = M.info->tensor_array[i];
    printf("   %-6s name=%-12s bytes=%-9zu dtype=%d shape=[", t.usage == NRT_TENSOR_USAGE_INPUT ? "input" : "output", t.name, t.size, (int)t.dtype);
    for (uint32_t d = 0; d < t.ndim; d++) printf("%s%u", d ? "," : "", t.shape[d]);
    printf("]\n");
    (t.usage == NRT_TENSOR_USAGE_INPUT ? M.ins : M.outs).push_back(t);
  }
  return M;
}

static nrt_tensor_t* dev_alloc(int vnc, size_t bytes, const char* name) {
  nrt_tensor_t* t; CK(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, vnc, bytes, name, &t)); return t;
}

static void stats(const char* label, std::vector<double>& v, double bytes = 0) {
  std::sort(v.begin(), v.end());
  double mean = 0; for (double x : v) mean += x; mean /= v.size();
  printf("%-44s n=%zu mean=%.2f us  p50=%.2f  p10=%.2f  p90=%.2f  min=%.2f", label, v.size(), mean, v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10], v[0]);
  if (bytes) printf("  -> %.1f GB/s (p50)", bytes / (v[v.size() / 2] * 1e3));
  printf("\n");
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s gemv.neff tiny.neff [iters]\n", argv[0]); return 2; }
  int iters = argc > 3 ? atoi(argv[3]) : 1000;
  const int vnc = 0;  // index within NEURON_RT_VISIBLE_CORES
  CK(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""));
  uint32_t nvis; CK(nrt_get_visible_vnc_count(&nvis)); printf("visible logical cores: %u\n", nvis);

  // ------------------------------------------------------------ GEMV
  Model G = load(argv[1], vnc);
  // NKI arg names become NEFF input names ("x", "wT"); return value -> "output_0"
  const nrt_tensor_info_t *xi = nullptr, *wi = nullptr, *gi = nullptr, *yi = &G.outs.at(0);
  for (auto& t : G.ins) { if (!strcmp(t.name, "x")) xi = &t; if (!strcmp(t.name, "wT") || !strcmp(t.name, "wS")) wi = &t; if (!strcmp(t.name, "g")) gi = &t; }
  if (!xi || !wi) { fprintf(stderr, "unexpected input names\n"); return 1; }
  const int K = xi->shape[xi->ndim - 1], N = wi->shape[1];
  printf("%s K=%d N=%d\n", gi ? "RMSNORM+GEMV" : "GEMV", K, N);

  std::mt19937 rng(123); std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<uint16_t> hx(K), hw((size_t)K * N), hy(N);
  for (auto& v : hx) v = f2bf(nd(rng));
  for (auto& v : hw) v = f2bf(nd(rng) * 0.02f);
  std::vector<uint16_t> hg(K); for (auto& v : hg) v = f2bf(1.f + 0.1f * nd(rng));
  std::vector<float> wf(hw.size()); for (size_t i = 0; i < hw.size(); i++) wf[i] = bf2f(hw[i]);

  auto cpu_gemv = [&](const std::vector<uint16_t>& x, std::vector<float>& y) {  // y[n] = sum_k x[k] * wT[k][n]
    y.assign(N, 0.f);
    float scale = 1.f;
    if (gi) { double ss = 0; for (int k = 0; k < K; k++) ss += (double)bf2f(x[k]) * bf2f(x[k]); scale = 1.f / sqrtf((float)(ss / K) + 1e-6f); }
    for (int k = 0; k < K; k++) { float xv = bf2f(x[k]) * (gi ? bf2f(hg[k]) : 1.f); const float* row = &wf[(size_t)k * N]; for (int n = 0; n < N; n++) y[n] += xv * row[n]; }
    for (auto& v : y) v *= scale;
  };

  // Allocate device tensors ONCE. The weight is uploaded once and stays resident.
  nrt_tensor_t* tw = dev_alloc(vnc, wi->size, "wT");
  nrt_tensor_t* tx = dev_alloc(vnc, xi->size, "x");
  nrt_tensor_t* ty = dev_alloc(vnc, yi->size, "y");
  auto t0 = clk::now();
  if (getenv("SWIZZLE")) {  // layout expected by gemv_swz*: (c, p, kt, n) <- wT[kt*128+p][c*NPER+n]
    const int P = 128, KT = K / P, C = atoi(getenv("SWIZZLE")), NPER = N / C;
    std::vector<uint16_t> s(hw.size()); size_t o = 0;
    for (int c = 0; c < C; c++) for (int p = 0; p < P; p++) for (int kt = 0; kt < KT; kt++) for (int n = 0; n < NPER; n++)
      s[o++] = hw[(size_t)(kt * P + p) * N + c * NPER + n];
    t0 = clk::now();
    CK(nrt_tensor_write(tw, s.data(), 0, wi->size));
  } else
  CK(nrt_tensor_write(tw, hw.data(), 0, wi->size));
  printf("weight upload %zu bytes: %.1f ms\n", wi->size, us_since(t0) / 1e3);
  CK(nrt_tensor_write(tx, hx.data(), 0, xi->size));
  nrt_tensor_t* tg = nullptr;
  if (gi) { tg = dev_alloc(vnc, gi->size, "g"); CK(nrt_tensor_write(tg, hg.data(), 0, gi->size)); }

  nrt_tensor_set_t *in, *out;
  CK(nrt_allocate_tensor_set(&in)); CK(nrt_allocate_tensor_set(&out));
  CK(nrt_add_tensor_to_tensor_set(in, xi->name, tx));
  CK(nrt_add_tensor_to_tensor_set(in, wi->name, tw));
  if (gi) CK(nrt_add_tensor_to_tensor_set(in, gi->name, tg));
  CK(nrt_add_tensor_to_tensor_set(out, yi->name, ty));

  CK(nrt_execute(G.m, in, out));
  CK(nrt_tensor_read(ty, hy.data(), 0, yi->size));
  std::vector<float> ref; cpu_gemv(hx, ref);
  double maxerr = 0, maxref = 0;
  for (int n = 0; n < N; n++) { maxerr = std::max(maxerr, (double)fabsf(bf2f(hy[n]) - ref[n])); maxref = std::max(maxref, (double)fabsf(ref[n])); }
  bool ok = maxerr < 0.02 * maxref;
  printf("GEMV check: max|dev-cpu|=%.4g  max|ref|=%.4g  y[0..3]=%.4f %.4f %.4f %.4f (cpu %.4f %.4f %.4f %.4f) => %s\n",
         maxerr, maxref, bf2f(hy[0]), bf2f(hy[1]), bf2f(hy[2]), bf2f(hy[3]), ref[0], ref[1], ref[2], ref[3], ok ? "PASS" : "FAIL");

  const double wbytes = (double)wi->size;
  for (int i = 0; i < 20; i++) CK(nrt_execute(G.m, in, out));
  std::vector<double> lat;
  for (int i = 0; i < iters; i++) { auto t = clk::now(); CK(nrt_execute(G.m, in, out)); lat.push_back(us_since(t)); }
  stats("gemv nrt_execute (sync, per call)", lat, wbytes);


  // Async one-at-a-time: schedule then spin on completion (latency of a single dependent call)
  {
    std::vector<double> l1; NRT_STATUS r; nrta_seq_t seq;
    for (int i = 0; i < iters; i++) { auto t = clk::now(); CK(nrta_execute_schedule(G.m, in, out, 0, &r, &seq));
      bool done = false; while (!done) CK(nrta_is_completed(seq, &done)); l1.push_back(us_since(t)); CK(r); }
    stats("gemv nrta schedule+spin (one at a time)", l1, wbytes);
  }
  // Async: enqueue many executions, wait on the last one.
  {
    std::vector<NRT_STATUS> rets(iters); nrta_seq_t seq = 0;
    for (int w = 0; w < 2; w++) {
      auto t = clk::now();
      for (int i = 0; i < iters; i++) {
        NRT_STATUS s;
        while ((s = nrta_execute_schedule(G.m, in, out, 0, &rets[i], &seq)) == NRT_QUEUE_FULL) {}
        CK(s);
      }
      bool done = false; while (!done) CK(nrta_is_completed(seq, &done));
      double us = us_since(t) / iters;
      for (auto r : rets) if (r != NRT_SUCCESS) { fprintf(stderr, "async exec ret %d\n", (int)r); return 1; }
      if (w) printf("%-44s %.2f us/call -> %.1f GB/s\n", "gemv nrta_execute_schedule (pipelined)", us, wbytes / (us * 1e3));
    }
  }

  // Chained: y_i feeds x_{i+1} entirely on device by ping-ponging two buffers between two tensor sets.
  {
    nrt_tensor_t* tb = dev_alloc(vnc, xi->size, "pingpong");
    nrt_tensor_set_t *inA, *outA, *inB, *outB;
    CK(nrt_allocate_tensor_set(&inA)); CK(nrt_allocate_tensor_set(&outA));
    CK(nrt_allocate_tensor_set(&inB)); CK(nrt_allocate_tensor_set(&outB));
    CK(nrt_add_tensor_to_tensor_set(inA, xi->name, tx)); CK(nrt_add_tensor_to_tensor_set(inA, wi->name, tw)); CK(nrt_add_tensor_to_tensor_set(outA, yi->name, tb));
    CK(nrt_add_tensor_to_tensor_set(inB, xi->name, tb)); CK(nrt_add_tensor_to_tensor_set(inB, wi->name, tw)); CK(nrt_add_tensor_to_tensor_set(outB, yi->name, tx));
    if (gi) { CK(nrt_add_tensor_to_tensor_set(inA, gi->name, tg)); CK(nrt_add_tensor_to_tensor_set(inB, gi->name, tg)); }
    const int steps = 8;  // even -> result ends in tx
    std::vector<uint16_t> cur = hx;
    CK(nrt_tensor_write(tx, cur.data(), 0, xi->size));
    auto t = clk::now();
    for (int s = 0; s < steps; s++) CK(nrt_execute(G.m, s % 2 ? inB : inA, s % 2 ? outB : outA));
    double us = us_since(t);
    CK(nrt_tensor_read(tx, hy.data(), 0, xi->size));
    std::vector<float> r;
    for (int s = 0; s < steps; s++) { cpu_gemv(cur, r); for (int n = 0; n < N; n++) cur[n] = f2bf(r[n]); }
    double me = 0, mr = 0; for (int n = 0; n < N; n++) { me = std::max(me, (double)fabsf(bf2f(hy[n]) - bf2f(cur[n]))); mr = std::max(mr, (double)fabsf(bf2f(cur[n]))); }
    printf("chain %d device-resident steps (sync, %.2f us/step): max|dev-cpu|=%.4g max|ref|=%.4g => %s\n", steps, us / steps, me, mr, me < 0.05 * mr ? "PASS" : "FAIL");

    // same chain, async: data dependency carried through the in-order COMPUTE queue
    CK(nrt_tensor_write(tx, hx.data(), 0, xi->size));
    std::vector<NRT_STATUS> rets(steps); nrta_seq_t seq = 0;
    t = clk::now();
    for (int s = 0; s < steps; s++) CK(nrta_execute_schedule(G.m, s % 2 ? inB : inA, s % 2 ? outB : outA, 0, &rets[s], &seq));
    bool done = false; while (!done) CK(nrta_is_completed(seq, &done));
    us = us_since(t);
    CK(nrt_tensor_read(tx, hy.data(), 0, xi->size));
    me = 0; for (int n = 0; n < N; n++) me = std::max(me, (double)fabsf(bf2f(hy[n]) - bf2f(cur[n])));
    printf("chain %d steps async-enqueued (%.2f us/step): max|dev-cpu|=%.4g => %s\n", steps, us / steps, me, me < 0.05 * mr ? "PASS" : "FAIL");
  }

  // ------------------------------------------------------------ tiny kernel: fixed overhead floor
  {
    Model T = load(argv[2], vnc);
    auto& ai = T.ins.at(0); auto& oi = T.outs.at(0);
    nrt_tensor_t* ta = dev_alloc(vnc, ai.size, "a");
    nrt_tensor_t* to = dev_alloc(vnc, oi.size, "o");
    std::vector<float> ha(ai.size / 4), ho(oi.size / 4);
    for (size_t i = 0; i < ha.size(); i++) ha[i] = (float)i;
    CK(nrt_tensor_write(ta, ha.data(), 0, ai.size));
    nrt_tensor_set_t *ti, *tout; CK(nrt_allocate_tensor_set(&ti)); CK(nrt_allocate_tensor_set(&tout));
    CK(nrt_add_tensor_to_tensor_set(ti, ai.name, ta)); CK(nrt_add_tensor_to_tensor_set(tout, oi.name, to));
    CK(nrt_execute(T.m, ti, tout)); CK(nrt_tensor_read(to, ho.data(), 0, oi.size));
    printf("tiny check: %s\n", ho == ha ? "PASS" : "FAIL");
    for (int i = 0; i < 50; i++) CK(nrt_execute(T.m, ti, tout));
    std::vector<double> l2;
    for (int i = 0; i < iters; i++) { auto t = clk::now(); CK(nrt_execute(T.m, ti, tout)); l2.push_back(us_since(t)); }
    stats("tiny nrt_execute (sync) = overhead floor", l2);
    {
      std::vector<double> l1; NRT_STATUS r; nrta_seq_t sq;
      for (int i = 0; i < iters; i++) { auto t = clk::now(); CK(nrta_execute_schedule(T.m, ti, tout, 0, &r, &sq));
        bool done = false; while (!done) CK(nrta_is_completed(sq, &done)); l1.push_back(us_since(t)); CK(r); }
      stats("tiny nrta schedule+spin (one at a time)", l1);
    }
    std::vector<NRT_STATUS> rets(iters); nrta_seq_t seq = 0;
    auto t = clk::now();
    for (int i = 0; i < iters; i++) { NRT_STATUS s; while ((s = nrta_execute_schedule(T.m, ti, tout, 0, &rets[i], &seq)) == NRT_QUEUE_FULL) {} CK(s); }
    bool done = false; while (!done) CK(nrta_is_completed(seq, &done));
    printf("%-44s %.2f us/call\n", "tiny nrta_execute_schedule (pipelined)", us_since(t) / iters);
    // host<->device small transfer costs (e.g. token ids in / logits out)
    std::vector<double> lw, lr;
    for (int i = 0; i < iters; i++) { auto t1 = clk::now(); CK(nrt_tensor_write(ta, ha.data(), 0, ai.size)); lw.push_back(us_since(t1));
                                      t1 = clk::now(); CK(nrt_tensor_read(to, ho.data(), 0, oi.size)); lr.push_back(us_since(t1)); }
    stats("nrt_tensor_write 2KB", lw); stats("nrt_tensor_read 2KB", lr);
  }
  nrt_close();
  return ok ? 0 : 1;
}
