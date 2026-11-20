// Standalone GPU regression executable; instrumentation is absent from the example binary.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define REQUIRE(test) do { if (!(test)) { \
    std::fprintf(stderr, "lifecycle check failed at line %d\n", __LINE__); \
    std::exit(1); } } while (false)

namespace probe {
int calls = 0, fail_at = 0;
int streams = 0, events = 0, buffers = 0, graphs = 0, executables = 0;
bool fail() { return ++calls == fail_at; }
bool empty() { return !(streams || events || buffers || graphs || executables); }
cudaError_t stream_create(cudaStream_t* p, unsigned flags) {
    if (fail()) return cudaErrorMemoryAllocation;
    auto e = cudaStreamCreateWithFlags(p, flags); if (e == cudaSuccess) ++streams; return e;
}
cudaError_t event_create(cudaEvent_t* p) {
    if (fail()) return cudaErrorMemoryAllocation;
    auto e = cudaEventCreate(p); if (e == cudaSuccess) ++events; return e;
}
cudaError_t allocate(float** p, std::size_t n) {
    if (fail()) return cudaErrorMemoryAllocation;
    auto e = cudaMalloc(p, n); if (e == cudaSuccess) ++buffers; return e;
}
cudaError_t begin_capture(cudaStream_t s, cudaStreamCaptureMode mode) {
    if (fail()) return cudaErrorMemoryAllocation;
    return cudaStreamBeginCapture(s, mode);
}
cudaError_t end_capture(cudaStream_t s, cudaGraph_t* p) {
    auto e = cudaStreamEndCapture(s, p); if (e == cudaSuccess) ++graphs; return e;
}
cudaError_t instantiate(cudaGraphExec_t* p, cudaGraph_t g, unsigned long long flags) {
    if (fail()) return cudaErrorMemoryAllocation;
    auto e = cudaGraphInstantiate(p, g, flags); if (e == cudaSuccess) ++executables; return e;
}
cudaError_t stream_destroy(cudaStream_t p) {
    auto e = cudaStreamDestroy(p); REQUIRE(e == cudaSuccess); --streams; return e;
}
cudaError_t event_destroy(cudaEvent_t p) {
    auto e = cudaEventDestroy(p); REQUIRE(e == cudaSuccess); --events; return e;
}
cudaError_t release(void* p) {
    auto e = cudaFree(p); REQUIRE(e == cudaSuccess); --buffers; return e;
}
cudaError_t graph_destroy(cudaGraph_t p) {
    auto e = cudaGraphDestroy(p); REQUIRE(e == cudaSuccess); --graphs; return e;
}
cudaError_t executable_destroy(cudaGraphExec_t p) {
    auto e = cudaGraphExecDestroy(p); REQUIRE(e == cudaSuccess); --executables; return e;
}
} // namespace probe

// Compile the unchanged runtime implementation with test-only API wrappers. The wrappers
// inject ordinary setup failures rather than causing OOM or a device fault on a shared GPU.
#define cudaStreamCreateWithFlags probe::stream_create
#define cudaEventCreate probe::event_create
#define cudaMalloc probe::allocate
#define cudaStreamBeginCapture probe::begin_capture
#define cudaStreamEndCapture probe::end_capture
#define cudaGraphInstantiate probe::instantiate
#define cudaStreamDestroy probe::stream_destroy
#define cudaEventDestroy probe::event_destroy
#define cudaFree probe::release
#define cudaGraphDestroy probe::graph_destroy
#define cudaGraphExecDestroy probe::executable_destroy
#include "kernel.cu"

int main() {
    State* state = nullptr;
    float elapsed = 0, value = 0;
    REQUIRE(runtime_create(0, &state) == cudaErrorInvalidValue);
    REQUIRE(runtime_create((1U << 24) + 1, &state) == cudaErrorInvalidValue);
    REQUIRE(runtime_create(1, nullptr) == cudaErrorInvalidValue);
    REQUIRE(runtime_run(nullptr, 1, 0, &elapsed) == cudaErrorInvalidValue);
    REQUIRE(runtime_read(nullptr, &value, 1) == cudaErrorInvalidValue);
    runtime_destroy(nullptr);

    // Fail each creation point: stream, two events, allocation, capture start, instantiate.
    for (int failure = 1; failure <= 6; ++failure) {
        probe::calls = 0; probe::fail_at = failure;
        REQUIRE(runtime_create(4096, &state) == cudaErrorMemoryAllocation);
        REQUIRE(state == nullptr);
        REQUIRE(probe::empty());
    }
    probe::fail_at = 0;
    int cycles = 0, checks = 0;
    for (std::size_t count : {1U, 255U, 256U, 257U, 4096U}) {
        for (int cycle = 0; cycle < 8; ++cycle) {
            REQUIRE(runtime_create(count, &state) == cudaSuccess);
            REQUIRE(runtime_run(state, 0, 0, &elapsed) == cudaErrorInvalidValue);
            REQUIRE(runtime_run(state, (1U << 24) + 1, 0, &elapsed) == cudaErrorInvalidValue);
            REQUIRE(runtime_run(state, 1, 2, &elapsed) == cudaErrorInvalidValue);
            REQUIRE(runtime_run(state, 1, 0, nullptr) == cudaErrorInvalidValue);
            REQUIRE(runtime_read(state, nullptr, count) == cudaErrorInvalidValue);
            REQUIRE(runtime_read(state, &value, 0) == cudaErrorInvalidValue);
            std::vector<float> values(count);
            for (unsigned repetitions : {1U, 17U, 100U}) {
                for (int graph : {0, 1}) {
                    REQUIRE(runtime_run(state, repetitions, graph, &elapsed) == cudaSuccess);
                    REQUIRE(runtime_read(state, values.data(), count) == cudaSuccess);
                    for (float v : values) REQUIRE(v == static_cast<float>(repetitions));
                    ++checks;
                }
            }
            runtime_destroy(state); state = nullptr;
            REQUIRE(probe::empty());
            ++cycles;
        }
    }
    std::printf("{\"correct\":true,\"lifecycle_cycles\":%d,\"correctness_checks\":%d,"
                "\"injected_setup_failures\":6,\"live_resources\":0}\n", cycles, checks);
}
