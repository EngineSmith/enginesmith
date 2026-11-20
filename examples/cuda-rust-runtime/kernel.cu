#include <cuda_runtime.h>
#include <cstddef>
#include <new>

// This kernel is compiled from source with the example. No numerical library is called.
__global__ void increment(float* values, std::size_t count) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) values[i] += 1.0f;
}

struct State {
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    float* values = nullptr;
    std::size_t count = 0;
};

extern "C" void runtime_destroy(State* state) {
    if (!state) return;
    if (state->stream) cudaStreamSynchronize(state->stream);
    if (state->executable) cudaGraphExecDestroy(state->executable);
    if (state->graph) cudaGraphDestroy(state->graph);
    if (state->values) cudaFree(state->values);
    if (state->begin) cudaEventDestroy(state->begin);
    if (state->end) cudaEventDestroy(state->end);
    if (state->stream) cudaStreamDestroy(state->stream);
    delete state;
}

extern "C" const char* runtime_error(int status) {
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}

extern "C" int runtime_create(std::size_t count, State** output) {
    if (!output || !count || count > (1U << 24)) return cudaErrorInvalidValue;
    *output = nullptr;
    // Never let a C++ allocation exception unwind through the Rust C ABI.
    auto* state = new (std::nothrow) State{};
    if (!state) return cudaErrorMemoryAllocation;
    state->count = count;
    cudaError_t status = cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking);
    if (status == cudaSuccess) status = cudaEventCreate(&state->begin);
    if (status == cudaSuccess) status = cudaEventCreate(&state->end);
    if (status == cudaSuccess) status = cudaMalloc(&state->values, count * sizeof(float));
    if (status == cudaSuccess) status = cudaMemsetAsync(state->values, 0, count * sizeof(float), state->stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(state->stream);
    if (status == cudaSuccess) status = cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeThreadLocal);
    if (status == cudaSuccess) {
        increment<<<(count + 255) / 256, 256, 0, state->stream>>>(state->values, count);
        cudaError_t launch = cudaGetLastError();
        cudaError_t capture = cudaStreamEndCapture(state->stream, &state->graph);
        status = launch != cudaSuccess ? launch : capture;
    }
    if (status == cudaSuccess) status = cudaGraphInstantiate(&state->executable, state->graph, 0);
    if (status != cudaSuccess) {
        runtime_destroy(state);
        return status;
    }
    *output = state;
    return cudaSuccess;
}

extern "C" int runtime_run(State* state, unsigned repetitions, int graph, float* elapsed_ms) {
    if (!state || !repetitions || repetitions > (1U << 24) || !elapsed_ms ||
        (graph != 0 && graph != 1)) return cudaErrorInvalidValue;
    cudaError_t status = cudaMemsetAsync(state->values, 0, state->count * sizeof(float), state->stream);
    if (status == cudaSuccess) status = cudaEventRecord(state->begin, state->stream);
    for (unsigned i = 0; i < repetitions && status == cudaSuccess; ++i) {
        if (graph) status = cudaGraphLaunch(state->executable, state->stream);
        else {
            increment<<<(state->count + 255) / 256, 256, 0, state->stream>>>(state->values, state->count);
            status = cudaGetLastError();
        }
    }
    if (status == cudaSuccess) status = cudaEventRecord(state->end, state->stream);
    if (status == cudaSuccess) status = cudaEventSynchronize(state->end);
    if (status == cudaSuccess) status = cudaEventElapsedTime(elapsed_ms, state->begin, state->end);
    return status;
}

extern "C" int runtime_read(State* state, float* destination, std::size_t count) {
    if (!state || !destination || count != state->count) return cudaErrorInvalidValue;
    cudaError_t status = cudaMemcpyAsync(destination, state->values, count * sizeof(float),
                                        cudaMemcpyDeviceToHost, state->stream);
    if (status == cudaSuccess) status = cudaStreamSynchronize(state->stream);
    return status;
}
