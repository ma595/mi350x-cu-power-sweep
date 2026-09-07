#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


constexpr int COMPONENT_LOOPCOUNT = 4096;
constexpr int WAVE_SIZE = 64;

#define REPEAT2(X) X X
#define REPEAT4(X) REPEAT2(X) REPEAT2(X)
#define REPEAT8(X) REPEAT4(X) REPEAT4(X)
#define REPEAT16(X) REPEAT8(X) REPEAT8(X)
#define REPEAT32(X) REPEAT16(X) REPEAT16(X)
#define REPEAT64(X) REPEAT32(X) REPEAT32(X)

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        const hipError_t status = (call);                                      \
        if (status != hipSuccess) {                                            \
            std::cerr << "HIP error: " << hipGetErrorString(status)            \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";       \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)


__global__ void scalar_add(float *output) {
    asm volatile("s_mov_b32 s40, 0\n" :::);
    for (int index = 0; index < COMPONENT_LOOPCOUNT * 32; ++index) {
        asm volatile("s_add_u32 s40, s40, 1\n" :::);
    }
    if ((threadIdx.x & (WAVE_SIZE - 1)) == 0) {
        output[blockIdx.x] = static_cast<float>(blockIdx.x);
    }
}


__global__ void sfu_transcendental(float *output) {
    float x = 0.001f * static_cast<float>(threadIdx.x + blockIdx.x + 1);
    float y = 0.0f;
    float z = 0.0f;
    for (int index = 0; index < COMPONENT_LOOPCOUNT; ++index) {
        asm volatile(
            "v_exp_f32 %[y], %[x]\n"
            "v_log_f32 %[z], %[y]\n"
            "v_sin_f32 %[y], %[z]\n"
            "v_cos_f32 %[z], %[y]\n"
            : [y] "=&v"(y), [z] "=&v"(z)
            : [x] "v"(x)
            :
        );
        x = y + z + 0.001f;
    }
    output[blockIdx.x * blockDim.x + threadIdx.x] = x;
}


__global__ void matrix_mfma_fp32(const float *input_a,
                                 const float *input_b,
                                 float *output) {
    using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    float4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    const int lane = threadIdx.x & (WAVE_SIZE - 1);
    const float a = input_a[lane & 3];
    const float b = input_b[(lane >> 2) & 3];

    for (int index = 0; index < COMPONENT_LOOPCOUNT * 8; ++index) {
        asm volatile(
            "v_mfma_f32_4x4x1f32 %[dst], %[a], %[b], %[src]\n"
            : [dst] "=v"(accumulator)
            : [a] "v"(a), [b] "v"(b), [src] "v"(accumulator)
        );
    }
    if (threadIdx.x < 4) {
        output[blockIdx.x * 4 + threadIdx.x] = accumulator[threadIdx.x];
    }
}


#define V_FMA_F64_8                                                          \
    "v_fma_f64 v[0:1],   v[16:17], v[32:33], v[0:1]\n"                    \
    "v_fma_f64 v[2:3],   v[18:19], v[34:35], v[2:3]\n"                    \
    "v_fma_f64 v[4:5],   v[20:21], v[36:37], v[4:5]\n"                    \
    "v_fma_f64 v[6:7],   v[22:23], v[38:39], v[6:7]\n"                    \
    "v_fma_f64 v[8:9],   v[24:25], v[40:41], v[8:9]\n"                    \
    "v_fma_f64 v[10:11], v[26:27], v[42:43], v[10:11]\n"                  \
    "v_fma_f64 v[12:13], v[28:29], v[44:45], v[12:13]\n"                  \
    "v_fma_f64 v[14:15], v[30:31], v[46:47], v[14:15]\n"

__global__ void vector_fp64_fma(float *output) {
    asm volatile(
        "v_mov_b32 v0,  0x00000000\n" "v_mov_b32 v1,  0x3ff00000\n"
        "v_mov_b32 v2,  0x00000000\n" "v_mov_b32 v3,  0x3ff00000\n"
        "v_mov_b32 v4,  0x00000000\n" "v_mov_b32 v5,  0x3ff00000\n"
        "v_mov_b32 v6,  0x00000000\n" "v_mov_b32 v7,  0x3ff00000\n"
        "v_mov_b32 v8,  0x00000000\n" "v_mov_b32 v9,  0x3ff00000\n"
        "v_mov_b32 v10, 0x00000000\n" "v_mov_b32 v11, 0x3ff00000\n"
        "v_mov_b32 v12, 0x00000000\n" "v_mov_b32 v13, 0x3ff00000\n"
        "v_mov_b32 v14, 0x00000000\n" "v_mov_b32 v15, 0x3ff00000\n"
        "v_mov_b32 v16, 0x00000000\n" "v_mov_b32 v17, 0x40000000\n"
        "v_mov_b32 v18, 0x00000000\n" "v_mov_b32 v19, 0x40000000\n"
        "v_mov_b32 v20, 0x00000000\n" "v_mov_b32 v21, 0x40000000\n"
        "v_mov_b32 v22, 0x00000000\n" "v_mov_b32 v23, 0x40000000\n"
        "v_mov_b32 v24, 0x00000000\n" "v_mov_b32 v25, 0x40000000\n"
        "v_mov_b32 v26, 0x00000000\n" "v_mov_b32 v27, 0x40000000\n"
        "v_mov_b32 v28, 0x00000000\n" "v_mov_b32 v29, 0x40000000\n"
        "v_mov_b32 v30, 0x00000000\n" "v_mov_b32 v31, 0x40000000\n"
        "v_mov_b32 v32, 0x00000000\n" "v_mov_b32 v33, 0x3fe00000\n"
        "v_mov_b32 v34, 0x00000000\n" "v_mov_b32 v35, 0x3fe00000\n"
        "v_mov_b32 v36, 0x00000000\n" "v_mov_b32 v37, 0x3fe00000\n"
        "v_mov_b32 v38, 0x00000000\n" "v_mov_b32 v39, 0x3fe00000\n"
        "v_mov_b32 v40, 0x00000000\n" "v_mov_b32 v41, 0x3fe00000\n"
        "v_mov_b32 v42, 0x00000000\n" "v_mov_b32 v43, 0x3fe00000\n"
        "v_mov_b32 v44, 0x00000000\n" "v_mov_b32 v45, 0x3fe00000\n"
        "v_mov_b32 v46, 0x00000000\n" "v_mov_b32 v47, 0x3fe00000\n"
        :::
    );
    for (int index = 0; index < COMPONENT_LOOPCOUNT; ++index) {
        asm volatile(REPEAT64(V_FMA_F64_8) :::);
    }
    if ((threadIdx.x & (WAVE_SIZE - 1)) == 0) {
        output[blockIdx.x] = static_cast<float>(blockIdx.x);
    }
}


#define V_FMA_F32_16                                                         \
    "v_fma_f32 v0,  v32, v48, v0\n"                                         \
    "v_fma_f32 v1,  v33, v49, v1\n"                                         \
    "v_fma_f32 v2,  v34, v50, v2\n"                                         \
    "v_fma_f32 v3,  v35, v51, v3\n"                                         \
    "v_fma_f32 v4,  v36, v52, v4\n"                                         \
    "v_fma_f32 v5,  v37, v53, v5\n"                                         \
    "v_fma_f32 v6,  v38, v54, v6\n"                                         \
    "v_fma_f32 v7,  v39, v55, v7\n"                                         \
    "v_fma_f32 v8,  v40, v56, v8\n"                                         \
    "v_fma_f32 v9,  v41, v57, v9\n"                                         \
    "v_fma_f32 v10, v42, v58, v10\n"                                        \
    "v_fma_f32 v11, v43, v59, v11\n"                                        \
    "v_fma_f32 v12, v44, v60, v12\n"                                        \
    "v_fma_f32 v13, v45, v61, v13\n"                                        \
    "v_fma_f32 v14, v46, v62, v14\n"                                        \
    "v_fma_f32 v15, v47, v63, v15\n"

__global__ void vector_fp32_fma(float *output) {
    const uint32_t seed = threadIdx.x + blockIdx.x;
    asm volatile(
        "v_mov_b32 v0,  %[seed]\n"  "v_mov_b32 v1,  %[seed]\n"
        "v_mov_b32 v2,  %[seed]\n"  "v_mov_b32 v3,  %[seed]\n"
        "v_mov_b32 v4,  %[seed]\n"  "v_mov_b32 v5,  %[seed]\n"
        "v_mov_b32 v6,  %[seed]\n"  "v_mov_b32 v7,  %[seed]\n"
        "v_mov_b32 v8,  %[seed]\n"  "v_mov_b32 v9,  %[seed]\n"
        "v_mov_b32 v10, %[seed]\n"  "v_mov_b32 v11, %[seed]\n"
        "v_mov_b32 v12, %[seed]\n"  "v_mov_b32 v13, %[seed]\n"
        "v_mov_b32 v14, %[seed]\n"  "v_mov_b32 v15, %[seed]\n"
        "v_mov_b32 v32, 0x3f800000\n" "v_mov_b32 v33, 0x3f800000\n"
        "v_mov_b32 v34, 0x3f800000\n" "v_mov_b32 v35, 0x3f800000\n"
        "v_mov_b32 v36, 0x3f800000\n" "v_mov_b32 v37, 0x3f800000\n"
        "v_mov_b32 v38, 0x3f800000\n" "v_mov_b32 v39, 0x3f800000\n"
        "v_mov_b32 v40, 0x40000000\n" "v_mov_b32 v41, 0x40000000\n"
        "v_mov_b32 v42, 0x40000000\n" "v_mov_b32 v43, 0x40000000\n"
        "v_mov_b32 v44, 0x40000000\n" "v_mov_b32 v45, 0x40000000\n"
        "v_mov_b32 v46, 0x40000000\n" "v_mov_b32 v47, 0x40000000\n"
        "v_mov_b32 v48, 0x3f000000\n" "v_mov_b32 v49, 0x3f000000\n"
        "v_mov_b32 v50, 0x3f000000\n" "v_mov_b32 v51, 0x3f000000\n"
        "v_mov_b32 v52, 0x3f000000\n" "v_mov_b32 v53, 0x3f000000\n"
        "v_mov_b32 v54, 0x3f000000\n" "v_mov_b32 v55, 0x3f000000\n"
        "v_mov_b32 v56, 0x3fc00000\n" "v_mov_b32 v57, 0x3fc00000\n"
        "v_mov_b32 v58, 0x3fc00000\n" "v_mov_b32 v59, 0x3fc00000\n"
        "v_mov_b32 v60, 0x3fc00000\n" "v_mov_b32 v61, 0x3fc00000\n"
        "v_mov_b32 v62, 0x3fc00000\n" "v_mov_b32 v63, 0x3fc00000\n"
        :
        : [seed] "v"(seed)
        :
    );
    for (int index = 0; index < COMPONENT_LOOPCOUNT; ++index) {
        asm volatile(REPEAT64(V_FMA_F32_16) :::);
    }
    if ((threadIdx.x & (WAVE_SIZE - 1)) == 0) {
        output[blockIdx.x] = static_cast<float>(seed);
    }
}


struct Options {
    std::string benchmark;
    int blocks = 304;
    int threads = 1024;
    double duration_s = 1.0;
    bool list = false;
};

const std::vector<std::string> BENCHMARKS = {
    "SCALAR ADD",
    "SFU TRANSCENDENTAL",
    "MATRIX MFMA FP32",
    "VECTOR FP64 FMA",
    "VECTOR FP32 FMA",
};

void usage(const char *program) {
    std::cerr
        << "Usage: " << program << " --exact-filter NAME [options]\n"
        << "  --blocks N       Block count (default: 304)\n"
        << "  --threads N      Threads per block (default: 1024)\n"
        << "  --duration-s S   Repeat for at least S seconds\n"
        << "  --list           List the five workloads\n";
}

int positive_int(const char *option, const char *text) {
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0) {
        throw std::runtime_error(std::string("invalid ") + option);
    }
    return static_cast<int>(value);
}

double positive_double(const char *option, const char *text) {
    char *end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || value <= 0.0) {
        throw std::runtime_error(std::string("invalid ") + option);
    }
    return value;
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        if (argument == "--list") {
            options.list = true;
        } else if (argument == "--exact-filter" && index + 1 < argc) {
            options.benchmark = argv[++index];
        } else if (argument == "--blocks" && index + 1 < argc) {
            options.blocks = positive_int("--blocks", argv[++index]);
        } else if (argument == "--threads" && index + 1 < argc) {
            options.threads = positive_int("--threads", argv[++index]);
        } else if (argument == "--duration-s" && index + 1 < argc) {
            options.duration_s = positive_double("--duration-s", argv[++index]);
        } else if (argument == "--help" || argument == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete option: " + argument);
        }
    }
    return options;
}

void launch(const std::string &name,
            dim3 grid,
            dim3 block,
            float *input_a,
            float *input_b,
            float *output) {
    if (name == "SCALAR ADD") {
        hipLaunchKernelGGL(scalar_add, grid, block, 0, 0, output);
    } else if (name == "SFU TRANSCENDENTAL") {
        hipLaunchKernelGGL(sfu_transcendental, grid, block, 0, 0, output);
    } else if (name == "MATRIX MFMA FP32") {
        hipLaunchKernelGGL(matrix_mfma_fp32, grid, block, 0, 0,
                           input_a, input_b, output);
    } else if (name == "VECTOR FP64 FMA") {
        hipLaunchKernelGGL(vector_fp64_fma, grid, block, 0, 0, output);
    } else if (name == "VECTOR FP32 FMA") {
        hipLaunchKernelGGL(vector_fp32_fma, grid, block, 0, 0, output);
    } else {
        throw std::runtime_error("unknown benchmark: " + name);
    }
    HIP_CHECK(hipGetLastError());
}

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.list) {
            for (const std::string &name : BENCHMARKS) {
                std::cout << name << "\n";
            }
            return 0;
        }
        if (options.benchmark.empty()) {
            usage(argv[0]);
            return 2;
        }
        if (options.threads > 1024) {
            throw std::runtime_error("MI300X supports at most 1024 threads per block");
        }

        int device_count = 0;
        HIP_CHECK(hipGetDeviceCount(&device_count));
        if (device_count != 1) {
            throw std::runtime_error(
                "expose exactly one GPU with ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES"
            );
        }
        HIP_CHECK(hipSetDevice(0));

        const std::size_t elements = std::max<std::size_t>(
            4096,
            static_cast<std::size_t>(options.blocks) * options.threads
        );
        const std::size_t bytes = elements * sizeof(float);
        float *input_a = nullptr;
        float *input_b = nullptr;
        float *output = nullptr;
        HIP_CHECK(hipMalloc(&input_a, bytes));
        HIP_CHECK(hipMalloc(&input_b, bytes));
        HIP_CHECK(hipMalloc(&output, bytes));
        HIP_CHECK(hipMemset(input_a, 0, bytes));
        HIP_CHECK(hipMemset(input_b, 0, bytes));
        HIP_CHECK(hipMemset(output, 0, bytes));
        HIP_CHECK(hipDeviceSynchronize());

        std::cout << "BENCHMARK_READY " << options.benchmark << std::endl;
        const auto start = std::chrono::steady_clock::now();
        const auto deadline = start + std::chrono::duration<double>(options.duration_s);
        std::uint64_t launches = 0;
        do {
            launch(options.benchmark,
                   dim3(options.blocks),
                   dim3(options.threads),
                   input_a,
                   input_b,
                   output);
            HIP_CHECK(hipDeviceSynchronize());
            ++launches;
        } while (std::chrono::steady_clock::now() < deadline);

        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start
        ).count();
        std::cout << "BENCHMARK_DONE " << options.benchmark
                  << " launches=" << launches
                  << " elapsed_s=" << elapsed << "\n";

        HIP_CHECK(hipFree(output));
        HIP_CHECK(hipFree(input_b));
        HIP_CHECK(hipFree(input_a));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 2;
    }
}

