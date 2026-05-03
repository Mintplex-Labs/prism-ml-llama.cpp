# llama.cpp (Mintplex Bonsai Fork)

> **Important**
> This is a fork of the Prism-ML fork of llama.cpp that is synced to the main llama.cpp repo. It is not yet ready for production use and should be considered experimental.
>
> The primary benefit of this fork is an up-to-date version of llama.cpp that also merges the capabilities of the Prism-ML fork to support the Bonsai 1-bit models.
>
> **Note:** this is not an official fork and is not supported by the Prism-ML team - this is just a personal fork to demo Bonsai until official support is added.

---

## What PR #1 Adds — Intel Arc SYCL Support

Bonsai 8B uses a custom quantization format called **Q1_0_g128** — a 1-bit weight
format where every weight is either `+d` or `-d` (stored as a single bit), grouped
in blocks of 128. This format is defined by PrismML and does not exist in upstream
llama.cpp.

The Mintplex fork has CUDA kernels for Q1_0_g128, but the **SYCL/oneAPI backend**
(used by Intel Arc GPUs) had no kernels for this type. Attempting to run Bonsai on
Intel Arc via SYCL crashed immediately:

```
fatal error: unsupport data type=q1_0_g128
Aborted
```

PR #1 adds three sets of changes to fix this:

**`ggml/src/ggml-sycl/vecdotq.hpp`** — The core math kernel. Takes a Q1_0_g128
weight block and a Q8_1 activation block, computes the dot product correctly
(applying the Q8_1 scale factor per block), and returns the result. This is the
innermost operation of every matrix multiply.

**`ggml/src/ggml-sycl/mmvq.cpp`** — The dispatch layer. When the SYCL scheduler
needs to multiply a Q1_0_g128 weight matrix by an activation vector, this routes
the call to the correct kernel. Without this the scheduler has no path and aborts.

**`ggml/src/ggml-sycl/convert.cpp`** — The type conversion path. When SYCL needs
to convert Q1_0_g128 weights to fp16 or fp32 (for KV cache and other operations),
these functions handle it. Without this a second crash occurs during graph setup.

**`common/chat-auto-parser-generator.cpp`** — A fix for the thinking suppression
bug. Bonsai 8B is built on Qwen3 which has chain-of-thought thinking baked into
its training. When `--reasoning off` is used, the server injects a suppression
block (`<think>\n\n</think>`) into the assistant turn. The original code used
`reasoning.start` as the delimiter even when thinking was disabled, causing the
parser to extract the suppression block as spurious `reasoning_content`. The fix
uses an empty delimiter when `enable_thinking` is false so the suppression block
is treated as a literal prefix and ignored.

**Result:** All 37 Bonsai 8B layers offload to the Intel Arc GPU via SYCL/oneAPI
Level Zero, achieving ~46 tok/s generation — approximately 24x faster than
CPU-only inference on the same machine.

---

## How to use this fork

### On macOS

```bash
git clone https://github.com/Mintplex-Labs/prism-ml-llama.cpp
cd prism-ml-llama.cpp
cmake -B build && cmake --build build -j
```

### On Linux — Intel Arc GPU (SYCL/oneAPI)

> **Note:** The macOS build command above will not enable GPU acceleration on Linux/Intel.
> Intel Arc users must use the oneAPI toolkit for full SYCL GPU offload.

**Prerequisites:**
- Ubuntu 22.04 / 24.04
- [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
- Intel Arc GPU with Level Zero driver

```bash
# Install Level Zero driver
sudo apt install -y intel-level-zero-gpu level-zero

# Clone the repo
git clone https://github.com/Mintplex-Labs/prism-ml-llama.cpp
cd prism-ml-llama.cpp

# Source oneAPI environment
source /opt/intel/oneapi/setvars.sh --force

# Configure with SYCL enabled, Vulkan disabled
# (Vulkan lacks integer dot product support on Arc — use SYCL for full speed)
cmake -B build \
    -DGGML_SYCL=ON \
    -DGGML_VULKAN=OFF \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)
```

## Download the Bonsai model

```bash
wget https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B.gguf \
    -O Bonsai-8B.gguf
```

---

## Run the model

### llama-cli (macOS)

```bash
./build/bin/llama-cli \
    -m Bonsai-8B.gguf \
    -p "Explain quantum computing in simple terms." \
    -n 256 \
    --temp 0.5 \
    --top-p 0.85 \
    --top-k 20 \
    -ngl 99
```

### llama-cli (Linux — Intel Arc via SYCL)

```bash
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export GGML_SYCL_DEVICE=0
export ZES_ENABLE_SYSMAN=1

./build/bin/llama-cli \
    -m Bonsai-8B.gguf \
    -p "Explain quantum computing in simple terms." \
    -n 256 \
    --temp 0.5 \
    --top-p 0.85 \
    --top-k 20 \
    -ngl 99
```

### llama-server (macOS)

```bash
./build/bin/llama-server \
    -m Bonsai-8B.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \
    --ctx-size 65536
```

### llama-server (Linux — Intel Arc via SYCL)

Bonsai GGUFs embed a Qwen3 thinking-capable chat template. To suppress
chain-of-thought output, create a plain ChatML template file first:

```bash
cat > chatml-nothink.jinja << 'EOF'
{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}
EOF
```

Then launch the server:

```bash
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export GGML_SYCL_DEVICE=0
export ZES_ENABLE_SYSMAN=1

./build/bin/llama-server \
    -m Bonsai-8B.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \
    --ctx-size 16384 \
    --reasoning off \
    --chat-template-file chatml-nothink.jinja
```

> **Note:** `--reasoning off` combined with `--chat-template-file` is required
> to suppress thinking mode. `--no-jinja` and `--reasoning-budget 0` alone
> are insufficient — the model generates `<think>` tokens regardless due to
> training.

---

## Performance (Intel Arc Pro B50)

| Metric | Value |
|--------|-------|
| GPU | Intel Arc Pro B50 (BMG G21, 16 GB) |
| Backend | SYCL / oneAPI Level Zero |
| Model | Bonsai-8B Q1_0_g128 (1.08 GB) |
| Layers offloaded | 37 / 37 |
| Prompt eval (cold) | ~51.7 tok/s |
| Prompt eval (warm) | ~98.0 tok/s |
| Generation | ~41.0 tok/s |
| CPU-only baseline | ~1.2 tok/s prompt / ~1.9 tok/s generation |
| Speedup vs CPU | ~43x prompt / ~22x generation |
| 500 token response | ~12.9s total |

---

## Troubleshooting

**`fatal error: unsupport data type=q1_0_g128`**
You are using a build without the SYCL Q1_0_g128 kernels. Make sure you
built with `-DGGML_SYCL=ON` using the oneAPI compilers. See PR #1 for details.

**Model generates `<think>` blocks**
Use `--reasoning off` and `--chat-template-file chatml-nothink.jinja` as
shown above. The thinking template is embedded in the GGUF and requires
both flags to suppress.

**Vulkan finds no GPU / slow inference**
Do not use Vulkan on Intel Arc for Bonsai models — Mesa ANV lacks
`VK_KHR_shader_integer_dot_product` for Battlemage, so Q1_0_g128 weights
fall back to CPU. Always use the SYCL build with `ONEAPI_DEVICE_SELECTOR=level_zero:0`.
