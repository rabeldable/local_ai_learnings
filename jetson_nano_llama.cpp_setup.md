# Jetson Nano + llama.cpp (CUDA) — Clean Install Guide

This guide recreates your working setup exactly, on a **Jetson Nano (t210ref)** with:

- **L4T**: 32.7.1 (JetPack 4.6.1 era)  
- **OS**: Ubuntu 18.04.6 LTS (bionic)  
- **CUDA**: 10.2 (default on JP4.6.x)  
- **llama.cpp commit**: `81bc921` (built on a local branch `llamaForJetsonNano`)  
- **CMake**: 3.31.6 (built from source)  
- **GCC/G++**: 8.5.0 (built from source)  
- **CUDA Arch**: **53** (Maxwell, Nano)

# Inspiration from Anurag Dogra
https://medium.com/@anuragdogra2192/llama-cpp-on-nvidia-jetson-nano-a-complete-guide-fb178530bc35



---

## 0) One‑time system prep

```bash
sudo -i
apt update
apt install -y build-essential git curl libssl-dev software-properties-common
```

**Set CUDA env (JetPack 4.6.x):**
```bash
echo 'export PATH=/usr/local/cuda-10.2/bin:$PATH' >> /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

nvcc --version   # sanity check
```

**(Recommended) Add swap** (Jetson Nano is 4GB RAM; builds benefit from swap):
```bash
fallocate -l 6G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

---

## 1) Build and install **CMake 3.31.6**

Ubuntu 18.04’s cmake is too old for recent llama.cpp. Build from source:

```bash
mkdir -p ~/builder && cd ~/builder
wget https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6.tar.gz
tar -xvf cmake-3.31.6.tar.gz
cd cmake-3.31.6
./bootstrap
make -j$(nproc)
sudo make install

cmake --version  # expect 3.31.6
which cmake      # /usr/local/bin/cmake
```

---

## 2) Build and set **GCC/G++ 8.5.0**

GCC on Nano can be too old/incompatible with some CUDA/CMake combos. Your working build used GCC 8.5.

```bash
cd ~/builder
wget http://ftp.gnu.org/gnu/gcc/gcc-8.5.0/gcc-8.5.0.tar.gz
tar -xvzf gcc-8.5.0.tar.gz
cd gcc-8.5.0
./contrib/download_prerequisites
mkdir build && cd build
../configure --enable-languages=c,c++ --disable-multilib
make -j$(nproc)
sudo make install
```

**Point system to the new compilers:**
```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/local/bin/gcc 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/local/bin/g++ 100
sudo update-alternatives --config gcc   # choose /usr/local/bin/gcc (8.5)
sudo update-alternatives --config g++   # choose /usr/local/bin/g++ (8.5)

gcc --version
g++ --version
```

---

## 3) Clone **llama.cpp** at the known‑good commit

```bash
mkdir -p ~/LOCAL_LLM && cd ~/LOCAL_LLM
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
git checkout 81bc921
git checkout -b llamaForJetsonNano
```

---

## 4) Build llama.cpp with CUDA (Nano = **SM 53**)

Use cuBLAS and set the architecture explicitly:

```bash
mkdir -p build && cd build

cmake ..   -DLLAMA_CUBLAS=ON   -DCMAKE_CUDA_ARCHITECTURES=53   -DCMAKE_C_COMPILER=/usr/local/bin/gcc   -DCMAKE_CXX_COMPILER=/usr/local/bin/g++   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.2/bin/nvcc

make -j2
```

**Notes**
- `-j2` is safer on Nano to avoid OOM.
- If you ever rebuild, `rm -rf build` then repeat cmake/make.

---

## 5) Models directory & downloads

```bash
mkdir -p ~/LOCAL_LLM/models && cd ~/LOCAL_LLM/models

# TinyLlama (lightweight, good smoke test)
curl -LO https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# CodeLlama 7B instruct (Q2_K for Nano memory constraints)
curl -L -o codellama-7b-instruct.Q2_K.gguf   "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q2_K.gguf?download=true"
```

---

## 6) First run / quick tests

```bash
cd ~/LOCAL_LLM/llama.cpp/build

# Option A: tinyllama quick prompt
./bin/main   -m ~/LOCAL_LLM/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf   -n 128   -p "Tell me a joke."

# Option B: CodeLlama 7B instruct (heavier)
./bin/main   -m ~/LOCAL_LLM/models/codellama-7b-instruct.Q2_K.gguf   -n 256   -p "Write a complete Python script that gets the top resource usage processes. Include error handling."
```

Tweak typical flags for Nano:
- `-t <threads>` (try 2–4)
- `-ngl <layers_offloaded>` (GPU offload depth; experiment if supported at this commit)
- `--temp`, `-n`, `--top-k`, etc.

---

## 7) Quality-of-life (optional but useful)

**Persistent env:**
```bash
cat /etc/profile.d/cuda.sh
# verify PATH/LD_LIBRARY_PATH lines are present (from step 0)
```

**Swap check:**
```bash
free -m
# confirm Swap shows the size you created
```

**Disk housekeeping:**
- Keep only one working `llama.cpp` tree (avoid multiple clones that waste space).
- Prune old models; Q2/Q3/Q4 quant variants add up quickly on a 64GB card.

---

## 8) Troubleshooting (Jetson/Nano specifics)

- **`nvcc not found`** → re‑source `/etc/profile.d/cuda.sh`, confirm `/usr/local/cuda-10.2/bin/nvcc`.
- **CMake too old** → `cmake --version` must show 3.31.6 (or ≥3.26).
- **Compute capability mismatch** → Nano is **SM 53**. Ensure `-DCMAKE_CUDA_ARCHITECTURES=53`.
- **OOM during build** → use `-j2` and ensure swap is enabled.
- **Runtime slow/unstable** → try smaller model (TinyLlama), reduce `-n`, lower context, reduce `-t`, or use lower quant (e.g., Q2_K).
- **Link errors with BLAS** → for this path we relied on cuBLAS only; if you add CPU BLAS, install `libopenblas-dev` and rebuild.

---

## 9) What this locks in (so future you doesn’t guess)

- L4T 32.7.1 / Ubuntu 18.04.6 / CUDA 10.2
- CMake 3.31.6 from source
- GCC/G++ 8.5.0 from source via update‑alternatives
- llama.cpp **commit `81bc921`**
- `cmake .. -DLLAMA_CUBLAS=ON -DCMAKE_CUDA_ARCHITECTURES=53 -DCMAKE_*_COMPILER=...`
- `make -j2` on Nano
- Known‑good model files & test prompts
