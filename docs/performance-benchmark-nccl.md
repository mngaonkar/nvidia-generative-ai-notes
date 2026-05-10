# Performance benchmark with NCCL

NCCL test (https://github.com/NVIDIA/nccl-tests) provide benchmarking tools for NCCL operations over TCP/IP or RDMA interconnects.

## Install steps
1. git clone https://github.com/NVIDIA/nccl-tests.git
2. make MPI=1 (turn on MPI for distributed testing)

## Example run

```
mpirun --prefix /usr/local \                                                       
  --launch-agent prted \
  -np 2 -host 10.0.0.131,10.0.0.147 -N 1 \
  ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 1

```

**Command breakdown:**

- **`mpirun`**: MPI launcher that starts the distributed job across multiple ranks (processes).
- **`--prefix /usr/local`**: Adds `/usr/local` to the MPI runtime's search path so that the same MPI installation (binaries, libraries) is used on all hosts.
- **`--launch-agent prted`**: Tells `mpirun` to use the Open MPI `prted` daemon on each node to spawn and manage the processes.
- **`-np 2`**: Run with 2 MPI processes (2 ranks in total).
- **`-host 10.0.0.131,10.0.0.147`**: Use these two machines as the hosts for the MPI ranks.
- **`-N 1`**: Launch 1 MPI process per node (so 1 rank on each host for a total of 2).
- **`./build/all_reduce_perf`**: NCCL test binary that benchmarks the all-reduce collective.
- **`-b 8`**: Minimum message size is 8 bytes.
- **`-e 1G`**: Maximum message size is 1 GiB.
- **`-f 2`**: Use message sizes that increase by a factor of 2 between tests (geometric progression).
- **`-g 1`**: Use 1 GPU per process (rank) in the benchmark.

## Common pitfalls

- **Mixed NCCL/MPI installations across nodes**
  - Symptom: jobs hang at startup or fail with obscure symbol/version errors.
  - Fix: ensure all nodes use the **same** CUDA, NCCL, and MPI versions. Use `--prefix` or module systems so `mpirun` picks up identical installs everywhere.
  ```
  prte --version                                              
  prte (PRRTE) 3.0.13

  mpirun --version
  mpirun (Open MPI) 5.0.9

  nvcc --version
  vcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2025 NVIDIA Corporation
  Built on Tue_Dec_16_07:23:41_PM_PST_2025
  Cuda compilation tools, release 13.1, V13.1.115
  Build cuda_13.1.r13.1/compiler.37061995_0

  python -c "import torch; print(torch.cuda.nccl.version())"
  (2, 26, 2)
  ```

- **Firewalls or closed ports blocking communication**
  - Symptom: ranks hang on initialization; only rank 0 prints logs.
  - Fix: open the relevant TCP/InfiniBand ports between nodes or set `NCCL_IB_DISABLE=1` (to force TCP) while debugging. Verify basic connectivity with `ping` and `ssh` before running NCCL tests.

- **Incorrect host mapping vs. physical GPUs**
  - Symptom: oversubscribed GPUs, poor performance, or `CUDA_ERROR_INVALID_DEVICE`.
  - Fix: confirm each host has at least `-N` visible GPUs and that `CUDA_VISIBLE_DEVICES` or `nvidia-smi topo -m` matches your intended mapping. Start with `-g 1` and scale up once a 1-GPU-per-rank test is stable.

- **Forgetting NCCL environment tuning**
  - Symptom: bandwidth much lower than expected even though the test runs.
  - Fix: explicitly set key env vars for your fabric, for example:
    - `NCCL_DEBUG=INFO` (or `WARN`) to see algorithm/topology choices
    - `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX` for InfiniBand
    - `NCCL_SOCKET_IFNAME` for TCP/IP interfaces

- **Benchmarks competing with other workloads**
  - Symptom: noisy, highly variable bandwidth/latency numbers.
  - Fix: run on otherwise idle GPUs/nodes, disable power-saving modes that downclock GPUs, and pin processes to specific GPUs/CPUs where possible.

- **Misinterpreting results**
  - Symptom: comparing numbers from different message ranges or topologies as if they were equivalent.
  - Fix: always note `-b`, `-e`, `-f`, topology (intra-node vs inter-node), and number of ranks when comparing runs. Small messages are latency-dominated; large messages are bandwidth-dominated.
