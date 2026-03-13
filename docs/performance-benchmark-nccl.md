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

