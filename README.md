Model: Llama-3.2-3B-Instruct

#### Current progress
Naive cold start: **~81s**  
`criu` (patched to use `mmap` instead of `preadv`) + overlapped `vmtouch` on container start + vLLM sleep level 2 + `reload_weights_fast`: **~6s**

#### Experiment 1 - cache mounting
Simply remount the caches (`~/.cache/vllm/* ~/.triton/* ~/.cache/flashinfer/* ~/.nv/ComputeCache`) generated during vllm init.  
Cold start, weights on disk, all caches cleared: **81 seconds**  
Warm start, weights in page cache (RAM), caches filled: **38 seconds**

#### Experiment 2 - criu, vllm sleep level 1
Capture vLLM at sleep level 1. See [Sleep Mode](https://github.com/vllm-project/vllm/blob/main/docs/features/sleep_mode.md).  
Necessary criu flags:
- `--enable-external-masters`
- `--tcp-established`
- `--shell-job`
- `--link-remap`

Checkpointing fails on io_uring -- run `sudo sysctl kernel.io_uring_disabled=2`

Checkpoint + restore successful. However, takes 15-30s to complete (first response). Disk constrained transfer.

Separately, criu loads via `preadv` - we should be able to speed this up with `mmap` (see doubleword criu fork). `preadv` iteratively reads through batches of pages and stages the result in kernel space -> user space (copy required), adding significant overhead per call. `mmap` instead allows us to map the pages directly into the process's virtual address space (zero copy).

criu restore time is bounded by how fast we can read the pages-img file from disk into the restored process memory.
To force slow case, clear page cache (`criu restore` takes **20.4s**): 
```
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
./run.sh 2 
```
Force prefetch image into page cache to get fast case (**10.7s**):
```
vmtouch -t experiments/[DUMP DIR]/pages-*.img
./run.sh 2
```

Switching to [doubleword criu fork zero-copy branch](https://github.com/doublewordai/criu/tree/warmstart/zero-copy-restore) (`restorer.c` rewritten to use `mmap` instead of `preadv`) reduces `criu restore` to **8.1s** when warm (image pages already in page cache):
```
docker build \
  --build-arg CRIU_REPO=https://github.com/doublewordai/criu \
  --build-arg CRIU_REF=warmstart/zero-copy-restore .
```

We can overlap `vmtouch` (to move `*.img` into page cache) with container load, so the cache is warmed when `criu` loads the pages.

#### Experiment 3 - criu, vllm sleep level 2
All else applies as above, but capture vLLM at sleep level 2.

Using sleep level 2 reduced the checkpoint size from 11.4gb to 3.4gb. `criu restore` now takes **~2.5-3s** (with `vmtouch` overlapped), and reloading weights takes **~6-8s** for a total of **~10-11s**.

#### Experiment 4 - as above, but custom weight reloader
See [`exp/fast-weights`](https://github.com/j9smith/vllm/tree/exp/fast-weights) vLLM fork (particularly [ad13a45](https://github.com/j9smith/vllm/commit/ad13a4535a8fc037c8d823cdf2c119099c1ecaf7)).

Firstly serialise the loaded weights onto disk (via `_dump_weights`) before sleep + checkpoint, then on restore reload them via a parallelised double-buffered weight loader (`reload_weights_fast`).

Reloading the checkpoint via vLLM's path (`reload_weights`) takes **~8s** - reloading via `reload_weights_fast` takes **~2.5-3s**. With `criu restore` taking ~2.5-3s, we now have a new reload time of **~6s**.

#### Progress
- ~~Exploring how to release pinned anonymous memory (~8.5gb in /dev/zero); current hypothesis is that it is PyTorch allocator/CUDA -- possibly maintaining weight buffer alloc even after weights are released from device memory. `criu` dumps this memory, inflating checkpoint sizes. If we could decrease checkpoint size, restore should be faster.~~ -- vLLM was in sleep mode 1 (not 2), and so the weights were being stored in the checkpoint. Rectifying this reduced from **16s** to **11s**.
- ~~`criu restore` dropped to ~3s. Weight loading taking up to 8s. Need to address this bottleneck.~~ Implemented new weight loader, `reload_weights_fast`; reduced weight load from ~6-8s -> ~2.5-3s.