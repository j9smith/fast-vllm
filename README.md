#### Experiment 1 - cache mounting
Simply remount the caches (`~/.cache/vllm/* ~/.triton/* ~/.cache/flashinfer/* ~/.nv/ComputeCache`) generated during vllm init.
Cold start, weights on disk, all caches cleared: **81 seconds**
Warm start, weights in RAM, caches filled: **38 seconds**

#### Experiment 2 - criu, vllm sleep level 1
Capture vLLM at sleep level 1. 
Necessary criu flags:
- `--enable-external-masters`
- `--tcp-established`
- `--shell-job`
- `--link-remap`

Checkpointing fails on io_uring -- run `sudo sysctl kernel.io_uring_disabled=2`

Checkpoint + restore successful. However, takes 15-30s to complete (first response). Disk constrained transfer (criu loads via `preadv` - we should be able to speed this up with `mmap` (see doubleword criu fork)). 

criu restore time is bounded by how fast we can read the pages-img file from disk into the restored process memory.
To force slow case, clear page cache (20.4s): 
```
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
./run.sh 2 
```
Force prefetch to get fast case (10.7s):
```
vmtouch -t experiments/2/checkpoints/dump-1779184471/pages-*.img
./run.sh 2  # should be fast
```

#### Experiment 3 - criu, vllm sleep level 2
All else applies as above, but capture vLLM at sleep level 2.