# Environment (Phase A) — recorded 2026-10-09T19:27:15+05:30

## System
```
Linux edgexpert-bc70 6.17.0-1021-nvidia #21-Ubuntu SMP PREEMPT_DYNAMIC Wed May 27 19:14:05 UTC 2026 aarch64 aarch64 aarch64 GNU/Linux
PRETTY_NAME="Ubuntu 24.04.4 LTS"

CPU(s):                                  20
Model name:                              Cortex-X925
Core(s) per socket:                      10
CPU max MHz:                             3900.0000
Model name:                              Cortex-A725
Core(s) per socket:                      10
CPU max MHz:                             2808.0000
NUMA node0 CPU(s):                       0-19

Per-CPU part numbers (0xd85 = Cortex-X925, 0xd87 = Cortex-A725):
0xd87 0xd87 0xd87 0xd87 0xd87 0xd85 0xd85 0xd85 0xd85 0xd85 0xd87 0xd87 0xd87 0xd87 0xd87 0xd85 0xd85 0xd85 0xd85 0xd85 
cpu0 max_freq_khz 2808000
cpu5 max_freq_khz 3900000

               total        used        free      shared  buff/cache   available
Mem:           119Gi       6.5Gi       7.9Gi        11Mi       106Gi       113Gi
Swap:           15Gi       209Mi        15Gi

Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  3.7T  1.9T  1.7T  54% /

sudo -n true: sudo: a password is required
```
Core map: Cortex-X925 (part 0xd85, 3.9 GHz max) = CPUs 5-9,15-19; Cortex-A725 (part 0xd87, 2.808 GHz max) = CPUs 0-4,10-14. 20 cores total (10+10) — matches expectation.
Memory: 119 GiB visible to Linux (128 GB unified LPDDR5x expected; part reserved). Passwordless sudo: NOT available (no sudo used).

## GPU
```
Fri Oct  9 19:27:15 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.03             Driver Version: 580.159.03     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GB10                    On  |   0000000F:01:00.0 Off |                  N/A |
| N/A   44C    P8              4W /  N/A  | Not Supported          |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3048      G   /usr/lib/xorg/Xorg                       43MiB |
|    0   N/A  N/A            3194      G   /usr/bin/gnome-shell                     18MiB |
+-----------------------------------------------------------------------------------------+
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0

permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```
Docker 29.2.1 installed but the user cannot access /var/run/docker.sock (not in docker group) → NGC container path unavailable.

## Toolchain
```
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Ubuntu clang version 18.1.3 (1ubuntu1)
perf_event_paranoid=4
Error:
Access to performance monitoring and observability operations is limited.
```
perf: paranoid level 4 → hardware counters unavailable to unprivileged users; instruction counting via perf is NOT possible (see E5 for the fallback).

## Power telemetry (no root)
```
power.draw [W]
4.65 W
/sys/class/hwmon/hwmon0: acpi_fan
/sys/class/hwmon/hwmon1: acpitz
/sys/class/hwmon/hwmon2: nvme
/sys/class/hwmon/hwmon3: mlx5
/sys/class/hwmon/hwmon4: mlx5
/sys/class/hwmon/hwmon5: mlx5
/sys/class/hwmon/hwmon6: mlx5
/sys/class/hwmon/hwmon7: mt7925_phy0
```
Only GPU power via nvidia-smi power.draw; hwmon exposes temperatures/fan only (no CPU/SoC power rails).
