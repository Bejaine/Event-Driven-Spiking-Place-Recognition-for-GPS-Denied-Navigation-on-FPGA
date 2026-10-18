# Environment, Jetson Orin Nano (Phase A) — recorded 2026-10-19T05:19:52+05:30

## System
```
Linux ubuntu 5.15.199-tegra #1 SMP PREEMPT Thu Jul 16 11:27:38 PDT 2026 aarch64 aarch64 aarch64 GNU/Linux
PRETTY_NAME="Ubuntu 22.04.5 LTS"
# R36 (release), REVISION: 5.2, GCID: 46426093, BOARD: generic, EABI: aarch64, DATE: Thu Jul 16 18:56:22 UTC 2026
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
nvidia-l4t-core 36.5.2-20260716114719
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
CPU(s):                                  6
Model name:                              Cortex-A78AE
CPU max MHz:                             1728.0000
CPU min MHz:                             115.2000
               total        used        free      shared  buff/cache   available
Mem:           7.4Gi       1.2Gi       1.9Gi        37Mi       4.4Gi       6.0Gi
Swap:          3.7Gi          0B       3.7Gi
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1  116G  105G  4.8G  96% /
sudo -n true: sudo: a password is required
```

## Power mode and clocks
```
NV Power Mode: MAXN_SUPER
2
cpu0 governor=schedutil cur=1190400 min=729600 max=1728000 avail_max=1728000
cpu1 governor=schedutil cur=1190400 min=729600 max=1728000 avail_max=1728000
cpu2 governor=schedutil cur=1190400 min=729600 max=1728000 avail_max=1728000
cpu3 governor=schedutil cur=1267200 min=729600 max=1728000 avail_max=1728000
cpu4 governor=schedutil cur=729600 min=729600 max=1728000 avail_max=1728000
cpu5 governor=schedutil cur=729600 min=729600 max=1728000 avail_max=1728000
gpu governor=nvhost_podgov cur=306000000 min=306000000 max=1020000000
gpu available: 306000000 408000000 510000000 612000000 714000000 816000000 918000000 1020000000
```

## Telemetry (INA3221, unprivileged)
```
hwmon: /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1
ch1 VDD_IN: in=4896 mV curr=1080 mA
ch2 VDD_CPU_GPU_CV: in=4888 mV curr=168 mA
ch3 VDD_SOC: in=4896 mV curr=296 mA
10-19-2026 05:19:52 RAM 1414/7607MB (lfb 21x4MB) SWAP 0/3804MB (cached 0MB) CPU [9%@1036,1%@1036,0%@1036,0%@1036,5%@729,0%@729] GR3D_FREQ 0% cpu@58C soc2@57.343C soc0@57.437C gpu@58.812C tj@59.031C soc1@59.031C VDD_IN 5052mW/5052mW VDD_CPU_GPU_CV 664mW/664mW VDD_SOC 1449mW/1449mW
10-19-2026 05:19:53 RAM 1414/7607MB (lfb 21x4MB) SWAP 0/3804MB (cached 0MB) CPU [2%@729,0%@729,0%@729,3%@729,0%@729,0%@729] GR3D_FREQ 0% cpu@58.125C soc2@57.281C soc0@57.406C gpu@58.312C tj@59C soc1@59C VDD_IN 4935mW/4994mW VDD_CPU_GPU_CV 586mW/625mW VDD_SOC 1410mW/1430mW
cpu-thermal 58250
gpu-thermal 58250
cv0-thermal 
cv1-thermal 
cv2-thermal 
soc0-thermal 57593
soc1-thermal 58937
soc2-thermal 57281
tj-thermal 59093
```

## Toolchain
```
gcc (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0
perf_event_paranoid=2
perf: not installed
valgrind: not installed
python 3.10.12 numpy 2.2.6 brian2 2.9.0
torch 2.11.0 cuda 12.6 available True
```

## Load at capture
```
 05:19:59 up  6:23,  1 user,  load average: 0.47, 0.14, 0.05
```
