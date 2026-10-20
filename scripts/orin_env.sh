#!/bin/bash
# Phase A (Jetson Orin Nano): record system, power mode, clocks, telemetry and toolchain
# into reports/environment_orin.md. Unprivileged; uses `sudo -n` only to test for sudo.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/reports/environment_orin.md
HW=$(ls -d /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon* 2>/dev/null | head -1)
sec() { echo; echo "## $1"; echo '```'; }
end() { echo '```'; }
{
echo "# Environment, Jetson Orin Nano (Phase A) — recorded $(date -Iseconds)"
sec System
uname -a
grep PRETTY_NAME /etc/os-release
cat /etc/nv_tegra_release
dpkg -l | grep -E 'nvidia-l4t-core|nvidia-jetpack ' | awk '{print $2, $3}'
cat /proc/device-tree/model 2>/dev/null | tr -d '\0'; echo
lscpu | grep -E 'Model name|^CPU\(s\)|max MHz|min MHz'
free -h
df -h /
echo -n "sudo -n true: "; sudo -n true 2>&1 && echo OK
end
sec "Power mode and clocks"
nvpmodel -q 2>&1
for c in /sys/devices/system/cpu/cpu[0-9]*; do
  echo "$(basename $c) governor=$(cat $c/cpufreq/scaling_governor) cur=$(cat $c/cpufreq/scaling_cur_freq) min=$(cat $c/cpufreq/scaling_min_freq) max=$(cat $c/cpufreq/scaling_max_freq) avail_max=$(cat $c/cpufreq/cpuinfo_max_freq)"
done
G=/sys/class/devfreq/17000000.gpu
echo "gpu governor=$(cat $G/governor) cur=$(cat $G/cur_freq) min=$(cat $G/min_freq) max=$(cat $G/max_freq)"
echo "gpu available: $(cat $G/available_frequencies)"
end
sec "Telemetry (INA3221, unprivileged)"
echo "hwmon: $HW"
for i in 1 2 3; do echo "ch$i $(cat $HW/in${i}_label): in=$(cat $HW/in${i}_input) mV curr=$(cat $HW/curr${i}_input) mA"; done
timeout 2 tegrastats --interval 500 2>&1 | head -2
for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type) $(cat $z/temp)"; done
end
sec Toolchain
gcc --version | head -1
/usr/local/cuda/bin/nvcc --version | tail -2
echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"
echo -n "perf: "; command -v perf || echo "not installed"
echo -n "valgrind: "; command -v valgrind || echo "not installed"
"$ROOT/envs/vprsnn/bin/python" -c "import sys,numpy,brian2;print('python',sys.version.split()[0],'numpy',numpy.__version__,'brian2',brian2.__version__)" 2>&1
"$ROOT/envs/vprsnn/bin/python" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available())" 2>&1 | tail -1
end
sec "Load at capture"
uptime
end
} > "$OUT"
echo "wrote $OUT"
