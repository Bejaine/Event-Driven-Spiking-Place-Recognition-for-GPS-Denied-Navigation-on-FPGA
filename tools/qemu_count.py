"""Count guest instructions from `qemu-riscv64 -d in_asm,exec,nochain` output on stdin.

in_asm blocks: "IN: <sym>" then lines "0x<addr>:  <hex>  <insn>"; each block ends with a blank line.
exec lines: "Trace <cpu>: <host ptr> [<flags>/<pc>/<...>] <sym>" -> one execution of TB at pc.
Total = sum over executions of (#instructions of the most recent translation starting at pc).
"""
import re
import sys

tb_len, cur_pc, cur_n = {}, None, 0
total = execs = 0
insn_re = re.compile(r"^0x([0-9a-f]+):\s")
trace_re = re.compile(r"^Trace \d+: 0x[0-9a-f]+ \[[0-9a-f]+/([0-9a-f]+)/")
for line in sys.stdin:
    if line.startswith("Trace"):
        m = trace_re.match(line)
        if m:
            total += tb_len.get(int(m.group(1), 16), 0); execs += 1
        continue
    m = insn_re.match(line)
    if m:
        if cur_pc is None:
            cur_pc = int(m.group(1), 16); cur_n = 0
        cur_n += 1
    elif line.strip() == "" or line.startswith("IN:") or line.startswith("----"):
        if cur_pc is not None:
            tb_len[cur_pc] = cur_n
        cur_pc = None
print(f"total insns: {total} (tb executions {execs}, tbs {len(tb_len)})")
