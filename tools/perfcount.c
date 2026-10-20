/* Minimal `perf stat -e instructions,cycles` replacement for machines without the perf tool.
 * Counts user-space retired instructions and cycles of a child command (and its threads)
 * via perf_event_open; needs perf_event_paranoid <= 2.
 * Build: gcc -O2 -o perfcount perfcount.c     Usage: ./perfcount <cmd> [args...]
 * Prints "instructions N" and "cycles N" to stderr. */
#define _GNU_SOURCE
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

static int open_counter(uint64_t config, pid_t pid) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = PERF_TYPE_HARDWARE;
    a.config = config;
    a.disabled = 1;
    a.enable_on_exec = 1;
    a.inherit = 1;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    return (int)syscall(SYS_perf_event_open, &a, pid, -1, -1, 0);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s cmd [args...]\n", argv[0]); return 2; }
    int go[2];
    if (pipe(go)) { perror("pipe"); return 1; }
    pid_t pid = fork();
    if (pid == 0) {           /* child: wait until counters are attached, then exec */
        char c;
        close(go[1]);
        if (read(go[0], &c, 1) != 1) _exit(127);
        execvp(argv[1], argv + 1);
        perror("execvp");
        _exit(127);
    }
    close(go[0]);
    int fi = open_counter(PERF_COUNT_HW_INSTRUCTIONS, pid);
    int fc = open_counter(PERF_COUNT_HW_CPU_CYCLES, pid);
    if (fi < 0 || fc < 0) perror("perf_event_open");
    if (write(go[1], "x", 1) != 1) perror("write");
    close(go[1]);
    int st;
    waitpid(pid, &st, 0);
    uint64_t ni = 0, nc = 0;
    if (fi >= 0 && read(fi, &ni, sizeof ni) != sizeof ni) ni = 0;
    if (fc >= 0 && read(fc, &nc, sizeof nc) != sizeof nc) nc = 0;
    fprintf(stderr, "instructions %llu\ncycles %llu\n", (unsigned long long)ni, (unsigned long long)nc);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}
