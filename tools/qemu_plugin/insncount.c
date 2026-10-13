/* Minimal QEMU TCG plugin: counts guest instructions executed (per translated block,
 * inline add of the block's instruction count). Prints "total insns: N" at exit. */
#include <stdio.h>
#include <inttypes.h>
#include "qemu-plugin.h"

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;
static uint64_t insn_count;

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
    size_t n = qemu_plugin_tb_n_insns(tb);
    qemu_plugin_register_vcpu_tb_exec_inline(tb, QEMU_PLUGIN_INLINE_ADD_U64, &insn_count, n);
}

static void at_exit(qemu_plugin_id_t id, void *p) {
    fprintf(stderr, "total insns: %" PRIu64 "\n", insn_count);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc, char **argv) {
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
