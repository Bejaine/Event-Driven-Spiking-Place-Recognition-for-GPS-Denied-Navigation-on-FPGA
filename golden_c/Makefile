# Portable C99 build of the SpikeNav golden model (single-threaded, no intrinsics)
CC ?= gcc
CFLAGS ?= -O3 -std=c99 -Wall -Wextra -pedantic
all: libsnngolden.so snn_golden_bench snn_golden_verify
libsnngolden.so: snn_golden.c snn_golden.h
	$(CC) $(CFLAGS) -fPIC -shared -o $@ snn_golden.c -lm
snn_golden_bench: bench_main.c snn_golden.c snn_golden.h
	$(CC) $(CFLAGS) -o $@ bench_main.c snn_golden.c -lm
snn_golden_verify: verify_main.c snn_golden.c snn_golden.h
	$(CC) $(CFLAGS) -o $@ verify_main.c snn_golden.c -lm
clean:
	rm -f libsnngolden.so snn_golden_bench snn_golden_verify
