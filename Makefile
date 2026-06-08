.PHONY: run
run:
	third_party/gem5/build/RISCV/gem5.opt gem5_config.py --binary sqlite_bench --l1i-size 64kB

.PHONY: build_gem5
build_gem5:
	cd third_party/gem5
	echo "" | scons build/RISCV/gem5.opt -j$(nproc)

.PHONY: build_sqlite
build_sqlite:
	cd third_party/sqlite-amalgamation
	riscv64-linux-gnu-gcc -static -O2 -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION -I third_party/sqlite-amalgamation -o sqlite_bench sqlite_bench.c third_party/sqlite-amalgamation/sqlite3.c -lm
