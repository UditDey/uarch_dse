/*
 * sqlite_bench.c — Minimal SQLite benchmark for gem5 O3CPU.
 * Keeps working set small, finishes in reasonable sim time.
 *
 * Usage: sqlite_bench [N]   (default N=100)
 */

#include "sqlite3.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned long rng = 12345;
static unsigned long rng_next(void) {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return rng >> 33;
}

static void exec(sqlite3 *db, const char *sql) {
    sqlite3_exec(db, sql, 0, 0, 0);
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 100;
    sqlite3 *db;
    sqlite3_stmt *stmt;

    sqlite3_open(":memory:", &db);
    printf("phase: schema\n");

    exec(db, "CREATE TABLE t(id INTEGER PRIMARY KEY, dept INT, salary REAL, name TEXT)");

    /* Insert */
    printf("phase: insert\n");
    exec(db, "BEGIN");
    sqlite3_prepare_v2(db, "INSERT INTO t VALUES(?,?,?,?)", -1, &stmt, 0);
    for (int i = 0; i < N; i++) {
        char name[32];
        snprintf(name, sizeof(name), "emp_%04d", i);
        sqlite3_bind_int(stmt, 1, i);
        sqlite3_bind_int(stmt, 2, rng_next() % 5);
        sqlite3_bind_double(stmt, 3, 40000.0 + (rng_next() % 60000));
        sqlite3_bind_text(stmt, 4, name, -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);
    exec(db, "COMMIT");

    printf("phase: index\n");
    exec(db, "CREATE INDEX idx_dept ON t(dept)");

    /* Point lookups */
    printf("phase: point lookups\n");
    sqlite3_prepare_v2(db, "SELECT salary FROM t WHERE id=?", -1, &stmt, 0);
    volatile double sink = 0;
    for (int i = 0; i < N; i++) {
        sqlite3_bind_int(stmt, 1, rng_next() % N);
        if (sqlite3_step(stmt) == SQLITE_ROW)
            sink += sqlite3_column_double(stmt, 0);
        sqlite3_reset(stmt);
    }
    sqlite3_finalize(stmt);

    /* Aggregate */
    printf("phase: aggregate\n");
    sqlite3_prepare_v2(db,
        "SELECT dept, avg(salary) FROM t GROUP BY dept", -1, &stmt, 0);
    while (sqlite3_step(stmt) == SQLITE_ROW) {}
    sqlite3_finalize(stmt);

    /* Update */
    printf("phase: update\n");
    exec(db, "UPDATE t SET salary = salary * 1.05 WHERE dept = 0");

    /* String scan */
    printf("phase: string scan\n");
    sqlite3_prepare_v2(db,
        "SELECT count(*) FROM t WHERE name LIKE '%emp_00%'", -1, &stmt, 0);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
    printf("done N=%d sink=%f\n", N, (double)sink);
    return 0;
}
