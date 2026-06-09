/*
 * sqlite_bench.c — SQLite benchmark for gem5 DSE.
 * Runs all query patterns in sequence to stress different
 * microarchitectural features.
 *
 * Usage: sqlite_bench [N]   (default N=200)
 *
 * Compile:
 *   riscv64-linux-gnu-gcc -static -O2 -DSQLITE_THREADSAFE=0 \
 *       -DSQLITE_OMIT_LOAD_EXTENSION -I third_party/sqlite3 \
 *       -o sqlite_bench sqlite_bench.c third_party/sqlite3/sqlite3.c -lm
 */

#include "sqlite3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long rng_state = 12345;
static unsigned long rng_next(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return rng_state >> 33;
}

static sqlite3 *db;

static void exec(const char *sql) {
    sqlite3_exec(db, sql, NULL, NULL, NULL);
}

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 200;
    sqlite3_stmt *s;

    sqlite3_open(":memory:", &db);

    /* ── Setup: create tables, insert data, build indexes ──────── */
    printf("setup N=%d\n", N);

    exec("CREATE TABLE emp("
         "id INTEGER PRIMARY KEY, dept INT, salary REAL,"
         "rating INT, name TEXT, notes TEXT)");
    exec("CREATE TABLE orders("
         "id INTEGER PRIMARY KEY, emp_id INT, amount REAL,"
         "quarter INT, tag TEXT)");

    exec("BEGIN");
    sqlite3_prepare_v2(db, "INSERT INTO emp VALUES(?,?,?,?,?,?)", -1, &s, 0);
    for (int i = 0; i < N; i++) {
        char name[48], notes[96];
        snprintf(name, sizeof(name), "employee_%06d", i);
        snprintf(notes, sizeof(notes), "region_%d_batch_%d_year_%d",
                 (int)(rng_next()%12), (int)(rng_next()%50), 2010+(int)(rng_next()%15));
        sqlite3_bind_int(s, 1, i);
        sqlite3_bind_int(s, 2, rng_next() % 8);
        sqlite3_bind_double(s, 3, 40000.0 + (rng_next() % 80000));
        sqlite3_bind_int(s, 4, 1 + (rng_next() % 5));
        sqlite3_bind_text(s, 5, name, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(s, 6, notes, -1, SQLITE_TRANSIENT);
        sqlite3_step(s); sqlite3_reset(s);
    }
    sqlite3_finalize(s);

    sqlite3_prepare_v2(db, "INSERT INTO orders VALUES(?,?,?,?,?)", -1, &s, 0);
    for (int i = 0; i < N * 3; i++) {
        char tag[32];
        snprintf(tag, sizeof(tag), "type_%d_cat_%d",
                 (int)(rng_next()%20), (int)(rng_next()%10));
        sqlite3_bind_int(s, 1, i);
        sqlite3_bind_int(s, 2, rng_next() % N);
        sqlite3_bind_double(s, 3, 10.0 + (rng_next() % 10000));
        sqlite3_bind_int(s, 4, 1 + (rng_next() % 4));
        sqlite3_bind_text(s, 5, tag, -1, SQLITE_TRANSIENT);
        sqlite3_step(s); sqlite3_reset(s);
    }
    sqlite3_finalize(s);
    exec("COMMIT");

    exec("CREATE INDEX idx_dept ON emp(dept)");
    exec("CREATE INDEX idx_salary ON emp(salary)");
    exec("CREATE INDEX idx_rating ON emp(rating)");
    exec("CREATE INDEX idx_emp ON orders(emp_id)");
    exec("CREATE INDEX idx_qtr ON orders(quarter)");

    /* ── Point lookups: pointer chasing, ROB depth, L1D ────────── */
    printf("phase: point lookups\n");
    sqlite3_prepare_v2(db,
        "SELECT salary, name FROM emp WHERE id = ?", -1, &s, 0);
    volatile double sink = 0;
    for (int i = 0; i < N; i++) {
        sqlite3_bind_int(s, 1, rng_next() % N);
        if (sqlite3_step(s) == SQLITE_ROW)
            sink += sqlite3_column_double(s, 0);
        sqlite3_reset(s);
    }
    sqlite3_finalize(s);

    /* ── Sequential scan: streaming access, prefetch, bandwidth ── */
    printf("phase: scan\n");
    sqlite3_prepare_v2(db,
        "SELECT sum(salary), avg(rating), count(*) FROM emp "
        "WHERE salary BETWEEN 50000 AND 100000", -1, &s, 0);
    sqlite3_step(s);
    sqlite3_finalize(s);

    /* ── JOIN: large working set, L2, LSQ depth ────────────────── */
    printf("phase: join\n");
    sqlite3_prepare_v2(db,
        "SELECT e.dept, sum(o.amount), count(*), avg(e.salary) "
        "FROM emp e JOIN orders o ON e.id = o.emp_id "
        "WHERE o.quarter IN (1,2) AND e.rating >= 3 "
        "GROUP BY e.dept ORDER BY sum(o.amount) DESC", -1, &s, 0);
    while (sqlite3_step(s) == SQLITE_ROW) {}
    sqlite3_finalize(s);

    /* ── Aggregation: hash collisions, L1D associativity ───────── */
    printf("phase: agg\n");
    sqlite3_prepare_v2(db,
        "SELECT dept, rating, count(*), avg(salary), min(salary), max(salary) "
        "FROM emp GROUP BY dept, rating ORDER BY avg(salary) DESC", -1, &s, 0);
    while (sqlite3_step(s) == SQLITE_ROW) {}
    sqlite3_finalize(s);

    /* ── String matching: branch-heavy, BP type, issue width ───── */
    printf("phase: like\n");
    const char *patterns[] = {
        "%region_3%", "%batch_1%", "%year_2020%",
        "%region_7%batch_4%", "%region_1%year_201%"
    };
    for (int p = 0; p < 5; p++) {
        sqlite3_prepare_v2(db,
            "SELECT count(*) FROM emp WHERE notes LIKE ?", -1, &s, 0);
        sqlite3_bind_text(s, 1, patterns[p], -1, SQLITE_STATIC);
        sqlite3_step(s);
        sqlite3_finalize(s);
    }
    sqlite3_prepare_v2(db,
        "SELECT count(*) FROM orders WHERE tag LIKE '%type_1%cat_5%'",
        -1, &s, 0);
    sqlite3_step(s);
    sqlite3_finalize(s);

    /* ── Nested subqueries: deep call stack, I-cache, RAS ──────── */
    printf("phase: nest\n");
    sqlite3_prepare_v2(db,
        "SELECT count(*) FROM emp e1 "
        "WHERE e1.salary > ("
        "  SELECT avg(e2.salary) FROM emp e2 WHERE e2.dept = e1.dept"
        ")", -1, &s, 0);
    sqlite3_step(s);
    sqlite3_finalize(s);

    sqlite3_prepare_v2(db,
        "SELECT dept, top_sal, total FROM ("
        "  SELECT e.dept, max(e.salary) as top_sal, sum(o.amount) as total "
        "  FROM emp e JOIN orders o ON e.id = o.emp_id GROUP BY e.dept"
        ") WHERE total > (SELECT avg(amount)*10 FROM orders) "
        "ORDER BY top_sal DESC", -1, &s, 0);
    while (sqlite3_step(s) == SQLITE_ROW) {}
    sqlite3_finalize(s);

    /* ── Bulk insert: store buffer, SQ entries, write-back ──────── */
    printf("phase: insert\n");
    exec("BEGIN");
    sqlite3_prepare_v2(db,
        "INSERT INTO emp VALUES(?,?,?,?,?,?)", -1, &s, 0);
    for (int i = 0; i < N; i++) {
        char name[48];
        snprintf(name, sizeof(name), "new_%06d", N + i);
        sqlite3_bind_int(s, 1, N + i);
        sqlite3_bind_int(s, 2, rng_next() % 8);
        sqlite3_bind_double(s, 3, 40000.0 + (rng_next() % 80000));
        sqlite3_bind_int(s, 4, 1 + (rng_next() % 5));
        sqlite3_bind_text(s, 5, name, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(s, 6, "bulk_row", -1, SQLITE_STATIC);
        sqlite3_step(s); sqlite3_reset(s);
    }
    sqlite3_finalize(s);
    exec("COMMIT");

    sqlite3_close(db);
    printf("done sink=%f\n", (double)sink);
    return 0;
}
