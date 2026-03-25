/* /life_mpi.c
   MPI Game of Life (row-wise 1D decomposition) with ghost cells + non-blocking halos.

   Build:
     mpicc -O3 -Wall -Wextra -std=c11 -o life_mpi life_mpi.c

   Run:
     mpirun -np <P> ./life_mpi <N> <max_iters> <num_procs_expected> <output_dir>

   Output:
     Writes final board to: <output_dir>/life_final_N<N>_P<P>.txt
*/
/* mpi_life1.c
   MPI Game of Life (row-wise distribution) with ghost cells + non-blocking halo exchange.
   FIX: removes drand48/srand48 to avoid implicit-declaration runtime crashes (SIGBUS).
   Uses portable SplitMix64 RNG for deterministic initialization.

   Build:
     mpicc -O3 -Wall -Wextra -std=c11 -o life_mpi mpi_life1.c

   Run:
     mpiexec -n P ./life_mpi N max_iters P out_dir
*/

#include <mpi.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIES  0u
#define ALIVE 1u

static uint8_t **alloc_u8_2d(int rows, int cols) {
    uint8_t *p = (uint8_t *)malloc((size_t)rows * (size_t)cols * sizeof(uint8_t));
    uint8_t **a = (uint8_t **)malloc((size_t)rows * sizeof(uint8_t *));
    if (!p || !a) {
        free(p);
        free(a);
        return NULL;
    }
    for (int i = 0; i < rows; i++) a[i] = &p[(size_t)i * (size_t)cols];
    return a;
}

static void free_u8_2d(uint8_t **a) {
    if (!a) return;
    free(&a[0][0]);
    free(a);
}

static void fill_ghost_cols(uint8_t **grid, int local_rows, int N) {
    for (int i = 0; i < local_rows + 2; i++) {
        grid[i][0] = DIES;
        grid[i][N + 1] = DIES;
    }
}

static void fill_row_u8(uint8_t *row, int len, uint8_t value) {
    for (int i = 0; i < len; i++) row[i] = value;
}

static void decompose_rows(int N, int size, int rank, int *local_rows, int *start_row_global) {
    int base = N / size;
    int rem  = N % size;
    *local_rows = base + (rank < rem ? 1 : 0);
    *start_row_global = rank * base + (rank < rem ? rank : rem);
}

/* SplitMix64: small, fast, deterministic RNG (portable). */
static inline uint64_t splitmix64_next(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* Deterministic init: for each global row i, seed = (base_seed ^ (i+1)). */
static void init_global_board(uint8_t *global, int N) {
    const uint64_t base_seed = 0xD0C0A1B2C3D4E5F6ULL;
    for (int i = 0; i < N; i++) {
        uint64_t state = base_seed ^ (uint64_t)(i + 1);
        for (int j = 0; j < N; j++) {
            uint64_t r = splitmix64_next(&state);
            global[(size_t)i * (size_t)N + (size_t)j] = (uint8_t)((r & 1ULL) ? ALIVE : DIES);
        }
    }
}

/* Compute next state for owned rows [1..local_rows], cols [1..N].
   Returns count of changed cells locally and sets alive count. */
static int compute_local(uint8_t **life, uint8_t **temp, int local_rows, int N, int *alive_local) {
    int changed = 0;
    int alive = 0;

    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= N; j++) {
            int neighbors =
                life[i - 1][j - 1] + life[i - 1][j] + life[i - 1][j + 1] +
                life[i][j - 1]     +                 life[i][j + 1] +
                life[i + 1][j - 1] + life[i + 1][j] + life[i + 1][j + 1];

            uint8_t cur = life[i][j];
            uint8_t nxt;
            if (cur == ALIVE) {
                nxt = (neighbors < 2 || neighbors > 3) ? DIES : ALIVE;
            } else {
                nxt = (neighbors == 3) ? ALIVE : DIES;
            }

            temp[i][j] = nxt;
            changed += (nxt != cur);
            alive += nxt;
        }
    }

    *alive_local = alive;
    return changed;
}

static void write_final_board(const char *out_dir, int N, int P, const uint8_t *global_board) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/life_final_N%d_P%d.txt", out_dir, N, P);

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open output file '%s': %s\n", path, strerror(errno));
        return;
    }

    for (int i = 0; i < N; i++) {
        const uint8_t *row = global_board + (size_t)i * (size_t)N;
        for (int j = 0; j < N; j++) {
            fprintf(fp, "%u%c", (unsigned)row[j], (j == N - 1) ? '\n' : ' ');
        }
    }

    fclose(fp);
    printf("Wrote final board to %s\n", path);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <N> <max_iters> <num_procs_expected> <output_dir>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const int N = atoi(argv[1]);
    const int max_iters = atoi(argv[2]);
    const int expectedP = atoi(argv[3]);
    const char *out_dir = argv[4];

    if (N <= 0 || max_iters < 0) {
        if (rank == 0) fprintf(stderr, "ERROR: N must be > 0 and max_iters must be >= 0\n");
        MPI_Finalize();
        return 1;
    }

    if (expectedP != size) {
        if (rank == 0) fprintf(stderr, "ERROR: expected %d processes but MPI launched %d\n", expectedP, size);
        MPI_Finalize();
        return 1;
    }

    int local_rows = 0, start_row_global = 0;
    decompose_rows(N, size, rank, &local_rows, &start_row_global);

    uint8_t **life = alloc_u8_2d(local_rows + 2, N + 2);
    uint8_t **temp = alloc_u8_2d(local_rows + 2, N + 2);
    if (!life || !temp) {
        fprintf(stderr, "Rank %d: ERROR allocating arrays\n", rank);
        free_u8_2d(life);
        free_u8_2d(temp);
        MPI_Finalize();
        return 1;
    }

    for (int i = 0; i < local_rows + 2; i++) {
        for (int j = 0; j < N + 2; j++) {
            life[i][j] = DIES;
            temp[i][j] = DIES;
        }
    }
    fill_ghost_cols(life, local_rows, N);
    fill_ghost_cols(temp, local_rows, N);

    /* Scatterv metadata on root only. */
    int *sendcounts = NULL, *displs = NULL;
    uint8_t *global_init = NULL;

    if (rank == 0) {
        sendcounts = (int *)malloc((size_t)size * sizeof(int));
        displs = (int *)malloc((size_t)size * sizeof(int));
        global_init = (uint8_t *)malloc((size_t)N * (size_t)N * sizeof(uint8_t));

        if (!sendcounts || !displs || !global_init) {
            fprintf(stderr, "Rank 0: ERROR allocating init buffers\n");
            free(sendcounts);
            free(displs);
            free(global_init);
            free_u8_2d(life);
            free_u8_2d(temp);
            MPI_Finalize();
            return 1;
        }

        int disp = 0;
        for (int r = 0; r < size; r++) {
            int lr = 0, sr = 0;
            decompose_rows(N, size, r, &lr, &sr);
            (void)sr;
            sendcounts[r] = lr * N;
            displs[r] = disp;
            disp += sendcounts[r];
        }

        init_global_board(global_init, N);
    }

    uint8_t *recvbuf = (uint8_t *)malloc((size_t)local_rows * (size_t)N * sizeof(uint8_t));
    if (!recvbuf) {
        fprintf(stderr, "Rank %d: ERROR allocating recvbuf\n", rank);
        free(sendcounts);
        free(displs);
        free(global_init);
        free_u8_2d(life);
        free_u8_2d(temp);
        MPI_Finalize();
        return 1;
    }

    MPI_Scatterv(global_init, sendcounts, displs, MPI_UNSIGNED_CHAR,
                 recvbuf, local_rows * N, MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; i++) {
        memcpy(&life[i + 1][1], &recvbuf[(size_t)i * (size_t)N], (size_t)N);
    }

    free(global_init);
    free(sendcounts);
    free(displs);

    const int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    if (up == MPI_PROC_NULL) fill_row_u8(&life[0][0], N + 2, DIES);
    if (down == MPI_PROC_NULL) fill_row_u8(&life[local_rows + 1][0], N + 2, DIES);

    double t1 = MPI_Wtime();

    int global_changed = 1;
    int global_alive = 0;
    int iters_done = 0;

    for (int k = 0; k < max_iters; k++) {
        MPI_Request reqs[4];
        int rc = 0;

        if (up != MPI_PROC_NULL) {
            MPI_Irecv(&life[0][1], N, MPI_UNSIGNED_CHAR, up,   200, MPI_COMM_WORLD, &reqs[rc++]);
            MPI_Isend(&life[1][1], N, MPI_UNSIGNED_CHAR, up,   201, MPI_COMM_WORLD, &reqs[rc++]);
        } else {
            fill_row_u8(&life[0][0], N + 2, DIES);
        }

        if (down != MPI_PROC_NULL) {
            MPI_Irecv(&life[local_rows + 1][1], N, MPI_UNSIGNED_CHAR, down, 201, MPI_COMM_WORLD, &reqs[rc++]);
            MPI_Isend(&life[local_rows][1], N, MPI_UNSIGNED_CHAR, down, 200, MPI_COMM_WORLD, &reqs[rc++]);
        } else {
            fill_row_u8(&life[local_rows + 1][0], N + 2, DIES);
        }

        if (rc > 0) MPI_Waitall(rc, reqs, MPI_STATUSES_IGNORE);

        int alive_local = 0;
        int changed_local = compute_local(life, temp, local_rows, N, &alive_local);

        fill_ghost_cols(temp, local_rows, N);

        uint8_t **swap = life;
        life = temp;
        temp = swap;

        MPI_Allreduce(&changed_local, &global_changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&alive_local, &global_alive, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        iters_done = k + 1;
        if (global_changed == 0) {
            if (rank == 0) printf("Exiting after iteration %d (no change).\n", iters_done);
            break;
        }
    }

    double t2 = MPI_Wtime();
    if (rank == 0) {
        printf("Time taken %f seconds for %d iterations, cells alive = %d\n",
               (t2 - t1), iters_done, global_alive);
    }

    /* Gather final board on rank 0 */
    int *recvcounts = NULL, *rdispls = NULL;
    uint8_t *global_final = NULL;

    if (rank == 0) {
        recvcounts = (int *)malloc((size_t)size * sizeof(int));
        rdispls = (int *)malloc((size_t)size * sizeof(int));
        global_final = (uint8_t *)malloc((size_t)N * (size_t)N * sizeof(uint8_t));

        if (!recvcounts || !rdispls || !global_final) {
            fprintf(stderr, "Rank 0: ERROR allocating gather buffers\n");
            free(recvcounts);
            free(rdispls);
            free(global_final);
            recvcounts = NULL;
            rdispls = NULL;
            global_final = NULL;
        } else {
            int disp = 0;
            for (int r = 0; r < size; r++) {
                int lr = 0, sr = 0;
                decompose_rows(N, size, r, &lr, &sr);
                (void)sr;
                recvcounts[r] = lr * N;
                rdispls[r] = disp;
                disp += recvcounts[r];
            }
        }
    }

    for (int i = 0; i < local_rows; i++) {
        memcpy(&recvbuf[(size_t)i * (size_t)N], &life[i + 1][1], (size_t)N);
    }

    MPI_Gatherv(recvbuf, local_rows * N, MPI_UNSIGNED_CHAR,
                global_final, recvcounts, rdispls, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    if (rank == 0 && global_final) {
        write_final_board(out_dir, N, size, global_final);
    }

    free(global_final);
    free(recvcounts);
    free(rdispls);
    free(recvbuf);
    free_u8_2d(life);
    free_u8_2d(temp);

    MPI_Finalize();
    return 0;
}