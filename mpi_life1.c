/* /life_mpi.c
   MPI Game of Life (row-wise 1D decomposition) with ghost cells + non-blocking halos.

   Build:
     mpicc -O3 -Wall -Wextra -std=c11 -o life_mpi life_mpi.c

   Run:
     mpirun -np <P> ./life_mpi <N> <max_iters> <num_procs_expected> <output_dir>

   Output:
     Writes final board to: <output_dir>/life_final_N<N>_P<P>.txt
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define DIES  0
#define ALIVE 1

static int **allocarray(int rows, int cols) {
    int *p = (int *)malloc((size_t)rows * (size_t)cols * sizeof(int));
    int **a = (int **)malloc((size_t)rows * sizeof(int *));
    if (!p || !a) {
        free(p);
        free(a);
        return NULL;
    }
    for (int i = 0; i < rows; i++) a[i] = &p[i * cols];
    return a;
}

static void freearray(int **a) {
    if (!a) return;
    free(&a[0][0]);
    free(a);
}

static void fill_ghost_cols(int **grid, int local_rows, int N) {
    for (int i = 0; i < local_rows + 2; i++) {
        grid[i][0] = DIES;
        grid[i][N + 1] = DIES;
    }
}

static void fill_row(int *row, int len, int value) {
    for (int i = 0; i < len; i++) row[i] = value;
}

static void decompose_rows(int N, int size, int rank, int *local_rows, int *start_row_global) {
    int base = N / size;
    int rem  = N % size;

    *local_rows = base + (rank < rem ? 1 : 0);
    *start_row_global = rank * base + (rank < rem ? rank : rem);
}

/* Compute next state for owned rows [1..local_rows], cols [1..N].
   Returns number of cells whose value changed in this rank's owned region. */
static int compute_local(int **life, int **temp, int local_rows, int N, int *cellsalive_local) {
    int changed = 0;
    int alive = 0;

    for (int i = 1; i <= local_rows; i++) {
        for (int j = 1; j <= N; j++) {
            int value =
                life[i - 1][j - 1] + life[i - 1][j] + life[i - 1][j + 1] +
                life[i][j - 1]     +                 life[i][j + 1] +
                life[i + 1][j - 1] + life[i + 1][j] + life[i + 1][j + 1];

            int next;
            if (life[i][j] == ALIVE) {
                next = (value < 2 || value > 3) ? DIES : ALIVE;
            } else {
                next = (value == 3) ? ALIVE : DIES;
            }

            temp[i][j] = next;
            if (next != life[i][j]) changed++;
            alive += next;
        }
    }

    *cellsalive_local = alive;
    return changed;
}

static void write_final_board(const char *out_dir, int N, int P, const int *global_board) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/life_final_N%d_P%d.txt", out_dir, N, P);

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: failed to open output file '%s': %s\n", path, strerror(errno));
        return;
    }

    for (int i = 0; i < N; i++) {
        const int *row = global_board + (size_t)i * (size_t)N;
        for (int j = 0; j < N; j++) {
            fprintf(fp, "%d%c", row[j], (j == N - 1) ? '\n' : ' ');
        }
    }

    fclose(fp);
    fprintf(stdout, "Wrote final board to %s\n", path);
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
    const int num_procs_expected = atoi(argv[3]);
    const char *out_dir = argv[4];

    if (N <= 0 || max_iters < 0) {
        if (rank == 0) fprintf(stderr, "ERROR: N must be > 0 and max_iters must be >= 0.\n");
        MPI_Finalize();
        return 1;
    }

    if (num_procs_expected != size) {
        if (rank == 0) {
            fprintf(stderr, "ERROR: expected %d processes, but MPI launched %d.\n",
                    num_procs_expected, size);
        }
        MPI_Finalize();
        return 1;
    }

    int local_rows = 0, start_row_global = 0;
    decompose_rows(N, size, rank, &local_rows, &start_row_global);

    /* Local arrays: (local_rows + 2) x (N + 2) include ghost cells. */
    int **life = allocarray(local_rows + 2, N + 2);
    int **temp = allocarray(local_rows + 2, N + 2);
    if (!life || !temp) {
        fprintf(stderr, "Rank %d: ERROR allocating arrays.\n", rank);
        freearray(life);
        freearray(temp);
        MPI_Finalize();
        return 1;
    }

    /* Initialize all to DIES (including ghosts). */
    for (int i = 0; i < local_rows + 2; i++) {
        for (int j = 0; j < N + 2; j++) {
            life[i][j] = DIES;
            temp[i][j] = DIES;
        }
    }

    /* Build Scatterv/Gatherv counts/displs over N rows. */
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = (int *)malloc((size_t)size * sizeof(int));
        displs     = (int *)malloc((size_t)size * sizeof(int));
        if (!sendcounts || !displs) {
            fprintf(stderr, "Rank 0: ERROR allocating sendcounts/displs.\n");
            free(sendcounts);
            free(displs);
            freearray(life);
            freearray(temp);
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
    }

    /* Root initializes full board (no ghosts) and scatters rows once (outside loop). */
    int *global_init = NULL;
    if (rank == 0) {
        global_init = (int *)malloc((size_t)N * (size_t)N * sizeof(int));
        if (!global_init) {
            fprintf(stderr, "Rank 0: ERROR allocating global_init.\n");
            free(sendcounts);
            free(displs);
            freearray(life);
            freearray(temp);
            MPI_Finalize();
            return 1;
        }

        /* Same deterministic style as your serial code: seed per row index (1..N). */
        for (int i = 0; i < N; i++) {
            srand48((long)(54321 | (i + 1)));
            for (int j = 0; j < N; j++) {
                global_init[(size_t)i * (size_t)N + (size_t)j] = (drand48() < 0.5) ? ALIVE : DIES;
            }
        }
    }

    int *recvbuf = (int *)malloc((size_t)local_rows * (size_t)N * sizeof(int));
    if (!recvbuf) {
        fprintf(stderr, "Rank %d: ERROR allocating recvbuf.\n", rank);
        free(global_init);
        free(sendcounts);
        free(displs);
        freearray(life);
        freearray(temp);
        MPI_Finalize();
        return 1;
    }

    MPI_Scatterv(global_init, sendcounts, displs, MPI_INT,
                 recvbuf, local_rows * N, MPI_INT,
                 0, MPI_COMM_WORLD);

    /* Copy recvbuf into life owned rows [1..local_rows], cols [1..N]. */
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            life[i + 1][j + 1] = recvbuf[(size_t)i * (size_t)N + (size_t)j];
        }
    }
    fill_ghost_cols(life, local_rows, N);
    fill_ghost_cols(temp, local_rows, N);

    free(global_init);
    if (rank == 0) {
        free(sendcounts);
        free(displs);
        sendcounts = NULL;
        displs = NULL;
    }

    const int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    /* If no neighbor, keep ghost row = DIES forever. */
    if (up == MPI_PROC_NULL) fill_row(&life[0][0], N + 2, DIES);
    if (down == MPI_PROC_NULL) fill_row(&life[local_rows + 1][0], N + 2, DIES);

    double t1 = MPI_Wtime();

    int k = 0;
    int global_changed = 1;
    int global_alive = 0;

    for (k = 0; k < max_iters; k++) {
        /* Non-blocking halo exchange: only cols [1..N] are needed. */
        MPI_Request reqs[4];
        int req_count = 0;

        if (up != MPI_PROC_NULL) {
            MPI_Irecv(&life[0][1], N, MPI_INT, up, 200, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(&life[1][1], N, MPI_INT, up, 201, MPI_COMM_WORLD, &reqs[req_count++]);
        } else {
            fill_row(&life[0][0], N + 2, DIES);
        }

        if (down != MPI_PROC_NULL) {
            MPI_Irecv(&life[local_rows + 1][1], N, MPI_INT, down, 201, MPI_COMM_WORLD, &reqs[req_count++]);
            MPI_Isend(&life[local_rows][1], N, MPI_INT, down, 200, MPI_COMM_WORLD, &reqs[req_count++]);
        } else {
            fill_row(&life[local_rows + 1][0], N + 2, DIES);
        }

        if (req_count > 0) MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

        int local_alive = 0;
        int local_changed = compute_local(life, temp, local_rows, N, &local_alive);

        /* Ensure ghost cols remain DIES in temp. */
        fill_ghost_cols(temp, local_rows, N);

        /* Swap grids (no copying). */
        int **ptr = life;
        life = temp;
        temp = ptr;

        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_alive, &global_alive, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (global_changed == 0) {
            if (rank == 0) {
                fprintf(stdout, "Exiting after iteration %d (no change).\n", k + 1);
            }
            k++; /* iterations actually performed */
            break;
        }
    }

    double t2 = MPI_Wtime();

    if (rank == 0) {
        fprintf(stdout, "Time taken %f seconds for %d iterations, cells alive = %d\n",
                t2 - t1, k, global_alive);
    }

    /* Gather final board (outside loop). */
    int *global_final = NULL;
    int *recvcounts = NULL, *rdispls = NULL;

    if (rank == 0) {
        global_final = (int *)malloc((size_t)N * (size_t)N * sizeof(int));
        recvcounts = (int *)malloc((size_t)size * sizeof(int));
        rdispls    = (int *)malloc((size_t)size * sizeof(int));
        if (!global_final || !recvcounts || !rdispls) {
            fprintf(stderr, "Rank 0: ERROR allocating gather buffers.\n");
            free(global_final);
            free(recvcounts);
            free(rdispls);
            global_final = NULL;
            recvcounts = NULL;
            rdispls = NULL;
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

    /* Pack local owned region (no ghosts) into recvbuf for gather. */
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            recvbuf[(size_t)i * (size_t)N + (size_t)j] = life[i + 1][j + 1];
        }
    }

    MPI_Gatherv(recvbuf, local_rows * N, MPI_INT,
                global_final, recvcounts, rdispls, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0 && global_final) {
        write_final_board(out_dir, N, size, global_final);
    }

    free(global_final);
    free(recvcounts);
    free(rdispls);
    free(recvbuf);
    freearray(life);
    freearray(temp);

    MPI_Finalize();
    return 0;
}