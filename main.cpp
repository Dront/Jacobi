#include <iostream>
#include <string.h>
#include <cmath>
#include <cstdlib>
#include <mpi/mpi.h>

#define ROOT (0)

using namespace std;

double * new_vector(const int size){
    return new double[size];
}

double * new_matrix(const int n, const int m){
    return new double[n * m];
}

void read_matrix(double * matrix, const int n, const int m){
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            matrix[i * m + j] = 0;
        }
    }
    for (int i = 0; i < n; ++i){
        matrix[i * m + i] = 1;
        matrix[i * m + m - 1] = i + 1;
    }
}

void write_matrix(double * matrix, const int n, const int m){
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < m; ++j){
            cout << matrix[i * m + j] << ' ';
        }
        cout << endl;
    }
    cout.flush();
}

void write_vector(double * vector, const int size){
    for (int i = 0; i < size; ++i){
        cout << vector[i] << ' ';
    }
    cout << endl;
    cout.flush();
}

void jacobi_iter(double *A, double *x, double *x_old, int n_part, int n, int first){
    int i, j;
    double sum;
    for (i=0; i<n_part; i++) {
        sum = 0;
        for (j=0; j<i+first; j++) {
            sum += A[i*(n+1)+j] * x_old[j];
        }
        for (j=i+first+1; j<n; j++) {
            sum += A[i*(n+1)+j] * x_old[j];
        }
        x[i+first] = (A[i*(n+1)+n] - sum) / A[i*(n+1)+i+first];
    }
}

int jacobi_solve(double *A, double *x, double e, int n, int myid, int n_part, int numprocs){
    double *x_old;
    int i, iter = 0, first;
    double d_norm, d_val;
    int *sendcnts, *displs;
    displs = (int*)malloc(numprocs*sizeof(int));
    sendcnts = (int*)malloc(numprocs*sizeof(int));
    MPI_Scan(&n_part, &first, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    first -= n_part;
    MPI_Allgather(&n_part, 1, MPI_INT, sendcnts, 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (i=1; i<numprocs; i++) displs[i] = displs[i-1] + sendcnts[i-1];
    x_old = new_vector(n);
    do {
        iter++;
        memcpy(x_old, x, sizeof(double)*n);
        jacobi_iter(A, x, x_old, n_part, n, first);
        MPI_Allgatherv(x+first, n_part, MPI_DOUBLE, x, sendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        d_norm = 0;
        if (myid == ROOT) {
            for (i=0; i<n; i++) {
                d_val = fabs(x[i] - x_old[i]);
                if (d_norm < d_val) d_norm = d_val;
            }
        }
        MPI_Bcast(&d_norm, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    } while (e < d_norm);
    free(x_old);
    return iter;
}

int main(int argc, char **argv) {
    double *A = 0, *A_part, *x;
    int i, n, n_part, part_size, iter, myid, numprocs;
    double error = 0.1e-10;
    int *sendcnts, *displs;
    double t_start, t_end;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int namelen;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(processor_name, &namelen);
    printf("process %i on %s\n", myid, processor_name);
    n = 10;
    MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&error, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    if (myid == ROOT) {
        A = new_matrix(n, n+1);
        read_matrix(A, n, n+1);
    }
    cout << "lol!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    write_matrix(A, n, n+1);
    cout << "lol!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    x = new_vector(n);
    MPI_Bcast(x, n, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    n_part = (n/numprocs) + (n%numprocs > myid ? 1 : 0);
//
    printf("process: %i; num of rows: %i\n", myid, n_part);
    A_part = new_matrix(n_part, n+1);
    displs = (int*)malloc(numprocs*sizeof(int));
    sendcnts = (int*)malloc(numprocs*sizeof(int));
    part_size = (n+1) * n_part;
    MPI_Gather(&part_size, 1, MPI_INT, sendcnts, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (i=1; i<numprocs; i++) displs[i] = displs[i-1] + sendcnts[i-1];
    t_start = MPI_Wtime();
    MPI_Scatterv(A, sendcnts, displs, MPI_DOUBLE, A_part, part_size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
//
    //printf("#%i: ", myid);
    write_matrix(A_part, n_part, n+1);
    t_end = MPI_Wtime();
    if (myid == ROOT) printf("Time to scatter matrix to all processors: %f sec\n", t_end - t_start);
    t_start = MPI_Wtime();
    iter = jacobi_solve(A_part, x, error, n, myid, n_part, numprocs);
    t_end = MPI_Wtime();
    if (myid == ROOT) printf("Time to solve equation: %f sec\n", t_end - t_start);
    if (myid == ROOT) {
        printf("iter %i\n", iter);
        printf("ans = \n");
        write_vector(x, n);
    }
    if (myid == ROOT) {
        delete[] A;
    }
    free(x);
    delete[] A_part;
    free(sendcnts);
    free(displs);
    MPI_Finalize();
    return 0;
}