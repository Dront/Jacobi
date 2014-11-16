#include <iostream>
#include <string.h>
#include <cstdlib>
#include <cmath>
#include <mpi/mpi.h>

#define ROOT (0)
#define ERROR 0.1e-20
#define MAX_ITERATIONS 1000

static int M_SIZE;
static double * trueX;

using namespace std;

void printLine(){
    cout << "########################################" << endl;
    cout.flush();
}

double *newVector(const int size){
    return new double[size];
}

void writeVector(double *vector, const int size){
    for (int i = 0; i < size; ++i){
        cout << vector[i] << ' ';
    }
    cout << endl;
    cout.flush();
}

double *initVector(double *matrix, const int size){
    double * vec = new double[size];

    for (int i = 0; i < size; ++i){
        vec[i] = matrix[i * (size + 1) + size] / matrix[i * (size + 1) + i];
    }

    for (int i = 0; i < size; ++i){
        vec[i] = 0;
    }

    return vec;
}

double *newMatrix(const int n, const int m){
    return new double[n * m];
}

void initMatrix(double *matrix, const int n, const int m){
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            matrix[i * m + j] = (rand() / (double) RAND_MAX - 0.5) * 10;
        }
    }

    trueX = newVector(n);
    for (int i = 0; i < n; ++i){
        trueX[i] = n - i;
    }

    for (int i = 0; i < n; ++i){
        matrix[i * m + i] = (rand() / (double) RAND_MAX) * M_SIZE * 100;

        double elemX = 0;
        for (int j = 0; j < n; ++j){
            elemX += matrix[i * m + j] * trueX[j];
        }

        matrix[i * m + m - 1] = elemX;
    }
}

void writeMatrix(double *matrix, const int n, const int m){
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < m; ++j){
            cout << matrix[i * m + j] << ' ';
        }
        cout << endl;
    }
    cout.flush();
}

double countError(double * x1, double * x2, const int size){
    double maxError = 0;
    for (int i = 0; i < size; ++i) {
        double error = fabs(x1[i] - x2[i]);
        if (maxError < error){
            maxError = error;
        }
    }

    return maxError;
}

void jacobiIteration(double *A, double *x, double *x_old, int n_part, int n, int first){
    double sum;
    for (int i = 0; i < n_part; ++i) {
        sum = 0;
        for (int j = 0; j < i + first; ++j) {
            sum += A[i * (n + 1) + j] * x_old[j];
        }
        for (int j = i + first + 1; j < n; ++j) {
            sum += A[i * (n + 1) + j] * x_old[j];
        }
        x[i + first] = (A[i * (n + 1) + n] - sum) / A[i * (n + 1) + i + first];
    }
}

int jacobiSolve(double *A, double *x, int processNum, int rowCount, int processCount){
    int first;
    MPI_Scan(&rowCount, &first, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    first -= rowCount;

    int * displs = new int[processCount];
    int * sendcnts = new int[processCount];
    MPI_Allgather(&rowCount, 1, MPI_INT, sendcnts, 1, MPI_INT, MPI_COMM_WORLD);
    displs[0] = 0;
    for (int i = 1; i< processCount; ++i){
        displs[i] = displs[i-1] + sendcnts[i-1];
    }

    double * x_old = newVector(M_SIZE);
    //double localError;
    double globalError;
    int iterationsCount = 0;
    do {
        ++iterationsCount;
        memcpy(x_old, x, sizeof(double) * M_SIZE);
        jacobiIteration(A, x, x_old, rowCount, M_SIZE, first);

        //localError = countError(x + first, x_old + first, rowCount);

        //MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allgatherv(x + first, rowCount, MPI_DOUBLE, x, sendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        if (processNum == ROOT) {
            globalError = countError(x, x_old, M_SIZE);
        }
        MPI_Bcast(&globalError, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    } while (ERROR < globalError && iterationsCount < MAX_ITERATIONS);

    delete[] x_old;

    cout << "Process #" << processNum << ". Iterations: " << iterationsCount << endl;
    cout.flush();

    MPI_Allgatherv(x + first, rowCount, MPI_DOUBLE, x, sendcnts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    return iterationsCount;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int processNum, processCount;
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processNum);

    double *A = NULL, *x;

    if (processNum == ROOT) {
        cout << "Enter matrix size: ";
        cin >> M_SIZE;
        MPI_Bcast(&M_SIZE, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
        A = newMatrix(M_SIZE, M_SIZE + 1);
        initMatrix(A, M_SIZE, M_SIZE + 1);
        x = initVector(A, M_SIZE);

        printLine();
        cout << "Initial matrix: " << endl;
        //writeMatrix(A, M_SIZE, M_SIZE + 1);
        cout << "Initial vector x: " << endl;
        //writeVector(x, M_SIZE);
    } else {
        MPI_Bcast(&M_SIZE, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
        x = new double[M_SIZE];
    }

    MPI_Bcast(x, M_SIZE, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    int rowCount = (M_SIZE / processCount) + (M_SIZE % processCount > processNum ? 1 : 0);
    double * A_part = newMatrix(rowCount, M_SIZE + 1);

    int * sendcnts = new int[processCount];
    int part_size = (M_SIZE + 1) * rowCount;
    MPI_Gather(&part_size, 1, MPI_INT, sendcnts, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    int * displs = new int[processCount];
    displs[0] = 0;
    for (int i = 1; i < processCount; i++){
        displs[i] = displs[i - 1] + sendcnts[i - 1];
    }

    double time_start = MPI_Wtime();
    MPI_Scatterv(A, sendcnts, displs, MPI_DOUBLE, A_part, part_size, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    double time_end = MPI_Wtime();

    delete[] sendcnts;
    delete[] displs;

    if (processNum == ROOT){
        printLine();
        cout << "Time used to send matrix: " << (time_end - time_start) << endl;
        delete[] A;
    }

    //printf("#%i: ", processNum);
    //writeMatrix(A_part, rowCount, M_SIZE+1);

    time_start = MPI_Wtime();
    int iterationsCount = jacobiSolve(A_part, x, processNum, rowCount, processCount);
    time_end = MPI_Wtime();

    int maxIter;
    MPI_Reduce(&iterationsCount, &maxIter, 1, MPI_INT, MPI_MAX, ROOT, MPI_COMM_WORLD);

    if (processNum == ROOT) {
        printLine();

        if (maxIter == MAX_ITERATIONS){
            cout << "Reached limit of iterations (" << MAX_ITERATIONS << ")." << endl;
        } else {
            cout << "X: ";
            writeVector(x, M_SIZE);
            cout << "Iterations: " << iterationsCount << endl;
            double err = countError(x, trueX, M_SIZE);
            cout << "Max error: " << err << endl;
        }
        cout << "Time used: " << time_end - time_start << "seconds" << endl;
        printLine();
        cout.flush();
    }

    delete[] x;
    delete[] A_part;

    MPI_Finalize();
    return 0;
}