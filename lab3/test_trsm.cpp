#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "cblas.h"

static const int BS = 128;

void my_trsm(int n, int m,
double alpha,
const double* A, int lda,
double* B, int ldb)
{
for (int i = 0; i < n; i += BS) {
int ib = std::min(BS, n - i);

    for (int ii = 0; ii < ib; ii++) {
        int row = i + ii;

        for (int j = 0; j < m; j++) {
            double sum = B[row*m + j];

            for (int k = i; k < row; k++)
                sum -= A[row*n + k] * B[k*m + j];

            B[row*m + j] = alpha * sum / A[row*n + row];
        }
    }

    int rest = n - i - ib;
    if (rest > 0) {
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            rest, m, ib,
            -1.0,
            A + (i+ib)*n + i, n,
            B + i*m, m,
            1.0,
            B + (i+ib)*m, m
        );
    }
}

}

using Clock=std::chrono::high_resolution_clock;
using Sec=std::chrono::duration<double>;

static void fill(double*p,int n){
for(int i=0;i<n;i++) p[i]=(double)rand()/RAND_MAX;
}

static double geomean(double*v,int n){
double s=0;
for(int i=0;i<n;i++) s+=log(v[i]);
return exp(s/n);
}

int main(){
int N=512;
int sz=N*N;


double *A=(double*)aligned_alloc(64,sz*sizeof(double));
double *B=(double*)aligned_alloc(64,sz*sizeof(double));
double *Bo=(double*)aligned_alloc(64,sz*sizeof(double));

for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
        A[i*N+j]=(j<=i)?((double)rand()/RAND_MAX+(i==j?5:0)):0;

fill(B,sz);

const int R=6;
double ob[R];

for(int r=0;r<R;r++){
    memcpy(Bo,B,sz*sizeof(double));
    auto t0=Clock::now();
    cblas_dtrsm(CblasRowMajor,CblasLeft,CblasLower,
                CblasNoTrans,CblasNonUnit,
                N,N,1.0,A,N,Bo,N);
    ob[r]=Sec(Clock::now()-t0).count();
}

double ogm=geomean(ob,R);
printf("\n TRSM N=512 \nOpenBLAS=%.4f\n\n",ogm);

double rel[R];
for(int r=0;r<R;r++){
    memcpy(Bo,B,sz*sizeof(double));
    auto t0=Clock::now();
    my_trsm(N,N,1.0,A,N,Bo,N);
    double tt=Sec(Clock::now()-t0).count();
    rel[r]=ogm/tt*100.0;
    printf("run=%d %.4f %.1f%%\n",r+1,tt,rel[r]);
}

printf(">> gm=%.1f%%\n\n",geomean(rel,R));


}
