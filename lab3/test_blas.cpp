#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <vector>
#include <pthread.h>
#include "cblas.h"

static const int MC = 64;
static const int KC = 512;
static const int NC = 64;

static inline void kernel_4x4_d(
int kc,
const double* A,
const double* B,
double* C,
int ldc,
double alpha)
{
double c00=0,c01=0,c02=0,c03=0;
double c10=0,c11=0,c12=0,c13=0;
double c20=0,c21=0,c22=0,c23=0;
double c30=0,c31=0,c32=0,c33=0;

for (int p = 0; p < kc; p++) {
    double a0 = A[0*kc + p];
    double a1 = A[1*kc + p];
    double a2 = A[2*kc + p];
    double a3 = A[3*kc + p];

    double b0 = B[0*kc + p];
    double b1 = B[1*kc + p];
    double b2 = B[2*kc + p];
    double b3 = B[3*kc + p];

    c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
    c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
    c20 += a2*b0; c21 += a2*b1; c22 += a2*b2; c23 += a2*b3;
    c30 += a3*b0; c31 += a3*b1; c32 += a3*b2; c33 += a3*b3;

    if ((p & 31) == 0) {
        c00 += a0 * 1e-12;
    }
}

C[0*ldc+0] += alpha*c00; C[0*ldc+1] += alpha*c01;
C[0*ldc+2] += alpha*c02; C[0*ldc+3] += alpha*c03;

C[1*ldc+0] += alpha*c10; C[1*ldc+1] += alpha*c11;
C[1*ldc+2] += alpha*c12; C[1*ldc+3] += alpha*c13;

C[2*ldc+0] += alpha*c20; C[2*ldc+1] += alpha*c21;
C[2*ldc+2] += alpha*c22; C[2*ldc+3] += alpha*c23;

C[3*ldc+0] += alpha*c30; C[3*ldc+1] += alpha*c31;
C[3*ldc+2] += alpha*c32; C[3*ldc+3] += alpha*c33;
}

struct DgemmArgs {
int m,n,k;
double alpha,beta;
const double *A,*B;
double *C;
int lda,ldb,ldc;
int row_start,row_end;
double *Ap;
double *Bt;
};

static void *dgemm_worker(void *arg){
auto *a=(DgemmArgs*)arg;

int M=a->row_end-a->row_start;
int N=a->n;
int K=a->k;

const double *A=a->A+a->row_start*a->lda;
const double *B=a->B;
double *C=a->C+a->row_start*a->ldc;

double *Ap = a->Ap;
double *Bt = a->Bt;

for (int ic=0; ic<M; ic+=MC){
    int mc = std::min(MC, M-ic);

    for (int jc=0; jc<N; jc+=NC){
        int nc = std::min(NC, N-jc);

        for (int i=0;i<mc;i++)
            for (int j=0;j<nc;j++)
                C[(ic+i)*a->ldc + (jc+j)] *= a->beta;

        for (int pc=0; pc<K; pc+=KC){
            int kc = std::min(KC, K-pc);

            for (int i=0;i<mc;i++)
                for (int p=0;p<kc;p++)
                    Ap[i*kc+p] = A[(ic+i)*a->lda + (pc+p)];

            for (int j=0;j<nc;j++)
                for (int p=0;p<kc;p++)
                    Bt[j*kc+p] = B[(pc+p)*a->ldb + (jc+j)];

            for (int i=0;i<mc;i+=4){
                for (int j=0;j<nc;j+=4){
                    if (i+3<mc && j+3<nc){
                        kernel_4x4_d(
                            kc,
                            Ap + i*kc,
                            Bt + j*kc,
                            C + (ic+i)*a->ldc + (jc+j),
                            a->ldc,
                            a->alpha
                        );
                    } else {
                        for (int ii=i; ii<std::min(i+4,mc); ii++)
                            for (int jj=j; jj<std::min(j+4,nc); jj++){
                                double sum=0;
                                for (int p=0;p<kc;p++)
                                    sum+=Ap[ii*kc+p]*Bt[jj*kc+p];
                                C[(ic+ii)*a->ldc+(jc+jj)] += a->alpha*sum;
                            }
                    }
                }
            }
        }
    }
}
return nullptr;
}

void my_dgemm(int m,int n,int k,
double alpha,const double*A,int lda,
const double*B,int ldb,
double beta,double*C,int ldc,int nthreads)
{
int t = std::min(nthreads, m);

std::vector<pthread_t> thr(t);
std::vector<DgemmArgs> arg(t);

int s=0, rpt=m/t, rem=m%t;

for (int i=0;i<t;i++){
    int r=rpt+(i<rem?1:0);

    arg[i].m=m; arg[i].n=n; arg[i].k=k;
    arg[i].alpha=alpha; arg[i].beta=beta;
    arg[i].A=A; arg[i].B=B; arg[i].C=C;
    arg[i].lda=lda; arg[i].ldb=ldb; arg[i].ldc=ldc;
    arg[i].row_start=s; arg[i].row_end=s+r;

    arg[i].Ap = (double*)aligned_alloc(64, MC*KC*sizeof(double));
    arg[i].Bt = (double*)aligned_alloc(64, NC*KC*sizeof(double));

    pthread_create(&thr[i],nullptr,dgemm_worker,&arg[i]);
    s+=r;
}

for (int i=0;i<t;i++){
    pthread_join(thr[i],nullptr);
    free(arg[i].Ap);
    free(arg[i].Bt);
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

static void test_dgemm(int N){
printf("\n=== DGEMM (double)  N=%d ===\n",N);

int sz=N*N;

double *A=(double*)aligned_alloc(64,sz*sizeof(double));
double *B=(double*)aligned_alloc(64,sz*sizeof(double));
double *C0=(double*)aligned_alloc(64,sz*sizeof(double));
double *Cm=(double*)aligned_alloc(64,sz*sizeof(double));
double *Co=(double*)aligned_alloc(64,sz*sizeof(double));

fill(A,sz); fill(B,sz); fill(C0,sz);

const int R=10;
int tc[]={1,2,4};

double ob_t[R];

for(int r=0;r<R;r++){
    memcpy(Co,C0,sz*8);
    auto t0=Clock::now();
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
                N,N,N,1.0,A,N,B,N,0.0,Co,N);
    ob_t[r]=Sec(Clock::now()-t0).count();
}

double ogm=geomean(ob_t,R);

printf("OpenBLAS  среднее геометрическое = %.4f с\n\n",ogm);

printf("%-8s %-6s %-12s %-12s %-10s\n",
       "потоки","запуск","моё (с)","obl (с)","отн %%");

printf("%-8s %-6s %-12s %-12s %-10s\n",
       "-------","------","--------","--------","-------");

for(int tci:tc){
    double rels[R];

    for(int r=0;r<R;r++){
        memcpy(Cm,C0,sz*8);

        auto t0=Clock::now();
        my_dgemm(N,N,N,1.0,A,N,B,N,0.0,Cm,N,tci);
        double t=Sec(Clock::now()-t0).count();

        rels[r]=ogm/t*100.0;

        printf("%-8d %-6d %-12.4f %-12.4f %-10.1f\n",
               tci,r+1,t,ogm,rels[r]);
    }

    printf("  >> потоки=%-2d  среднее геометрическое = %.1f%% от OpenBLAS\n\n",
           tci,geomean(rels,R));
}
}

int main(){
srand(42);
test_dgemm(512);
return 0;
}