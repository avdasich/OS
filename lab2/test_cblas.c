//ubuntu 24.04

#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <cblas.h>

int passed = 0;
int failed = 0;

void check_f(const char *name, float expected, float got) {
    if (fabsf(expected - got) < 1e-4f) {
        printf("[PASS] %s\n", name);
        passed++;
    } else {
        printf("[FAIL] %s: ожидалось %.4f, получено %.4f\n", name, expected, got);
        failed++;
    }
}

void check_d(const char *name, double expected, double got) {
    if (fabs(expected - got) < 1e-9) {
        printf("[PASS] %s\n", name);
        passed++;
    } else {
        printf("[FAIL] %s: ожидалось %.6f, получено %.6f\n", name, expected, got);
        failed++;
    }
}

void check_i(const char *name, int expected, int got) {
    if (expected == got) {
        printf("[PASS] %s\n", name);
        passed++;
    } else {
        printf("[FAIL] %s: ожидалось %d, получено %d\n", name, expected, got);
        failed++;
    }
}

//level 1

void test_sdot() {
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {4.0f, 5.0f, 6.0f};
    //1*4 + 2*5 + 3*6 = 32
    check_f("sdot", 32.0f, cblas_sdot(3, x, 1, y, 1));
}

void test_ddot() {
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    check_d("ddot", 32.0, cblas_ddot(3, x, 1, y, 1));
}

void test_sdsdot() {
    float x[] = {1.0f, 2.0f};
    float y[] = {3.0f, 4.0f};
    //10 + (1*3 + 2*4) = 21
    check_f("sdsdot", 21.0f, cblas_sdsdot(2, 10.0f, x, 1, y, 1));
}

void test_dsdot() {
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {1.0f, 1.0f, 1.0f};
    check_d("dsdot", 6.0, cblas_dsdot(3, x, 1, y, 1));
}

void test_snrm2() {
    float x[] = {3.0f, 4.0f};
    //sqrt(9+16) = 5
    check_f("snrm2", 5.0f, cblas_snrm2(2, x, 1));
}

void test_dnrm2() {
    double x[] = {3.0, 4.0};
    check_d("dnrm2", 5.0, cblas_dnrm2(2, x, 1));
}

void test_sasum() {
    float x[] = {1.0f, -2.0f, 3.0f};
    //|1|+|-2|+|3| = 6
    check_f("sasum", 6.0f, cblas_sasum(3, x, 1));
}

void test_dasum() {
    double x[] = {1.0, -2.0, 3.0};
    check_d("dasum", 6.0, cblas_dasum(3, x, 1));
}

void test_isamax() {
    float x[] = {1.0f, -5.0f, 3.0f};
    //макс по модулю индекс = 1
    check_i("isamax", 1, (int)cblas_isamax(3, x, 1));
}

void test_idamax() {
    double x[] = {1.0, 2.0, -7.0, 3.0};
    //макс по модулю индекс = 2
    check_i("idamax", 2, (int)cblas_idamax(4, x, 1));
}

void test_scopy() {
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[3] = {0};
    cblas_scopy(3, x, 1, y, 1);
    check_f("scopy y[0]", 1.0f, y[0]);
    check_f("scopy y[1]", 2.0f, y[1]);
    check_f("scopy y[2]", 3.0f, y[2]);
}

void test_dcopy() {
    double x[] = {4.0, 5.0, 6.0};
    double y[3] = {0};
    cblas_dcopy(3, x, 1, y, 1);
    check_d("dcopy y[0]", 4.0, y[0]);
    check_d("dcopy y[2]", 6.0, y[2]);
}

void test_sswap() {
    float x[] = {1.0f, 2.0f};
    float y[] = {9.0f, 8.0f};
    cblas_sswap(2, x, 1, y, 1);
    check_f("sswap x[0]", 9.0f, x[0]);
    check_f("sswap y[0]", 1.0f, y[0]);
}

void test_dswap() {
    double x[] = {1.0, 2.0};
    double y[] = {9.0, 8.0};
    cblas_dswap(2, x, 1, y, 1);
    check_d("dswap x[0]", 9.0, x[0]);
    check_d("dswap y[0]", 1.0, y[0]);
}

void test_saxpy() {
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[] = {4.0f, 5.0f, 6.0f};
    //y = 2*x + y = [6, 9, 12]
    cblas_saxpy(3, 2.0f, x, 1, y, 1);
    check_f("saxpy y[0]",  6.0f, y[0]);
    check_f("saxpy y[1]",  9.0f, y[1]);
    check_f("saxpy y[2]", 12.0f, y[2]);
}

void test_daxpy() {
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {4.0, 5.0, 6.0};
    cblas_daxpy(3, 2.0, x, 1, y, 1);
    check_d("daxpy y[0]",  6.0, y[0]);
    check_d("daxpy y[2]", 12.0, y[2]);
}

void test_sscal() {
    float x[] = {1.0f, 2.0f, 3.0f};
    cblas_sscal(3, 3.0f, x, 1);
    check_f("sscal x[0]", 3.0f, x[0]);
    check_f("sscal x[1]", 6.0f, x[1]);
    check_f("sscal x[2]", 9.0f, x[2]);
}

void test_dscal() {
    double x[] = {2.0, 4.0, 6.0};
    cblas_dscal(3, 0.5, x, 1);
    check_d("dscal x[0]", 1.0, x[0]);
    check_d("dscal x[2]", 3.0, x[2]);
}

void test_srotg() {
    float a = 3.0f, b = 4.0f, c, s;
    cblas_srotg(&a, &b, &c, &s);
    //r = 5, c^2+s^2 = 1
    check_f("srotg |r|",     5.0f, fabsf(a));
    check_f("srotg c^2+s^2", 1.0f, c*c + s*s);
}

void test_drotg() {
    double a = 3.0, b = 4.0, c, s;
    cblas_drotg(&a, &b, &c, &s);
    check_d("drotg |r|",     5.0, fabs(a));
    check_d("drotg c^2+s^2", 1.0, c*c + s*s);
}

void test_srot() {
    float x[] = {1.0f, 0.0f};
    float y[] = {0.0f, 1.0f};
    //поворот 90 градусов c=0 s=1
    cblas_srot(2, x, 1, y, 1, 0.0f, 1.0f);
    check_f("srot x[0]",  0.0f, x[0]);
    check_f("srot y[0]", -1.0f, y[0]);
}

//комплексные
void test_cdotu_sub() {
    //[1+0i, 0+1i] . [0+1i, 1+0i] = 0+2i
    float x[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float y[4] = {0.0f, 1.0f, 1.0f, 0.0f};
    float r[2]  = {0};
    cblas_cdotu_sub(2, x, 1, y, 1, r);
    check_f("cdotu Re", 0.0f, r[0]);
    check_f("cdotu Im", 2.0f, r[1]);
}

void test_cdotc_sub() {
    //conj(0+1i) * (0+1i) = 1
    float x[2] = {0.0f, 1.0f};
    float y[2] = {0.0f, 1.0f};
    float r[2]  = {0};
    cblas_cdotc_sub(1, x, 1, y, 1, r);
    check_f("cdotc Re", 1.0f, r[0]);
    check_f("cdotc Im", 0.0f, r[1]);
}

void test_scnrm2() {
    float x[2] = {3.0f, 4.0f}; //|3+4i| = 5
    check_f("scnrm2", 5.0f, cblas_scnrm2(1, x, 1));
}

void test_scasum() {
    float x[2] = {1.0f, 2.0f}; //|1|+|2| = 3
    check_f("scasum", 3.0f, cblas_scasum(1, x, 1));
}

void test_csscal() {
    float x[2] = {1.0f, 2.0f}; //(1+2i)*2 = 2+4i
    cblas_csscal(1, 2.0f, x, 1);
    check_f("csscal Re", 2.0f, x[0]);
    check_f("csscal Im", 4.0f, x[1]);
}

//level 2

void test_sgemv() {
    //A=[[1,2],[3,4]], x=[1,1] => y=[3,7]
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x[] = {1.0f, 1.0f};
    float y[2] = {0};
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 2, 1.0f, A, 2, x, 1, 0.0f, y, 1);
    check_f("sgemv y[0]", 3.0f, y[0]);
    check_f("sgemv y[1]", 7.0f, y[1]);
}

void test_dgemv() {
    double A[] = {1.0, 2.0, 3.0, 4.0};
    double x[] = {1.0, 1.0};
    double y[2] = {0};
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 2, 2, 1.0, A, 2, x, 1, 0.0, y, 1);
    check_d("dgemv y[0]", 3.0, y[0]);
    check_d("dgemv y[1]", 7.0, y[1]);
}

void test_sger() {
    //A = x*y^T = [[3,4],[6,8]]
    float A[4] = {0};
    float x[] = {1.0f, 2.0f};
    float y[] = {3.0f, 4.0f};
    cblas_sger(CblasRowMajor, 2, 2, 1.0f, x, 1, y, 1, A, 2);
    check_f("sger A[0][0]", 3.0f, A[0]);
    check_f("sger A[1][1]", 8.0f, A[3]);
}

void test_ssymv() {
    //A=[[2,1],[1,3]], x=[1,1] => y=[3,4]
    float A[] = {2.0f, 1.0f, 1.0f, 3.0f};
    float x[] = {1.0f, 1.0f};
    float y[2] = {0};
    cblas_ssymv(CblasRowMajor, CblasUpper, 2, 1.0f, A, 2, x, 1, 0.0f, y, 1);
    check_f("ssymv y[0]", 3.0f, y[0]);
    check_f("ssymv y[1]", 4.0f, y[1]);
}

void test_strmv() {
    //A=[[2,1],[0,3]], x=[1,2] => x=[4,6]
    float A[] = {2.0f, 1.0f, 0.0f, 3.0f};
    float x[] = {1.0f, 2.0f};
    cblas_strmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2, A, 2, x, 1);
    check_f("strmv x[0]", 4.0f, x[0]);
    check_f("strmv x[1]", 6.0f, x[1]);
}

//level 3

void test_sgemm() {
    //C = A*B = [[19,22],[43,50]]
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C[4] = {0};
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0f, A, 2, B, 2, 0.0f, C, 2);
    check_f("sgemm C[0][0]", 19.0f, C[0]);
    check_f("sgemm C[0][1]", 22.0f, C[1]);
    check_f("sgemm C[1][0]", 43.0f, C[2]);
    check_f("sgemm C[1][1]", 50.0f, C[3]);
}

void test_dgemm() {
    double A[] = {1.0, 2.0, 3.0, 4.0};
    double B[] = {5.0, 6.0, 7.0, 8.0};
    double C[4] = {0};
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0, A, 2, B, 2, 0.0, C, 2);
    check_d("dgemm C[0][0]", 19.0, C[0]);
    check_d("dgemm C[1][1]", 50.0, C[3]);
}

void test_ssymm() {
    //A симм, B=единичная => C=A
    float A[] = {1.0f, 2.0f, 2.0f, 3.0f};
    float B[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float C[4] = {0};
    cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, 2, 2, 1.0f, A, 2, B, 2, 0.0f, C, 2);
    check_f("ssymm C[0][0]", 1.0f, C[0]);
    check_f("ssymm C[0][1]", 2.0f, C[1]);
}

void test_ssyrk() {
    //A=[1,2], C = A*A^T = [[5]]
    float A[] = {1.0f, 2.0f};
    float C[] = {0.0f};
    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans, 1, 2, 1.0f, A, 2, 0.0f, C, 1);
    check_f("ssyrk C=5", 5.0f, C[0]);
}

void test_strmm() {
    //A=[[2,1],[0,3]], B=единичная => B=A
    float A[] = {2.0f, 1.0f, 0.0f, 3.0f};
    float B[] = {1.0f, 0.0f, 0.0f, 1.0f};
    cblas_strmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 2, 2, 1.0f, A, 2, B, 2);
    check_f("strmm B[0][0]", 2.0f, B[0]);
    check_f("strmm B[0][1]", 1.0f, B[1]);
    check_f("strmm B[1][1]", 3.0f, B[3]);
}

void test_strsm() {
    //A=[[2,0],[0,4]], B=[4,8] => X=[2,2]
    float A[] = {2.0f, 0.0f, 0.0f, 4.0f};
    float B[] = {4.0f, 8.0f};
    cblas_strsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, 2, 1, 1.0f, A, 2, B, 1);
    check_f("strsm B[0]", 2.0f, B[0]);
    check_f("strsm B[1]", 2.0f, B[1]);
}

//поток запускает несколько тестов параллельно с main
void *thread_func(void *arg) {
    test_ddot();
    test_dgemm();
    test_dscal();
    return NULL;
}

int main() {
    printf("интерфейсные тесты OpenBLAS\n\n");

    //запускаем поток
    pthread_t t;
    pthread_create(&t, NULL, thread_func, NULL);

    //тесты в основном потоке
    test_sdot();    test_ddot();
    test_sdsdot();  test_dsdot();
    test_snrm2();   test_dnrm2();
    test_sasum();   test_dasum();
    test_isamax();  test_idamax();
    test_scopy();   test_dcopy();
    test_sswap();   test_dswap();
    test_saxpy();   test_daxpy();
    test_sscal();   test_dscal();
    test_srotg();   test_drotg();
    test_srot();
    test_cdotu_sub(); test_cdotc_sub();
    test_scnrm2();  test_scasum();
    test_csscal();

    test_sgemv();   test_dgemv();
    test_sger();    test_ssymv();
    test_strmv();

    test_sgemm();   test_dgemm();
    test_ssymm();   test_ssyrk();
    test_strmm();   test_strsm();

    pthread_join(t, NULL);

    printf("\nитог: %d пройдено, %d провалено\n", passed, failed);
    return (failed == 0) ? 0 : 1;
}
