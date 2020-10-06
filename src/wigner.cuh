//
// Created by pierfied on 10/5/20.
//

#ifndef SHOCC_WIGNER_CUH
#define SHOCC_WIGNER_CUH

__device__
double lnfac(double x);

__device__
double emmRecursionSeed(double j, double m, double beta);

__device__
double spinRecursionSeed(double j, double m, double beta);

__global__
void recursionCoeffKernel(int lmax, int spin, double *fac1, double *fac2, double *fac3);

void computeRecursionCoeffs(int lmax, int spin, double **fac1, double **fac2, double **fac3);

#endif //SHOCC_WIGNER_CUH
