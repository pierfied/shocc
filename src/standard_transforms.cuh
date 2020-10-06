//
// Created by pierfied on 10/5/20.
//

#ifndef SHOCC_STANDARD_TRANSFORMS_CUH
#define SHOCC_STANDARD_TRANSFORMS_CUH

#include <torch/extension.h>
#include <cuComplex.h>

__global__
void FKernel(int lmax, int nrings, cuDoubleComplex *F, cuDoubleComplex *alm,
             double *ringTheta, double *ringPhi0, double *fac1, double *fac2, double *fac3);

void alm2map(torch::Tensor alm, int nside, int lmax);

#endif //SHOCC_STANDARD_TRANSFORMS_CUH
