//
// Created by pierfied on 10/5/20.
//

#ifndef SHOCC_STANDARD_TRANSFORMS_CUH
#define SHOCC_STANDARD_TRANSFORMS_CUH

#include <torch/extension.h>
#include <cuComplex.h>

__global__
void alm2mapKernel();

void alm2map(torch::Tensor alm, int nside);

#endif //SHOCC_STANDARD_TRANSFORMS_CUH
