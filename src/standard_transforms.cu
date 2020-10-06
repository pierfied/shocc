//
// Created by pierfied on 10/5/20.
//

#include "standard_transforms.cuh"
#include <healpix_cxx/healpix_base.h>
#include <iostream>

__global__ void alm2mapKernel() {
}

void alm2map(torch::Tensor alm, int nside) {
    auto size = alm.numel();

    std::cout << "size: " << size << std::endl;

    int nrings = 3 * nside;
    int startpix, ringpix;
    bool shifted;

    T_Healpix_Base<int> base(nside, RING);

    for (int i = 0; i < nrings; i++){
        base.get_ring_info_small(i, startpix, ringpix, shifted);

        std::cout << i << " " << startpix << " " << ringpix << " " << shifted << std::endl;
    }


//    alm2mapKernel<<<1, 1>>>(size, (cuDoubleComplex *) alm.data<torch::complex<double>>());
}
