//
// Created by pierfied on 10/5/20.
//

#include <healpix_cxx/healpix_base.h>
#include <iostream>
#include <cuComplex.h>
#include <cufft.h>

#include "standard_transforms.cuh"
#include "wigner.cuh"

__global__ void FKernel(int lmax, int nrings, cuDoubleComplex *F, cuDoubleComplex *alm,
                        double *ringTheta, double *ringPhi0, double *fac1, double *fac2, double *fac3) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < (lmax + 1) * nrings; i += stride) {
        int y = i / nrings;
        int m = i % nrings;

        // Initialize F[m,y] to zero.
        F[i].x = 0;
        F[i].y = 0;

        // Seed the recursion.
        double dlm = emmRecursionSeed(m, 0., ringTheta[y]);
        double prevDlm = 0;

        // Loop over l to compute F.
        double cosTheta = cos(ringTheta[y]);
        for (int l = m; l <= lmax; l++) {
            int ind = m * (lmax + 1) - (m - 1) * m / 2 + (l - m);

            // Compute the contributions to F.
            double sLambdalm = pow(-1, m) * sqrt((2. * l + 1) / (4 * M_PI)) * dlm;
            F[i].x += alm[ind].x * sLambdalm;
            F[i].y += alm[ind].y * sLambdalm;

            // Compute the next wigner d value.
            double tmpDlm = fac1[ind] * cosTheta * dlm - fac3[ind] * prevDlm;
            prevDlm = dlm;
            dlm = tmpDlm;
        }

        // Apply the phase shift for the ring.
        F[i] = cuCmul(F[i], make_cuDoubleComplex(cos(m * ringPhi0[y]), sin(m * ringPhi0[y])));
    }
}

torch::Tensor alm2map(torch::Tensor alm, int nside, int lmax) {
    // Start computing the recursion coefficients on the GPU now while we compute Healpix stuff next.
    double *fac1, *fac2, *fac3;
    computeRecursionCoeffs(lmax, 0, &fac1, &fac2, &fac3);

    // Create the base Healpix class for useful routines later.
    nside_dummy dummy;
    Healpix_Base base(nside, RING, dummy);

    // Compute the size of the map and number of rings.
    int npix = 12 * nside * nside;
    int nrings = base.pix2ring(npix - 1);

    // Create the CUDA arrays for the ring info.
    int *ringPix, *ringStart;
    double *ringTheta, *ringPhi0;
    cudaMallocManaged(&ringPix, sizeof(int) * nrings);
    cudaMallocManaged(&ringStart, sizeof(int) * nrings);
    cudaMallocManaged(&ringTheta, sizeof(double) * nrings);
    cudaMallocManaged(&ringPhi0, sizeof(double) * nrings);

    // Get all of the relevant info for the ring.
#pragma omp parallel for
    for (int i = 0; i < nrings; i++) {
        bool shifted;
        base.get_ring_info2(i + 1, ringStart[i], ringPix[i], ringTheta[i], shifted);
        ringPhi0[i] = base.pix2ang(ringStart[i]).phi;
    }

    // Create the F array and get the pointer to the alm data.
    cuDoubleComplex *almPtr, *F;
    almPtr = (cuDoubleComplex *) alm.data<torch::complex<double>>();
    cudaMallocManaged(&F, sizeof(cuDoubleComplex) * (lmax + 1) * nrings);

    // Launch the kernel to compute F.
    int blockSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, FKernel, 0, 0);
    FKernel<<<gridSize, blockSize>>>(lmax, nrings, F, almPtr, ringTheta, ringPhi0, fac1, fac2, fac3);

    // Create the map tensor.
    torch::Tensor map = torch::zeros(npix, torch::dtype(torch::kFloat64).device(torch::kCUDA));
    double *mapPtr = map.data<double>();

    // Perform the FFTs to build the map.
    for (int i = 0; i < nrings; i++) {
        cufftHandle plan;
        double *ringPtr = &mapPtr[ringStart[i]];
        cufftDoubleComplex *data = &F[i * (lmax + 1)];
        cufftPlan1d(&plan, ringPix[i], CUFFT_Z2D, 1);
        cufftExecZ2D(plan, data, ringPtr);
    }

    // Free arrays.
    cudaFree(fac1);
    cudaFree(fac2);
    cudaFree(fac3);
    cudaFree(ringPix);
    cudaFree(ringStart);
    cudaFree(ringTheta);
    cudaFree(ringPhi0);
    cudaFree(F);

    return map;
}
