//
// Created by pierfied on 10/5/20.
//

#include <healpix_cxx/healpix_base.h>
#include <iostream>
#include <cuComplex.h>
#include <cufft.h>

#include "standard_transforms.cuh"
#include "wigner.cuh"

#define CHUNKSIZE 16

__global__ void FKernel(int lmax, int nrings, int nchunks, cuDoubleComplex *F, cuDoubleComplex *alm,
                        double *ringTheta, double *ringPhi0, double *fac1, double *fac2, double *fac3) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < (lmax + 1) * nchunks; i += stride) {
        int m = i / nchunks;
        int c = i % nchunks;
        int yOffset = c * CHUNKSIZE;

        cuDoubleComplex FVals[CHUNKSIZE];
        double dlm[CHUNKSIZE], prevDlm[CHUNKSIZE], cosTheta[CHUNKSIZE], cosPhase[CHUNKSIZE], sinPhase[CHUNKSIZE];

#pragma unroll
        for (int j = 0; j < CHUNKSIZE; j++) {
            dlm[j] = emmRecursionSeed(m, 0, ringTheta[yOffset + j]);
            prevDlm[j] = 0;
            cosTheta[j] = cos(ringTheta[yOffset + j]);
            cosPhase[j] = cos(ringPhi0[yOffset + j]);
            sinPhase[j] = sin(ringPhi0[yOffset + j]);
            FVals[j].x = 0;
            FVals[j].y = 0;
        }

        int ind = m * (lmax + 1) - (m - 1) * m / 2;
        double sLambdalm[CHUNKSIZE], tmpDlm[CHUNKSIZE];

        for (int l = m; l <= lmax; l++){
            double prefac = pow(-1, m) * sqrt((2. * l + 1) / (4 * M_PI));
            cuDoubleComplex almVal = alm[ind];
            double fac1Val = fac1[ind];
            double fac3Val = fac3[ind];

//#pragma unroll
//            for (int j = 0; j < CHUNKSIZE; j++){
//                sLambdalm[j] = prefac * dlm[j];
//            }

#pragma unroll
            for (int j = 0; j < CHUNKSIZE; j++){
                sLambdalm[j] = prefac * dlm[j];
                tmpDlm[j] = fac1Val * cosTheta[j] * dlm[j] - fac3Val * prevDlm[j];
                prevDlm[j] = dlm[j];
                dlm[j] = tmpDlm[j];
                FVals[j].x += almVal.x * sLambdalm[j];
                FVals[j].y += almVal.y * sLambdalm[j];
            }

//#pragma unroll
//            for (int j = 0; j < CHUNKSIZE; j++){
//                tmpDlm[j] = fac1Val * cosTheta[j] * dlm[j] - fac3Val * prevDlm[j];
//                prevDlm[j] = dlm[j];
//                dlm[j] = tmpDlm[j];
//            }

//#pragma unroll
//            for (int j = 0; j < CHUNKSIZE; j++){
//            }

            ind++;
        }

#pragma unroll
        for (int j = 0; j < CHUNKSIZE; j++){
            F[(yOffset + j) * (lmax + 1) + m] = cuCmul(FVals[j], make_cuDoubleComplex(cosPhase[j], sinPhase[j]));
        }
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
    int nchunks = (nrings + CHUNKSIZE - 1) / CHUNKSIZE;
    int nringsPad = nchunks * CHUNKSIZE;

    // Create the CUDA arrays for the ring info.
    int *ringPix, *ringStart;
    double *ringTheta, *ringPhi0;
    cudaMallocManaged(&ringPix, sizeof(int) * nringsPad);
    cudaMallocManaged(&ringStart, sizeof(int) * nringsPad);
    cudaMallocManaged(&ringTheta, sizeof(double) * nringsPad);
    cudaMallocManaged(&ringPhi0, sizeof(double) * nringsPad);

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
    cudaMallocManaged(&F, sizeof(cuDoubleComplex) * (lmax + 1) * nringsPad);

    // Launch the kernel to compute F.
    int blockSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, FKernel, 0, 0);
    FKernel<<<gridSize, blockSize>>>(lmax, nrings, nchunks, F, almPtr, ringTheta, ringPhi0, fac1, fac2, fac3);

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
        cufftDestroy(plan);
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
