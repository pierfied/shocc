//
// Created by pierfied on 10/5/20.
//

#include "wigner.cuh"

// Compute the log factorial via the gamma function.
__device__ double lnfac(double x) {
    return lgamma(x + 1);
}

// Compute the recursion seed for the case d(j, -j, m, beta).
__device__ double emmRecursionSeed(double j, double m, double beta) {
    double prefac = (lnfac(2 * j) - lnfac(j + m) - lnfac(j - m)) / 2;
    double cosfac = (j - m) * log(cos(beta / 2));
    double sinfac = (j + m) * log(sin(beta / 2));

    double d = exp(prefac + cosfac + sinfac);
    return d;
}

// Compute the recursion seed for the case d(j, m, j, beta).
__device__ double spinRecursionSeed(double j, double m, double beta) {
    double prefac = (lnfac(2 * j) - lnfac(j + m) - lnfac(j - m)) / 2;
    double cosfac = (j + m) * log(cos(beta / 2));
    double sinfac = (j - m) * log(sin(beta / 2));

    double d = exp(prefac + cosfac + sinfac);
    return d;
}

__global__ void recursionCoeffKernel(int lmax, int spin, double *fac1, double *fac2, double *fac3) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int nEll = lmax + 1;
    for (int i = index; i < nEll * nEll; i += stride) {
        // Get the l & m values from the index.
        int im = i / nEll;
        int il = i % nEll;
        double m = im;
        double l = il;

        // Only compute the coefficients for l >= m.
        if (l < m) continue;

        // Compute the recursion coefficients.
        int ind = im * nEll - (im - 1) * im / 2 + (il - im);
        double denomFac = sqrt(((l + 1) * (l + 1) - m * m) * ((l + 1) * (l + 1) - spin * spin));
        fac1[ind] = (l + 1) * (2 * l + 1) / denomFac;
        fac2[ind] = m * spin / (l * (l + 1));
        fac3[ind] = (l + 1) * sqrt((l * l - m * m) * (l * l - spin * spin)) / (l * denomFac);
    }

    // Set the second and third l,m = 0 coefficients to 0 because of divide by zero error.
    fac2[0] = 0;
    fac3[0] = 0;
}

void computeRecursionCoeffs(int lmax, int spin, double **fac1, double **fac2, double **fac3) {
    // Create the arrays for the coefficients.
    int facSize = (lmax + 1) * (lmax + 2) / 2;
    cudaMallocManaged(fac1, sizeof(double) * facSize);
    cudaMallocManaged(fac2, sizeof(double) * facSize);
    cudaMallocManaged(fac3, sizeof(double) * facSize);

    // Launch the kernel to compute the coefficients.
    int blockSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, recursionCoeffKernel, 0, 0);
    recursionCoeffKernel<<<gridSize, blockSize>>>(lmax, spin, *fac1, *fac2, *fac3);
}


