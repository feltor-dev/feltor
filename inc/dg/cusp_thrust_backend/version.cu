#include <thrust/version.h>
#include <cusp/version.h>
#include <iostream>

#ifndef CUDA_VERSION
#define CUDA_VERSION 0
#endif

int main(void)
{
    int cuda_major =  CUDA_VERSION / 1000;
    int cuda_minor = (CUDA_VERSION % 1000) / 10;

    int thrust_major = THRUST_MAJOR_VERSION;
    int thrust_minor = THRUST_MINOR_VERSION;

    int cusp_major = CUSP_MAJOR_VERSION;
    int cusp_minor = CUSP_MINOR_VERSION;

    if( CUDA_VERSION==0)
        std::cout << "Cuda is not needed for host-only compilation!\n";
    else
        std::cout << "CUDA   v" << cuda_major   << "." << cuda_minor   << std::endl;
    std::cout << "Thrust v" << thrust_major << "." << thrust_minor << std::endl;
    std::cout << "Cusp   v" << cusp_major   << "." << cusp_minor   << std::endl;

    return 0;
}
