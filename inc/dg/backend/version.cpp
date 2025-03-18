#include <thrust/version.h>
#include <cusp/version.h>
#include <iostream>

int main(void)
{

    int thrust_major = THRUST_MAJOR_VERSION;
    int thrust_minor = THRUST_MINOR_VERSION;
    int thrust_subminor = THRUST_SUBMINOR_VERSION;

    int cusp_major = CUSP_MAJOR_VERSION;
    int cusp_minor = CUSP_MINOR_VERSION;
    int cusp_subminor = CUSP_SUBMINOR_VERSION;

#ifdef __NVCC__
    int cuda_major = __CUDACC_VER_MAJOR__;
    int cuda_minor = __CUDACC_VER_MINOR__;
    std::cout << "CUDA   v" << cuda_major   << "." << cuda_minor   << std::endl;
#else
    std::cout << "Cuda is not needed for host-only compilation!\n";
#endif
    std::cout << "Thrust v" << thrust_major << "." << thrust_minor << "."<<thrust_subminor << std::endl;
    std::cout << "Cusp   v" << cusp_major   << "." << cusp_minor   << "."<<cusp_subminor<< std::endl;

    return 0;
}
