#pragma once
//See also exblas/config.h for more preprocessor configurations

//%%%%%%%%%%%%define RESTRICT keyword for host compiler%%%%%%%%%%%%%%%%%%%%%%%%%
#if defined(__INTEL_COMPILER)
// On Intel compiler, you need to pass the -restrict compiler flag in addition to your own compiler flags.
# define RESTRICT restrict
#elif defined(__GNUG__)
# define RESTRICT __restrict__
#elif defined( _MSC_VER)
# define RESTRICT __restrict
#else
#pragma message( "WARNING: Don't know restrict keyword for this compiler!")
# define RESTRICT
#endif

//%%%%%%%%%%%%%%%%check for fast FMAs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#include <cmath>
#ifndef FP_FAST_FMA
#pragma message( "NOTE: Fast std::fma(a,b,c) not activated! Using a*b+c instead!")
#define DG_FMA(a,b,c) (a*b+c)
#else
#define DG_FMA(a,b,c) (dg::detail::dg_fma(a,b,c))
#endif

//%%%%%%%%%%%%%check for SIMD support in OpenMP4 if device system is OMP%%%%%%%%%%
#include "thrust/device_vector.h"//the <thrust/device_vector.h> header must be included for the THRUST_DEVICE_SYSTEM macros to work
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
#if defined(__INTEL_COMPILER)

#if __INTEL_COMPILER < 1500
#pragma message( "NOTE: icc version >= 15.0 recommended (to activate OpenMP 4 support)")
#define SIMD
#else//>1500
#define SIMD simd
#endif//__INTEL_COMPILER

#elif defined(__ibmxl__)
#define SIMD simd
#pragma message( "Using IBM compiler!")

#elif defined(__GNUG__)

#ifndef GCC_VERSION
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif //GCC_VERSION
#if GCC_VERSION < 40900
#pragma message( "gcc version >= 4.9 recommended to activate OpenMP 4 support")
#define SIMD
#else
#define SIMD simd
#endif //GCC_VERSION

#elif defined(_MSC_VER)
#pragma message( "WARNING: No OpenMP 4 support on your compiler")
#define SIMD
#endif //compilers
#endif //THRUST_DEVICE_SYSTEM


//%%%%%%%%%%%%%%%try to check for cuda-aware MPI support%%%%%%%%%%%%%%%%%%%%%%%%%%
//TODO This should be tested somewhere
namespace dg{
#ifdef MPI_VERSION
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA // cuda is involved
//{;

#ifdef DG_CUDA_UNAWARE_MPI // TODO document in config README
//{;
#pragma message( "Assume CUDA-unaware MPI support as per user indication!")
inline constexpr bool cuda_aware_mpi = false;
//}
#else // ! DG_CUDA_UNAWARE_MPI
//{;
#if defined(OPEN_MPI) && OPEN_MPI
//{;
#include "mpi-ext.h"
// Open-MPI header defines MPIX_CUDA_AWARE if compiled with cuda
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
#pragma message( "CUDA-aware MPI support detected! Yay!")
inline constexpr bool cuda_aware_mpi = true;
#else
#pragma message( "No CUDA aware MPI installation! Falling back to regular MPI!")
inline constexpr bool cuda_aware_mpi = false;
#endif
//}
#else // Other than open-mpi there seems no way to determine cuda support
//{;
#pragma message( "We assume CUDA-aware MPI support! Compile with -DDG_CUDA_UNAWARE_MPI if we should not!")
inline constexpr bool cuda_aware_mpi = true;
//}
#endif // OPEN_MPI
//}
#endif // DG_CUDA_UNAWARE_MPI
//}
#else // THRUST != CUDA
//{;
inline constexpr bool cuda_aware_mpi = false;
//}
#endif //THRUST == CUDA
#endif //MPI_VERSION
} // namespace dg

//%%%%%%%%%%%%%%%Define DG_DEVICE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
///@brief Expands to \__host__ \__device__ if compiled with nvcc else is empty
#define DG_DEVICE
#ifdef __CUDACC__
#undef DG_DEVICE
#define DG_DEVICE __host__ __device__
#endif
#include "fma.h"
