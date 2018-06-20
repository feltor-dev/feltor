#ifndef _DG_BLAS_OMP_
#define _DG_BLAS_OMP_
#include <omp.h>
#include "config.h"
#include "blas1_serial.h"
#include "exblas/exdot_omp.h"
namespace dg
{
namespace blas1
{
namespace detail
{
const int MIN_SIZE=100;//don't parallelize if work is too small

static inline std::vector<int64_t> doDot_dispatch( OmpTag, int size, const double* x_ptr, const double * y_ptr) {
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    if(size<MIN_SIZE)
        exblas::exdot_cpu( size, x_ptr,y_ptr, &h_superacc[0]);
    else
        exblas::exdot_omp( size, x_ptr,y_ptr, &h_superacc[0]);
    return h_superacc;
}
static inline std::vector<int64_t> doDot_dispatch( OmpTag, unsigned size, const double* x_ptr, const double * y_ptr, const double* z_ptr) {
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    if(size<MIN_SIZE)
        exblas::exdot_cpu( size, x_ptr,y_ptr,z_ptr, &h_superacc[0]);
    else
        exblas::exdot_omp( size, x_ptr,y_ptr,z_ptr, &h_superacc[0]);
    return h_superacc;
}

template< class Subroutine, class T, class ...Ts>
inline void doSubroutine_omp( int size, Subroutine f, T x, Ts... xs)
{
#pragma omp for nowait
    for( int i=0; i<size; i++)
        //f(x[i], xs[i]...);
        //f(thrust::raw_reference_cast(*(x+i)), thrust::raw_reference_cast(*(xs+i))...);
        f(static_cast<typename std::iterator_traits<T>::reference>(*(x+i)), static_cast<typename std::iterator_traits<Ts>::reference>(*(xs+i))...);
}

template< class Subroutine, class T, class ...Ts>
inline void doSubroutine_dispatch( OmpTag, int size, Subroutine f, T x, Ts... xs)
{
    if(omp_in_parallel())
    {
        doSubroutine_omp( size, f, x, xs... );
        return;
    }
    if(size>MIN_SIZE)
    {
        #pragma omp parallel
        {
            doSubroutine_omp( size, f, x, xs...);
        }
    }
    else
        doSubroutine_dispatch( SerialTag(), size, f, x, xs...);
}

}//namespace detail
}//namespace blas1
}//namespace dg
#endif //_DG_BLAS_OMP_
