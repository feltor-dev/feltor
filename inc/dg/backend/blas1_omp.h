#ifndef _DG_BLAS_OMP_
#define _DG_BLAS_OMP_
#include "blas1_serial.h"
#include "exblas/exdot.fpe.cpp"
namespace dg
{
namespace blas1
{
namespace detail
{
const unsigned MIN_SIZE=100;//don't parallelize if work is too small 
exblas::Superaccumulator doDot_dispatch( OmpTag, unsigned size, const double* x_ptr, const double * y_ptr) {
    return exblas::Superaccumulator(  exblas::exdot_omp( size, x_ptr,y_ptr,8,true)) ;
}
template< class Vector, class UnaryOp>
inline void doTransform_dispatch( OmpTag, const Vector& x, Vector& y, UnaryOp op) {
    thrust::transform( thrust::omp::tag(), x.begin(), x.end(), y.begin(), op);
}
template< class T>
inline void doScal_dispatch( OmpTag, unsigned size, T* x, T alpha)
{
    if(size<MIN_SIZE) {
        doScal_dispatch( SerialTag(), size, x, alpha);
        return;
    }
    #pragma omp parallel for SIMD
    for( unsigned i=0; i<size; i++)
        x[i]*=alpha;
}
template<class T>
inline void doPlus_dispatch( OmpTag, unsigned size, T* x, T alpha)
{
    if(size<MIN_SIZE)
    {
        for( unsigned i=0; i<size; i++)
            x[i]+=alpha;
        return;
    }
    #pragma omp parallel for SIMD
    for( unsigned i=0; i<size; i++)
        x[i]+=alpha;
}
template<class T>
void doAxpby_dispatch( OmpTag, unsigned size, T alpha, const T * RESTRICT x_ptr, T beta, T* RESTRICT y_ptr)
{
    if(size<MIN_SIZE) 
    {
        for( unsigned i=0; i<size; i++)
            y_ptr[i] = alpha*x_ptr[i] + beta*y_ptr[i];
        return;
    }
    #pragma omp parallel for SIMD
    for( unsigned i=0; i<size; i++)
        y_ptr[i] = alpha*x_ptr[i] + beta*y_ptr[i];
}
template<class T>
void doAxpbypgz_dispatch( OmpTag, unsigned size, T alpha, const T * RESTRICT x_ptr, T beta, const T* RESTRICT y_ptr, T gamma, T* RESTRICT z_ptr)
{
    if(size<MIN_SIZE)
    {
        for( unsigned i=0; i<size; i++)
            z_ptr[i] = alpha*x_ptr[i] + beta*y_ptr[i] + gamma*z_ptr[i];
        return;
    }
    #pragma omp parallel for SIMD
    for( unsigned i=0; i<size; i++)
        z_ptr[i] = alpha*x_ptr[i] + beta*y_ptr[i] + gamma*z_ptr[i];
}
template<class value_type>
inline void doPointwiseDot_dispatch( OmpTag, unsigned size, 
              value_type alpha, 
              const value_type* x_ptr,
              const value_type* y_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    if( size<MIN_SIZE)
    {
        for( unsigned i=0; i<size; i++)
            z_ptr[i] = alpha*x_ptr[i]*y_ptr[i]+gamma*z_ptr[i];
        return; 
    }
    #pragma omp parallel for SIMD
    for( unsigned i=0; i<size; i++)
        z_ptr[i] = alpha*x_ptr[i]*y_ptr[i]+gamma*z_ptr[i];
}
template<class value_type>
inline void doPointwiseDivide_dispatch( OmpTag, unsigned size, 
              value_type alpha, 
              const value_type* x_ptr,
              const value_type* y_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    if( size<MIN_SIZE)
    {
        for( unsigned i=0; i<size; i++)
            z_ptr[i] = alpha*x_ptr[i]/y_ptr[i]+gamma*z_ptr[i];
        return;
    }
    #pragma omp parallel for SIMD
    for( unsigned i=0; i<size; i++)
        z_ptr[i] = alpha*x_ptr[i]/y_ptr[i]+gamma*z_ptr[i];
}

template<class value_type>
inline void doPointwiseDot_dispatch( OmpTag, unsigned size, 
              value_type alpha, 
              const value_type* x1_ptr,
              const value_type* y1_ptr,
              value_type beta, 
              const value_type* x2_ptr,
              const value_type* y2_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    if( size<MIN_SIZE)
    {
        for( unsigned i=0; i<size; i++)
            z_ptr[i] = alpha*x1_ptr[i]*y1_ptr[i]
                       +beta*x2_ptr[i]*y2_ptr[i]
                       +gamma*z_ptr[i];
        return;
    }
    #pragma omp parallel for SIMD
    for( unsigned i=0; i<size; i++)
        z_ptr[i] = alpha*x1_ptr[i]*y1_ptr[i] 
                   +beta*x2_ptr[i]*y2_ptr[i]
                   +gamma*z_ptr[i];
}

template<class value_type>
inline void doPointwiseDot_dispatch( OmpTag,unsigned size, 
              value_type alpha, 
              const value_type* x1,
              const value_type* x2,
              const value_type* x3,
              value_type beta, 
              value_type* y)
{
    if( size <MIN_SIZE) {
        for( unsigned i=0; i<size; i++)
            y[i] = alpha*x1[i]*x2[i]*x3[i]+beta*y[i];
        return;
    }
    #pragma omp parallel for SIMD
    for( unsigned i=0; i<size; i++)
        y[i] = alpha*x1[i]*x2[i]*x3[i] +beta*y[i];
}

}//namespace detail
}//namespace blas1
}//namespace dg
#endif //_DG_BLAS_OMP_
