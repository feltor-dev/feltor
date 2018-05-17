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

std::vector<int64_t> doDot_dispatch( OmpTag, int size, const double* x_ptr, const double * y_ptr) {
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    if(size<MIN_SIZE)
        exblas::exdot_cpu( size, x_ptr,y_ptr, &h_superacc[0]);
    else
        exblas::exdot_omp( size, x_ptr,y_ptr, &h_superacc[0]);
    return h_superacc;
}

template< class UnaryOp, class Binary, class T>
inline void doEvaluate_dispatch( OmpTag, int size, T* y, Binary f, UnaryOp op, const T* x) {
    if(size<MIN_SIZE) {
        doEvaluate_dispatch( SerialTag(), size, y, f, op, x);
        return;
    }
#pragma omp parallel for
    for( int i=0; i<size; i++)
        f(y[i], op(x[i]));
}

template< class UnaryOp, class Binary, class T>
inline void doEvaluate_dispatch( OmpTag, int size, T* z, Binary f, UnaryOp op, const T* x, const T* y) {
    if(size<MIN_SIZE) {
        doEvaluate_dispatch( SerialTag(), size, z, f, op, x, y);
        return;
    }
#pragma omp parallel for
    for( int i=0; i<size; i++)
        f(z[i], op(x[i], y[i]));
}

template< class T>
inline void doScal_omp( int size, T* x, T alpha)
{
    #pragma omp for SIMD nowait
    for( int i=0; i<size; i++)
        x[i]*=alpha;
}
template< class T>
inline void doScal_dispatch( OmpTag, int size, T* x, T alpha)
{
    if(omp_in_parallel())
    {
        doScal_omp( size, x, alpha);
        return;
    }
    if(size>MIN_SIZE)
    {
        #pragma omp parallel
        {
            doScal_omp( size, x, alpha);
        }
    }
    else
        doScal_dispatch( SerialTag(), size, x, alpha);
}
template<class T>
inline void doPlus_omp( int size, T* x, T alpha)
{
    #pragma omp for SIMD nowait
    for( int i=0; i<size; i++)
        x[i]+=alpha;
}
template<class T>
inline void doPlus_dispatch( OmpTag, int size, T* x, T alpha)
{
    if(omp_in_parallel())
    {
        doPlus_omp( size, x, alpha);
        return;
    }
    if(size>MIN_SIZE)
    {
        #pragma omp parallel
        {
            doPlus_omp( size, x, alpha);
        }
    }
    else
        doPlus_dispatch( SerialTag(), size, x, alpha);

}
template<class T>
void doAxpby_omp(int size, T alpha, const T * RESTRICT x_ptr, T beta, T* RESTRICT y_ptr)
{
    #pragma omp for SIMD nowait
    for( int i=0; i<size; i++)
    {
        double temp = y_ptr[i]*beta;
        y_ptr[i] = DG_FMA( alpha,x_ptr[i], temp);
    }
}
template<class T>
void doAxpby_dispatch( OmpTag, int size, T alpha, const T * RESTRICT x_ptr, T beta, T* RESTRICT y_ptr)
{
    if(omp_in_parallel())
    {
        doAxpby_omp(size, alpha, x_ptr, beta, y_ptr);
        return;
    }
    if(size>MIN_SIZE)
    {
        #pragma omp parallel
        {
            doAxpby_omp(size, alpha, x_ptr, beta, y_ptr);
        }
    }
    else
        doAxpby_dispatch(SerialTag(), size, alpha, x_ptr, beta, y_ptr);
}
template<class T>
void doAxpbypgz_omp( int size, T alpha, const T * RESTRICT x_ptr, T beta, const T* RESTRICT y_ptr, T gamma, T* RESTRICT z_ptr)
{
    #pragma omp for SIMD nowait
    for( int i=0; i<size; i++)
    {
        double temp = z_ptr[i]*gamma;
        temp = DG_FMA( alpha,x_ptr[i], temp);
        temp = DG_FMA( beta, y_ptr[i], temp);
        z_ptr[i] = temp;
    }
}
template<class T>
void doAxpbypgz_dispatch( OmpTag, int size, T alpha, const T * RESTRICT x_ptr, T beta, const T* RESTRICT y_ptr, T gamma, T* RESTRICT z_ptr)
{
    if(omp_in_parallel())
    {
        doAxpbypgz_omp( size, alpha, x_ptr, beta, y_ptr, gamma, z_ptr);
        return;
    }
    if(size>MIN_SIZE)
    {
        #pragma omp parallel
        {
            doAxpbypgz_omp( size, alpha, x_ptr, beta, y_ptr, gamma, z_ptr);
        }
    }
    else
        doAxpbypgz_dispatch(SerialTag(), size, alpha, x_ptr, beta, y_ptr, gamma, z_ptr);
}
template<class value_type>
inline void doPointwiseDot_omp(int size,
              value_type alpha,
              const value_type* x_ptr,
              const value_type* y_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    #pragma omp for SIMD nowait
    for( int i=0; i<size; i++)
    {
        double temp = z_ptr[i]*gamma;
        z_ptr[i] = DG_FMA( (alpha*x_ptr[i]), y_ptr[i], temp);
    }
}
template<class value_type>
inline void doPointwiseDot_dispatch( OmpTag, int size,
              value_type alpha,
              const value_type* x_ptr,
              const value_type* y_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    if(omp_in_parallel())
    {
        doPointwiseDot_omp( size, alpha, x_ptr, y_ptr, gamma, z_ptr);
        return;
    }
    if(size>MIN_SIZE)
    {
        #pragma omp parallel
        {
            doPointwiseDot_omp( size, alpha, x_ptr, y_ptr, gamma, z_ptr);
        }
    }
    else
        doPointwiseDot_dispatch(SerialTag(), size, alpha, x_ptr, y_ptr, gamma, z_ptr);
}
template<class value_type>
inline void doPointwiseDivide_omp(int size,
              value_type alpha,
              const value_type* x_ptr,
              const value_type* y_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    #pragma omp for SIMD nowait
    for( int i=0; i<size; i++)
    {
        double temp = z_ptr[i]*gamma;
        z_ptr[i] = DG_FMA( alpha, (x_ptr[i]/y_ptr[i]), temp);
    }
}
template<class value_type>
inline void doPointwiseDivide_dispatch( OmpTag, int size,
              value_type alpha,
              const value_type* x_ptr,
              const value_type* y_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    if(omp_in_parallel())
    {
        doPointwiseDivide_omp( size, alpha, x_ptr, y_ptr, gamma, z_ptr);
        return;
    }
    if(size>MIN_SIZE)
    {
        #pragma omp parallel
        {
            doPointwiseDivide_omp( size, alpha, x_ptr, y_ptr, gamma, z_ptr);
        }
    }
    else
        doPointwiseDivide_dispatch(SerialTag(), size, alpha, x_ptr, y_ptr, gamma, z_ptr);
}
template<class value_type>
inline void doPointwiseDot_omp( int size,
              value_type alpha,
              const value_type* x1_ptr,
              const value_type* y1_ptr,
              value_type beta,
              const value_type* x2_ptr,
              const value_type* y2_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    #pragma omp for SIMD nowait
    for( int i=0; i<size; i++)
    {
        double temp = z_ptr[i]*gamma;
        temp = DG_FMA( (alpha*x1_ptr[i]), y1_ptr[i], temp);
        temp = DG_FMA(  (beta*x2_ptr[i]), y2_ptr[i], temp);
        z_ptr[i] = temp;
    }
}

template<class value_type>
inline void doPointwiseDot_dispatch( OmpTag, int size,
              value_type alpha,
              const value_type* x1_ptr,
              const value_type* y1_ptr,
              value_type beta,
              const value_type* x2_ptr,
              const value_type* y2_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    if(omp_in_parallel())
    {
        doPointwiseDot_omp( size, alpha, x1_ptr, y1_ptr, beta, x2_ptr, y2_ptr, gamma, z_ptr);
        return;
    }
    if(size>MIN_SIZE)
    {
        #pragma omp parallel
        {
            doPointwiseDot_omp( size, alpha, x1_ptr, y1_ptr, beta, x2_ptr, y2_ptr, gamma, z_ptr);
        }
    }
    else
        doPointwiseDot_dispatch( SerialTag(), size, alpha, x1_ptr, y1_ptr, beta, x2_ptr, y2_ptr, gamma, z_ptr);
}
template<class value_type>
inline void doPointwiseDot_omp(int size,
              value_type alpha,
              const value_type* x1,
              const value_type* x2,
              const value_type* x3,
              value_type beta,
              value_type* y)
{
    #pragma omp for SIMD nowait
    for( int i=0; i<size; i++)
    {
        double temp = y[i]*beta;
        y[i] = DG_FMA( (alpha*x1[i]), (x2[i]*x3[i]), temp);
    }
}

template<class value_type>
inline void doPointwiseDot_dispatch( OmpTag,int size,
              value_type alpha,
              const value_type* x1_ptr,
              const value_type* x2_ptr,
              const value_type* x3_ptr,
              value_type beta,
              value_type* y_ptr)
{
    if(omp_in_parallel())
    {
        doPointwiseDot_omp( size, alpha, x1_ptr, x2_ptr, x3_ptr, beta, y_ptr);
        return;
    }
    if(size>MIN_SIZE)
    {
        #pragma omp parallel
        {
            doPointwiseDot_omp( size, alpha, x1_ptr, x2_ptr, x3_ptr, beta, y_ptr);
        }
    }
    else
        doPointwiseDot_dispatch( SerialTag(),size, alpha, x1_ptr, x2_ptr, x3_ptr, beta, y_ptr);
}

}//namespace detail
}//namespace blas1
}//namespace dg
#endif //_DG_BLAS_OMP_
