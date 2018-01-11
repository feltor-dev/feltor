#ifndef _DG_BLAS_SERIAL_
#define _DG_BLAS_SERIAL_
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include "execution_policy.h"
#include "exblas/exdot.fpe_cpu.cpp"
namespace dg
{
namespace blas1
{
namespace detail
{
std::vector<int64_t> doDot_dispatch( SerialTag, unsigned size, const double* x_ptr, const double * y_ptr) {
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    exblas::exdot_cpu( size, x_ptr,y_ptr, &h_superacc[0]) ;
    return h_superacc;
}
template< class Vector, class UnaryOp>
inline void doTransform_dispatch( SerialTag, const Vector& x, Vector& y, UnaryOp op) {
    thrust::transform( thrust::cpp::tag(), x.begin(), x.end(), y.begin(), op);
}
template< class T>
inline void doScal_dispatch( SerialTag, unsigned size, T* x, T alpha)
{
    for( unsigned i=0; i<size; i++)
        x[i]*=alpha;
}
template<class T>
inline void doPlus_dispatch( SerialTag, unsigned size, T* x, T alpha)
{
    for( unsigned i=0; i<size; i++)
        x[i]+=alpha;
}
template<class T>
void doAxpby_dispatch( SerialTag, unsigned size, T alpha, const T * RESTRICT x, T beta, T* RESTRICT y)
{
    for( unsigned i=0; i<size; i++)
        y[i] = alpha*x[i] + beta*y[i];
}
template<class T>
void doAxpbypgz_dispatch( SerialTag, unsigned size, T alpha, const T * RESTRICT x, T beta, const T* RESTRICT y, T gamma, T* RESTRICT z)
{
    for( unsigned i=0; i<size; i++)
        z[i] = alpha*x[i] + beta*y[i] + gamma*z[i];
}
template<class value_type>
inline void doPointwiseDot_dispatch( SerialTag, unsigned size, 
              value_type alpha, 
              const value_type* x_ptr,
              const value_type* y_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    for( unsigned i=0; i<size; i++)
        z_ptr[i] = alpha*x_ptr[i]*y_ptr[i]+gamma*z_ptr[i];
}
template<class value_type>
inline void doPointwiseDivide_dispatch( SerialTag, unsigned size, 
              value_type alpha, 
              const value_type* x_ptr,
              const value_type* y_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    for( unsigned i=0; i<size; i++)
        z_ptr[i] = alpha*x_ptr[i]/y_ptr[i]+gamma*z_ptr[i];
}

template<class value_type>
inline void doPointwiseDot_dispatch( SerialTag, unsigned size, 
              value_type alpha, 
              const value_type* x1_ptr,
              const value_type* y1_ptr,
              value_type beta, 
              const value_type* x2_ptr,
              const value_type* y2_ptr,
              value_type gamma,
              value_type* z_ptr)
{
    for( unsigned i=0; i<size; i++)
        z_ptr[i] = alpha*x1_ptr[i]*y1_ptr[i] 
                   +beta*x2_ptr[i]*y2_ptr[i]
                   +gamma*z_ptr[i];
}

template<class value_type>
inline void doPointwiseDot_dispatch( SerialTag, unsigned size, 
              value_type alpha, 
              const value_type* x1,
              const value_type* x2,
              const value_type* x3,
              value_type beta, 
              value_type* y)
{
    for( unsigned i=0; i<size; i++)
        y[i] = alpha*x1[i]*x2[i]*x3[i]+beta*y[i];
}




}//namespace detail
}//namespace blas1
}//namespace dg
#endif //_DG_BLAS_SERIAL_
