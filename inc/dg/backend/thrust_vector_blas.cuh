#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "vector_categories.h"
#include "vector_traits.h"

#include "blas1_serial.h"
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include "blas1_cuda.cuh"
#else
#include "blas1_omp.h"
#endif


///@cond
namespace dg
{
template<class T>
struct VectorTraits<thrust::host_vector<T>, 
    typename std::enable_if< std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using vector_category   = ThrustVectorTag;
    using execution_policy  = SerialTag;
};
template<class T>
struct VectorTraits<thrust::host_vector<T>, 
    typename std::enable_if< !std::is_arithmetic<T>::value>::type>
{
    using value_type        = get_value_type<T>;
    using vector_category   = VectorVectorTag;
    using execution_policy  = get_execution_policy<T>;
};

template<class T>
struct VectorTraits<thrust::device_vector<T>, typename std::enable_if<std::is_arithmetic<T>::value>::type>
{
    using value_type        = T;
    using vector_category   = ThrustVectorTag;
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    using execution_policy  = CudaTag ; 
#else
    using execution_policy  = OmpTag ; 
#endif
};
//resize
//size
//data
//begin
//end

namespace blas1
{
namespace detail
{

template< class To, class From>
To doTransfer( const From& in, ThrustVectorTag, ThrustVectorTag)
{
    To t( in.begin(), in.end());
    return t;
}

template< class Vector>
std::vector<int64_t> doDot_superacc( const Vector& x, const Vector& y, ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    const double* x_ptr = thrust::raw_pointer_cast( x.data());
    const double* y_ptr = thrust::raw_pointer_cast( y.data());
    return doDot_dispatch( get_execution_policy<Vector>(), x.size(), x_ptr, y_ptr);
}

template<class Vector>
get_value_type<Vector> doDot( const Vector& x, const Vector& y, ThrustVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,y,ThrustVectorTag());
    return exblas::Round(acc.data());
}

template< class Vector, class UnaryOp>
inline void doTransform(  const Vector& x, Vector& y, UnaryOp op, ThrustVectorTag) {
    doTransform_dispatch( get_execution_policy<Vector>(), x,y,op);
}

template< class Vector>
inline void doScal( Vector& x, get_value_type<Vector> alpha, ThrustVectorTag)
{
    if( alpha == 1.) 
        return;
    get_value_type<Vector> * x_ptr = thrust::raw_pointer_cast( x.data());
    doScal_dispatch( get_execution_policy<Vector>(), x.size(), x_ptr, alpha);
}

template< class Vector>
inline void doPlus(  Vector& x, get_value_type<Vector> alpha, ThrustVectorTag)
{
    if(alpha==0)
        return;
    get_value_type<Vector> * x_ptr = thrust::raw_pointer_cast( x.data());
    return doPlus_dispatch( get_execution_policy<Vector>(), x.size(), x_ptr, alpha);
}

template< class Vector>
inline void doAxpby( get_value_type<Vector> alpha, 
              const Vector& x, 
              get_value_type<Vector> beta, 
              Vector& y, 
              ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    if( alpha == 0) {
        doScal( y, beta, ThrustVectorTag());
        return;
    }
    if( &x == &y) {
        doScal( y, (alpha+beta), ThrustVectorTag());
        return;
    }
    if( alpha==1. && beta == 0) {
        thrust::copy( x.begin(), x.end(), y.begin());
        return; 
    }
    const get_value_type<Vector> * x_ptr = thrust::raw_pointer_cast( x.data());
    get_value_type<Vector> * y_ptr = thrust::raw_pointer_cast( y.data());
    doAxpby_dispatch( get_execution_policy<Vector>(), x.size(), alpha, x_ptr, beta, y_ptr);
}

template< class Vector>
inline void doAxpbypgz( get_value_type<Vector> alpha, 
              const Vector& x, 
              get_value_type<Vector> beta, 
              const Vector& y, 
              get_value_type<Vector> gamma, 
              Vector& z, 
              ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
    assert( x.size() == z.size() );
#endif //DG_DEBUG
    if( alpha == 0)
    {
        doAxpby( beta, y, gamma, z, ThrustVectorTag());
        return;
    }
    else if( beta == 0)
    {
        doAxpby( alpha, x, gamma, z, ThrustVectorTag());
        return;
    }
    if( &x==&y)
    {
        doAxpby( alpha+beta, x, gamma, z, ThrustVectorTag());
        return;
    }
    else if( &x==&z)
    {
        doAxpby( beta, y, alpha+gamma, z, ThrustVectorTag());
        return;
    }
    else if( &y==&z)
    {
        doAxpby( alpha, x, beta+gamma, z, ThrustVectorTag());
        return;
    }
    const get_value_type<Vector> * x_ptr = thrust::raw_pointer_cast( x.data());
    const get_value_type<Vector> * y_ptr = thrust::raw_pointer_cast( y.data());
    get_value_type<Vector> * z_ptr = thrust::raw_pointer_cast( z.data());
    unsigned size = x.size();
    doAxpbypgz_dispatch( get_execution_policy<Vector>(), size, alpha, x_ptr, beta, y_ptr, gamma, z_ptr);
}

template<class Vector>
inline void doPointwiseDot(  
              get_value_type<Vector> alpha, 
              const Vector& x1,
              const Vector& x2, 
              get_value_type<Vector> beta, 
              Vector& y, 
              ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x1.size() == y.size() && x2.size() == y.size() );
#endif //DG_DEBUG
    if( alpha == 0) {
        dg::blas1::detail::doScal(y, beta, dg::ThrustVectorTag());
        return;
    }
    const get_value_type<Vector> * x1_ptr = thrust::raw_pointer_cast( x1.data());
    const get_value_type<Vector> * x2_ptr = thrust::raw_pointer_cast( x2.data());
    get_value_type<Vector> * y_ptr = thrust::raw_pointer_cast( y.data());
    unsigned size = x1.size();
    doPointwiseDot_dispatch( get_execution_policy<Vector>(), size, alpha, x1_ptr, x2_ptr, beta, y_ptr);
}

template<class Vector>
inline void doPointwiseDivide(  
              get_value_type<Vector> alpha, 
              const Vector& x1,
              const Vector& x2, 
              get_value_type<Vector> beta, 
              Vector& y, 
              ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x1.size() == y.size() && x2.size() == y.size() );
#endif //DG_DEBUG
    if( alpha == 0) {
        dg::blas1::detail::doScal(y, beta, dg::ThrustVectorTag());
        return;
    }
    const get_value_type<Vector> * x1_ptr = thrust::raw_pointer_cast( x1.data());
    const get_value_type<Vector> * x2_ptr = thrust::raw_pointer_cast( x2.data());
    get_value_type<Vector> * y_ptr = thrust::raw_pointer_cast( y.data());
    unsigned size = x1.size();
    doPointwiseDivide_dispatch( get_execution_policy<Vector>(), size, alpha, x1_ptr, x2_ptr, beta, y_ptr);
}

template<class Vector>
inline void doPointwiseDot(  
              get_value_type<Vector> alpha, 
              const Vector& x1,
              const Vector& y1, 
              get_value_type<Vector> beta, 
              const Vector& x2,
              const Vector& y2, 
              get_value_type<Vector> gamma, 
              Vector& z, 
              ThrustVectorTag)
{
    if( alpha==0){ 
        doPointwiseDot( beta, x2,y2, gamma, z, ThrustVectorTag());
        return;
    }
    else if( beta==0){
        doPointwiseDot( alpha, x1,y1, gamma, z, ThrustVectorTag());
        return;
    }
    const get_value_type<Vector> *x1_ptr = thrust::raw_pointer_cast( x1.data());
    const get_value_type<Vector> *x2_ptr = thrust::raw_pointer_cast( x2.data());
    const get_value_type<Vector> *y1_ptr = thrust::raw_pointer_cast( y1.data());
    const get_value_type<Vector> *y2_ptr = thrust::raw_pointer_cast( y2.data());
          get_value_type<Vector> * z_ptr = thrust::raw_pointer_cast( z.data());
    unsigned size = x1.size();
    doPointwiseDot_dispatch( get_execution_policy<Vector>(), size, alpha, x1_ptr, y1_ptr, beta, x2_ptr, y2_ptr, gamma, z_ptr);
}
template<class Vector>
inline void doPointwiseDot(  
              get_value_type<Vector> alpha, 
              const Vector& x1,
              const Vector& x2,
              const Vector& x3, 
              get_value_type<Vector> beta, 
              Vector& y, 
              ThrustVectorTag)
{
    if( alpha==0){ 
        doScal( y, beta, ThrustVectorTag());
        return;
    }
    const get_value_type<Vector> *x1_ptr = thrust::raw_pointer_cast( x1.data());
    const get_value_type<Vector> *x2_ptr = thrust::raw_pointer_cast( x2.data());
    const get_value_type<Vector> *x3_ptr = thrust::raw_pointer_cast( x3.data());
          get_value_type<Vector> * y_ptr = thrust::raw_pointer_cast( y.data());
    unsigned size = x1.size();
    doPointwiseDot_dispatch( get_execution_policy<Vector>(), size, alpha, x1_ptr, x2_ptr, x3_ptr, beta, y_ptr);
}

}//namespace detail

} //namespace blas1
} //namespace dg
///@endcond

#endif //_DG_BLAS_VECTOR_
