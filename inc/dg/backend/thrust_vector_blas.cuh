#ifndef _DG_BLAS_VECTOR_
#define _DG_BLAS_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "vector_categories.h"
#include "vector_traits.h"


namespace dg
{
namespace blas1
{
    ///@cond
namespace detail
{


template< typename value_type>
struct Axpby_Functor
{
    Axpby_Functor( value_type alpha, value_type beta): alpha(alpha), beta(beta) {}
    __host__ __device__
        value_type operator()( const value_type& x, const value_type& y)
        {
            return alpha*x+beta*y;
        }
    __host__ __device__
        value_type operator()( const value_type& y)
        {
            return beta*y;
        }
  private:
    value_type alpha, beta;
};

template <class value_type>
struct Plus_Functor
{
    Plus_Functor( value_type alpha): alpha(alpha){}
    
    __host__ __device__
        value_type operator()( const value_type& x)
        {
            return alpha+x;
        }
  private:
    value_type alpha;
};

template< class Vector1, class Vector2>
void doTransfer( const Vector1& in, Vector2& out, ThrustVectorTag, ThrustVectorTag)
{
    out.resize(in.size());
    thrust::copy( in.begin(), in.end(), out.begin());
}

template< class Vector>
typename Vector::value_type doDot( const Vector& x, const Vector& y, ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    typedef typename Vector::value_type value_type;
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
    value_type sum = 0;
    unsigned size=x.size();
    #pragma omp parallel for simd reduction(+:sum) 
    for( unsigned i=0; i<size; i++)
        sum += x[i]*y[i];
    return sum;
#else
    return thrust::inner_product( x.begin(), x.end(),  y.begin(), value_type(0));
#endif
}

template< class Vector, class UnaryOp>
inline void doTransform(  const Vector& x, Vector& y,
                          UnaryOp op,
                          ThrustVectorTag)
{
    thrust::transform( x.begin(), x.end(), y.begin(), op);
}

template< class Vector>
inline void doScal(  Vector& x, 
              typename Vector::value_type alpha, 
              ThrustVectorTag)
{
    if( alpha == 1.) 
        return;
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
    unsigned size=x.size();
    #pragma omp parallel for simd
    for( unsigned i=0; i<size; i++)
        x[i]*=alpha;
#else
    thrust::transform( x.begin(), x.end(), x.begin(), 
            detail::Axpby_Functor<typename Vector::value_type>( 0, alpha));
#endif
}
template< class Vector>
inline void doPlus(  Vector& x, 
              typename Vector::value_type alpha, 
              ThrustVectorTag)
{
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
    unsigned size=x.size();
    #pragma omp parallel for simd
    for( unsigned i=0; i<size; i++)
        x[i]+=alpha;
#else
    thrust::transform( x.begin(), x.end(), x.begin(), 
            detail::Plus_Functor<typename Vector::value_type>( alpha));
#endif
}

template< class Vector>
inline void doAxpby( typename Vector::value_type alpha, 
              const Vector& x, 
              typename Vector::value_type beta, 
              Vector& y, 
              ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() );
#endif //DG_DEBUG
    if( alpha == 0)
    {
        doScal( y, beta, ThrustVectorTag());
        return;
    }
    if( &x == &y)
    {
        doScal( y, (alpha+beta), ThrustVectorTag());
        return;
    }
    if( alpha==1. && beta == 0) 
    {
        thrust::copy( x.begin(), x.end(), y.begin());
        return; 
    }
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
    const typename Vector::value_type * RESTRICT x_ptr = thrust::raw_pointer_cast( &x.data()[0]);
    typename Vector::value_type * RESTRICT y_ptr = thrust::raw_pointer_cast( &y.data()[0]);
    unsigned size = x.size();
    #pragma omp parallel for simd
    for( unsigned i=0; i<size; i++)
        y_ptr[i] = alpha*x_ptr[i] + beta*y_ptr[i];
#else
    if( beta != 0)
        thrust::transform( x.begin(), x.end(), y.begin(), y.begin(), 
            detail::Axpby_Functor< typename Vector::value_type>( alpha, beta) );
    else 
        thrust::transform( x.begin(), x.end(), y.begin(),
            detail::Axpby_Functor< typename Vector::value_type>( 0., alpha) );
#endif
}

template< class Vector>
inline void doAxpby( typename Vector::value_type alpha, 
              const Vector& x, 
              typename Vector::value_type beta, 
              const Vector& y, 
              typename Vector::value_type gamma, 
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
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
    const typename Vector::value_type * RESTRICT x_ptr = thrust::raw_pointer_cast( &x.data()[0]);
    const typename Vector::value_type * RESTRICT y_ptr = thrust::raw_pointer_cast( &y.data()[0]);
    typename Vector::value_type * RESTRICT z_ptr = thrust::raw_pointer_cast( &z.data()[0]);
    unsigned size = x.size();
    #pragma omp parallel for simd
    for( unsigned i=0; i<size; i++)
        z_ptr[i] = alpha*x_ptr[i] + beta*y_ptr[i] + gamma*z_ptr[i];
#else
    if( gamma==0)
    {
        thrust::transform( x.begin(), x.end(), y.begin(), z.begin(), 
            detail::Axpby_Functor< typename Vector::value_type>( alpha, beta) );
    }
    else 
    {
        doAxpby( alpha, x, gamma, z, ThrustVectorTag());
        doAxpby( beta, y, 1., z, ThrustVectorTag());
    }

#endif
}

template< class Vector>
inline void doAxpby( typename Vector::value_type alpha, 
              const Vector& x, 
              typename Vector::value_type beta, 
              const Vector& y, 
              Vector& z, 
              ThrustVectorTag)
{
    doAxpby( alpha, x, beta, y, 0., z, ThrustVectorTag());
}


template < class Vector>
struct ThrustVectorDoSymv
{
    typedef typename Vector::value_type value_type;
    typedef thrust::tuple< value_type, value_type> Pair; 
    __host__ __device__
        ThrustVectorDoSymv( value_type alpha, value_type beta): alpha_(alpha), beta_(beta){}

    __host__ __device__
        value_type operator()( const value_type& y, const Pair& p) 
        {
            return alpha_*thrust::get<0>(p)*thrust::get<1>(p) + beta_*y;
        }
  private:
    value_type alpha_, beta_;
};

template<class Vector>
inline void doPointwiseDot(  
              typename Vector::value_type alpha, 
              const Vector& x1,
              const Vector& x2, 
              typename Vector::value_type beta, 
              Vector& y, 
              ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x1.size() == y.size() && x2.size() == y.size() );
#endif //DG_DEBUG
    if( alpha == 0)
    {
        dg::blas1::detail::doScal(y, beta, dg::ThrustVectorTag());
        return;
    }
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
    const typename Vector::value_type * x1_ptr = thrust::raw_pointer_cast( &(x1.data()[0]));
    const typename Vector::value_type * x2_ptr = thrust::raw_pointer_cast( &(x2.data()[0]));
     typename Vector::value_type * y_ptr = thrust::raw_pointer_cast( &(y.data()[0]));
    unsigned size = x1.size();
    #pragma omp parallel for simd
    for( unsigned i=0; i<size; i++)
    {
        y_ptr[i] = alpha*x1_ptr[i]*x2_ptr[i]+beta*y_ptr[i];
    }
#else
    thrust::transform( 
        y.begin(), y.end(),
        thrust::make_zip_iterator( thrust::make_tuple( x1.begin(), x2.begin() )),  
        y.begin(),
        detail::ThrustVectorDoSymv<Vector>( alpha, beta)
    ); 
#endif
}

template< class Vector>
inline void doPointwiseDot( const Vector& x1, const Vector& x2, Vector& y, ThrustVectorTag)
{
    doPointwiseDot( 1., x1, x2, 0., y, ThrustVectorTag());
}

template< class Vector>
inline void doPointwiseDivide( const Vector& x1, const Vector& x2, Vector& y, ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x1.size() == x2.size() );
    assert( x1.size() == y.size() );
#endif //DG_DEBUG
    thrust::transform( x1.begin(), x1.end(), x2.begin(), y.begin(), 
                        thrust::divides<typename VectorTraits<Vector>::value_type>());
}


#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
template<class value_type>
 __global__ void pointwiseDot_kernel( value_type alpha, value_type beta, value_type gamma,
         const value_type*  x1, const value_type* y1, const value_type* x2, 
         const value_type*  y2, value_type* z,  
         const int size
         )
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_rows/grid_size rows
    for( int row = thread_id; row<size; row += grid_size)
    {
        z[row]=alpha*x1[row]*y1[row]+beta*x2[row]*y2[row]+gamma*z[row];
    }
}
#endif

template<class value_type>
inline void doPointwiseDot(  
              value_type alpha, 
              const thrust::device_vector<value_type>& x1,
              const thrust::device_vector<value_type>& y1, 
              value_type beta, 
              const thrust::device_vector<value_type>& x2,
              const thrust::device_vector<value_type>& y2, 
              value_type gamma, 
              thrust::device_vector<value_type>& z, 
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
    const value_type *x1_ptr = thrust::raw_pointer_cast( x1.data());
    const value_type *x2_ptr = thrust::raw_pointer_cast( x2.data());
    const value_type *y1_ptr = thrust::raw_pointer_cast( y1.data());
    const value_type *y2_ptr = thrust::raw_pointer_cast( y2.data());
          value_type * z_ptr = thrust::raw_pointer_cast( z.data());
    unsigned size = x1.size();
#if THRUST_DEVICE_SYSTEM!=THRUST_DEVICE_SYSTEM_CUDA
    #pragma omp parallel for simd
    for( unsigned i=0; i<size; i++)
    {
        z_ptr[i] = alpha*x1_ptr[i]*y1_ptr[i] 
                    +beta*x2_ptr[i]*y2_ptr[i]
                    +gamma*z_ptr[i];
    }
#else
    //set up kernel parameters
    const size_t BLOCK_SIZE = 256; 
    const size_t NUM_BLOCKS = std::min<size_t>((size-1)/BLOCK_SIZE+1, 65000);
    pointwiseDot_kernel<value_type><<<NUM_BLOCKS, BLOCK_SIZE>>>( alpha, beta, gamma, x1_ptr, y1_ptr, x2_ptr, y2_ptr, z_ptr, size);
#endif
}
template<class value_type>
inline void doPointwiseDot(  
              value_type alpha, 
              const thrust::host_vector<value_type>& x1,
              const thrust::host_vector<value_type>& y1, 
              value_type beta, 
              const thrust::host_vector<value_type>& x2,
              const thrust::host_vector<value_type>& y2, 
              value_type gamma, 
              thrust::host_vector<value_type>& z, 
              ThrustVectorTag)
{
    unsigned size=x1.size();
    for( unsigned i=0; i<size; i++)
    {
        z[i] = alpha*x1[i]*y1[i] 
                    +beta*x2[i]*y2[i]
                    +gamma*z[i];
    }
}

}//namespace detail

///@endcond
} //namespace blas1
} //namespace dg

#endif //_DG_BLAS_VECTOR_
