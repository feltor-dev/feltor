#ifndef _DG_BLAS_PRECONDITIONER_
#define _DG_BLAS_PRECONDITIONER_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

#include "thrust_vector.cuh" //load thrust_vector BLAS1 routines
#include "../vector_categories.h"

namespace dg{
namespace blas2{
namespace detail{

//thrust vector preconditioner

template < class Vector>
struct ThrustVectorDoDot
{
    typedef typename Vector::value_type value_type;
    typedef thrust::tuple< value_type, value_type> Pair; 
    __host__ __device__
        value_type operator()( const value_type & x, const Pair& p) {
            return thrust::get<0>(p)*thrust::get<1>(p)*x;
        }
    __host__ __device__
        value_type operator()( const value_type& x, const value_type& p) {
            return p*x*x;
        }
};

template< class Matrix, class Vector>
inline typename Matrix::value_type doDot( const Vector& x, const Matrix& m, const Vector& y, ThrustMatrixTag, ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() && x.size() == m.size() );
#endif //DG_DEBUG
    return thrust::inner_product(  x.begin(), x.end(), 
                            thrust::make_zip_iterator( thrust::make_tuple( y.begin(), m.begin())  ), 
                            0.0,
                            thrust::plus<double>(),
                            detail::ThrustVectorDoDot<Matrix>()
                            );
}
template< class Matrix, class Vector>
inline typename Matrix::value_type doDot( const Matrix& m, const Vector& x, dg::ThrustMatrixTag, dg::ThrustVectorTag)
{
    return thrust::inner_product( x.begin(), x.end(),
                                  m.begin(),
                                  0.0,
                                  thrust::plus<double>(),
                                  detail::ThrustVectorDoDot<Matrix>()
            ); //very fast
}

template < class Vector>
struct ThrustVectorDoSymv
{
    typedef typename Vector::value_type value_type;
    typedef thrust::tuple< value_type, value_type> Pair; 
    __host__ __device__
        ThrustVectorDoSymv( value_type alpha, value_type beta): alpha_(alpha), beta_(beta){}

    __host__ __device__
        value_type operator()(const value_type& x, const Pair& p) 
        {
            return alpha_*thrust::get<0>(p)*x + beta_*thrust::get<1>(p);
        }
  private:
    value_type alpha_, beta_;
};

template< class Matrix, class Vector>
inline void doSymv(  
              typename Matrix::value_type alpha, 
              const Matrix& m,
              const Vector& x, 
              typename Matrix::value_type beta, 
              Vector& y, 
              ThrustMatrixTag,
              ThrustVectorTag)
{
#ifdef DG_DEBUG
    assert( x.size() == y.size() && x.size() == m.size() );
#endif //DG_DEBUG
    if( alpha == 0)
    {
        if( beta == 1) 
            return;
        dg::blas1::detail::doAxpby( 0., x, beta, y, dg::ThrustVectorTag());
        return;
    }
    thrust::transform( x.begin(), x.end(), 
                       thrust::make_zip_iterator( thrust::make_tuple( m.begin(), y.begin() )),  
                       y.begin(),
                       detail::ThrustVectorDoSymv<Matrix>( alpha, beta)
                      ); 
}
template< class Matrix, class Vector>
inline void doSymv(  
              const Matrix& m, 
              const Vector& x,
              Vector& y, 
              ThrustMatrixTag,
              ThrustVectorTag)
{
    dg::blas1::detail::doPointwiseDot( m,x,y, dg::ThrustVectorTag());
}


}//namespace detail
} //namespace blas2
} //namespace dg
#endif //_DG_BLAS_PRECONDITIONER_
