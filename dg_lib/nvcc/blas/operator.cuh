#ifndef _DG_BLAS_OPERATOR_
#define _DG_BLAS_OPERATOR_

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

#include "../array.cuh"
#include "../operators.cuh"
#include "thrust_vector.cuh"
#include "../vector_categories.h"
#include "../matrix_categories.h"

namespace dg{
namespace blas2{
namespace detail{

template< class Op> 
struct Operator_Symv_Functor
{
    typedef typename Op::value_type value_type;
    typedef typename Op::array_type array_type;

    Operator_Symv_Functor( value_type alpha, value_type beta, const Op& op ): op_(op), alpha(alpha), beta(beta) {}
    __host__ __device__
        array_type operator()( const array_type& x, const array_type& y)
        {
            array_type tmp( (value_type)0);
            for( unsigned i=0; i<tmp.size(); i++)
            {
                for( unsigned j=0; j<tmp.size(); j++)
                    tmp[i] +=alpha*op_(i,j)*x[j];
                tmp[i] += beta*y[i];
            }
            return tmp;
        }
    __host__ __device__
        array_type operator()(const array_type& arr)
        {
            array_type tmp( (value_type)0);
            for( unsigned i=0; i<tmp.size(); i++)
                for( unsigned j=0; j<tmp.size(); j++)
                    tmp[i] +=op_(i,j)*arr[j];
            return tmp;
        }
  private:
    const Op op_;
    value_type alpha, beta;
};

template< class Ptr>
Ptr* recast( Ptr* const ptr, thrust::input_host_iterator_tag) { return ptr;}

template< class Ptr>
thrust::device_ptr< Ptr> recast( Ptr* const ptr, thrust::input_device_iterator_tag) 
{ 
    return thrust::device_ptr< Ptr>(ptr);
}


template< class Matrix, class Vector>
inline void doSymv(  
              typename Matrix::value_type alpha, 
              const Matrix& m,
              const Vector& x, 
              typename Matrix::value_type beta, 
              Vector& y, 
              OperatorMatrixTag,
              ThrustVectorTag)
{
    if( alpha == 0)
    {
        if( beta == 1) 
            return;
        dg::blas1::detail::doAxpby( 0., x, beta, y, dg::ThrustVectorTag());
        return;
    }
    typename thrust::iterator_traits< typename Vector::iterator>::iterator_category tag;
    typedef typename Matrix::array_type array_type;
    const array_type * xbegin = reinterpret_cast<array_type const *>(thrust::raw_pointer_cast( &x[0]));  
    array_type const * xend = xbegin + x.size()/xbegin->size() -1;  
    array_type * ybegin = reinterpret_cast<array_type *>(thrust::raw_pointer_cast(&y[0]));  

    thrust::transform( recast( xbegin, tag), recast(xend, tag), 
                       recast( ybegin, tag),
                       recast( ybegin, tag),
                       detail::Operator_Symv_Functor<Matrix>( alpha, beta, m)
                       );
}
template< class Matrix, class Vector>
inline void doSymv(  
              const Matrix& m, 
              const Vector& x,
              Vector& y, 
              OperatorMatrixTag,
              ThrustVectorTag)
{
    typename thrust::iterator_traits< typename Vector::iterator>::iterator_category tag;
    typedef typename Matrix::array_type array_type;
    const array_type * xbegin = reinterpret_cast<array_type const *>(thrust::raw_pointer_cast( &x[0]));  
    array_type const * xend = xbegin + x.size()/xbegin->size() -1;  
    array_type * ybegin = reinterpret_cast<array_type *>(thrust::raw_pointer_cast(&y[0]));  

    thrust::transform( recast( xbegin, tag), recast(xend, tag), 
                       recast( ybegin, tag),
                       detail::Operator_Symv_Functor<Matrix>( 0, 0, m)
                       ); //slightly faster
}


}//namespace detail
} //namespace blas2
} //namespace dg
#endif //_DG_BLAS_OPERATOR_
