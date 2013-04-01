#ifndef _DG_BLAS_OPERATOR_
#define _DG_BLAS_OPERATOR_

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

#include "../array.cuh"
#include "../operator.cuh"
#include "thrust_vector.cuh"
#include "../vector_categories.h"
#include "../matrix_categories.h"
#include "../operator_tuple.cuh"

namespace dg{
namespace blas2{
namespace detail{

template< class Op> 
struct Operator_Symv_Functor
{
    typedef typename Op::value_type value_type;
    typedef typename Op::array_type operand_type;

    Operator_Symv_Functor( value_type alpha, value_type beta, const Op& op ): op_(op), alpha(alpha), beta(beta) {}
    __host__ __device__
        operand_type operator()( const operand_type& x, const operand_type& y) const
        {
            operand_type tmp( (value_type)0);
            for( unsigned i=0; i<tmp.size(); i++)
            {
                for( unsigned j=0; j<tmp.size(); j++)
                    tmp[i] +=alpha*op_(i,j)*x[j];
                tmp[i] += beta*y[i];
            }
            return tmp;
        }
    __host__ __device__
        operand_type operator()(const operand_type& arr) const
        {
            operand_type tmp( (value_type)0);
            for( unsigned i=0; i<tmp.size(); i++)
                for( unsigned j=0; j<tmp.size(); j++)
                    tmp[i] +=op_(i,j)*arr[j];
            return tmp;
        }
  private:
    const Op op_;
    value_type alpha, beta;
};

//might become obsolete by the tensor function
template<class Op>
struct Operator_Symv_Functor<thrust::tuple<Op,Op> >
{
    typedef typename Op::value_type value_type;
    typedef typename Op::matrix_type matrix_type;
    typedef thrust::tuple< Op, Op> Pair;

    Operator_Symv_Functor( value_type alpha, value_type beta, const Pair& p ): 
                        op1( thrust::get<0>(p)), 
                        op2( thrust::get<1>(p)),  
                        alpha(alpha), beta(beta) {}
    __host__ __device__
        matrix_type operator()( const matrix_type& x, matrix_type& y) const
        {
            matrix_type tmp( (value_type)0);
            unsigned n = sqrt(tmp.size());
            //first transform each row
            for( unsigned i=0; i<n; i++) 
                for( unsigned j=0; j<n; j++)
                {
                    //multiply Op2 with each row k
                    for(  unsigned k=0; k<n; k++)
                        tmp[i*n+j] += op2(j, k)*x[ i*n+k];
                }
            //then transform each col
            for( unsigned i=0; i<n; i++) 
                for( unsigned j=0; j<n; j++)
                {
                    //multiply Op1 with each col 
                    y[i*n+j] *= beta;
                    for(  unsigned k=0; k<n; k++)
                        y[i*n+j] += alpha*op1(i,k)*tmp[k*n+j];
                }
            return y;
        }
    __host__ __device__
        matrix_type operator()(const matrix_type& x) const
        {
            matrix_type tmp( (value_type)0);
            unsigned n = sqrt(tmp.size());
            //first transform each row
            for( unsigned i=0; i<n; i++) 
                for( unsigned j=0; j<n; j++)
                    for(  unsigned k=0; k<n; k++)
                        tmp[i*n+j] += op2(j, k)*x[ i*n+k];
            //then transform each col
            matrix_type y( (value_type)0);
            for( unsigned i=0; i<n; i++) 
                for( unsigned j=0; j<n; j++)
                    for(  unsigned k=0; k<n; k++)
                        y[i*n+j] += op1(i,k)*tmp[k*n+j];
            return y;
        }
  private:
    const Op op1, op2; //neglecting op2 doesn't gain speed
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
              typename MatrixTraits<Matrix>::value_type alpha, 
              const Matrix& m,
              const Vector& x, 
              typename MatrixTraits<Matrix>::value_type beta, 
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
    typedef typename MatrixTraits<Matrix>::operand_type operand_type;
    const operand_type * xbegin = reinterpret_cast<operand_type const *>(thrust::raw_pointer_cast( &x[0]));  
    operand_type const * xend = xbegin + x.size()/xbegin->size();  
    operand_type * ybegin = reinterpret_cast<operand_type *>(thrust::raw_pointer_cast(&y[0]));  

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
    typedef typename MatrixTraits<Matrix>::operand_type operand_type;
    const operand_type * xbegin = reinterpret_cast<operand_type const *>(thrust::raw_pointer_cast( &x[0]));  
    operand_type const * xend = xbegin + x.size()/xbegin->size();  
    operand_type * ybegin = reinterpret_cast<operand_type *>(thrust::raw_pointer_cast(&y[0]));  

    thrust::transform( recast( xbegin, tag), recast(xend, tag), 
                       recast( ybegin, tag),
                       detail::Operator_Symv_Functor<Matrix>( 0, 0, m)
                       ); //slightly faster
}


}//namespace detail
} //namespace blas2
} //namespace dg
#endif //_DG_BLAS_OPERATOR_
