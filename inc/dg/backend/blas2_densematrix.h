#pragma once

#include "densematrix.h"

namespace dg{
namespace blas2{
namespace detail{

// s.a. the detail::gemm algorithm in runge_kutta.h
template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              get_value_type<Vector1> alpha,
              Matrix&& m,
              const Vector1& x,
              get_value_type<Vector1> beta,
              Vector2& y,
              DenseMatrixTag)
{
    static_assert( std::is_same<get_execution_policy<Matrix>,
                                get_execution_policy<Vector2>>::value,
                "Dense Matrix and output Vector type must have same execution policy");
    static_assert( std::is_same<get_tensor_category<
                            typename std::decay_t<Matrix>::container_type>,
                        get_tensor_category<Vector2>>::value,
                "Dense Matrix and output Vector type must have same data layout");
    static_assert( std::is_same<get_tensor_category<
                            typename std::decay_t<Matrix>::container_type>,
                        get_tensor_category<Vector2>>::value,
                "Dense Matrix and output Vector type must have same data layout");
    const unsigned size = x.size();
    if( size != m.num_cols()) {
        throw Error( Message(_ping_)<<"x has the wrong size "<<x.size()<<" Number of columns is "<<m.num_cols());
    }
    // naive summation
    //dg::blas1::scal( y, beta);
    //for( unsigned i=0; i<size; i++)
    //    dg::blas1::axpby( alpha*x[i], *m.get()[i], 1., y);
    unsigned i=0;
    if( size >= 8)
    {
        for( i=0; i<size/8; i++)
            dg::blas1::evaluate( y, dg::Axpby(alpha, i == 0 ? beta : 1.), dg::PairSum(),
                    x[i*8+0], *m.get()[i*8+0],
                    x[i*8+1], *m.get()[i*8+1],
                    x[i*8+2], *m.get()[i*8+2],
                    x[i*8+3], *m.get()[i*8+3],
                    x[i*8+4], *m.get()[i*8+4],
                    x[i*8+5], *m.get()[i*8+5],
                    x[i*8+6], *m.get()[i*8+6],
                    x[i*8+7], *m.get()[i*8+7]);
    }
    unsigned l=0;
    if( size%8 >= 4)
    {
        for( l=0; l<(size%8)/4; l++)
            dg::blas1::evaluate( y, dg::Axpby(alpha,size < 8 ? beta : 1.), dg::PairSum(),
                    x[i*8+l*4+0], *m.get()[i*8+l*4+0],
                    x[i*8+l*4+1], *m.get()[i*8+l*4+1],
                    x[i*8+l*4+2], *m.get()[i*8+l*4+2],
                    x[i*8+l*4+3], *m.get()[i*8+l*4+3]);
    }
    unsigned k=0;
    if( (size%8)%4 >= 2)
    {
        for( k=0; k<((size%8)%4)/2; k++)
            dg::blas1::evaluate( y, dg::Axpby(alpha, size < 4 ? beta : 1.), dg::PairSum(),
                    x[i*8+l*4+k*2+0], *m.get()[i*8+l*4+k*2+0],
                    x[i*8+l*4+k*2+1], *m.get()[i*8+l*4+k*2+1]);
    }
    if( ((size%8)%4)%2 == 1)
    {
        dg::blas1::axpby( alpha*x[i*8+l*4+k*2], *m.get()[i*8+l*4+k*2], size < 2 ? beta: 1., y);
    }
}

template< class Matrix, class Vector1, class Vector2>
inline void doSymv(
              Matrix&& m,
              const Vector1& x,
              Vector2& y,
              DenseMatrixTag)
{
    doSymv( 1., std::forward<Matrix>(m), x, 0., y, DenseMatrixTag());
}

} //namespace detail
} //namespace blas2
} //namespace dg
