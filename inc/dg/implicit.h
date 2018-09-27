#pragma once
#include "cg.h"

///@cond
namespace dg{
namespace detail{

//compute: y + alpha f(y,t)
template< class LinearOp, class ContainerType>
struct Implicit
{
    using real_type = get_value_type<ContainerType>;
    Implicit( real_type alpha, real_type t, LinearOp& f): f_(f), alpha_(alpha), t_(t){}
    void symv( const ContainerType& x, ContainerType& y)
    {
        if( alpha_ != 0)
            f_(t_,x,y);
        blas1::axpby( 1., x, alpha_, y, y);
        blas2::symv( f_.weights(), y, y);
    }
    //compute without weights
    void operator()( const ContainerType& x, ContainerType& y)
    {
        if( alpha_ != 0)
            f_(t_,x,y);
        blas1::axpby( 1., x, alpha_, y, y);
    }
    real_type& alpha( ){  return alpha_;}
    real_type alpha( ) const  {return alpha_;}
    real_type& time( ){  return t_;}
    real_type time( ) const  {return t_;}
  private:
    LinearOp& f_;
    real_type alpha_;
    real_type t_;
};

}//namespace detail
template< class M, class V>
struct TensorTraits< detail::Implicit<M, V> >
{
    using value_type = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
}//namespace dg
///@endcond
