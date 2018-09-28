#pragma once

#include "gridX.h"
#include "evaluation.h"


/*! @file
  @brief Function discretization routines on X-point topology
  */
namespace dg
{

///@cond
namespace create{
template<class real_type>
thrust::host_vector<real_type> abscissas( const RealGridX1d<real_type>& g)
{
    return abscissas(g.grid());
}
}//namespace create
///@endcond

///@addtogroup evaluation
///@{

///@copydoc dg::evaluate(UnaryOp,const RealGrid1d<real_type>&)
template< class UnaryOp,class real_type>
thrust::host_vector<real_type> evaluate( UnaryOp f, const RealGridX1d<real_type>& g)
{
    return evaluate( f, g.grid());
};
///@cond
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type (f)(real_type), const RealGridX1d<real_type>& g)
{
    return evaluate( *f, g.grid());
};
///@endcond

///@copydoc dg::evaluate(const BinaryOp&,const aRealTopology2d<real_type>&)
template< class BinaryOp, class real_type>
thrust::host_vector<real_type> evaluate( const BinaryOp& f, const aRealTopologyX2d<real_type>& g)
{
    return evaluate( f, g.grid());
};
///@cond
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type(f)(real_type, real_type), const aRealTopologyX2d<real_type>& g)
{
    return evaluate( *f, g.grid());
};
///@endcond

///@copydoc dg::evaluate(const TernaryOp&,const aRealTopology3d<real_type>&)
template< class TernaryOp, class real_type>
thrust::host_vector<real_type> evaluate( const TernaryOp& f, const aRealTopologyX3d<real_type>& g)
{
    return evaluate( f, g.grid());
};
///@cond
template<class real_type>
thrust::host_vector<real_type> evaluate( real_type(f)(real_type, real_type, real_type), const aRealTopologyX3d<real_type>& g)
{
    return evaluate( *f, g.grid());
};
///@endcond

///@}
}//namespace dg

