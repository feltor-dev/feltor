#pragma once

#include <cassert>

#include "../enums.h"

namespace dg{
namespace geo{

template<class container, class Geometry>
void attachVolume( container& inout, const Geometry& g)
{
    dg::geo::detail::doAttachVolume( inout, g, typename dg::GeometryTraits<Geometry>::metric_category());
}

template<class container, class Geometry>
void raiseIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g)
{
    assert( &in1 != &out1);
    assert( &in2 != &out2);
    assert( &in2 != &in1);
    dg::geo::detail::doRaiseIndex( in1, in2, out2, out2, g, typename dg::GeometryTraits<Geometry>::metric_category());

}

template<class TernaryOp, class Geometry> 
thrust::host_vector<double> pullback( dg::system sys, TernaryOp f, const Geometry& g)
{
    return doPullback( sys, f, g, typename GeometryTraits<Geometry>::metric_category()); 
}
template<class Geometry> 
thrust::host_vector<double> pullback( dg::system sys, double(f)(double, double, double), const Geometry& g)
{
    pullback<double(double, double, double), Geometry>( sys, f, g); 
}
template<class TernaryOp, class Geometry> 
void pushforward( dg::system sys, TernaryOp f1, TenaryOp& f2, thrust::host_vector<double>& out1, thrust::host_vector<double>& out2, const Geometry& g)
{
    return doPushForward( sys, f1, f2, out1, out2, g, typename GeometryTraits<Geometry>::metric_category()); 
}

template<class Geometry> 
void pushforward( dg::system sys, double(f1)(double,double,double), double(f2)(double, double, double), thrust::host_vector<double>& out1, thrust::host_vector<double>& out2, const Geometry& g)
{
    pushforward<double(double, double, double), Geometry>( sys, f1, f2, out1, out2, g); 
}

}//namespace geo

namespace create{

template< class Geometry>
thrust::host_vector<double> volume( const Geometry& g)
{
    thrust::host_vector<double> weights = dg::create::weights( g);
    dg::geo::attachVolume( weights, g);
    return weights;
}

template< class Geometry>
thrust::host_vector<double> inv_volume( const Geometry& g)
{
    thrust::host_vector<double> weights = dg::create::weights( g);
    dg::geo::attachVolume( weights, g);
    for( unsigned i=0; i<weights.size(); i++)
        weights[i] = 1./weights[i];
    return weights;
}

}//namespace create

}//namespace dg
