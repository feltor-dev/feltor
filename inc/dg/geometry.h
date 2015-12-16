#pragma once

#include <cassert>
#include "thrust/host_vector.h"
#include "backend/mpi_vector.h"

#include "../enums.h"
#include "geometry/geometry_traits.h"

namespace dg{
namespace geo{

template<class container, class Geometry>
void multiplyVolume( container& inout, const Geometry& g)
{
    dg::geo::detail::doMultiplyVolume( inout, g, typename dg::GeometryTraits<Geometry>::metric_category());
}
template<class container, class Geometry>
void divideVolume( container& inout, const Geometry& g)
{
    dg::geo::detail::doDivideVolume( inout, g, typename dg::GeometryTraits<Geometry>::metric_category());
}

template<class container, class Geometry>
void raisePerpIndex( container& in1, container& in2, container& out1, container& out2, const Geometry& g)
{
    assert( &in1 != &out1);
    assert( &in2 != &out2);
    assert( &in2 != &in1);
    dg::geo::detail::doRaisePerpIndex( in1, in2, out2, out2, g, typename dg::GeometryTraits<Geometry>::metric_category());

}
template<class container, class Geometry>
void multiplyPerpVolume( container& inout, const Geometry& g)
{
    dg::geo::detail::doMuliplyPerpVolume( inout, g, typename dg::GeometryTraits<Geometry>::metric_category());
}
template<class container, class Geometry>
void dividePerpVolume( container& inout, const Geometry& g)
{
    dg::geo::detail::doDividePerpVolume( inout, g, typename dg::GeometryTraits<Geometry>::metric_category());
}

template<class TernaryOp1, class TernaryOp2, class Geometry> 
void pushforwardPerp( TernaryOp1 f1, TenaryOp2& f2, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out1, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out2,
        const Geometry& g)
{
    return doPushForwardPerp( f1, f2, out1, out2, g, typename GeometryTraits<Geometry>::metric_category() ); 
}

template<class Geometry> 
void pushforwardPerp( dg::system sys, 
        double(f1)(double,double,double), double(f2)(double, double, double), 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out1, 
        typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector& out2,
        const Geometry& g)
{
    pushforwardPerp<container, double(double, double, double), double(double, double, double), Geometry>( f1, f2, out1, out2, g); 
}

}//namespace geo

namespace create{

namespace detail{

template<class MemoryTag>
struct HostVec {
}
template<>
struct HostVec< SharedTag>
{
    typedef thrust::host_vector<double> host_vector;
}
template<>
struct HostVec< MPITag>
{
    typedef MPI_Vector<thrust::host_vector<double> > host_vector;
}

}//namespace detail

template< class Geometry>
typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector volume( const Geometry& g)
{
    typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector weights = dg::create::weights( g);
    dg::geo::multiplyVolume( weights, g);
    return weights;
}

template< class Geometry>
typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector inv_volume( const Geometry& g)
{
    typename HostVec< typename GeometryTraits<Geometry>::memory_category>::host_vector weights = dg::create::inv_weights( g);
    dg::geo::divideVolume( weights, g);
    return weights;
}

}//namespace create

}//namespace dg
