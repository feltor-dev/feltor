#pragma once

#include "xspacelib.cuh"
#include "../blas1.h"

/*! @file 

  Contains classes for poloidal and toroidal average computations.

  */
namespace dg{

/**
 * @brief Class for y average computations
 *
 * @ingroup utilities
 * @tparam container Vector class to be used
 * @tparam IndexContainer Class for scatter maps
 */
template< class container, class IndexContainer>
struct PoloidalAverage
{
    /**
     * @brief Construct from grid mpi object
     *
     * @param g 2d MPIGrid
     */
    template<class Grid2d>
    PoloidalAverage( const Grid2d& g) : 
        g1d( g.x0(), g.x1(),g.n(), g.Nx(),g.bcx()),
        g2d( g.x0(), g.x1(),g.y0(), g.y1(),g.n(), g.Nx(), g.Ny(),g.bcx(),g.bcy()),
        dummy( g.n()*g.Nx()),
        helper1d(  g.n()*g.Nx()),
        helper( g.size()),
        ly_(g.ly()) 
    {
            invertxy = create::scatterMapInvertxy( g.n(), g.Nx(), g.Ny());
            lines = create::contiguousLineNumbers( g.n()*g.Nx(), g.n()*g.Ny());
            w2d = create::weights( g2d);
            v1d = create::inv_weights( g1d);
    }
    /**
     * @brief Compute the average in y-direction
     *
     * @param src 2D Source MPIvector 
     * @param res 2D result MPIvector (may not equal src), every line contains the x-dependent average over
     the y-direction of src 
     */
    void operator() (const container& src, container& res)
    {
        assert( &src != &res);
        const thrust::host_vector<double>& in = src.data();
        thrust::host_vector<double>& out = res.data();
//         res.resize( src.size());
        //weight to ensure correct integration
        blas1::pointwiseDot( in, w2d, helper);
        thrust::scatter( helper.begin(), helper.end(), invertxy.begin(), out.begin());
        thrust::reduce_by_key( lines.begin(), lines.end(), out.begin(), dummy.begin(), helper1d.begin());
        blas1::axpby( 1./ly_, helper1d, 0, helper1d);
        //remove weights in x-direction
        blas1::pointwiseDot( helper1d, v1d, helper1d);
        //copy to a full vector
        thrust::copy( helper1d.begin(), helper1d.end(), helper.begin());
        unsigned pos = helper1d.size();
        while ( 2*pos < helper.size() )
        {
            thrust::copy_n( helper.begin(), pos, helper.begin() + pos);
            pos*=2; 
        }
        thrust::copy_n( helper.begin(), helper.size() - pos, helper.begin() + pos);
        //copy to result
        thrust::copy( helper.begin(), helper.end(), out.begin());
        dg::blas1::axpby(1.0,out,0.0,res.data(),res.data());
    }
  private:
    dg::Grid1d<double> g1d;
    dg::Grid2d<double> g2d;
    HVec helper1d,dummy,invertxy, lines;
    HVec helper;
    HVec v1d;
    HVec w2d;
    double ly_;
};


}//namespace dg
