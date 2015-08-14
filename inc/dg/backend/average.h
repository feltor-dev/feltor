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
        g2d(g.ghostless()),
        g1d( g2d.x0(),g2d.x1(),g2d.n(),g2d.Nx(),g2d.bcx()),
        dummy( g2d.n()*g2d.Nx()),
        helper1d(  g2d.n()*g2d.Nx()),
        helper( g2d.size()),helper2( g2d.size()),
        ly_(g2d.ly()) 
    {
            invertxy = create::scatterMapInvertxy( g2d.n(), g2d.Nx(), g2d.Ny());
            lines = create::contiguousLineNumbers( g2d.n()*g2d.Nx(), g2d.n()*g2d.Ny());
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
        const thrust::host_vector<double>& in = src.cut_overlap(); // src.without ghost
        //weight to ensure correct integration
        blas1::pointwiseDot( in, w2d, helper);      
        thrust::scatter( helper.begin(), helper.end(), invertxy.begin(), helper2.begin());
        thrust::reduce_by_key( lines.begin(), lines.end(), helper2.begin(), dummy.begin(), helper1d.begin());

        blas1::axpby( 1./ly_, helper1d, 0, helper1d); // helper1d = helper1d/ly
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
        res.copy_into_interior(helper);
    }
  private:
    IndexContainer dummy, invertxy, lines;
    dg::Grid2d<double> g2d;
    dg::Grid1d<double> g1d;
    HVec helper1d ; //, dummy, invertxy, lines;
    HVec helper,helper2;
    HVec v1d;
    HVec w2d;
    double ly_;
};


}//namespace dg
