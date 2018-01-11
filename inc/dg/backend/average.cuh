#pragma once

#include "memory.h"
#include "../blas1.h"
#include "dg/geometry/split_and_join.h"

/*! @file 
  @brief contains classes for poloidal and toroidal average computations.
  */
namespace dg{

/**
 * @brief Class for y average computations
 *
 * @snippet backend/average_t.cu doxygen
 * @ingroup utilities
 * @tparam container Currently this is one of 
 *  - \c dg::HVec, \c dg::DVec, \c dg::MHVec or \c dg::MDVec  
 * @tparam IndexContainer Type of index vectors; May equal \c container
 */
template< class container, class IndexContainer>
struct PoloidalAverage
{
    /**
     * @brief Construct from grid object
     *
     * @param g 2d Grid
     */
    PoloidalAverage( const aTopology2d& g):
        dummy( g.n()*g.Nx()), 
        helper( g.size()), helper1d( g.n()*g.Nx()), ly_(g.ly())
    {
        invertxy = create::scatterMapInvertxy( g.n(), g.Nx(), g.Ny());
        lines = create::contiguousLineNumbers( g.n()*g.Nx(), g.n()*g.Ny());
        Grid2d gTr( g.y0(), g.y1(), g.x0(), g.x1(), g.n(), g.Ny(), g.Nx());
        w2d = dg::create::weights( gTr);
        Grid1d g1x( 0, g.lx(), g.n(), g.Nx());
        v1d = dg::create::inv_weights( g1x);

    }
    /**
     * @brief Compute the average in y-direction
     *
     * @param src 2D Source vector 
     * @param res 2D result vector (may not equal src), every line contains the x-dependent average over
     the y-direction of src 
     */
    void operator() (const container& src, container& res)
    {
        assert( &src != &res);
        res.resize( src.size());
        thrust::scatter( src.begin(), src.end(), invertxy.begin(), helper.begin());
        //weights to ensure correct integration
        blas1::pointwiseDot( helper, w2d, helper);
        thrust::reduce_by_key( lines.begin(), lines.end(), helper.begin(), dummy.begin(), helper1d.begin());
        blas1::scal( helper1d, 1./ly_);
        //remove 1d weights in x-direction
        blas1::pointwiseDot( helper1d, v1d, helper1d);
        //copy to a full vector
        thrust::copy( helper1d.begin(), helper1d.end(), res.begin());
        unsigned pos = helper1d.size();
        while ( 2*pos < res.size() )
        {
            thrust::copy_n( res.begin(), pos, res.begin() + pos);
            pos*=2; 
        }
        thrust::copy_n( res.begin(), res.size() - pos, res.begin() + pos);

    }
  private:
    IndexContainer invertxy, lines, dummy; //dummy contains output keys i.e. line numbers
    container helper, helper1d;
    container w2d, v1d;
    double ly_;
};

/**
 * @brief Compute the average in the 3rd direction
 *
 * @tparam container must be either \c dg::HVec or \c dg::DVec
 * @param src 3d Source vector 
 * @param res contains the 2d result on output (may not equal src, gets resized properly)
 * @param t Contains the grid sizes 
 * @ingroup utilities
 */
template<class container>
void toroidal_average( const container& src, container& res, const aTopology3d& t)
{
    std::vector<container> split_src;
    dg::split( src, split_src, t);
    res = split_src[0];
    for( unsigned k=1; k<t.Nz(); k++)
        dg::blas1::axpby(1.0,split_src[k],1.0,res); 
    dg::blas1::scal(res,1./(double)t.Nz()); //scale avg
}

}//namespace dg
