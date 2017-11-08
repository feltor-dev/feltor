#pragma once

#include "evaluation.cuh"
#include "xspacelib.cuh"
#include "memory.h"
#include "../blas1.h"

/*! @file 
  @brief contains classes for poloidal and toroidal average computations.
  */
namespace dg{
//struct printf_functor
//{
//__host__ __device__
//void operator()(double x)
//{
//    printf("%f\n",x);
//}
//};
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
 * @brief Class for phi average computations
 *
 * @ingroup utilities
 * @tparam container Vector class to be used
 */
template< class container = thrust::host_vector<double> >
struct ToroidalAverage
{
    /**
     * @brief Construct from grid object
     *
     * @param g3d 3d Grid
     */
    ToroidalAverage(const dg::aTopology3d& g3d):
        g3d_(g3d),
        sizeg2d_(g3d_.get().size()/g3d_.get().Nz())
    {        
    }
    /**
     * @brief Compute the average in phi-direction
     *
     * @param src 3d Source vector 
     * @param res contains the 2d result on output (may not equal src)
     */
    void operator()(const container& src, container& res)
    {
        for( unsigned k=0; k<g3d_.get().Nz(); k++)
        {
            container data2d(src.begin() + k*sizeg2d_,src.begin() + (k+1)*sizeg2d_);
            dg::blas1::axpby(1.0,data2d,1.0,res); 
        }
        dg::blas1::scal(res,1./g3d_.get().Nz()); //scale avg
    }
    private:
    Handle<dg::aTopology3d> g3d_;
    unsigned sizeg2d_;
};
}//namespace dg
