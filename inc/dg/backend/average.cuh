#pragma once

#include "evaluation.cuh"
#include "xspacelib.cuh"
#include "../blas1.h"

/*! @file 

  Contains classes for poloidal and toroidal average computations.

  */
namespace dg{
struct printf_functor
{
__host__ __device__
void operator()(double x)
{
    printf("%f\n",x);
}
};
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
    PoloidalAverage( const Grid2d<double>& g):dummy( g.n()*g.Nx()), helper( g.size()), helper1d( g.n()*g.Nx()), ly_(g.ly())
    {
        invertxy = create::scatterMapInvertxy( g.n(), g.Nx(), g.Ny());
        lines = create::contiguousLineNumbers( g.n()*g.Nx(), g.n()*g.Ny());
        w2d = dg::create::weights( g);
        Grid1d<double> g1x( 0, g.lx(), g.n(), g.Nx());
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
        //weight to ensure correct integration
        blas1::pointwiseDot( src, w2d, helper);
        thrust::scatter( helper.begin(), helper.end(), invertxy.begin(), res.begin());
        thrust::reduce_by_key( lines.begin(), lines.end(), res.begin(), dummy.begin(), helper1d.begin());
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
        thrust::copy( helper.begin(), helper.end(), res.begin());
        thrust::for_each(thrust::host,res.begin(), res.end(), printf_functor());


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
    ToroidalAverage(const dg::Grid3d<double>& g3d):
        g3d_(g3d),
        sizeg2d_(g3d_.size()/g3d_.Nz())
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
        for( unsigned k=0; k<g3d_.Nz(); k++)
        {
            container data2d(src.begin() + k*sizeg2d_,src.begin() + (k+1)*sizeg2d_);
            dg::blas1::axpby(1.0,data2d,1.0,res); 
        }
        dg::blas1::scal(res,1./g3d_.Nz()); //scale avg
    }
    private:
    const dg::Grid3d<double>& g3d_;
    unsigned sizeg2d_;
};
}//namespace dg
