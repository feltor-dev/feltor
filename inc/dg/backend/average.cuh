#pragma once

#include "xspacelib.cuh"

namespace dg{

/**
 * @brief Class for y average computations
 *
 * The problem is the dg format of the vector
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
        //reorder = create::scatterMap( g.n(), g.Nx(), g.Ny());
        lines = create::contiguousLineNumbers( g.n()*g.Nx(), g.n()*g.Ny());
        w2d = create::weights( g);
        Grid1d<double> g1x( 0, g.lx(), g.n(), g.Nx());
        v1d = create::inv_weights( g1x);

    }
    /**
     * @brief Compute the average in y-direction
     *
     * @param src Source vector 
     * @param res contains the result on output (may not equal src)
     */
    void operator() (const container& src, container& res)
    {
        assert( &src != &res);
        res.resize( src.size());
        //weight to ensure correct integration
        blas1::pointwiseDot( src, w2d, helper);
        thrust::scatter( helper.begin(), helper.end(), invertxy.begin(), res.begin());
        /*
        //std::cout << "Result of invertXY:\n";
        //for( unsigned i=0; i<9; i++)
        //{
        //    for( unsigned j=0; j<6; j++)
        //        std::cout << res[i*6+j]<<" ";
        //    std::cout<< "\n";
        //}
        //std::cout << "H1\n";
        //compute average
        //std::cout << "Line Numbers:\n";
        //for( unsigned i=0; i<6; i++)
        //{
        //    for( unsigned j=0; j<9; j++)
        //        std::cout << lines[i*9+j]<<" ";
        //    std::cout<< "\n";
        //}
        */
        //std::cout << "Result of Averaging:\n";
        thrust::reduce_by_key( lines.begin(), lines.end(), res.begin(), dummy.begin(), helper1d.begin());
        blas1::axpby( 1./ly_, helper1d, 0, helper1d);
        //remove weights in x-direction
        blas1::pointwiseDot( helper1d, v1d, helper1d);
        //for( unsigned i=0; i<helper1d.size(); i++)
            //std::cout<< helper1d[i]<<" ";
        //std::cout << "INNER integral: "<<thrust::reduce( helper1d.begin(), helper1d.end())<<"\n";
        thrust::copy( helper1d.begin(), helper1d.end(), helper.begin());
        //copy to a full vector
        unsigned pos = helper1d.size();
        while ( 2*pos < helper.size() )
        {
            thrust::copy_n( helper.begin(), pos, helper.begin() + pos);
            pos*=2; 
        }
        thrust::copy_n( helper.begin(), helper.size() - pos, helper.begin() + pos);
        //for( unsigned i=0; i<9; i++)
        //{
        //    for( unsigned j=0; j<6; j++)
        //        std::cout << helper[i*6+j]<<" ";
        //    std::cout<< "\n";
        //}

        //invert average vector
        //thrust::gather( reorder.begin(), reorder.end(), helper.begin(), res.begin());
        thrust::copy( helper.begin(), helper.end(), res.begin());
        //for( unsigned i=0; i<9; i++)
        //{
        //    for( unsigned j=0; j<6; j++)
        //        std::cout << res[i*6+j]<<" ";
        //    std::cout<< "\n";
        //}

    }
  private:
    IndexContainer invertxy, reorder, lines, dummy;
    container helper, helper1d;
    container w2d, v1d;
    double ly_;
};
/**
 * @brief Class for phi average computations
 *
 * The problem is the dg format of the vector
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
