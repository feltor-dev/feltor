#pragma once

#include "evaluation.cuh"
#include "xspacelib.cuh"
#include "average.cuh"
#include "../blas1.h"

/*! @file 
  @brief contains classes for poloidal and toroidal average computations.
  */
namespace dg{

/**
 * @brief MPI specialized class for y average computations
 *
 * @ingroup utilities
 * @tparam container Vector class to be used
 * @tparam IndexContainer Class for scatter maps
 */
template< class container, class IndexContainer>
struct PoloidalAverage<MPI_Vector<container>, MPI_Vector<IndexContainer> >
{
    /**
     * @brief Construct from grid mpi object
     * @param g 2d MPITopology
     */
    PoloidalAverage( const aMPITopology2d& g): 
        helper1d_( g.n()*g.Nx()), hhelper1d_(g.n()*g.Nx()),
        recv_(hhelper1d_),dummy( g.n()*g.Nx()), 
        helper_( g.size()), ly_(g.global().ly())
    {
        int remain[] = {false, true};
        MPI_Cart_sub( g.communicator(), remain, &comm1d_);

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
     * @param src 2D Source MPIvector 
     * @param res 2D result MPIvector (may not equal src), every line contains the x-dependent average over
     the y-direction of src 
     */
    void operator() (const MPI_Vector<container>& src, MPI_Vector<container>& res)
    {
        assert( &src != &res);
        res.data().resize( src.data().size());
        thrust::scatter( src.data().begin(), src.data().end(), invertxy.begin(), helper_.begin());
        //weights to ensure correct integration
        blas1::pointwiseDot( helper_, w2d, helper_);
        thrust::reduce_by_key( lines.begin(), lines.end(), helper_.begin(), dummy.begin(), helper1d_.begin());
        blas1::scal( helper1d_, 1./ly_);
        //remove 1d weights in x-direction
        blas1::pointwiseDot( helper1d_, v1d, helper1d_);
        //copy to host
        thrust::copy( helper1d_.begin(), helper1d_.end(), hhelper1d_.begin());
        //Reduce  
        MPI_Allreduce( hhelper1d_.data(), recv_.data(), helper1d_.size(), MPI_DOUBLE, MPI_SUM, comm1d_);
        //copy to device
        thrust::copy( recv_.begin(), recv_.end(), helper1d_.begin());
        //copy to a full vector
        thrust::copy( helper1d_.begin(), helper1d_.end(), res.data().begin());
        unsigned pos = helper1d_.size();
        while ( 2*pos < res.data().size() )
        {
            thrust::copy_n( res.data().begin(), pos, res.data().begin() + pos);
            pos*=2; 
        }
        thrust::copy_n( res.data().begin(), res.data().size() - pos, res.data().begin() + pos);
    }
  private:
    container helper1d_;
    thrust::host_vector<double> hhelper1d_, recv_;
    MPI_Comm comm1d_;
    IndexContainer invertxy, lines, dummy; //dummy contains output keys i.e. line numbers
    container helper_;
    container w2d, v1d;
    double ly_;
};


}//namespace dg
