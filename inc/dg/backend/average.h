#pragma once

#include "evaluation.cuh"
#include "grid.h"
#ifdef MPI_VERSION
#include "mpi_grid.h"
#endif
#include "../blas1.h"
#include "memory.h"
#include "split_and_join.h"
#include "functors.h"

/*! @file 
  @brief contains classes for poloidal and toroidal average computations.
  */
namespace dg{

namespace detail
{

thrust::host_vector<double> create_weightsY1d( const aTopology2d& g)
{
    dg::Grid1d g1d( g.x0(), g.x1(), g.n(), g.Ny());
    return dg::create::weights( g1d);
}
#ifdef MPI_VERSION
MPI_Vector<thrust::host_vector<double>> create_weightsY1d( const aMPITopology2d& g)
{
    thrust::host_vector<double> w = dg::create::weights( g.local());
    return MPI_Vector<thrust::host_vector<double> >( w, g.get_poloidal_comm());

}
#endif



}

/**
 * @brief MPI specialized class for y average computations
 *
 * @snippet backend/average_mpit.cu doxygen
 * @ingroup utilities
 * @tparam container Currently this is one of 
 *  - \c dg::HVec, \c dg::DVec, \c dg::MHVec or \c dg::MDVec  
 */
template< class Topology2d, class container>
struct PoloidalAverage
{
    /**
     * @brief Construct from grid mpi object
     * @param g 2d MPITopology
     */
    PoloidalAverage( const Topology2d& g): 
    m_g2d(g)
    {
        m_w1dy=dg::transfer<container>(dg::detail::create_weightsY1d(g));
        dg::blas1::scal( m_w1dy, 1./g.ly());
        container w2d = dg::transfer<container>(dg::create::weights(g));
        dg::split_poloidal( w2d, m_split, g);
    }
    /**
     * @brief Compute the average in y-direction
     *
     * @param src 2D Source Vector (must have the same size as the grid given in the constructor)
     * @param res 2D result Vector (may alias src), every line contains the x-dependent average over
     the y-direction of src 
     */
    void operator() (const container& src, container& res)
    {
        dg::split_poloidal( src, m_split, m_g2d);
        for( unsigned i=0; i<m_split.size(); i++)
        {
            double value = dg::blas1::dot( m_split[i], m_w1dy);
            dg::blas1::transform( m_split[i], m_split[i], dg::CONSTANT(value));
        }
        dg::join_poloidal(m_split, res, m_g2d);
    }
  private:
    container m_w1dy; 
    std::vector<container> m_split;
    get_host_grid<Topology2d> m_g2d;

};


}//namespace dg
