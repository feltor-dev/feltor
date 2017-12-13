#pragma once

#include "evaluation.cuh"
#include "grid.h"
#ifdef MPI_VERSION
#include "mpi_grid.h"
#endif
#include "../blas1.h"
#include "memory.h"
#include "split_and_join.h"

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
    {
        m_helper1d = dg::detail::create_weightsY( g);
            dg::blas1::transfer( m_helper1d , m_w1dy);
        dg::blas1::scal( w1dy, 1./g.ly());
    }
    /**
     * @brief Compute the average in y-direction
     *
     * @param src 2D Source Vector 
     * @param res 2D result Vector (may not equal src), every line contains the x-dependent average over
     the y-direction of src 
     */
    void operator() (const container& src, container& res)
    {
        dg::split( src, m_split);
        for( unsigned i=0; i<m_split.size(); i++)
        {
            m_helper1d[i] = dg::blas1::dot( m_split[i], w1dy);
            dg::blas1::transform( m_split[i], m_split[i], dg::CONSTANT<double>());
        }
        dg::join(m_split, res);
    }
  private:
    container m_w1dy; 
    thrust::host_vector<double> m_helper1d;
    std::vector<

};


}//namespace dg
