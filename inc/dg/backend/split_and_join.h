#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "grid.h"
#ifdef MPI_VERSION
#include "mpi_vector.h"
#include "mpi_grid.h"
#endif //MPI_VERSION
namespace dg
{
///@ingroup scatter
///@{
/** @brief  Split a vector into planes 
*
* @tparam thrust_vector1 either thrust::host_vector or thrust::device_vector
* @tparam thrust_vector2 either thrust::host_vector or thrust::device_vector
* @param in contiguous 3d vector (must be of size grid.size())
* @param out contains \c grid.Nz() 2d vectors of 2d size on output (gets resized if necessary)
* @param grid provide dimensions in 3rd and first two dimensions
*/
template<class thrust_vector1, class thrust_vector2>
void split( const thrust_vector1& in, std::vector<thrust_vector2>& out, const aTopology3d& grid)
{
    Grid3d l( grid);
    unsigned size2d=l.n()*l.n()*l.Nx()*l.Ny();
    out.resize( l.Nz());
    for(unsigned i=0; i<l.Nz(); i++)
        out[i].assign( in.begin() + i*size2d, in.begin()+(i+1)*size2d);
}
#ifdef MPI_VERSION
///@brief MPI Version of split
///@copydetails dg::split()
///@note every plane in out holds a 2d Cartesian MPI_Communicator 
///@note two seperately split vectors have congruent (not identical) MPI_Communicators (Note here the MPI concept of congruent vs. identical communicators)
template<class thrust_vector1, class thrust_vector2>
void split( const MPI_Vector<thrust_vector1>& in, std::vector<MPI_Vector<thrust_vector2> >& out, const aMPITopology3d& grid)
{
    int result;
    MPI_Comm_compare( in.communicator(), grid.communicator(), &result);
    assert( result == MPI_CONGRUENT || result == MPI_IDENT);
    MPI_Comm planeComm = grid.get_perp_comm();
    //local size2d
    Grid3d l = grid.local();
    unsigned size2d=l.n()*l.n()*l.Nx()*l.Ny();
    out.resize( l.Nz());
    for(unsigned i=0; i<l.Nz(); i++)
    {
        out[i].data().assign( in.data().begin() + i*size2d, in.data().begin()+(i+1)*size2d);
        out[i].set_communicator( planeComm);
    }
}
#endif //MPI_VERSION
/**
* @brief Revert split operation
*
* @tparam thrust_vector1 either thrust::host_vector or thrust::device_vector
* @tparam thrust_vector2 either thrust::host_vector or thrust::device_vector
* @param in \c grid.Nz() 2d vectors of 2d size 
* @param out contiguous 3d vector (gets resized if necessary) 
* @param grid provide dimensions in 3rd and first two dimensions
* @note split followed by join restores the original vector
*/
template<class thrust_vector1, class thrust_vector2>
void join( const std::vector<thrust_vector1>& in, thrust_vector2& out, const aTopology3d& grid)
{
    unsigned size2d=grid.n()*grid.n()*grid.Nx()*grid.Ny();
    out.resize( size2d*grid.Nz());
    for(unsigned i=0; i<grid.Nz(); i++)
        thrust::copy( in[i].begin(), in[i].end(), out.begin()+i*size2d);
}

#ifdef MPI_VERSION
///@brief MPI Version of join
///@copydetails dg::join()
template<class thrust_vector1, class thrust_vector2>
void join( const std::vector<MPI_Vector<thrust_vector1> >& in, MPI_Vector<thrust_vector2 >& out, const aMPITopology3d& grid)
{
    Grid3d l(grid.local());
    unsigned size2d=l.n()*l.n()*l.Nx()*l.Ny();
    out.data().resize( size2d*l.Nz());
    out.set_communicator( grid.communicator());
    for(unsigned i=0; i<l.Nz(); i++)
        thrust::copy( in[i].data().begin(), in[i].data().end(), out.data().begin()+i*size2d);
}
#endif //MPI_VERSION

///@}
}//namespace dg
