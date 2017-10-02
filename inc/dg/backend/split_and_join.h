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
* @tparam thrust_vector either thrust::host_vector or thrust::device_vector
* @param in contiguous 3d vector (must be of size grid.size())
* @param out contains grid.Nz() 2d vectors of 2d size on output (gets resized if necessary)
* @param grid provide dimensions in 3rd and first two dimensions
*/
template<class thrust_vector>
void split( const thrust_vector& in, std::vector<thrust_vector>& out, const aTopology3d& grid)
{
    unsigned size2d=grid.n()*grid.n()*grid.Nx()*grid.Ny();
    out.resize( grid.Nz());
    for(unsigned i=0; i<grid.Nz(); i++)
        out[i].assign( in.begin() + i*size2d, in.begin()+(i+1)*size2d);
}
#ifdef MPI_VERSION
///@brief MPI Version of split
///@copydetails dg::split()
///@note every plane in out gets its own 2d Cartesian communicator
template <class thrust_vector>
void split( const MPI_Vector<thrust_vector>& in, std::vector<MPI_Vector<thrust_vector> >& out, const aMPITopology3d& grid)
{
    int result;
    MPI_Comm_compare( x.communicator(), grid.communicator(), &result);
    assert( result == MPI_IDENT);
    MPI_Comm planeComm;
    int remain_dims[] = {true,true,false}; 
    MPI_Cart_sub( in.communicator(), remain_dims, &planeComm);
    //local size2d
    unsigned size2d=grid.n()*grid.n()*grid.Nx()*grid.Ny();
    out.resize( grid.Nz());
    for(unsigned i=0; i<grid.Nz(); i++)
    {
        out[i].data().assign( in.data().begin() + i*size2d, in.data().begin()+(i+1)*size2d);
        out[i].communicator() = planeComm;
    }
}
#endif //MPI_VERSION
/**
* @brief Revert split operation
*
* @tparam thrust_vector either thrust::host_vector or thrust::device_vector
* @param in grid.Nz() 2d vectors of 2d size 
* @param out contiguous 3d vector (gets resized if necessary) 
* @param grid provide dimensions in 3rd and first two dimensions
* @note split followed by join restores the original vector
*/
template<class thrust_vector>
void join( const std::vector<thrust_vector>& in, thrust_vector& out, const aTopology3d& grid)
{
    unsigned size2d=grid.n()*grid.n()*grid.Nx()*grid.Ny();
    out.resize( size2d*grid.Nz());
    for(unsigned i=0; i<grid.Nz(); i++)
        thrust::copy( in[i].begin(), in[i].end(), out.begin()+i*size2d);
}

#ifdef MPI_VERSION
///@brief MPI Version of join
///@copydetails dg::join()
template<class thrust_vector>
void join( const std::vector<MPI_Vector<thrust_vector> >& in, MPI_Vector<thrust_vector >& out, const aMPITopology3d& grid)
{
    unsigned size2d=grid.n()*grid.n()*grid.Nx()*grid.Ny();
    out.data().resize( size2d*grid.Nz());
    out.communicator() = grid.communicator();
    for(unsigned i=0; i<grid.Nz(); i++)
        thrust::copy( in[i].data().begin(), in[i].data().end(), out.data().begin()+i*size2d);
}
#endif //MPI_VERSION

/**
* @brief Expand 2d to 3d vector
* @tparam thrust_vector1 either thrust::host_vector or thrust::device_vector
* @tparam thrust_vector2 either thrust::host_vector or thrust::device_vector
* @param in 2d vector of 2d size 
* @param out contiguous 3d vector (gets resized if necessary) 
* @param grid provide dimensions in 3rd and first two dimensions
*/
template<class thrust_vector1, class thrust_vector2>
void expand( const thrust_vector1& in, thrust_vector2& out, const aTopology3d& grid)
{
    unsigned perp_size = grid.n()*grid.n()*grid.Nx()*grid.Ny();
    out.resize( grid.size());
    for( unsigned i=0; i<grid.Nz(); i++)
        thrust::copy( in.begin(), in.end(), out.begin() + i*perp_size);
}
#ifdef MPI_VERSION
///@brief MPI Version of expand
///@copydetails dg::expand()
template<class thrust_vector1, class thrust_vector2>
void expand( const MPI_Vector<thrust_vector1>& in, MPI_Vector<thrust_vector2>& out, const aMPITopology3d& grid)
{
    unsigned perp_size = grid.n()*grid.n()*grid.Nx()*grid.Ny();
    out.data().resize( grid.size());
    out.communicator() = grid.communicator();
    for( unsigned i=0; i<grid.Nz(); i++)
        thrust::copy( in.data().begin(), in.data().end(), out.data().begin() + i*perp_size);
}
#endif //MPI_VERSION


///@}
}//namespace dg
