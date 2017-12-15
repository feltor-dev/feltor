#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "thrust_vector_blas.cuh"
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
* @tparam thrust_vector1 either thrust::host_vector or \c thrust::device_vector
* @tparam thrust_vector2 either thrust::host_vector or \c thrust::device_vector
* @param in contiguous 3d vector (must be of size \c grid.size())
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
* @tparam thrust_vector1 either \c thrust::host_vector or \c thrust::device_vector
* @tparam thrust_vector2 either \c thrust::host_vector or \c thrust::device_vector
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
/////////////////////////////////////////////poloidal split/////////////////////
///@cond
void transpose_dispatch( SerialTag, unsigned nx, unsigned ny, const double* in, double* out)
{
    for( unsigned i=0; i<ny; i++)
        for( unsigned j=0; j<nx; j++)
            out[j*ny+i] = in[i*nx+j];
}
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
__global__
void transpose_gpu_kernel( unsigned nx, unsigned ny, const double* in, double* out)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    const int size = nx*ny;
    for( int row = thread_id; row<size; row += grid_size)
    {
        int i=row/nx, j = row%nx;
        out[j*ny+i] = in[i*nx+j];
    }
}
void transpose_dispatch( CudaTag, unsigned nx, unsigned ny, const double* in, double* out){
    const size_t BLOCK_SIZE = 256; 
    const size_t NUM_BLOCKS = std::min<size_t>((nx*ny-1)/BLOCK_SIZE+1, 65000);
    transpose_gpu_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>( nx, ny, in, out);
}
#else
void transpose_dispatch( OmpTag, unsigned nx, unsigned ny, const double* in, double* out)
{
#pragma omp parallel for
    for( unsigned i=0; i<ny; i++)
        for( unsigned j=0; j<nx; j++)
            out[j*ny+i] = in[i*nx+j];
}
#endif
/////////////join
///@endcond
/** @brief  Split a vector poloidally into lines
*
* @tparam thrust_vector either \c thrust::host_vector or \c thrust::device_vector
* @param in contiguous 2d vector (must be of size \c grid.size())
* @param out contains \c grid.n()*grid.Ny() 1d vectors of size \c grid.n()*grid.Nx() on output (gets resized if necessary)
* @param grid provide dimensions in 1st and 2nd dimensions
*/
template<class thrust_vector>
void split_poloidal( const thrust_vector& in, std::vector<thrust_vector>& out, const aTopology2d& grid)
{
    Grid2d l( grid);
    out.resize( l.n()*l.Nx());
    const double* in_ptr = thrust::raw_pointer_cast(in.data());
    std::vector<double*> out_ptrs(l.n()*l.Nx());
    double** out_ptr = out_ptrs.data();
    for( unsigned i=0; i<out.size(); i++)
    {
        out[i].resize( l.n()*l.Ny());
        out_ptrs[i] = thrust::raw_pointer_cast( out[i].data());
    }
    split_poloidal_dispatch( l.n()*l.Nx(), l.n()*l.Ny(), in_ptr, out_ptr, get_execution_policy<thrust_vector>());
}
#ifdef MPI_VERSION
///@brief MPI Version of split
///@copydetails dg::split_poloidal()
///@note every plane in out holds a 1d Cartesian MPI_Communicator 
///@note two seperately split vectors have congruent (not identical) MPI_Communicators (Note here the MPI concept of congruent vs. identical communicators)
template<class thrust_vector>
void split_poloidal( const MPI_Vector<thrust_vector>& in, std::vector<MPI_Vector<thrust_vector> >& out, const aMPITopology2d& grid)
{
    int result;
    MPI_Comm_compare( in.communicator(), grid.communicator(), &result);
    assert( result == MPI_CONGRUENT || result == MPI_IDENT);
    MPI_Comm poloidalComm = grid.get_poloidal_comm();
    //local size2d
    Grid2d l = grid.local();
    std::vector<thrust_vector> out_local;
    split_poloidal( in.data(), out_local, l);

    out.resize( l.n()*l.Nx());
    for(unsigned i=0; i<out.size(); i++)
    {
        out[i].data() = out_local[i];
        out[i].set_communicator( poloidalComm);
    }
}
#endif //MPI_VERSION
/**
* @brief Revert split operation
*
* @tparam thrust_vector either \c thrust::host_vector or \c thrust::device_vector
* @param in \c grid.nx()*grid.Nx() 1d vectors of 1d size 
* @param out contiguous 2d vector (gets resized if necessary) 
* @param grid provide dimensions in first and 2nd dimensions
* @note split_poloidal followed by join_poloidal restores the original vector
*/
template<class thrust_vector>
void join_poloidal( const std::vector<thrust_vector>& in, thrust_vector& out, const aTopology2d& grid)
{
    std::vector<const double*> in_ptrs( in.size());
    for( unsigned i=0; i<in.size(); i++)
        in_ptrs[i] = thrust::raw_pointer_cast(in[i].data());
    const double ** in_ptr = in_ptrs.data();
    double* out_ptr = thrust::raw_pointer_cast( out.data());
    join_poloidal_dispatch( grid.n()*grid.Nx(), grid.n()*grid.Ny(), in_ptr, out_ptr, get_execution_policy<thrust_vector>());
}

#ifdef MPI_VERSION
///@brief MPI Version of join
///@copydetails dg::join_poloidal()
template<class thrust_vector>
void join( const std::vector<MPI_Vector<thrust_vector> >& in, MPI_Vector<thrust_vector >& out, const aMPITopology2d& grid)
{
    //local size2d

    Grid2d l = grid.local();
    thrust_vector out_local(l.size());

    std::vector<double*> in_ptrs( in.size());
    for( unsigned i=0; i<in.size(); i++)
        in_ptrs[i] = thrust::raw_pointer_cast(in[i].data().data());
    const double ** in_ptr = in_ptrs.data();

    double* out_ptr = thrust::raw_pointer_cast( out_local.data());
    join_poloidal_dispatch( l.n()*l.Nx(), l.n()*l.Ny(), in_ptr, out_ptr, typename VectorTraits<thrust_vector>::vector_category());
    out.set_communicator( grid.communicator());
    out.data() = out_local;
}
#endif //MPI_VERSION

///@}
}//namespace dg
