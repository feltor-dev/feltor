#pragma once

#include <iostream>
#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h> //declare THRUST_DEVICE_SYSTEM
#include "config.h"
#include "mpi_datatype.h"
#include "../enums.h"

/*!@file
@brief convenience mpi init functions

enums need to be included before this
*/

namespace dg
{

///@addtogroup mpi_utility
///@{

/**
 * @brief Convencience shortcut: Calls MPI_Init or MPI_Init_thread and inits CUDA devices
 *
 * Shortcut for
 * @code
#ifdef _OPENMP
    int provided, error;
    error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    assert( error == MPI_SUCCESS && "Threaded MPI lib required!\n");
#else
    MPI_Init(&argc, &argv);
#endif
    //... init cuda devices if THRUST_DEVICE_SYSTEM_CUDA
 * @endcode
 * @note Also sets the GPU a process should use via <tt> cudaSetDevice( rank
 * \% num_devices_per_node) </tt> if <tt> THRUST_DEVICE_SYSTEM ==
 * THRUST_DEVICE_SYSTEM_CUDA </tt>.  We assume that the number of GPUs per node
 * is fixed.
 * @attention Abort program if MPI does not support OpenMP and \c _OPENMP is
 * defined or if no CUDA capable devices are found in case of \c
 * THRUST_DEVICE_SYSTEM_CUDA
 * @param argc command line argument number
 * @param argv command line arguments
 */
inline void mpi_init( int argc, char* argv[])
{
#ifdef _OPENMP
    int provided, error;
    error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    assert( error == MPI_SUCCESS && "Threaded MPI lib required!\n");
#else
    MPI_Init(&argc, &argv);
#endif
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    int num_devices=0;
    cudaGetDeviceCount(&num_devices);
    if(num_devices == 0)
    {
        std::cerr << "# No CUDA capable devices found on rank "<<rank<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(-1);
    }
    int device = rank % num_devices; //assume # of gpus/node is fixed
    cudaSetDevice( device);
#endif//THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
}

/*! @brief Read \c num values from \c is and broadcast to all processes as type \c T
 *
 * Only the rank0 in \c comm reads from \c is
 * @tparam T the type of value to read from \c is
 * @param num The number of values to read
 * @param comm The communicator to which to broadcast
 * @param is Input stream rank 0 reads \c num parameters of type \c T from
 */
template<class T>
std::vector<T> mpi_read_as( unsigned num, MPI_Comm comm, std::istream& is = std::cin)
{
    int rank;
    MPI_Comm_rank( comm, &rank);
    std::vector<T> nums(num);
    if( rank == 0)
    {
        for( unsigned u=0; u<num; u++)
            is >> nums[u];
    }
    MPI_Bcast( &nums[0], num, getMPIDataType<T>(), 0, comm);
    return nums;
}

/**
 * @brief Read in grid sizes from \c is
 *
 * The intended use for this function is as a setup module for application codes that
 * use our \c dg::Grid s. Basically just
 *  -# [verbose, rank0] print a query to \c os
 *  -# Call <tt>auto vals = dg::mpi_read_as<unsigned>( 1 + N.size(), comm, is);</tt>
 *  -# [verbose, rank0] print read values back to \c os
 *  .
 * @param n rank 0 reads in from \c is and broadcasts to all processes in \c MPI_COMM_WORLD
 * @param N rank 0 reads in from \c is and broadcasts to all processes in \c MPI_COMM_WORLD
 * @param comm communicator (handle) to which rank0 broadcasts
 * @param is Input stream rank 0 reads parameters (\c n, \c N)
 * @param verbose If true, rank 0 prints queries and information on \c os
 * @param os Output stream used if \c verbose==true
 * @sa dg::aRealTopologyNd
 */
inline void mpi_read_grid( unsigned& n, std::vector<unsigned>& N, MPI_Comm comm,
    std::istream& is = std::cin, bool verbose = true, std::ostream& os = std::cout)
{
    int rank;
    MPI_Comm_rank( comm, &rank);
    const std::string Ns[6] = {"Nx", "Ny", "Nz", "Nu", "Nv", "Nw"};
    unsigned ndims = N.size();
    assert( ndims > 0);
    if(rank == 0 and verbose)
    {
        os << "# Type n, "<<Ns[0];
        for( unsigned u=1; u<ndims; u++)
            os << " and "<< Ns[u];
        os << std::endl;
    }
    auto vals = mpi_read_as<unsigned>( 1 + ndims, comm, is);
    n = vals[0];
    for( unsigned u=0; u<ndims; u++)
        N[u] = vals[1+u];
    if(rank == 0 and verbose)
    {
        os << "# On the grid "<<n;
        for( unsigned u=1; u<ndims; u++)
            os << " x "<< N[u];
        os << std::endl;
    }
}

/*!@brief Convenience shortcut allowing a call like
 * @code{.cpp}
 *  mpi_read_grid( n, {&Nx, &Ny}, comm, is, verbose, os);
 * @endcode
 */
inline void mpi_read_grid( unsigned& n, std::vector<unsigned*> N, MPI_Comm comm,
    std::istream& is = std::cin, bool verbose = true, std::ostream& os = std::cout)
{
    std::vector<unsigned> Ns( N.size());
    mpi_read_grid( n, Ns, comm, is, verbose, os);
    for( unsigned u=0; u<Ns.size(); u++)
        *N[u] = Ns[u];
}

/*! @brief Convenience call to \c MPI_Cart_create preceded by \c MPI_Dims_create
 *
 * This function is equivalent to
 * @code{.cpp}
 *   int size;
 *   MPI_Comm_size( comm_old, &size);
 *   int ndims = dims.size();
 *   MPI_Dims_create( size, ndims, &dims[0]);
 *   MPI_Comm comm_cart;
 *   MPI_Cart_create( comm_old, dims.size(), &dims[0], periods, reorder, &comm_cart);
 *   return comm_cart;
 * @endcode
 * @param comm_old input communicator (handle) (parameter used in \c MPI_Cart_create)
 * @param dims specify number of processes in each dimensions. (\c dims.size()
 * determines \c ndims parameter used in \c MPI_Cart_create). Elements can be 0
 * in which case a distribution is automatically chosen in that direction.
 * @param periods logical array of size \c ndims specifying whether the grid is
 * periodic (true) or not (false) in each dimension (parameter used in \c
 * MPI_Cart_create
 * @param reorder (parameter used in \c MPI_Cart_create)
 * @note most MPI libraries ignore the \c reorder parameter
 * @return communicator with new Cartesian topology (handle)
 */
inline MPI_Comm mpi_cart_create( MPI_Comm comm_old, std::vector<int> dims,
                    std::vector<int> periods, bool reorder = true)
{
    int size;
    MPI_Comm_size( comm_old, &size);
    assert( dims.size() == periods.size());
    int ndims = dims.size();
    int re = (int)reorder;
    int err = MPI_Dims_create( size, ndims, &dims[0]);
    if( err != MPI_SUCCESS)
        throw Error(Message(_ping_)<<
                "Cannot create Cartesian dimensions from given dimensions and size "<<size);
    int reduce = 1;
    for( int u=0; u<(int)ndims; u++)
        reduce *= dims[u];
    if( size != reduce)
    {
        throw Error(Message(_ping_)<<
            "ERROR: Process partition needs to match total number of processes! "
            <<size<< " vs "<<reduce);
    }
    MPI_Comm comm_cart;
    err = MPI_Cart_create( comm_old, ndims, &dims[0], &periods[0], re, &comm_cart);
    if( err != MPI_SUCCESS)
        throw Error(Message(_ping_)<<
                "Cannot create Cartesian comm from given communicator");
    return comm_cart;
}

/**
 * @brief Convenience call: read in number of processses from istream and
 * create Cartesian MPI communicator
 *
 * The intended use for this function is as a setup module for application codes.
 * This function does:
 *  -# [verbose, rank0] print a query to \c os
    -# Call <tt>auto np = mpi_read_as<int>( bcs.size(), comm_old, is);</tt>
 *  -# [rank0] read \c bcs.size() integers from \c is
 *  -# [verbose, rank0] print read integers back to \c os
 *  -# [verbose, rank0, cuda_aware_mpi] print GPU information to \c os
 *  -# Call \c MPI_Cart_create and return Cartesian comm
 *  .
 *
 * @param bcs if <tt>bcs[u]==dg::PER</tt> then the communicator is periodic in that dimension
 * @param is Input stream rank 0 reads \c bcs.size() parameters (\c np[u])
 * @param comm_old input communicator (handle) (parameter used in \c MPI_Cart_create)
 * @param reorder (parameter used in \c MPI_Cart_create)
 * @note most MPI libraries ignore the \c reorder parameter
 * @param verbose If true, rank 0 prints queries and information to \c os
 * In this case <tt>bcs.size()<=6</tt>
 * @param os Output stream used if \c verbose==true
 * @return communicator with new Cartesian topology (handle)
 */
inline MPI_Comm mpi_cart_create(
    std::vector<dg::bc> bcs,
    std::istream& is = std::cin,
    MPI_Comm comm_old = MPI_COMM_WORLD,
    bool reorder = true,
    bool verbose = true,
    std::ostream& os = std::cout)
{
    int rank, size;
    MPI_Comm_rank( comm_old, &rank);
    MPI_Comm_size( comm_old, &size);
    if(rank==0 && verbose)os << "# MPI v"<<MPI_VERSION<<"."<<MPI_SUBVERSION<<std::endl;
    unsigned ndims = bcs.size();
    assert( ndims != 0);
    std::vector<int> periods( ndims);
    for( unsigned u=0; u<ndims; u++)
    {
        if(bcs[u] == dg::PER)
            periods[u] = true;
        else
            periods[u] = false;
    }
    if( rank == 0)
    {
        if(verbose)
        {
            const std::string nps[6] = {"npx", "npy", "npz", "npu", "npv", "npw"};
            os << "# Type "<<nps[0];
            for( unsigned u=1; u<ndims; u++)
                os << " and "<< nps[u];
            os << std::endl;
        }
    }
    auto np = mpi_read_as<int>( ndims, comm_old, is);
    if( rank == 0)
    {
        if(verbose)
        {
            int num_threads = 1;
#ifdef _OPENMP
            num_threads = omp_get_max_threads( );
#endif //omp
            os << "# Computing with "<<np[0];
            for( unsigned u=1; u<ndims; u++)
                os << " x" <<np[u];
             os << " processes x " << num_threads<<" threads = "
                << size*num_threads<<" total"<<std::endl;
        }
    }
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int device=0;
    cudaGetDevice( &device);
    if( rank==0 and verbose)
    {
        std::cout << "# MPI is "
                  <<(cuda_aware_mpi ? "cuda-aware" : "NOT cuda-aware")
                  <<"!\n";
    }
    if(verbose)std::cout << "# Rank "<<rank<<" computes with device "<<device<<" !"<<std::endl;
#endif//THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    return dg::mpi_cart_create( comm_old, np, periods, reorder);
}


/**
 * @brief DEPRECATED: Short for
 * @code{.cpp}
 *   comm = mpi_cart_create( {bcx}, is, MPI_COMM_WORLD, true, verbose,
 *   std::cout);
 * @endcode
 * @ingroup mpi_legacy
 */
inline void mpi_init1d( dg::bc bcx, MPI_Comm& comm, std::istream& is =
std::cin, bool verbose = true)
{
    comm = mpi_cart_create( {bcx}, is, MPI_COMM_WORLD, true, verbose,
    std::cout);
}
/**
 * @brief DEPRECATED: Short for
 * @code{.cpp}
 *   comm = mpi_cart_create( {bcx,bcy}, is, MPI_COMM_WORLD, true, verbose,
 *   std::cout);
 * @endcode
 * @ingroup mpi_legacy
 */
inline void mpi_init2d( dg::bc bcx, dg::bc bcy, MPI_Comm& comm, std::istream&
is = std::cin, bool verbose = true)
{
    comm = mpi_cart_create( {bcx, bcy}, is, MPI_COMM_WORLD, true, verbose,
    std::cout);
}

/**
 * @brief DEPRECATED: Short for
 * @code{.cpp}
 *   comm = mpi_cart_create( {bcx,bcy,bcz}, is, MPI_COMM_WORLD, true, verbose,
 *   std::cout);
 * @endcode
 * @ingroup mpi_legacy
 */
inline void mpi_init3d( dg::bc bcx, dg::bc bcy, dg::bc bcz, MPI_Comm& comm,
std::istream& is = std::cin, bool verbose = true  )
{
    comm = mpi_cart_create( {bcx, bcy, bcz}, is, MPI_COMM_WORLD, true, verbose,
    std::cout);
}

/**
 * @brief DEPRECATED
 *
 * Short for
 * @code{.cpp}
 *   comm = mpi_cart_create( {bcx}, is, MPI_COMM_WORLD, true, verbose,
 *       std::cout);
 *   mpi_read_grid( n, {&N}, comm, is, verbose, std::cout);
 * @endcode
 * @ingroup mpi_legacy
 */
inline void mpi_init1d( dg::bc bcx, unsigned& n, unsigned& N, MPI_Comm& comm,
std::istream& is = std::cin, bool verbose = true  )
{
    comm = mpi_cart_create( {bcx}, is, MPI_COMM_WORLD, true, verbose,
        std::cout);
    mpi_read_grid( n, {&N}, comm, is, verbose, std::cout);
}

/**
 * @brief DEPRECATED
 *
 * Short for
 * @code{.cpp}
 *   comm = mpi_cart_create( {bcx, bcy}, is, MPI_COMM_WORLD, true, verbose,
 *       std::cout);
 *   mpi_read_grid( n, {&Nx, &Ny}, comm, is, verbose, std::cout);
 * @endcode
 * @ingroup mpi_legacy
 */
inline void mpi_init2d( dg::bc bcx, dg::bc bcy, unsigned& n, unsigned& Nx,
unsigned& Ny, MPI_Comm& comm, std::istream& is = std::cin, bool verbose = true
)
{
    comm = mpi_cart_create( {bcx, bcy}, is, MPI_COMM_WORLD, true, verbose,
        std::cout);
    mpi_read_grid( n, {&Nx, &Ny}, comm, is, verbose, std::cout);
}


/**
 * @brief DEPRECATED
 *
 * Short for
 * @code{.cpp}
 *   comm = mpi_cart_create( {bcx, bcy, bcz}, is, MPI_COMM_WORLD, true, verbose,
 *       std::cout);
 *   mpi_read_grid( n, {&Nx, &Ny, &Nz}, comm, is, verbose, std::cout);
 * @endcode
 * @ingroup mpi_legacy
 */
inline void mpi_init3d( dg::bc bcx, dg::bc bcy, dg::bc bcz, unsigned& n,
unsigned& Nx, unsigned& Ny, unsigned& Nz, MPI_Comm& comm, std::istream& is =
std::cin, bool verbose = true  )
{
    comm = mpi_cart_create( {bcx, bcy, bcz}, is, MPI_COMM_WORLD, true, verbose,
        std::cout);
    mpi_read_grid( n, {&Nx, &Ny, &Nz}, comm, is, verbose, std::cout);
}

///@}

} //namespace dg
