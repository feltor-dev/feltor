#pragma once

#include <iostream>
#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h> //declare THRUST_DEVICE_SYSTEM
#include "../enums.h"

/*!@file
@brief convenience mpi init functions

enums need to be included before this
*/

namespace dg
{

/**
 * @brief Convencience shortcut: Calls MPI_Init or MPI_Init_thread
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
 * @endcode
 * @param argc command line argument number
 * @param argv command line arguments
 * @ingroup misc
 */
static inline void mpi_init( int argc, char* argv[])
{
#ifdef _OPENMP
    int provided, error;
    error = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    assert( error == MPI_SUCCESS && "Threaded MPI lib required!\n");
#else
    MPI_Init(&argc, &argv);
#endif
}

/** @class hide_cart_warning
* @attention Before creating a second Cartesian communicator consider freeing existing ones with \c MPI_Comm_free. (Using \c mpi_init2d and \c mpi_init3d in the same program has sometimes led to Segmentation faults in the past)
  */
/**
* @brief Read in number of processses and create Cartesian MPI communicator
*
* Also sets the GPU a process should use via \c rank\% num_devices_per_node if \c THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
* @param bcx if \c bcx==dg::PER then the communicator is periodic in x
* @param bcy if \c bcy==dg::PER then the communicator is periodic in y
* @param comm (write only) \c MPI_COMM_WORLD as a 2d Cartesian MPI communicator
* @param is Input stream rank 0 reads parameters (\c npx, \c npy)
* @param verbose If true, rank 0 prints queries and information on \c std::cout
* @ingroup misc
* @copydoc hide_cart_warning
*/
static inline void mpi_init2d( dg::bc bcx, dg::bc bcy, MPI_Comm& comm, std::istream& is = std::cin, bool verbose = true  )
{
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    if(rank==0 && verbose)std::cout << "# MPI v"<<MPI_VERSION<<"."<<MPI_SUBVERSION<<std::endl;
    int periods[2] = {false,false};
    if( bcx == dg::PER) periods[0] = true;
    if( bcy == dg::PER) periods[1] = true;
    int np[2];
    if( rank == 0)
    {
        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads( );
#endif //omp
        if(verbose)std::cout << "# Type npx and npy\n";
        is >> np[0] >> np[1];
        if(verbose)std::cout << "# Computing with "
                  << np[0]<<" x "<<np[1]<<" processes x "
                  << num_threads<<" threads = "
                  <<size*num_threads<<" total"<<std::endl;
        if( size != np[0]*np[1])
        {
            std::cerr << "ERROR: Process partition needs to match total number of processes!"<<std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
            exit(-1);
        }
    }
    MPI_Bcast( np, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, &comm);

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int num_devices=0;
    cudaGetDeviceCount(&num_devices);
    if(num_devices == 0)
    {
        std::cerr << "# No CUDA capable devices found on rank "<<rank<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(-1);
    }
    int device = rank % num_devices; //assume # of gpus/node is fixed
    if(verbose)std::cout << "# Rank "<<rank<<" computes with device "<<device<<" !"<<std::endl;
    cudaSetDevice( device);
#endif//THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
}
/**
* @brief Read in number of processes and broadcast to process group
*
* @param n  rank 0 reads in from \c is and broadcasts to all processes in \c comm
* @param Nx rank 0 reads in from \c is and broadcasts to all processes in \c comm
* @param Ny rank 0 reads in from \c is and broadcasts to all processes in \c comm
* @param comm (read only) a 2d Cartesian MPI communicator
* @param is Input stream rank 0 reads parameters (\c n, \c Nx, \c Ny)
* @param verbose If true, rank 0 prints queries and information on \c std::cout
* @ingroup misc
*/
static inline void mpi_init2d(unsigned& n, unsigned& Nx, unsigned& Ny, MPI_Comm comm, std::istream& is = std::cin, bool verbose = true  )
{
    int rank;
    MPI_Comm_rank( comm, &rank);
    if( rank == 0)
    {
        if(verbose)std::cout << "# Type n, Nx and Ny\n";
        is >> n >> Nx >> Ny;
        if(verbose)std::cout<< "# On the grid "<<n <<" x "<<Nx<<" x "<<Ny<<std::endl;
    }
    MPI_Bcast(  &n,1 , MPI_UNSIGNED, 0, comm);
    MPI_Bcast( &Nx,1 , MPI_UNSIGNED, 0, comm);
    MPI_Bcast( &Ny,1 , MPI_UNSIGNED, 0, comm);
}

/**
* @brief Read in number of processses and grid size and create Cartesian MPI communicator
*
* Also sets the GPU a process should use via \c rank\% num_devices_per_node if \c THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
* @param bcx if \c bcx==dg::PER then the communicator is periodic in x
* @param bcy if \c bcy==dg::PER then the communicator is periodic in y
* @param n  rank 0 reads in from \c is and broadcasts to all processes in \c MPI_COMM_WORLD
* @param Nx rank 0 reads in from \c is and broadcasts to all processes in \c MPI_COMM_WORLD
* @param Ny rank 0 reads in from \c is and broadcasts to all processes in \c MPI_COMM_WORLD
* @param comm (write only) \c MPI_COMM_WORLD as a 2d Cartesian MPI communicator
* @param is Input stream rank 0 reads parameters (\c npx, \c npy, \c n, \c Nx, \c Ny)
* @param verbose If true, rank 0 prints queries and information on \c std::cout
* @ingroup misc
* @copydoc hide_cart_warning
*/
static inline void mpi_init2d( dg::bc bcx, dg::bc bcy, unsigned& n, unsigned& Nx, unsigned& Ny, MPI_Comm& comm, std::istream& is = std::cin, bool verbose = true  )
{
    mpi_init2d( bcx, bcy, comm, is, verbose);
    mpi_init2d( n, Nx, Ny, comm, is, verbose);
}


/**
* @brief Read in number of processses and create Cartesian MPI communicator
*
* Also sets the GPU a process should use via \c rank\% num_devices_per_node if \c THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
* @param bcx if \c bcx==dg::PER then the communicator is periodic in x
* @param bcy if \c bcy==dg::PER then the communicator is periodic in y
* @param bcz if \c bcz==dg::PER then the communicator is periodic in z
* @param comm (write only) \c MPI_COMM_WORLD as a 3d Cartesian MPI communicator
* @param is Input stream rank 0 reads parameters (\c npx, \c npy, \c npz)
* @param verbose If true, rank 0 prints queries and information on \c std::cout
* @ingroup misc
* @copydoc hide_cart_warning
*/
static inline void mpi_init3d( dg::bc bcx, dg::bc bcy, dg::bc bcz, MPI_Comm& comm, std::istream& is = std::cin, bool verbose = true  )
{
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    int periods[3] = {false,false, false};
    if( bcx == dg::PER) periods[0] = true;
    if( bcy == dg::PER) periods[1] = true;
    if( bcz == dg::PER) periods[2] = true;
    int np[3];
    if( rank == 0)
    {
        int num_threads = 1;
#ifdef _OPENMP
        num_threads = omp_get_max_threads( );
#endif //omp
        if(verbose) std::cout << "# Type npx and npy and npz\n";
        is >> np[0] >> np[1]>>np[2];
        if(verbose) std::cout << "# Computing with "
                  << np[0]<<" x "<<np[1]<<" x "<<np[2] << " processes x "
                  << num_threads<<" threads = "
                  <<size*num_threads<<" total"<<std::endl;
        if( size != np[0]*np[1]*np[2])
        {
            std::cerr << "ERROR: Process partition needs to match total number of processes!"<<std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
            exit(-1);
        }
    }
    MPI_Bcast( np, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Cart_create( MPI_COMM_WORLD, 3, np, periods, true, &comm);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int num_devices=0;
    cudaGetDeviceCount(&num_devices);
    if(num_devices == 0)
    {
        std::cerr << "# No CUDA capable devices found on rank "<<rank<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
        exit(-1);
    }
    int device = rank % num_devices; //assume # of gpus/node is fixed
    if(verbose)std::cout << "# Rank "<<rank<<" computes with device "<<device<<" !"<<std::endl;
    cudaSetDevice( device);
#endif//THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
}
/**
* @brief Read in number of processes and broadcast to process group
*
* @param n  rank 0 reads in from \c is and broadcasts to all processes in \c comm
* @param Nx rank 0 reads in from \c is and broadcasts to all processes in \c comm
* @param Ny rank 0 reads in from \c is and broadcasts to all processes in \c comm
* @param Nz rank 0 reads in from \c is and broadcasts to all processes in \c comm
* @param comm (read only) a 3d Cartesian MPI communicator
* @param is Input stream rank 0 reads parameters (\c n, \c Nx, \c Ny, \c Nz )
* @param verbose If true, rank 0 prints queries and information on \c std::cout
* @ingroup misc
*/
static inline void mpi_init3d(unsigned& n, unsigned& Nx, unsigned& Ny, unsigned& Nz, MPI_Comm comm, std::istream& is = std::cin, bool verbose = true  )
{
    int rank;
    MPI_Comm_rank( comm, &rank);
    if( rank == 0)
    {
        if(verbose)std::cout << "# Type n, Nx and Ny and Nz\n";
        is >> n >> Nx >> Ny >> Nz;
        if(verbose)std::cout<< "# On the grid "<<n <<" x "<<Nx<<" x "<<Ny<<" x "<<Nz<<std::endl;
    }
    MPI_Bcast(  &n,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast( &Nx,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast( &Ny,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast( &Nz,1 , MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}

/**
* @brief Read in number of processses and grid size and create Cartesian MPI communicator
*
* Also sets the GPU via \c rank\% num_devices_per_node a process should use if \c THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
* @param bcx if \c bcx==dg::PER then the communicator is periodic in x
* @param bcy if \c bcy==dg::PER then the communicator is periodic in y
* @param bcz if \c bcz==dg::PER then the communicator is periodic in z
* @param n  rank 0 reads in from \c is and broadcasts to all processes in \c MPI_COMM_WORLD
* @param Nx rank 0 reads in from \c is and broadcasts to all processes in \c MPI_COMM_WORLD
* @param Ny rank 0 reads in from \c is and broadcasts to all processes in \c MPI_COMM_WORLD
* @param Nz rank 0 reads in from \c is and broadcasts to all processes in \c MPI_COMM_WORLD
* @param comm (write only) \c MPI_COMM_WORLD as a 3d Cartesian MPI communicator
* @param is Input stream rank 0 reads parameters (\c npx, \c npy, \c npz, \c n, \c Nx, \c Ny, \c Nz)
* @param verbose If true, rank 0 prints queries and information on \c std::cout
* @ingroup misc
* @copydoc hide_cart_warning
*/
static inline void mpi_init3d( dg::bc bcx, dg::bc bcy, dg::bc bcz, unsigned& n, unsigned& Nx, unsigned& Ny, unsigned& Nz, MPI_Comm& comm, std::istream& is = std::cin, bool verbose = true  )
{
    mpi_init3d( bcx, bcy, bcz, comm, is, verbose);
    mpi_init3d( n, Nx, Ny, Nz, comm, is, verbose);
}
} //namespace dg
