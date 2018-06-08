/**
 *  @file mpi_accumulate.h
 *  @brief Primitives for an MPI Reduction
 *
 *  @authors
 *        Matthias Wiesenberger -- mattwi@fysik.dtu.dk
 */
#pragma once
#include <mpi.h>
#include "accumulate.h"

namespace exblas {

/**
 * @brief This function can be used to partition communicators for the \c exblas::reduce_mpi_cpu function
 *
 * @ingroup highlevel
 * @param comm the input communicator (unmodified)
 * @param comm_mod a subgroup of comm (comm is split)
 * @param comm_mod_reduce a subgroup of comm, consists of all rank 0 processes in comm_mod
 * @note the creation of new communicators involves communication between all participation processes (comm in this case)
 */
static void mpi_reduce_communicator(MPI_Comm comm, MPI_Comm* comm_mod, MPI_Comm* comm_mod_reduce){
    int mod = 128;
    int rank, size;
    MPI_Comm_rank( comm, &rank);
    MPI_Comm_size( comm, &size);
    MPI_Comm_split( comm, rank/mod, rank%mod, comm_mod);
    MPI_Group group, reduce_group;
    MPI_Comm_group( comm, &group);
    int reduce_size=(int)ceil((double)size/(double)mod);
    int reduce_ranks[reduce_size];
    for( int i=0; i<reduce_size; i++)
        reduce_ranks[i] = i*mod;
    MPI_Group_incl( group, reduce_size, reduce_ranks, &reduce_group);
    MPI_Comm_create( comm, reduce_group, comm_mod_reduce);
    //returns MPI_COMM_NULL to processes that are not in the group
}

/*! @brief reduce a number of superaccumulators distributed among mpi processes

We cannot sum more than 256 accumulators before we need to normalize again, so we need to split the reduction into several steps if more than 256 processes are involved. This function normalizes,
reduces, normalizes, reduces and broadcasts the result to all participating
processes.  As usual the resulting superaccumulator is unnormalized.
 * @ingroup highlevel
@param num_superacc number of Superaccumulators eaach process holds
@param in unnormalized input superaccumulators ( must be of size num_superacc*\c exblas::BIN_COUNT, allocated on the cpu) (read/write, undefined on out)
@param out each process contains the result on output( must be of size num_superacc*\c exblas::BIN_COUNT, allocated on the cpu) (write, may not alias in)
@param comm The complete MPI communicator
@param comm_mod This is comm modulo 128 ( or any other number <256)
@param comm_mod_reduce This is the communicator consisting of all rank 0 processes in comm_mod, may be \c MPI_COMM_NULL
@sa \c exblas::mpi_reduce_communicator to generate the required communicators
*/
static void reduce_mpi_cpu(  unsigned num_superacc, int64_t* in, int64_t* out, MPI_Comm comm, MPI_Comm comm_mod, MPI_Comm comm_mod_reduce )
{
    for( unsigned i=0; i<num_superacc; i++)
    {
        int imin=exblas::IMIN, imax=exblas::IMAX;
        cpu::Normalize(&in[i*exblas::BIN_COUNT], imin, imax);
    }
    MPI_Reduce(in, out, num_superacc*exblas::BIN_COUNT, MPI_LONG, MPI_SUM, 0, comm_mod);
    int rank;
    MPI_Comm_rank( comm_mod, &rank);
    if(comm_mod_reduce != MPI_COMM_NULL)
    {
        for( unsigned i=0; i<num_superacc; i++)
        {
            int imin=exblas::IMIN, imax=exblas::IMAX;
            cpu::Normalize(&out[i*exblas::BIN_COUNT], imin, imax);
            for( int k=0; k<exblas::BIN_COUNT; k++)
                in[i*BIN_COUNT+k] = out[i*BIN_COUNT+k];
        }
        MPI_Reduce(in, out, num_superacc*exblas::BIN_COUNT, MPI_LONG, MPI_SUM, 0, comm_mod_reduce);
    }
    MPI_Bcast( out, num_superacc*exblas::BIN_COUNT, MPI_LONG, 0, comm);
}

}//namespace exblas
