#pragma once
#include <mpi.h>
#include "accumulate.h"

namespace exblas
{

/*! @brief reduce a number of superaccumulators distributed among mpi processes

We cannot sum more than 256 accumulators before we need to normalize again, so we need to split the reduction. This function normalizes, 
reduces, normalizes, reduces and broadcasts the result to all participating
processes. 
@param num_superacc number of Superaccumulators eaach process holds
@param in unnormalized input superaccumulators ( read/write, must be of size num_superacc*BIN_COUNT)
@param out each process contains the result on output( write, must be of size num_superacc*BIN_COUNT)
@param comm The complete MPI communicator
@param comm_mod This is comm modulo 128 ( or any other number <256)
@param comm_mod_reduce This is the communicator consisting of all rank 0 processes in comm_mod, may be MPI_COMM_NULL
*/
void reduce_mpi_cpu(  unsigned num_superacc, int64_t* in, int64_t* out, MPI_Comm comm, MPI_Comm comm_mod, MPI_Comm comm_mod_reduce )
{
    for( unsigned i=0; i<num_superacc; i++)
    {
        int imin=exblas::IMIN, imax=exblas::IMAX;
        Normalize(&in[i*exblas::BIN_COUNT], &imin, &imax);
    }
    MPI_Reduce(in, out, num_superacc*exblas::BIN_COUNT, MPI_LONG, MPI_SUM, 0, comm_mod); 
    int rank;
    MPI_Comm_rank( comm_mod, &rank);
    if(comm_mod_reduce != MPI_COMM_NULL)
    {
        for( unsigned i=0; i<num_superacc; i++)
        {
            int imin=exblas::IMIN, imax=exblas::IMAX;
            Normalize(&out[i*exblas::BIN_COUNT], &imin, &imax);
            for( int k=0; k<BIN_COUNT; k++)
                in[i*BIN_COUNT+k] = out[i*BIN_COUNT+k];
        }
        MPI_Reduce(in, out, num_superacc*exblas::BIN_COUNT, MPI_LONG, MPI_SUM, 0, comm_mod_reduce); 
    }
    MPI_Bcast( out, num_superacc*exblas::BIN_COUNT, MPI_LONG, 0, comm);
}

}//namespace exblas
