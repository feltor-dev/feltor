#pragma once

#include "mpi_config.h"

#include <thrust/host_vector.h>
#include "vector_traits.h"

namespace dg
{

struct MPI_Vector
{
    thrust::host_vector<double> data; //!< thrust host vector as data type
    unsigned stride, Nx, Ny, Nz; //!< has to know interior 
};

template<> 
struct VectorTraits<MPI_Vector> {
    typedef double value_type;
    typedef MPIVectorTag vector_category;
};


void x_row( MPI_Vector& vec, MPI_comm comm)
{
    MPI_Status status;
    int cols = vec.Nx*vec.stride;
    int rows = vec.Ny;
    int source, dest;
    status = MPI_Cart_shift( comm, 1, +1, &source, &dest);
    //MPI_Sendrecv is good for sending in a "chain"
    MPI_Sendrecv(   &vec.data[cols], cols, MPI_DOUBLE,  //sender
                    dest, 7,  //destination
                    &vec.data[cols*(rows-1)], cols, MPI_DOUBLE, //receiver
                    source, 7, //source
                    MPI_COMM_WORLD, &status);

    MPI_Cart_shift( comm, 1, -1, &source, &dest);
    MPI_Sendrecv(   vec.data.data(), cols, MPI_DOUBLE,  //sender
                    dest, 1,  //destination
                    &vec.data[(rows-2)*cols], cols, MPI_DOUBLE, //receiver
                    source, 1, //source
                    MPI_COMM_WORLD, &status);


}
void x_col( MPI_Vector& vec, MPI_comm comm)
{
    MPI_Status status;
    int stride = vec.stride;
    int cols = vec.Nx;
    int rows = vec.Ny;
    //create buffer before sending single cells (1 is left side, 2 is right side)
    thrust::host_vector<double> sendbuffer1( stride*rows);
    thrust::host_vector<double> recvbuffer1( stride*rows);
    thrust::host_vector<double> sendbuffer2( stride*rows);
    thrust::host_vector<double> recvbuffer2( stride*rows);
    //copy into buffers
    for( int i=0; i<rows; i++)
    {
        for( int j=0; j<stride; j++)
        {
            sendbuffer1[i] = vec.data[(i*cols + 1       )*stride+j];
            sendbuffer2[i] = vec.data[(i*cols + cols - 2)*stride+j];
        }
    }
    int source, dest;
    MPI_Comm_rank( comm, rank); 
    MPI_Cart_coords( comm, rank, maxdims, coords);
    MPI_Cart_shift( comm, 0, -1, &source, &dest);
    MPI_Sendrecv(   sendbuffer1.data(), rows*stride, MPI_DOUBLE,  //sender
                    dest, 3,  //destination
                    recvbuffer2.data(), rows*stride, MPI_DOUBLE, //receiver
                    source, 3, //source
                    MPI_COMM_WORLD, &status);
    MPI_Cart_shift( comm, 0, +1, &source, &dest);
    MPI_Sendrecv(   sendbuffer2.data(), rows*stride, MPI_DOUBLE,  //sender
                    dest, 9,  //destination
                    recvbuffer1.data(), rows*stride, MPI_DOUBLE, //receiver
                    source, 9, //source
                    MPI_COMM_WORLD, &status);
    //copy back into vector
    for( int i=0; i<rows; i++)
    {
        for( int j=0; j<stride; j++)
        {
            vec.data[(i*cols           )*stride+j] = recvbuffer1[i];
            vec.data[(i*cols + cols - 1)*stride+j] = recvbuffer2[i];
        }
    }
}


}//namespace dg
