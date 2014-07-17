#pragma once

#include "mpi_config.h"

#include <thrust/host_vector.h>
#include "vector_traits.h"

namespace dg
{

struct MPI_Vector
{
    MPI_Vector( unsigned stride, unsigned Nx, unsigned Ny): 
        stride_(stride), Nx_(Nx), Ny_(Ny), Nz_(1), data_( stride*Nx*Ny) {}
    MPI_Vector( unsigned stride, unsigned Nx, unsigned Ny, unsigned Nz): 
        stride_(stride), Nx_(Nx), Ny_(Ny), Nz_(Nz), data_( stride*Nx*Ny*Nz) {}
    thrust::host_vector<double>& data() {return data_;}
    const thrust::host_vector<double>& data() const {return data_;}
    unsigned stride()const {return stride_;}
    unsigned Nx()const {return Nx_;}
    unsigned Ny()const {return Ny_;}
    unsigned Nz()const {return Nz_;}
    double operator[]( unsigned idx) const {return data_[idx];}
    void x_row( MPI_Comm comm);
    void x_col( MPI_Comm comm);
  private:
    unsigned stride_, Nx_, Ny_, Nz_; //!< has to know interior 
    thrust::host_vector<double> data_; //!< thrust host vector as data type
};

template<> 
struct VectorTraits<MPI_Vector> {
    typedef double value_type;
    typedef MPIVectorTag vector_category;
};


void MPI_Vector::x_row( MPI_Comm comm)
{
    MPI_Status status;
    int cols = Nx_*stride_;
    int rows = Ny_;
    int source, dest;
    MPI_Cart_shift( comm, 1, +1, &source, &dest);
    //MPI_Sendrecv is good for sending in a "chain"
    MPI_Sendrecv(   &data_[cols], cols, MPI_DOUBLE,  //sender
                    dest, 7,  //destination
                    &data_[cols*(rows-1)], cols, MPI_DOUBLE, //receiver
                    source, 7, //source
                    MPI_COMM_WORLD, &status);

    MPI_Cart_shift( comm, 1, -1, &source, &dest);
    MPI_Sendrecv(   data_.data(), cols, MPI_DOUBLE,  //sender
                    dest, 1,  //destination
                    &data_[(rows-2)*cols], cols, MPI_DOUBLE, //receiver
                    source, 1, //source
                    MPI_COMM_WORLD, &status);


}
void MPI_Vector::x_col( MPI_Comm comm)
{
    MPI_Status status;
    int stride = stride_;
    int cols = Nx_;
    int rows = Ny_;
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
            sendbuffer1[i] = data_[(i*cols + 1       )*stride+j];
            sendbuffer2[i] = data_[(i*cols + cols - 2)*stride+j];
        }
    }
    int source, dest;
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
            data_[(i*cols           )*stride+j] = recvbuffer1[i];
            data_[(i*cols + cols - 1)*stride+j] = recvbuffer2[i];
        }
    }
}


}//namespace dg
