#pragma once

#include "mpi_config.h"

#include <thrust/host_vector.h>
#include "vector_traits.h"

namespace dg
{

struct MPI_Vector
{
    MPI_Vector( unsigned n, unsigned Nx, unsigned Ny): 
        n_(n), Nx_(Nx), Ny_(Ny), Nz_(1), data_( n*n*Nx*Ny) {}
    MPI_Vector( unsigned n, unsigned Nx, unsigned Ny, unsigned Nz): 
        n_(n), Nx_(Nx), Ny_(Ny), Nz_(Nz), data_( n*n*Nx*Ny*Nz) {}
    thrust::host_vector<double>& data() {return data_;}
    const thrust::host_vector<double>& data() const {return data_;}
    unsigned n() const {return n_;}
    unsigned Nx()const {return Nx_;}
    unsigned Ny()const {return Ny_;}
    unsigned Nz()const {return Nz_;}
    unsigned size() const{return n_*n_*Nx_*Ny_*Nz_;}
    double operator[]( unsigned idx) const {return data_[idx];}
    void x_row( MPI_Comm comm);
    void x_col( MPI_Comm comm);
    void display( std::ostream& os)
    {
        for( unsigned i=0; i<Nx_; i++)
        {
            for( unsigned j=0; j<Ny_; j++)
            {
                for( unsigned k=0; k<n_; k++)
                    os << data_[(i*Nx_ + j)*n_+k] << " ";
                os << " ";
            }
            os << "\n";
        }
    }
    friend std::ostream& operator<<( std::ostream& os, const MPI_Vector& v)
    {
        std::cout << "Vector with "<<v.Ny_<<" rows and "<<v.Nx_<<" columns: \n";
        for( unsigned i=0; i<v.Ny_; i++)
        {
            for( unsigned j=0; j<v.Nx_; j++)
            {
                for( unsigned k=0; k<v.n_; k++)
                    os << v.data_[(i*v.Nx_ + j)*v.n_+k] << " ";
                os << " ";
            }
            os << "\n";
        }
    }
  private:
    unsigned n_, Nx_, Ny_, Nz_; //!< has to know interior 
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
    int cols = Nx_*n_;
    int rows = Ny_*n_;
    int source, dest;
    MPI_Cart_shift( comm, 1, -1, &source, &dest);
    //MPI_Sendrecv is good for sending in a "chain"
    MPI_Sendrecv(   &data_[n_*cols], n_*cols, MPI_DOUBLE,  //sender
                    dest, 7,  //destination
                    &data_[cols*(rows-n_)], n_*cols, MPI_DOUBLE, //receiver
                    source, 7, //source
                    MPI_COMM_WORLD, &status);

    MPI_Cart_shift( comm, 1, +1, &source, &dest);
    MPI_Sendrecv(   &data_[(rows-2*n_)*cols], n_*cols, MPI_DOUBLE,  //sender
                    dest, 1,  //destination
                    &data_[0], n_*cols, MPI_DOUBLE, //receiver
                    source, 1, //source
                    MPI_COMM_WORLD, &status);


}
void MPI_Vector::x_col( MPI_Comm comm)
{
    MPI_Status status;
    int n = n_;
    int cols = Nx_;
    int rows = n_*Ny_;
    //create buffer before sending single cells (1 is left side, 2 is right side)
    thrust::host_vector<double> sendbuffer1( rows*n);
    thrust::host_vector<double> recvbuffer1( rows*n);
    thrust::host_vector<double> sendbuffer2( rows*n);
    thrust::host_vector<double> recvbuffer2( rows*n);
    //copy into buffers
    for( int i=0; i<rows; i++)
    {
        for( int j=0; j<n; j++)
        {
            sendbuffer1[i*n+j] = data_[(i*cols + 1       )*n+j];
            sendbuffer2[i*n+j] = data_[(i*cols + cols - 2)*n+j];
        }
    }
    int source, dest;
    MPI_Cart_shift( comm, 0, -1, &source, &dest);
    MPI_Sendrecv(   sendbuffer1.data(), rows*n, MPI_DOUBLE,  //sender
                    dest, 3,  //destination
                    recvbuffer2.data(), rows*n, MPI_DOUBLE, //receiver
                    source, 3, //source
                    MPI_COMM_WORLD, &status);
    MPI_Cart_shift( comm, 0, +1, &source, &dest);
    MPI_Sendrecv(   sendbuffer2.data(), rows*n, MPI_DOUBLE,  //sender
                    dest, 9,  //destination
                    recvbuffer1.data(), rows*n, MPI_DOUBLE, //receiver
                    source, 9, //source
                    MPI_COMM_WORLD, &status);
    //copy back into vector
    for( int i=0; i<rows; i++)
    {
        for( int j=0; j<n; j++)
        {
            data_[(i*cols           )*n+j] = recvbuffer1[i*n+j];
            data_[(i*cols + cols - 1)*n+j] = recvbuffer2[i*n+j];
        }
    }
}


}//namespace dg
