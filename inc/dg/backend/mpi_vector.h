#pragma once

#include <cassert>
#include <thrust/host_vector.h>
#include "vector_traits.h"

namespace dg
{

struct MPI_Vector
{
    /**
     * @brief construct a vector
     *
     * @param n polynomial coefficients
     * @param Nx local # of cells in x 
     * @param Ny local # of cells in y
     * @param comm MPI communicator
     */
    MPI_Vector( unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm): 
        n_(n), Nx_(Nx), Ny_(Ny), Nz_(1), data_( n*n*Nx*Ny), comm_(comm) {}
    MPI_Vector( unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm): 
        n_(n), Nx_(Nx), Ny_(Ny), Nz_(Nz), data_( n*n*Nx*Ny*Nz), comm_(comm) {}
    thrust::host_vector<double>& data() {return data_;}
    const thrust::host_vector<double>& data() const {return data_;}
    /**
     * @brief Cut the ghostcells and leave interior
     *
     * @return  The interior without ghostcells
     */
    thrust::host_vector<double> cut_overlap() const;
    /**
     * @brief Opposite of cut_overlap, copies values into interior
     *
     * a cut_overlap followed by a copy_into_interior leaves the values unchanged
     * @param src The source values
     */
    void copy_into_interior( const thrust::host_vector<double>& src);
    unsigned n() const {return n_;}
    unsigned Nx()const {return Nx_;}
    unsigned Ny()const {return Ny_;}
    unsigned Nz()const {return Nz_;}
    /**
     * @brief Return local size
     * 
     * @return local size
     */
    unsigned size() const{return n_*n_*Nx_*Ny_*Nz_;}
    double operator[]( unsigned idx) const {return data_[idx];}
    /**
     * @brief exchanged data of overlapping rows
     *
     * @param comm Communicator
     */
    void x_row( MPI_Comm comm);
    /**
     * @brief exchange data of overlapping columns
     *
     * @param comm Communicator
     */
    void x_col( MPI_Comm comm);
    void display( std::ostream& os) const
    {
        for( unsigned s=0; s<Nz_; s++)
            for( unsigned i=0; i<n_*Ny_; i++)
            {
                for( unsigned j=0; j<n_*Nx_; j++)
                    os << data_[(s*n_*Ny_+i)*n_*Nx_ + j] << " ";
                os << "\n";
            }
    }
    friend std::ostream& operator<<( std::ostream& os, const MPI_Vector& v)
    {
        os << "Vector with Nz = "<<v.Nz_<<", Ny = "<<v.Ny_
           <<" Nx = "<<v.Nx_<<" and n = "<<v.n_<<": \n";
        v.display(os);
        return os;
    }
    void swap( MPI_Vector& that){ 
#ifdef DG_DEBUG
        assert( n_ == that.n_);
        assert( Nx_ == that.Nx_);
        assert( Ny_ == that.Ny_);
        assert( Nz_ == that.Nz_);
        assert( comm_ == that.comm_);
#endif
        data_.swap(that.data_);
    }
  private:
    unsigned n_, Nx_, Ny_, Nz_; //!< has to know interior 
    thrust::host_vector<double> data_; //!< thrust host vector as data type
    MPI_Comm comm_;
};

typedef MPI_Vector MVec;
template<> 
struct VectorTraits<MPI_Vector> {
    typedef double value_type;
    typedef MPIVectorTag vector_category;
};
template<> 
struct VectorTraits<const MPI_Vector> {
    typedef double value_type;
    typedef MPIVectorTag vector_category;
};


void MPI_Vector::x_col( MPI_Comm comm)
{
    //shift data in zero-th dimension
    MPI_Status status;
    int n = n_;
    int cols = Nx_;
    int rows = n_*Ny_*Nz_;
    //create buffer before sending single cells (1 is left side, 2 is right side)
    thrust::host_vector<double> sendbuffer1( rows*n, 0);
    thrust::host_vector<double> recvbuffer1( rows*n, 0);
    thrust::host_vector<double> sendbuffer2( rows*n, 0);
    thrust::host_vector<double> recvbuffer2( rows*n, 0);
    //copy into buffers
    for( int i=0; i<rows; i++)
        for( int j=0; j<n; j++)
        {
            sendbuffer1[i*n+j] = data_[(i*cols + 1       )*n+j];
            sendbuffer2[i*n+j] = data_[(i*cols + cols - 2)*n+j];
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
        for( int j=0; j<n; j++)
        {
            data_[(i*cols           )*n+j] = recvbuffer1[i*n+j];
            data_[(i*cols + cols - 1)*n+j] = recvbuffer2[i*n+j];
        }
}

void MPI_Vector::x_row( MPI_Comm comm)
{
    //shift data in first dimension
    MPI_Status status;
    unsigned n = n_;
    unsigned cols = Nx_*n_;
    unsigned number = Nz_*Nx_*n;

    thrust::host_vector<double> sendbuffer1( n*number, 0);
    thrust::host_vector<double> recvbuffer1( n*number, 0);
    thrust::host_vector<double> sendbuffer2( n*number, 0);
    thrust::host_vector<double> recvbuffer2( n*number, 0);
    //copy into buffers
    for( unsigned s=0; s<Nz_; s++)
        for( unsigned k=0; k<n; k++)
            for( unsigned j=0; j<cols; j++)
            {
                sendbuffer1[(s*n+k)*cols+j] = 
                    data_[((s*Ny_ + 1)*n + k)*cols + j];
                sendbuffer2[(s*n+k)*cols+j] = 
                    data_[((s*Ny_ + Ny_ - 2)*n + k)*cols + j];
            }
    int source, dest;
    MPI_Cart_shift( comm, 1, -1, &source, &dest);
    //MPI_Sendrecv is good for sending in a "chain"
    MPI_Sendrecv(   sendbuffer1.data(), n*number, MPI_DOUBLE,  //sender
                    dest, 7,  //destination
                    recvbuffer2.data(), n*number, MPI_DOUBLE, //receiver
                    source, 7, //source
                    comm, &status);

    MPI_Cart_shift( comm, 1, +1, &source, &dest);
    MPI_Sendrecv(   sendbuffer2.data(), n*number, MPI_DOUBLE,  //sender
                    dest, 1,  //destination
                    recvbuffer1.data(), n*number, MPI_DOUBLE, //receiver
                    source, 1, //source
                    comm, &status);
    //copy back into vector
    for( unsigned s=0; s<Nz_; s++)
        for( unsigned k=0; k<n; k++)
            for( unsigned j=0; j<cols; j++)
            {
                data_[((s*Ny_    )*n + k)*cols + j] = 
                    recvbuffer1[(s*n+k)*cols+j];
                data_[((s*Ny_ + Ny_ - 1)*n + k)*cols + j] =
                    recvbuffer2[(s*n+k)*cols+j]; 
            }

}

thrust::host_vector<double> MPI_Vector::cut_overlap() const
{
    thrust::host_vector<double> reduce( n_*n_*(Nx_-2)*(Ny_-2)*Nz_, 1.);
    for( unsigned s=0; s<Nz_; s++)
        for( unsigned i=n_; i<(Ny_-1)*n_; i++)
            for( unsigned j=n_; j<(Nx_-1)*n_; j++)
                reduce[ j-n_ + (Nx_-2)*n_*( i-n_ + (Ny_-2)*n_*s)] = 
                    data_[ j + Nx_*n_*(i + Ny_*n_*s)];
    return reduce;
}

void MPI_Vector::copy_into_interior( const thrust::host_vector<double>& src)
{
    assert( src.size() == n_*n_*(Nx_-2)*(Ny_-2)*Nz_);
    for( unsigned s=0; s<Nz_; s++)
        for( unsigned i=n_; i<(Ny_-1)*n_; i++)
            for( unsigned j=n_; j<(Nx_-1)*n_; j++)
                data_[ j + Nx_*n_*(i + Ny_*n_*s)] =
                    src[ j-n_ + (Nx_-2)*n_*( i-n_ + (Ny_-2)*n_*s)];
}

}//namespace dg
