#pragma once

#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include "vector_traits.h"
#include "thrust_vector_blas.cuh"

namespace dg
{

/**
 * @brief mpi Vector class 
 *
 * @ingroup mpi_structures
 * The idea of this Vector class is to simply base it on an existing container class
 * that fully supports the blas functionality. In a computation we use mpi to 
 communicate (e.g. boundary points in matrix-vector multiplications) and use
 the existing blas functions for the local computations. 
 * (At the blas level 1 level communication is needed for scalar products)
 * @tparam container underlying local container class
 */
template<class container>
struct MPI_Vector
{
    typedef container container_type;//!< typedef to acces underlying container
    MPI_Vector(){}
    /**
     * @brief construct a vector
     *
     * @param data internal data
     * @param comm MPI communicator
     */
    MPI_Vector( const container& data, MPI_Comm comm): 
        data_( data), comm_(comm) {}
    
    /**
    * @brief Conversion operator
    *
    * uses conversion between compatible containers
    * @tparam OtherContainer Another container class
    * @param src the source 
    */
    template<class OtherContainer>
    MPI_Vector( const MPI_Vector<OtherContainer>& src){ data_ = src.data(); comm_ = src.communicator();} 

    /**
     * @brief Set the communicator to which this vector belongs
     *
     * @return MPI communicator reference
     */
    MPI_Comm& communicator(){return comm_;}

    /**
     * @brief Set underlying data
     *
     * @return 
     */
    container& data() {return data_;}
    /**
     * @brief Get underlying data
     *
     * @return 
     */
    const container& data() const {return data_;}
    /**
     * @brief Return local size
     * 
     * @return local size
     */
    unsigned size() const{return data_.size();}

    /**
     * @brief The communicator to which this vector belongs
     *
     * @return MPI communicator
     */
    MPI_Comm communicator() const{return comm_;}

    /**
     * @brief Display local data
     *
     * @param os outstream
     */
    void display( std::ostream& os) const
    {
        for( unsigned j=0; j<data_.size(); j++)
            os << data_[j] << " ";
        os << "\n";
    }
    /**
    * @brief Access single data points
    *
    * @param i index
    *
    * @return a value
    * @attention if the container class is a device vector communication is needed to transfer the value from the device to the host
    */
    double operator[](int i) const{return data_[i];}
    /**
     * @brief Disply local data
     *
     * @param os outstream
     * @param v a vector
     *
     * @return  the outsream
     */
    friend std::ostream& operator<<( std::ostream& os, const MPI_Vector& v)
    {
        os << "Vector of size  "<<v.size()<<"\n";
        v.display(os);
        return os;
    }
    /**
     * @brief Swap data 
     *
     * @param that must have equal sizes and communicator
     */
    void swap( MPI_Vector& that){ 
        assert( comm_ == that.comm_);
        data_.swap(that.data_);
    }
  private:
    container data_; 
    MPI_Comm comm_;
};

///@cond

template<class container> 
struct VectorTraits<MPI_Vector<container> > {
    typedef double value_type;
    typedef MPIVectorTag vector_category;
};
template<class container> 
struct VectorTraits<const MPI_Vector<container> > {
    typedef double value_type;
    typedef MPIVectorTag vector_category;
};
///@endcond

/////////////////////////////communicator exchanging columns//////////////////

/**
* @brief Communicator for nearest neighbor communication
*
* exchanges a halo of given size among next neighbors in a given direction
* @ingroup mpi_structures
* @tparam Index the type of index container
* @tparam Vector the vector container type
* @note models aCommunicator
*/
template<class Index, class Vector>
struct NearestNeighborComm
{
    NearestNeighborComm(){silent_ = true;}
    /**
    * @brief Construct 
    *
    * @param n size of the halo
    * @param vector_dimensions {x, y, z} dimension (total number of points)
    * @param comm the (cartesian) communicator 
    * @param direction 0 is x, 1 is y, 2 is z
    */
    NearestNeighborComm( int n, const int vector_dimensions[3], MPI_Comm comm, int direction)
    {
        construct( n, vector_dimensions, comm, direction);
    }

    /**
    * @brief Construct from other Communicator
    *
    * Simply copies halo size, dimensions, communicator and direction and 
    constructs a new object
    * @tparam OtherIndex other index type
    * @tparam OtherVector other container type
    * @param src source object
    */
    template< class OtherIndex, class OtherVector>
    NearestNeighborComm( const NearestNeighborComm<OtherIndex, OtherVector>& src){
        construct( src.n(), src.dims(), src.communicator(), src.direction());
    }

    /**
    * @brief Construct a vector containing halo cells of neighboring processes
    *
    * No inner points are stored
    * @param input local input vector
    *
    * @return new container
    */
    Vector collect( const Vector& input)const;
    /**
    * @brief Size of the output of collect
    *
    * @return size
    */
    int size()const; //size of values is size of input plus ghostcells
    /**
    * @brief The communicator used
    *
    * @return MPI communicator
    */
    MPI_Comm communicator() const {return comm_;}
    /**
    * @brief halo size
    *
    * @return  halo size
    */
    int n() const{return n_;}
    /**
    * @brief  The dimensionality of the input vector
    *
    * @return dimensions
    */
    const int* dims() const{return dim_;}
    /**
    * @brief The direction of communication
    *
    * @return direction
    */
    int direction() const {return direction_;}
    private:
    void construct( int n, const int vector_dimensions[3], MPI_Comm comm, int direction);
    typedef thrust::host_vector<double> HVec;

    int n_, dim_[3]; //deepness, dimensions
    MPI_Comm comm_;
    int direction_;
    bool silent_;
    Index buffer_gather1, buffer_gather2, buffer_scatter1, buffer_scatter2;

    void sendrecv( HVec&, HVec&, HVec& , HVec&)const;
    int buffer_size() const;
};

typedef NearestNeighborComm<thrust::host_vector<int>, thrust::host_vector<double> > NNCH; //!< host Communicator for the use in an mpi matrix for derivatives

///@cond

template<class I, class V>
void NearestNeighborComm<I,V>::construct( int n, const int dimensions[3], MPI_Comm comm, int direction)
{
    silent_=false;
        int ndims;
        MPI_Cartdim_get( comm, &ndims);
        int dims[ndims], periods[ndims], coords[ndims];
        MPI_Cart_get( comm, ndims, dims, periods, coords);
        if( dims[direction] == 1) silent_ = true;
    n_=n;
    dim_[0] = dimensions[0], dim_[1] = dimensions[1], dim_[2] = dimensions[2];
    comm_ = comm;
    direction_ = direction;
    assert( 0<=direction);
    assert( direction <3);
    thrust::host_vector<int> hbgather1(buffer_size()), hbgather2(hbgather1), hbscattr1(buffer_size()), hbscattr2(hbscattr1);
    switch( direction)
    {
        case( 0):
        for( int i=0; i<dim_[2]*dim_[1]; i++)
        {
            for( int j=0; j<n_; j++)
            {
                hbgather1[i*n+j] = (i*dim_[0]               + j);
                hbgather2[i*n+j] = (i*dim_[0] + dim_[0] - n + j);
                hbscattr1[i*n+j] = (i*(2*n)                      + j);
                hbscattr2[i*n+j] = (i*(2*n)+ (2*n) - n + j);
            }
        }
        break;
        case( 1):
        for( int i=0; i<dim_[2]; i++)
        {
            for( int j=0; j<n; j++)
                for( int k=0; k<dim_[0]; k++)
                {
                    hbgather1[(i*n+j)*dim_[0]+k] = 
                        (i*dim_[1] +               j)*dim_[0] + k;
                    hbgather2[(i*n+j)*dim_[0]+k] = 
                        (i*dim_[1] + dim_[1] - n + j)*dim_[0] + k;
                    hbscattr1[(i*n+j)*dim_[0]+k] = 
                        (i*(          2*n) +                       j)*dim_[0] + k;
                    hbscattr2[(i*n+j)*dim_[0]+k] = 
                        (i*(          2*n) + (          2*n) - n + j)*dim_[0] + k;
                }
        }
        break;
        case( 2):
        for( int i=0; i<n; i++)
        {
            for( int j=0; j<dim_[0]*dim_[1]; j++)
            {
                hbgather1[i*dim_[0]*dim_[1]+j] =  i*dim_[0]*dim_[1]            + j;
                hbgather2[i*dim_[0]*dim_[1]+j] = (i+dim_[2]-n)*dim_[0]*dim_[1] + j;
                hbscattr1[i*dim_[0]*dim_[1]+j] =  i*dim_[0]*dim_[1]            + j;
                hbscattr2[i*dim_[0]*dim_[1]+j] = (i+(  2*n)-n)*dim_[0]*dim_[1] + j;
            }
        }
        break;
    }
    buffer_gather1 =hbgather1, buffer_gather2 =hbgather2;
    buffer_scatter1=hbscattr1, buffer_scatter2=hbscattr2;
}

template<class I, class V>
int NearestNeighborComm<I,V>::size() const
{
    if( silent_) return 0;
    return 2*buffer_size();
}

template<class I, class V>
int NearestNeighborComm<I,V>::buffer_size() const
{
    switch( direction_)
    {
        case( 0): //x-direction
            return n_*dim_[1]*dim_[2];
        case( 1): //y-direction
            return n_*dim_[0]*dim_[2];
        case( 2): //z-direction
            return n_*dim_[0]*dim_[1]; //no further n_ (hide in dim_)
        default: 
            return 0;
    }
}

template<class I, class V>
V NearestNeighborComm<I,V>::collect( const V& input) const
{
    if( silent_) return V();
        //int rank;
        //MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        //dg::Timer t;
        //t.tic();
    V values( size());
    V buffer1( buffer_size());
    V buffer2( buffer_size());
    //V sendbuffer2( buffer_size());
    //V recvbuffer2( buffer_size());
        //t.toc();
        //if(rank==0)std::cout << "Allocation   took "<<t.diff()<<"s\n";
        //t.tic();
    //gather values from input into sendbuffer
    thrust::gather( buffer_gather1.begin(), buffer_gather1.end(), input.begin(), buffer1.begin());
    thrust::gather( buffer_gather2.begin(), buffer_gather2.end(), input.begin(), buffer2.begin());
        //t.toc();
        //if(rank==0)std::cout << "Gather       took "<<t.diff()<<"s\n";
        //t.tic();
    //copy to host 
    HVec sb1,sb2;
    dg::blas1::detail::doTransfer( buffer1, sb1, typename VectorTraits<V>::vector_category(), ThrustVectorTag());
    dg::blas1::detail::doTransfer( buffer2, sb2, typename VectorTraits<V>::vector_category(), ThrustVectorTag());
    HVec rb1(buffer_size(), 0), rb2( buffer_size(), 0);
        //t.toc();
        //if(rank==0)std::cout << "Copy to host took "<<t.diff()<<"s\n";
        //t.tic();
    //mpi sendrecv
    sendrecv( sb1, sb2, rb1, rb2);
        //t.toc();
        //if(rank==0)std::cout << "MPI sendrecv took "<<t.diff()<<"s\n";
        //t.tic();
    //send data back to device
    dg::blas1::detail::doTransfer( rb1, buffer1, ThrustVectorTag(), typename VectorTraits<V>::vector_category());
    dg::blas1::detail::doTransfer( rb2, buffer2, ThrustVectorTag(), typename VectorTraits<V>::vector_category());
        //t.toc();
        //if(rank==0)std::cout << "Copy to devi took "<<t.diff()<<"s\n";
        //t.tic();
    //scatter received values into values array
    thrust::scatter( buffer1.begin(), buffer1.end(), buffer_scatter1.begin(), values.begin());
    thrust::scatter( buffer2.begin(), buffer2.end(), buffer_scatter2.begin(), values.begin());
        //t.toc();
        //if(rank==0)std::cout << "Scatter      took "<<t.diff()<<"s\n";
    return values;
}

template<class I, class V>
void NearestNeighborComm<I,V>::sendrecv( HVec& sb1, HVec& sb2 , HVec& rb1, HVec& rb2) const
{
    int source, dest;
    MPI_Status status;
    //mpi_cart_shift may return MPI_PROC_NULL then the receive buffer is not modified 
    MPI_Cart_shift( comm_, direction_, -1, &source, &dest);
    MPI_Sendrecv(   sb1.data(), buffer_size(), MPI_DOUBLE,  //sender
                    dest, 3,  //destination
                    rb2.data(), buffer_size(), MPI_DOUBLE, //receiver
                    source, 3, //source
                    comm_, &status);
    MPI_Cart_shift( comm_, direction_, +1, &source, &dest);
    MPI_Sendrecv(   sb2.data(), buffer_size(), MPI_DOUBLE,  //sender
                    dest, 9,  //destination
                    rb1.data(), buffer_size(), MPI_DOUBLE, //receiver
                    source, 9, //source
                    comm_, &status);
}


///@endcond
}//namespace dg
