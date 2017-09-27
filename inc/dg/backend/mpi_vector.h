#pragma once

#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include "vector_traits.h"
#include "thrust_vector_blas.cuh"
#include "mpi_communicator.h"
#include "memory.h"

namespace dg
{

/**
 * @brief mpi Vector class 
 *
 * @ingroup mpi_structures
 *
 * This class is a simple wrapper around a container object and an MPI_Comm. 
 * The blas1 and blas2 functionality is available iff it is available for the container type.
 * We use mpi to communicate (e.g. boundary points in matrix-vector multiplications) 
 * and use the existing blas functions for the local computations. 
 * (At the blas level 1 communication is needed only for scalar products)
 * @tparam container local container type. Must have a size() and a swap() member function.
 */
template<class container>
struct MPI_Vector
{
    typedef container container_type;//!< typedef to acces underlying container
    ///no data is allocated, communicator is MPI_COMM_WORLD
    MPI_Vector(){ comm_ = MPI_COMM_WORLD;}
    /**
     * @brief construct a vector
     * @param data internal data copy 
     * @param comm MPI communicator
     */
    MPI_Vector( const container& data, MPI_Comm comm): 
        data_( data), comm_(comm) {}
    
    /**
    * @brief Conversion operator
    *
    * uses conversion between compatible containers
    * @tparam OtherContainer another container class (container must be copy constructible from OtherContainer)
    * @param src the source 
    */
    template<class OtherContainer>
    MPI_Vector( const MPI_Vector<OtherContainer>& src){ data_ = src.data(); comm_ = src.communicator();} 

    ///@brief Get underlying data
    ///@return read access to data
    const container& data() const {return data_;}
    ///@brief Set underlying data
    ///@return write access to data
    container& data() {return data_;}

    ///@brief Get the communicator to which this vector belongs
    ///@return read access to MPI communicator
    MPI_Comm communicator() const{return comm_;}
    ///@brief Set the communicator to which this vector belongs
    ///@return write access to MPI communicator
    MPI_Comm& communicator(){return comm_;}

    ///@brief Return the size of the data object
    ///@return local size
    unsigned size() const{return data_.size();}

    ///@brief Swap data  and communicator
    ///@param src communicator and data is swapped
    void swap( MPI_Vector& src){ 
        std::swap( comm_ , src.comm_);
        data_.swap(src.data_);
    }
  private:
    container data_; 
    MPI_Comm comm_;
};

///@cond

template<class container> 
struct VectorTraits<MPI_Vector<container> > {
    typedef typename container::value_type value_type;
    typedef MPIVectorTag vector_category;
};
template<class container> 
struct VectorTraits<const MPI_Vector<container> > {
    typedef typename container::value_type value_type;
    typedef MPIVectorTag vector_category;
};

///@endcond

/////////////////////////////communicator exchanging columns//////////////////
/**
* @brief Communicator for nearest neighbor communication
*
* exchanges a halo of given depth among neighboring processes in a given direction 
* (the corresponding gather map is of general type and the communication 
*  can also be modeled in GeneralComm, but not BijectiveComm or SurjectiveComm )
* @ingroup mpi_structures
* @tparam Index the type of index container (must be either thrust::host_vector<int> or thrust::device_vector<int>)
* @tparam Vector the vector container type must have a resize() function and work 
* in the thrust library functions ( i.e. must a thrust::host_vector or thrust::device_vector)
* @note models aCommunicator
*/
template<class Index, class Vector>
struct NearestNeighborComm : public aCommunicator<Vector>
{
    ///@brief no communication
    NearestNeighborComm(){
        silent_ = true;
    }
    /**
    * @brief Construct 
    *
    * @param n size of the halo
    * @param vector_dimensions {x, y, z} dimension (total number of points)
    * @param comm the (cartesian) communicator 
    * @param direction 0 is x, 1 is y, 2 is z
    */
    NearestNeighborComm( unsigned n, const unsigned vector_dimensions[3], MPI_Comm comm, unsigned direction)
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
        if( src.size() == 0)  silent_=true;
        else
            construct( src.n(), src.dims(), src.communicator(), src.direction());
    }

    /**
    * @brief halo size
    * @return  halo size
    */
    unsigned n() const{return n_;}
    /**
    * @brief  The dimensionality of the input vector
    * @return dimensions ( 3)
    */
    const unsigned* dims() const{return dim_;}
    /**
    * @brief The direction of communication
    *
    * @return direction
    */
    unsigned direction() const {return direction_;}
    NearestNeighborComm* clone()const{return new NearestNeighborComm(*this);}
    private:
    MPI_Comm do_communicator() const {return comm_;}
    unsigned do_size()const; //size of values is size of input plus ghostcells
    Vector do_make_buffer( )const{
        Vector tmp( do_size());
        return tmp;
    }
    void do_global_gather( const Vector& values, Vector& gather)const;
    void do_global_scatter_reduce( const Vector& toScatter, Vector& values)const;
    void construct( unsigned n, const unsigned vector_dimensions[3], MPI_Comm comm, unsigned direction);

    unsigned n_, dim_[3]; //deepness, dimensions
    MPI_Comm comm_;
    unsigned direction_;
    bool silent_;
    Index gather_map1, gather_map2, scatter_map1, scatter_map2; //buffer_size
    Buffer<Vector> buffer1, buffer2, rb1, rb2;  //buffer_size

    void sendrecv( Vector&, Vector&, Vector& , Vector&)const;
    unsigned buffer_size() const;
};

typedef NearestNeighborComm<thrust::host_vector<int>, thrust::host_vector<double> > NNCH; //!< host Communicator for the use in an mpi matrix for derivatives

///@cond

template<class I, class V>
void NearestNeighborComm<I,V>::construct( unsigned n, const unsigned dimensions[3], MPI_Comm comm, unsigned direction)
{
    silent_=false;
    n_=n;
    dim_[0] = dimensions[0], dim_[1] = dimensions[1], dim_[2] = dimensions[2];
    comm_ = comm;
    direction_ = direction;
    {
        int ndims;
        MPI_Cartdim_get( comm, &ndims);
        int dims[ndims], periods[ndims], coords[ndims];
        MPI_Cart_get( comm, ndims, dims, periods, coords);
        if( dims[direction] == 1) silent_ = true;
    }
    assert( direction <3);
    thrust::host_vector<int> hbgather1(buffer_size()), hbgather2(hbgather1), hbscattr1(buffer_size()), hbscattr2(hbscattr1);
    switch( direction)
    {
        case( 0):
        for( unsigned i=0; i<dim_[2]*dim_[1]; i++)
        {
            for( unsigned j=0; j<n_; j++)
            {
                hbgather1[i*n+j] = (i*dim_[0]               + j);
                hbgather2[i*n+j] = (i*dim_[0] + dim_[0] - n + j);
                hbscattr1[i*n+j] = (i*(2*n)                      + j);
                hbscattr2[i*n+j] = (i*(2*n)+ (2*n) - n + j);
            }
        }
        break;
        case( 1):
        for( unsigned i=0; i<dim_[2]; i++)
        {
            for( unsigned j=0; j<n; j++)
                for( unsigned k=0; k<dim_[0]; k++)
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
        for( unsigned i=0; i<n; i++)
        {
            for( unsigned j=0; j<dim_[0]*dim_[1]; j++)
            {
                hbgather1[i*dim_[0]*dim_[1]+j] =  i*dim_[0]*dim_[1]            + j;
                hbgather2[i*dim_[0]*dim_[1]+j] = (i+dim_[2]-n)*dim_[0]*dim_[1] + j;
                hbscattr1[i*dim_[0]*dim_[1]+j] =  i*dim_[0]*dim_[1]            + j;
                hbscattr2[i*dim_[0]*dim_[1]+j] = (i+(  2*n)-n)*dim_[0]*dim_[1] + j;
            }
        }
        break;
    }
    gather_map1 =hbgather1, gather_map2 =hbgather2;
    scatter_map1=hbscattr1, scatter_map2=hbscattr2;
    buffer1.data().resize( buffer_size()), buffer2.data().resize( buffer_size());
    rb1.data().resize( buffer_size()), rb2.data().resize( buffer_size());
}

template<class I, class V>
unsigned NearestNeighborComm<I,V>::do_size() const
{
    if( silent_) return 0;
    return 2*buffer_size();
}

template<class I, class V>
unsigned NearestNeighborComm<I,V>::buffer_size() const
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
void NearestNeighborComm<I,V>::do_global_gather( const V& input, V& values) const
{
    //gather values from input into sendbuffer
    thrust::gather( gather_map1.begin(), gather_map1.end(), input.begin(), buffer1.data().begin());
    thrust::gather( gather_map2.begin(), gather_map2.end(), input.begin(), buffer2.data().begin());
    //mpi sendrecv
    sendrecv( buffer1.data(), buffer2.data(), rb1.data(), rb2.data());
    //scatter received values into values array
    thrust::scatter( rb1.data().begin(), rb1.data().end(), scatter_map1.begin(), values.begin());
    thrust::scatter( rb2.data().begin(), rb2.data().end(), scatter_map2.begin(), values.begin());
}
template<class I, class V>
void NearestNeighborComm<I,V>::do_global_scatter_reduce( const V& values, V& input) const
{
    //scatter received values into values array
    thrust::gather( scatter_map1.begin(), scatter_map1.end(), values.begin(), rb1.data().begin());
    thrust::gather( scatter_map2.begin(), scatter_map2.end(), values.begin(), rb2.data().begin());
    //mpi sendrecv
    sendrecv( rb1.data(), rb2.data(), buffer1.data(), buffer2.data());
    //gather values from input into sendbuffer
    thrust::scatter( buffer1.data().begin(), buffer1.data().end(), gather_map1.begin(), input.begin());
    thrust::scatter( buffer2.data().begin(), buffer2.data().end(), gather_map2.begin(), input.begin());
}

template<class I, class V>
void NearestNeighborComm<I,V>::sendrecv( V& sb1, V& sb2 , V& rb1, V& rb2) const
{
    int source, dest;
    MPI_Status status;
    //mpi_cart_shift may return MPI_PROC_NULL then the receive buffer is not modified 
    MPI_Cart_shift( comm_, direction_, -1, &source, &dest);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    cudaDeviceSynchronize(); //wait until device functions are finished before sending data
#endif //THRUST_DEVICE_SYSTEM
    MPI_Sendrecv(   thrust::raw_pointer_cast(sb1.data()), buffer_size(), getMPIDataType<typename VectorTraits<V>::value_type>(),  //sender
                    dest, 3,  //destination
                    thrust::raw_pointer_cast(rb2.data()), buffer_size(), getMPIDataType<typename VectorTraits<V>::value_type>(), //receiver
                    source, 3, //source
                    comm_, &status);
    MPI_Cart_shift( comm_, direction_, +1, &source, &dest);
    MPI_Sendrecv(   thrust::raw_pointer_cast(sb2.data()), buffer_size(), getMPIDataType<typename VectorTraits<V>::value_type>(),  //sender
                    dest, 9,  //destination
                    thrust::raw_pointer_cast(rb1.data()), buffer_size(), getMPIDataType<typename VectorTraits<V>::value_type>(), //receiver
                    source, 9, //source
                    comm_, &status);
}


///@endcond
}//namespace dg
