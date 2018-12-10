#pragma once

#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include "exblas/mpi_accumulate.h"
#include "tensor_traits.h"
#include "blas1_dispatch_shared.h"
#include "mpi_communicator.h"
#include "memory.h"

//TODO: should we catch the cases where outer_size \in {1,2,3} in NearestNeighborComm?
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
 * @tparam container local container type. Must have a \c size() and a \c swap() member function and a specialization of the \c TensorTraits class.
 */
template<class container>
struct MPI_Vector
{
    typedef container container_type;//!< typedef to acces underlying container
    ///no data is allocated, communicators are \c MPI_COMM_NULL
    MPI_Vector(){
        comm_ = comm128_ = comm128Reduce_ = MPI_COMM_NULL;
    }
    /**
     * @brief construct a vector
     *
     * calls \c exblas::mpi_reduce_communicator() (collective call)
     * @param data internal data copy
     * @param comm MPI communicator (may not be \c MPI_COMM_NULL)
     */
    MPI_Vector( const container& data, MPI_Comm comm): data_( data), comm_(comm) {
        exblas::mpi_reduce_communicator( comm, &comm128_, &comm128Reduce_);
    }

    /**
    * @brief Conversion operator
    *
    * uses conversion between compatible containers
    * @tparam OtherContainer another container class (container must be copy constructible from OtherContainer)
    * @param src the source
    */
    template<class OtherContainer>
    MPI_Vector( const MPI_Vector<OtherContainer>& src){
        data_ = src.data();
        comm_ = src.communicator();
        comm128_ = src.communicator_mod();
        comm128Reduce_ = src.communicator_mod_reduce();
    }

    ///@brief Get underlying data
    ///@return read access to data
    const container& data() const {return data_;}
    ///@brief Set underlying data
    ///@return write access to data
    container& data() {return data_;}

    ///@brief Get the communicator to which this vector belongs
    ///@return read access to MPI communicator
    MPI_Comm communicator() const{return comm_;}
    ///@brief Returns a communicator of fixed size 128
    MPI_Comm communicator_mod() const{return comm128_;}

    /**
     * @brief Returns a communicator consisting of all processes with rank 0 in \c communicator_mod()
     *
     * @return returns MPI_COMM_NULL to processes not part of that group
     */
    MPI_Comm communicator_mod_reduce() const{return comm128Reduce_;}
    /**
    * @brief Set the communicators with \c exblas::mpi_reduce_communicator
    */
    void set_communicator(MPI_Comm comm, MPI_Comm comm_mod, MPI_Comm comm_mod_reduce){
        comm_ = comm;
        comm128_ = comm_mod;
        comm128Reduce_ = comm_mod_reduce;
    }

    ///@brief Return the size of the data object
    ///@return local size
    unsigned size() const{return data_.size();}

    ///@brief Swap data  and communicator
    ///@param src communicator and data is swapped
    void swap( MPI_Vector& src){
        data_.swap(src.data_);
        std::swap( comm_ , src.comm_);
        std::swap( comm128_ , src.comm128_);
        std::swap( comm128Reduce_ , src.comm128Reduce_);
    }
  private:
    container data_;
    MPI_Comm comm_, comm128_, comm128Reduce_;

};

///@addtogroup dispatch
///@{

///@brief prototypical MPI vector
template<class container>
struct TensorTraits<MPI_Vector<container> > {
    using value_type = get_value_type<container>;
    using tensor_category = MPIVectorTag;
    using execution_policy = get_execution_policy<container>;
};
///@}

/////////////////////////////communicator//////////////////////////
/**
* @brief Communicator for asynchronous nearest neighbor communication
*
* Imagine a communicator with Cartesian topology and further imagine that the
* grid topology is also Cartesian (vectors form a box) in two or three dimensions.
* In each direction this box has a boundary layer (the halo) of a depth given by
* the user. Each boundary layer has two neighboring layers, one on the same process
* and one lying on the neighboring process.
* What this class does is to gather these six layers (three for each side)
* into one buffer vector. The layout of the buffer vector is independently of the
* direction contiguous in the layer.
*
* If the number of neighboring processes in the given direction is 1,
* the buffer size is 0 and all members return immediately.
*
* This is done asynchronously i.e. the user can initiate the communication
* and signal when the results are needed at a later stage.
* @note the corresponding gather map is of general type and the communication
*  can also be modeled in \c GeneralComm, but not \c BijectiveComm or \c SurjectiveComm
*  @attention Currently we cannot handle the case where the whole vector is the boundary layer (i.e. \c size() == local_vector_size) i.e. both neighboring layers are on different processes
* @ingroup mpi_structures
* @tparam Index the type of index container (must be either thrust::host_vector<int> or thrust::device_vector<int>)
* @tparam Vector the vector container type must have a resize() function and work
* in the thrust library functions ( i.e. must a thrust::host_vector or thrust::device_vector)
*/
template<class Index, class Buffer, class Vector>
struct NearestNeighborComm
{
    using container_type = Vector;
    using buffer_type = Buffer;
    using pointer_type = get_value_type<Vector>*;
    using const_pointer_type = get_value_type<Vector> const *;
    ///@brief no communication
    NearestNeighborComm(){
        silent_ = true;
    }
    /**
    * @brief Construct
    *
    * @param n depth of the halo
    * @param vector_dimensions {x, y, z} dimension (total number of points)
    * @param comm the (cartesian) communicator
    * @param direction 0 is x, 1 is y, 2 is z
    */
    NearestNeighborComm( unsigned n, const unsigned vector_dimensions[3], MPI_Comm comm, unsigned direction)
    {
        static_assert( std::is_same<const_pointer_type, get_value_type<Buffer>>::value, "Must be same pointer types");
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
    template< class OtherIndex, class OtherBuffer, class OtherVector>
    NearestNeighborComm( const NearestNeighborComm<OtherIndex, OtherBuffer, OtherVector>& src){
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

    /**
     * @brief Allocate a buffer object
     * @return a buffer object on the stack
     * @note if \c size()==0 the default constructor of \c Buffer is called
     */
    Buffer allocate_buffer( )const{
        if( do_size() == 0 ) return Buffer();
        return Buffer(6);
    }

    /**
    * @brief Gather values from given Vector and initiate asynchronous MPI communication
    * @param input from which to gather data (it is safe to change values on return since values to communicate are copied into \c buffer)
    * @param buffer (write only) where received data resides after \c global_gather_wait() was called (must be of size \c size())
    * @param rqst four request variables that can be used to call MPI_Waitall
    */
    void global_gather_init( const_pointer_type input, const_pointer_type* buffer, MPI_Request rqst[4])const
    {
        unsigned size = buffer_size();
        if(trivial_)
        {
            buffer[0] = thrust::raw_pointer_cast(&internal_buffer_.data()[0*size]);
            buffer[1] = input;
            buffer[2] = input+size;
            buffer[3] = input+(outer_size_-2)*size;
            buffer[4] = input+(outer_size_-1)*size;
            buffer[5] = thrust::raw_pointer_cast(&internal_buffer_.data()[5*size]);
        }
        else
        {
            buffer[0] = thrust::raw_pointer_cast(&internal_buffer_.data()[0*size]);
            buffer[1] = thrust::raw_pointer_cast(&internal_buffer_.data()[1*size]);
            buffer[2] = thrust::raw_pointer_cast(&internal_buffer_.data()[2*size]);
            buffer[3] = thrust::raw_pointer_cast(&internal_buffer_.data()[3*size]);
            buffer[4] = thrust::raw_pointer_cast(&internal_buffer_.data()[4*size]);
            buffer[5] = thrust::raw_pointer_cast(&internal_buffer_.data()[5*size]);
        }
        do_global_gather_init( get_execution_policy<Vector>(), input, buffer, rqst);
        sendrecv( buffer[1], buffer[4],
                  thrust::raw_pointer_cast(&internal_buffer_.data()[0*size]),
                  thrust::raw_pointer_cast(&internal_buffer_.data()[5*size]),
                  rqst);
    }
    /**
    * @brief Wait for asynchronous communication to finish and gather received data into buffer
    *
    * @param input from which to gather data (it is safe to change values on return since values to communicate are copied into \c buffer)
    * @param buffer (write only) where received data resides on return (must be of size \c size())
    * @param rqst the same four request variables that were used in global_gather_init
    */
    void global_gather_wait(const_pointer_type input, const_pointer_type* buffer, MPI_Request rqst[4])const
    {
        MPI_Waitall( 4, rqst, MPI_STATUSES_IGNORE );
    }
    ///@copydoc aCommunicator::buffer_size()
    unsigned size() const{return do_size();}
    ///@copydoc aCommunicator::isCommunicating()
    bool isCommunicating() const{
        if( do_size() == 0) return false;
        return true;
    }
    ///@copydoc aCommunicator::isCommunicating()
    MPI_Comm communicator() const{return comm_;}
    private:
    void do_global_gather_init( OmpTag, const_pointer_type, const_pointer_type*, MPI_Request rqst[4])const;
    void do_global_gather_init( SerialTag, const_pointer_type, const_pointer_type*, MPI_Request rqst[4])const;
    void do_global_gather_init( CudaTag, const_pointer_type, const_pointer_type*, MPI_Request rqst[4])const;
    unsigned do_size()const;
    void construct( unsigned n, const unsigned vector_dimensions[3], MPI_Comm comm, unsigned direction);

    unsigned n_, dim_[3]; //deepness, dimensions
    MPI_Comm comm_;
    unsigned direction_;
    bool silent_, trivial_=false; //silent -> no comm, trivial -> comm in last dim
    unsigned outer_size_ = 1; //size of vector in units of buffer_size
    Index gather_map_middle_;
    dg::Buffer<Vector> internal_buffer_;

    void sendrecv(const_pointer_type, const_pointer_type, pointer_type, pointer_type, MPI_Request rqst[4])const;
    unsigned buffer_size() const;
    int m_source[2], m_dest[2];
};

///@cond

template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::construct( unsigned n, const unsigned dimensions[3], MPI_Comm comm, unsigned direction)
{
    static_assert( std::is_base_of<SharedVectorTag, get_tensor_category<V>>::value,
               "Only Shared vectors allowed");
    silent_=false;
    n_=n;
    dim_[0] = dimensions[0], dim_[1] = dimensions[1], dim_[2] = dimensions[2];
    direction_ = direction; //now buffer_size() is callable
    if( dimensions[2] == 1 && direction == 1) trivial_ = true;
    else if( direction == 2) trivial_ = true;
    else trivial_ = false;
    if( !silent_)
    {
        outer_size_ = dimensions[0]*dimensions[1]*dimensions[2]/buffer_size();
        assert( outer_size_ > 1 && "Parallelization too fine grained!"); //right now we cannot have that
    }
    comm_ = comm;
    {
        int ndims;
        MPI_Cartdim_get( comm, &ndims);
        int dims[ndims], periods[ndims], coords[ndims];
        MPI_Cart_get( comm, ndims, dims, periods, coords);
        if( dims[direction] == 1) silent_ = true;
    }
    //mpi_cart_shift may return MPI_PROC_NULL then the receive buffer is not modified
    MPI_Cart_shift( comm_, direction_, -1, &m_source[0], &m_dest[0]);
    MPI_Cart_shift( comm_, direction_, +1, &m_source[1], &m_dest[1]);
    assert( direction <3);
    thrust::host_vector<int> mid_gather( 4*buffer_size());
    switch( direction)
    {
        case( 0):
        for( unsigned i=0; i<dim_[2]*dim_[1]; i++)
            for( unsigned j=0; j<n; j++)
            {
                mid_gather[(0*n+j)*dim_[2]*dim_[1]+i] = i*dim_[0]               + j;
                mid_gather[(1*n+j)*dim_[2]*dim_[1]+i] = i*dim_[0] + n           + j;
                mid_gather[(2*n+j)*dim_[2]*dim_[1]+i] = i*dim_[0] + dim_[0]-2*n + j;
                mid_gather[(3*n+j)*dim_[2]*dim_[1]+i] = i*dim_[0] + dim_[0]-  n + j;
            }
        break;
        case( 1):
        for( unsigned i=0; i<dim_[2]; i++)
            for( unsigned j=0; j<n; j++)
                for( unsigned k=0; k<dim_[0]; k++)
                {
                    mid_gather[((0*n+j)*dim_[2]+i)*dim_[0] + k] = (i*dim_[1]               + j)*dim_[0] + k;
                    mid_gather[((1*n+j)*dim_[2]+i)*dim_[0] + k] = (i*dim_[1] + n           + j)*dim_[0] + k;
                    mid_gather[((2*n+j)*dim_[2]+i)*dim_[0] + k] = (i*dim_[1] + dim_[1]-2*n + j)*dim_[0] + k;
                    mid_gather[((3*n+j)*dim_[2]+i)*dim_[0] + k] = (i*dim_[1] + dim_[1]-  n + j)*dim_[0] + k;
                }
        break;
        case( 2):
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<dim_[0]*dim_[1]; j++)
            {
                mid_gather[(0*n+i)*dim_[0]*dim_[1]+j] = (i               )*dim_[0]*dim_[1] + j;
                mid_gather[(1*n+i)*dim_[0]*dim_[1]+j] = (i + n           )*dim_[0]*dim_[1] + j;
                mid_gather[(2*n+i)*dim_[0]*dim_[1]+j] = (i + dim_[2]-2*n )*dim_[0]*dim_[1] + j;
                mid_gather[(3*n+i)*dim_[0]*dim_[1]+j] = (i + dim_[2]-  n )*dim_[0]*dim_[1] + j;
            }
        break;
    }
    gather_map_middle_ = mid_gather;
    internal_buffer_.data().resize( 6*buffer_size() );
}

template<class I, class B, class V>
unsigned NearestNeighborComm<I,B,V>::do_size() const
{
    if( silent_) return 0;
    return 6*buffer_size(); //3 buffers on each side
}

template<class I, class B, class V>
unsigned NearestNeighborComm<I,B,V>::buffer_size() const
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

template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::do_global_gather_init( SerialTag, const_pointer_type input, const_pointer_type* buffer, MPI_Request rqst[4]) const
{
    if( !trivial_)
    {
        unsigned size = buffer_size();
        for( unsigned i=0; i<4*size; i++)
            internal_buffer_.data()[i+size] = input[gather_map_middle_[i]];
    }
}
#ifdef _OPENMP
template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::do_global_gather_init( OmpTag, const_pointer_type input, const_pointer_type* buffer, MPI_Request rqst[4]) const
{
    if(!trivial_)
    {
        unsigned size = buffer_size();
        #pragma omp parallel for
        for( unsigned i=0; i<4*size; i++)
            internal_buffer_.data()[size+i] = input[gather_map_middle_[i]];
    }
}
#endif
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::do_global_gather_init( CudaTag, const_pointer_type input, const_pointer_type* buffer, MPI_Request rqst[4]) const
{
    //gather values from input into sendbuffer
    if(!trivial_)
    {
        unsigned size = buffer_size();
        thrust::gather( thrust::cuda::tag(), gather_map_middle_.begin(), gather_map_middle_.end(), input, internal_buffer_.data().begin()+size);
    }
    cudaDeviceSynchronize(); //wait until device functions are finished before sending data
}
#endif

template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::sendrecv( const_pointer_type sb1_ptr, const_pointer_type sb2_ptr, pointer_type rb1_ptr, pointer_type rb2_ptr, MPI_Request rqst[4]) const
{
    MPI_Isend( sb1_ptr, buffer_size(),
               getMPIDataType<get_value_type<V>>(),  //sender
               m_dest[0], 3, comm_, &rqst[0]); //destination
    MPI_Irecv( rb2_ptr, buffer_size(),
               getMPIDataType<get_value_type<V>>(), //receiver
               m_source[0], 3, comm_, &rqst[1]); //source

    MPI_Isend( sb2_ptr, buffer_size(),
               getMPIDataType<get_value_type<V>>(),  //sender
               m_dest[1], 9, comm_, &rqst[2]);  //destination
    MPI_Irecv( rb1_ptr, buffer_size(),
               getMPIDataType<get_value_type<V>>(), //receiver
               m_source[1], 9, comm_, &rqst[3]); //source
}


///@endcond
}//namespace dg
