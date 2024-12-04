#pragma once

#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include "exceptions.h"
#include "exblas/mpi_accumulate.h"
#include "tensor_traits.h"
#include "blas1_dispatch_shared.h"
#include "mpi_communicator.h"
#include "memory.h"
#include "config.h"

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
        m_comm = m_comm128 = m_comm128Reduce = MPI_COMM_NULL;
    }
    /**
     * @brief construct a vector
     *
     * calls \c dg::exblas::mpi_reduce_communicator() (collective call)
     * @param data internal data copy
     * @param comm MPI communicator (may not be \c MPI_COMM_NULL)
     */
    MPI_Vector( const container& data, MPI_Comm comm): m_data( data), m_comm(comm) {
        exblas::mpi_reduce_communicator( comm, &m_comm128, &m_comm128Reduce);
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
        m_data = src.data();
        m_comm = src.communicator();
        m_comm128 = src.communicator_mod();
        m_comm128Reduce = src.communicator_mod_reduce();
    }

    ///@brief Get underlying data
    ///@return read access to data
    const container& data() const {return m_data;}
    ///@brief Set underlying data
    ///@return write access to data
    container& data() {return m_data;}

    ///@brief Get the communicator to which this vector belongs
    ///@return read access to MPI communicator
    MPI_Comm communicator() const{return m_comm;}
    ///@brief Returns a communicator of fixed size 128
    MPI_Comm communicator_mod() const{return m_comm128;}

    /**
     * @brief Returns a communicator consisting of all processes with rank 0 in \c communicator_mod()
     *
     * @return returns MPI_COMM_NULL to processes not part of that group
     */
    MPI_Comm communicator_mod_reduce() const{return m_comm128Reduce;}

    /**
    * @brief Set the communicators with \c dg::exblas::mpi_reduce_communicator
    *
    * The reason why you can't just set the comm and need three parameters is
    * that generating communicators involves communication, which you might want to
    * avoid when you do it many times. So you have to call the function as
    * @code
    * MPI_Comm comm = MPI_COMM_WORLD, comm_mod, comm_mod_reduce;
    * dg::exblas::mpi_reduce_communicator( comm, &comm_mod, &comm_mod_reduce);
    * mpi_vector.set_communicator( comm, comm_mod, comm_mod_reduce);
    * @endcode
    */
    void set_communicator(MPI_Comm comm, MPI_Comm comm_mod, MPI_Comm comm_mod_reduce){
        m_comm = comm;
        m_comm128 = comm_mod;
        m_comm128Reduce = comm_mod_reduce;
    }

    ///@brief Return the size of the data object
    ///@return \c data.size()
    unsigned size() const{return m_data.size();}

    ///@brief Swap data  and communicator
    ///@param src communicator and data is swapped
    void swap( MPI_Vector& src){
        m_data.swap(src.m_data);
        std::swap( m_comm , src.m_comm);
        std::swap( m_comm128 , src.m_comm128);
        std::swap( m_comm128Reduce , src.m_comm128Reduce);
    }
  private:
    container m_data;
    MPI_Comm m_comm, m_comm128, m_comm128Reduce;

};
///@cond
//free function as required by the std to be swappable
//https://en.cppreference.com/w/cpp/named_req/Swappable
//even though with move assignments std::swap also works as fast
template<class container>
void swap( MPI_Vector<container>& a, MPI_Vector<container>& b){
    a.swap(b);
}
///@endcond

///@addtogroup traits
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
* grid topology is also Cartesian (vectors form a box) in Nd dimensions.
* In each direction this box has a boundary layer (the halo) of a depth given by
* the user. Each boundary layer has two neighboring layers, one on the same process
* and one lying on the neighboring process.
* What this class does is to provide you with six pointers to each of these
* six layers (three on each side). The pointers either reference data in an
* internal communication buffer (since it involves communciation to get the
* layers from neighboring processes) another buffer (if mpi communication
* requires to reorder input data) or the input vector itself (if the
* communication goes along the last dimension there is no need to reorder,
* in fact, here is the main gain we get from the pointer approach, we save
* on unnecessary data copies, which might be significant in cases where
* the communication to computation ratio is high).
* The size of the data each pointer references is the halo size, \c buffer_size()
*
* The communication is done asynchronously i.e. the user can initiate
* the communication and signal when the results are needed at a later stage.
*
* @note If the number of neighboring processes in the given direction is 1,
* the buffer size is 0 and all members return immediately.
* @note the pointers may alias each other (if the input contains less than 4 layers)
*
* @note the corresponding gather map is of general type and the communication
*  can also be modeled in \c GeneralComm, but not \c BijectiveComm or \c SurjectiveComm
*  @attention Currently we cannot handle the case where the whole vector is
*  the boundary layer (i.e. \c buffer_size()==input.size() and both neighboring layers are on different processes)
* @ingroup mpi_structures
* @tparam Index the type of index container (must be either thrust::host_vector<int> or thrust::device_vector<int>)
* @tparam Buffer the container for the pointers to the buffer arrays
* @tparam Vector the vector container type must have a resize() function and work
* in the thrust library functions ( i.e. must a thrust::host_vector or thrust::device_vector)
* @sa dg::RowColDistMat
*/
template<class Index, class Buffer, class Vector>
struct NearestNeighborComm
{
    using container_type = Vector;
    using buffer_type = Buffer;
    using pointer_type = get_value_type<Vector>*;
    using const_pointer_type = get_value_type<Vector> const *;
    ///@brief no communication
    ///@param comm optional MPI communicator: the purpose is to be able to store MPI communicator even if no communication is involved in order to construct MPI_Vector with it
    NearestNeighborComm( MPI_Comm comm = MPI_COMM_NULL){
        m_n = 0;
        m_dim[0] = m_dim[1] = m_dim[2] = 0;
        m_comm = comm;
        m_silent = true;
    }
    /**
    * @brief Construct
    *
    * @tparam Nd the dimensionality of the vector and the MPI Communicator
    * @param n depth of the halo
    * @param shape local # of vector elements in each direction
    * @param comm the (cartesian) communicator (must be of dimension Nd)
    * @param direction coordinate along which to exchange halo e.g. 0 is x, 1 is y, 2 is z
    */
    template<size_t Nd>
    NearestNeighborComm( unsigned n, std::array<unsigned, Nd> shape, MPI_Comm
            comm, unsigned direction) : NearestNeighborComm( n, {shape.begin(),
                shape.end()}, comm, direction){}
    /**
    * @brief Construct
    *
    * @param n depth of the halo
    * @param shape Local number of points per dimension
    * @param comm the (cartesian) communicator, number of dimensions must match shape.size()
    * @param direction in which to exchange halo
    */
    NearestNeighborComm( unsigned n, const std::vector<unsigned>& shape,
            MPI_Comm comm, unsigned direction)
    {
        static_assert( std::is_same<const_pointer_type, get_value_type<Buffer>>::value, "Must be same pointer types");
        int ndims;
        MPI_Cartdim_get( comm, &ndims);
        assert( (unsigned)ndims == shape.size());
        std::vector<unsigned> vdims(3, 1);
        vdims[1] = shape[direction];
        for( unsigned u=0; u<direction; u++)
            vdims[0] *= shape[u];
        for( unsigned u=direction+1; u<(unsigned)ndims; u++)
            vdims[2] *= shape[u];
        construct( n, &vdims[0], comm, direction);
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
        if( src.buffer_size() == 0)  m_silent=true;
        else
            construct( src.n(), src.dims(), src.communicator(), src.direction());
    }

    /**
    * @brief halo size
    * @return  halo size
    */
    unsigned n() const{return m_n;}
    /**
    * @brief  The shape of the input vector as seen by this class
    *
    * All dimensions smaller than coord given in constructor are collapsed
    * to dims[0], dims[1] is the coord dimension in the constructor
    * and dims[2] are all dimensions larger than coord
    * @return dimensions ( 3)
    */
    const unsigned* dims() const{return m_dim;}
    /**
    * @brief The direction of communication
    *
    * @return direction
    */
    unsigned direction() const {return m_direction;}
    ///@copydoc aCommunicator::communicator()
    MPI_Comm communicator() const{return m_comm;}

    /**
     * @brief Allocate a buffer object
     *
     * The buffer object is only a colletion of pointers to the actual data
     * @return a buffer object on the stack
     * @note if \c buffer_size()==0 the default constructor of \c Buffer is called
     */
    Buffer allocate_buffer( )const{
        if( buffer_size() == 0 ) return Buffer();
        return Buffer(6);
    }
    /**  @brief The size of the halo
     * @return the size of the halo (0 if no communication)
     */
    unsigned buffer_size() const;
    ///@copydoc aCommunicator::isCommunicating()
    bool isCommunicating() const{
        if( buffer_size() == 0) return false;
        return true;
    }
    /**
     * @brief Map a local matrix index to a buffer index
     * @param i matrix index
     * @return buffer index (0,1,...,5)
     */
    int map_index(int i) const{
        if( i==-1) return 0;
        if( i== 0) return 1;
        if( i==+1) return 2;
        if( i==(int)m_outer_size-0) return 5;
        if( i==(int)m_outer_size-1) return 4;
        if( i==(int)m_outer_size-2) return 3;
        throw Error( Message(_ping_)<<"Index not mappable!");
        return -1;
    }

    /**
    * @brief Gather values from given Vector and initiate asynchronous MPI communication
    * @param input from which to gather data (it is @b unsafe to change values on return)
    * @param buffer (write only) pointers to the received data after \c global_gather_wait() was called (must be allocated by \c allocate_buffer())
    * @param rqst four request variables that can be used to call MPI_Waitall
    */
    void global_gather_init( const_pointer_type input, buffer_type& buffer, MPI_Request rqst[4])const
    {
        unsigned size = buffer_size();
        //init pointers on host
        const_pointer_type host_ptr[6];
        if(m_trivial)
        {
            host_ptr[0] = thrust::raw_pointer_cast(&m_internal_buffer[0*size]);
            host_ptr[1] = input;
            host_ptr[2] = input+size;
            host_ptr[3] = input+(m_outer_size-2)*size;
            host_ptr[4] = input+(m_outer_size-1)*size;
            host_ptr[5] = thrust::raw_pointer_cast(&m_internal_buffer[5*size]);
        }
        else
        {
            host_ptr[0] = thrust::raw_pointer_cast(&m_internal_buffer[0*size]);
            host_ptr[1] = thrust::raw_pointer_cast(&m_internal_buffer[1*size]);
            host_ptr[2] = thrust::raw_pointer_cast(&m_internal_buffer[2*size]);
            host_ptr[3] = thrust::raw_pointer_cast(&m_internal_buffer[3*size]);
            host_ptr[4] = thrust::raw_pointer_cast(&m_internal_buffer[4*size]);
            host_ptr[5] = thrust::raw_pointer_cast(&m_internal_buffer[5*size]);
        }
        //copy pointers to device
        thrust::copy( host_ptr, host_ptr+6, buffer.begin());
        //fill internal_buffer if !trivial
        do_global_gather_init( get_execution_policy<Vector>(), input, rqst);
        sendrecv( host_ptr[1], host_ptr[4],
                  thrust::raw_pointer_cast(&m_internal_buffer[0*size]), //host_ptr is const!
                  thrust::raw_pointer_cast(&m_internal_buffer[5*size]), //host_ptr is const!
                  rqst);
    }
    /**
    * @brief Wait for asynchronous communication to finish and gather received data into buffer
    *
    * Calls MPI_Waitall on the \c rqst variables and may do additional cleanup. After this call returns it is safe to use data the buffer points to.
    * @param input from which to gather data (it is safe to change values on return since values to communicate are copied into \c buffer)
    * @param buffer (write only) where received data resides on return (must be allocated by \c allocate_buffer())
    * @param rqst the same four request variables that were used in global_gather_init
    */
    void global_gather_wait(const_pointer_type input, const buffer_type& buffer, MPI_Request rqst[4])const
    {
        MPI_Waitall( 4, rqst, MPI_STATUSES_IGNORE );
#ifdef _DG_CUDA_UNAWARE_MPI
    if( std::is_same< get_execution_policy<Vector>, CudaTag>::value ) //could be serial tag
    {
        unsigned size = buffer_size();
        cudaError_t code = cudaGetLastError( );
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_buffer[0*size]), //dst
                    thrust::raw_pointer_cast(&m_internal_host_buffer[0*size]), //src
                    size*sizeof(get_value_type<Vector>), cudaMemcpyHostToDevice);
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));

        code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_buffer[5*size]), //dst
                    thrust::raw_pointer_cast(&m_internal_host_buffer[5*size]), //src
                    size*sizeof(get_value_type<Vector>), cudaMemcpyHostToDevice);
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    }
#endif
    }
    private:
    void do_global_gather_init( OmpTag, const_pointer_type, MPI_Request rqst[4])const;
    void do_global_gather_init( SerialTag, const_pointer_type, MPI_Request rqst[4])const;
    void do_global_gather_init( CudaTag, const_pointer_type, MPI_Request rqst[4])const;
    void construct( unsigned n, const unsigned vector_dimensions[3], MPI_Comm comm, unsigned direction);

    unsigned m_n, m_dim[3]; //deepness, [left,middle,right]
    MPI_Comm m_comm;
    unsigned m_direction = 0;
    bool m_silent, m_trivial=false; //silent -> no comm, m_trivial-> comm in last dim
    unsigned m_outer_size = 1; //size of vector in units of buffer_size
    Index m_gather_map_middle;
    mutable Vector m_internal_buffer;
#ifdef _DG_CUDA_UNAWARE_MPI
    //a copy of the data on the host (we need to send data manually through the host)
    mutable thrust::host_vector<get_value_type<Vector>> m_internal_host_buffer;
#endif

    void sendrecv(const_pointer_type, const_pointer_type, pointer_type, pointer_type, MPI_Request rqst[4])const;
    int m_source[2], m_dest[2];
};

///@cond

template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::construct( unsigned n, const unsigned dimensions[3], MPI_Comm comm, unsigned direction)
{
    static_assert( std::is_base_of<SharedVectorTag, get_tensor_category<V>>::value,
               "Only Shared vectors allowed");
    m_silent=false;
    m_n=n;
    m_dim[0] = dimensions[0], m_dim[1] = dimensions[1], m_dim[2] = dimensions[2];
    m_direction = direction;
    if( dimensions[2] == 1 ) m_trivial = true;
    else m_trivial = false;
    m_comm = comm;
    //mpi_cart_shift may return MPI_PROC_NULL then the receive buffer is not modified
    MPI_Cart_shift( m_comm, m_direction, -1, &m_source[0], &m_dest[0]);
    MPI_Cart_shift( m_comm, m_direction, +1, &m_source[1], &m_dest[1]);
    {
        int ndims;
        MPI_Cartdim_get( comm, &ndims);
        int dims[ndims], periods[ndims], coords[ndims];
        MPI_Cart_get( comm, ndims, dims, periods, coords);
        if( dims[m_direction] == 1) m_silent = true;
    }
    if( !m_silent)
    {
    m_outer_size = dimensions[0]*dimensions[1]*dimensions[2]/buffer_size();
    assert( m_outer_size > 1 && "Parallelization too fine grained!"); //right now we cannot have that
    thrust::host_vector<int> mid_gather( 4*buffer_size());
    for( unsigned i=0; i<m_dim[2]; i++)
        for( unsigned j=0; j<n; j++)
            for( unsigned k=0; k<m_dim[0]; k++)
            {
                mid_gather[((0*n+j)*m_dim[2]+i)*m_dim[0] + k] = (i*m_dim[1]                + j)*m_dim[0] + k;
                mid_gather[((1*n+j)*m_dim[2]+i)*m_dim[0] + k] = (i*m_dim[1] + n            + j)*m_dim[0] + k;
                mid_gather[((2*n+j)*m_dim[2]+i)*m_dim[0] + k] = (i*m_dim[1] + m_dim[1]-2*n + j)*m_dim[0] + k;
                mid_gather[((3*n+j)*m_dim[2]+i)*m_dim[0] + k] = (i*m_dim[1] + m_dim[1]-  n + j)*m_dim[0] + k;
            }
    m_gather_map_middle = mid_gather; //transfer to device
    m_internal_buffer.resize( 6*buffer_size() );
#ifdef _DG_CUDA_UNAWARE_MPI
    m_internal_host_buffer.resize( 6*buffer_size() );
#endif
    }
}

template<class I, class B, class V>
unsigned NearestNeighborComm<I,B,V>::buffer_size() const
{
    if( m_silent) return 0;
    return m_n*m_dim[0]*m_dim[2];
}

template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::do_global_gather_init( SerialTag, const_pointer_type input, MPI_Request rqst[4]) const
{
    if( !m_trivial)
    {
        unsigned size = buffer_size();
        for( unsigned i=0; i<4*size; i++)
            m_internal_buffer[i+size] = input[m_gather_map_middle[i]];
    }
}
#ifdef _OPENMP
template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::do_global_gather_init( OmpTag, const_pointer_type input, MPI_Request rqst[4]) const
{
    if(!m_trivial)
    {
        unsigned size = buffer_size();
        #pragma omp parallel for
        for( unsigned i=0; i<4*size; i++)
            m_internal_buffer[size+i] = input[m_gather_map_middle[i]];
    }
}
#endif
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::do_global_gather_init( CudaTag, const_pointer_type input, MPI_Request rqst[4]) const
{
    //gather values from input into sendbuffer
    if(!m_trivial)
    {
        unsigned size = buffer_size();
        thrust::gather( thrust::cuda::tag(), m_gather_map_middle.begin(), m_gather_map_middle.end(), input, m_internal_buffer.begin()+size);
    }
    cudaError_t code = cudaGetLastError( );
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    code = cudaDeviceSynchronize(); //wait until device functions are finished before sending data
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
}
#endif

template<class I, class B, class V>
void NearestNeighborComm<I,B,V>::sendrecv( const_pointer_type sb1_ptr, const_pointer_type sb2_ptr, pointer_type rb1_ptr, pointer_type rb2_ptr, MPI_Request rqst[4]) const
{
    unsigned size = buffer_size();
#ifdef _DG_CUDA_UNAWARE_MPI
    if( std::is_same< get_execution_policy<V>, CudaTag>::value ) //could be serial tag
    {
        cudaError_t code = cudaGetLastError( );
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_host_buffer[1*size]),//dst
            sb1_ptr, size*sizeof(get_value_type<V>), cudaMemcpyDeviceToHost); //src
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_host_buffer[4*size]),  //dst
            sb2_ptr, size*sizeof(get_value_type<V>), cudaMemcpyDeviceToHost); //src
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        sb1_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer[1*size]);
        sb2_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer[4*size]);
        rb1_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer[0*size]);
        rb2_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer[5*size]);
    }
//This is a mistake if called with a host_vector
#endif
    MPI_Isend( sb1_ptr, size,
               getMPIDataType<get_value_type<V>>(),  //sender
               m_dest[0], 3, m_comm, &rqst[0]); //destination
    MPI_Irecv( rb2_ptr, size,
               getMPIDataType<get_value_type<V>>(), //receiver
               m_source[0], 3, m_comm, &rqst[1]); //source

    MPI_Isend( sb2_ptr, size,
               getMPIDataType<get_value_type<V>>(),  //sender
               m_dest[1], 9, m_comm, &rqst[2]);  //destination
    MPI_Irecv( rb1_ptr, size,
               getMPIDataType<get_value_type<V>>(), //receiver
               m_source[1], 9, m_comm, &rqst[3]); //source
}


///@endcond
}//namespace dg
