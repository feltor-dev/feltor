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

namespace dg
{
/////////////////////////////communicator//////////////////////////
/**
* @brief Communicator for asynchronous communication of MPISparseBlockMat
*
* Imagine a communicator with Cartesian topology and further imagine that the
* grid topology is also Cartesian (vectors form a box) in Nd dimensions.  A
* Sparseblockmat typically requires to gather slices of given index from other
* processes in a 1d communicator.  This class provides pointers to these other
* indices.  The pointers either reference data in an
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
*  can also be modeled in \c MPIGather
* @ingroup mpi_structures
* @tparam Index the type of index container (must be either thrust::host_vector<int> or thrust::device_vector<int>)
* @tparam Buffer the container for the pointers to the buffer arrays
* @tparam Vector the vector container type must have a resize() function and work
* in the thrust library functions ( i.e. must a thrust::host_vector or thrust::device_vector)
* @sa dg::RowColDistMat
*/
template<template< typename> typename Vector>
struct KroneckerComm
{
    ///@brief no communication
    ///@param comm optional MPI communicator: the purpose is to be able to store MPI communicator even if no communication is involved in order to construct MPI_Vector with it
    KroneckerComm( MPI_Comm comm = MPI_COMM_NULL){
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
    // TODO comm is the communicator that the pids refer to
    //
    // hyperblocks contains 1d indices {start,end,pid}
    // collapsed shape contains {left_vector_size, direction_vector_size, right_vector_size} in the Sparseblockmat sense
    // not all participating pids need to have the same number of hyperblocks or have the same shape
    // even though shape[0] and shape[2] must be equal between any two communicating pids
    // Index map must be injective i.e. vector element cannot be sent to more than one other rank
    KroneckerComm( unsigned left,
    const thrust::host_vector<std::array<int,2>>& gIdx,
    const thrust::host_vector<int>& bufferIdx, unsigned n, unsigned right,
        MPI_Comm comm_1d)
    {
        auto recvIdx = detail::gIdx2recvIdx( gIdx, bufferIdx, comm_1d);
        auto sendIdx = detail::recvIdx2sendIdx ( recvIdx, comm_1d, m_communicating);
        // now get unique send indices
        std::map<int, thrust::host_vector<int>> send_bufferIdx;
        auto gsendIdx = sendIdx2gIdx( sendIdx, send_bufferIdx);

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
    template< template<typename > typename OtherVector, class value_type2>
    KroneckerComm( const KroneckerComm<OtherVector,value_type2>& src){
        // ...
        //if( src.buffer_size() == 0)  m_silent=true;
        //else
        //    construct( src.n(), src.dims(), src.communicator(), src.direction());
    }

    /**
    * @brief slice width
    * @return  slice width
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
        // This maps the constructors indices to a local buffer
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
            host_ptr[0] = thrust::raw_pointer_cast(&m_internal_buffer.data()[0*size]);
            host_ptr[1] = input;
            host_ptr[2] = input+size;
            host_ptr[3] = input+(m_outer_size-2)*size;
            host_ptr[4] = input+(m_outer_size-1)*size;
            host_ptr[5] = thrust::raw_pointer_cast(&m_internal_buffer.data()[5*size]);
        }
        else
        {
            host_ptr[0] = thrust::raw_pointer_cast(&m_internal_buffer.data()[0*size]);
            host_ptr[1] = thrust::raw_pointer_cast(&m_internal_buffer.data()[1*size]);
            host_ptr[2] = thrust::raw_pointer_cast(&m_internal_buffer.data()[2*size]);
            host_ptr[3] = thrust::raw_pointer_cast(&m_internal_buffer.data()[3*size]);
            host_ptr[4] = thrust::raw_pointer_cast(&m_internal_buffer.data()[4*size]);
            host_ptr[5] = thrust::raw_pointer_cast(&m_internal_buffer.data()[5*size]);
        }
        //copy pointers to device
        thrust::copy( host_ptr, host_ptr+6, buffer.begin());
        //fill internal_buffer if !trivial
        do_global_gather_init( get_execution_policy<Vector>(), input, rqst);
        sendrecv( host_ptr[1], host_ptr[4],
                  thrust::raw_pointer_cast(&m_internal_buffer.data()[0*size]), //host_ptr is const!
                  thrust::raw_pointer_cast(&m_internal_buffer.data()[5*size]), //host_ptr is const!
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
        code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_buffer.data()[0*size]), //dst
                    thrust::raw_pointer_cast(&m_internal_host_buffer.data()[0*size]), //src
                    size*sizeof(get_value_type<Vector>), cudaMemcpyHostToDevice);
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));

        code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_buffer.data()[5*size]), //dst
                    thrust::raw_pointer_cast(&m_internal_host_buffer.data()[5*size]), //src
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

    unsigned m_local_left_size, m_local_right_size;
    MPI_Comm m_comm; // typically 1d
    bool m_silent, m_trivial=false; //silent -> no comm, m_trivial-> comm in last dim
    unsigned m_outer_size = 1; //size of vector in units of buffer_size
    Vector<int> m_gather_map_middle;
    dg::Buffer<Vector<value_type>> m_internal_buffer;
#ifdef _DG_CUDA_UNAWARE_MPI
    //a copy of the data on the host (we need to send data manually through the host)
    dg::Buffer<thrust::host_vector<value_type>> m_internal_host_buffer;
#endif

    void sendrecv(const_pointer_type, const_pointer_type, pointer_type, pointer_type, MPI_Request rqst[4])const;
};

///@cond

template<template<class> class V, class value_type>
void KroneckerComm<V, value_type>::KroneckerComm(
    std::vector<std::array<int,3>> bbs, std::array<unsigned,3> collapsed_shape, MPI_Comm comm)
{
    static_assert( std::is_base_of<SharedVectorTag, get_tensor_category<V<value_type>>>::value,
               "Only Shared vectors allowed");
    // First convert bbs to a flat array
    std::vector<int> starts ( bbs.size()), ends(bbs.size()), pids( bbs.size());
    for( unsigned u=0;u<bbs.size(); u++)
        starts[u] = bbs[u][0], ends[u] = bbs[u][1], pids[u] = bbs[u][2];
    thrust::vector<int> localIndexMap






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
    m_internal_buffer.data().resize( 6*buffer_size() );
#ifdef _DG_CUDA_UNAWARE_MPI
    m_internal_host_buffer.data().resize( 6*buffer_size() );
#endif
    }
}

template<class I, class B, class V>
unsigned KroneckerComm<I,B,V>::buffer_size() const
{
    if( m_silent) return 0;
    return m_n*m_dim[0]*m_dim[2];
}

template<class I, class B, class V>
void KroneckerComm<I,B,V>::do_global_gather_init( SerialTag, const_pointer_type input, MPI_Request rqst[4]) const
{
    if( !m_trivial)
    {
        unsigned size = buffer_size();
        for( unsigned i=0; i<4*size; i++)
            m_internal_buffer.data()[i+size] = input[m_gather_map_middle[i]];
    }
}
#ifdef _OPENMP
template<class I, class B, class V>
void KroneckerComm<I,B,V>::do_global_gather_init( OmpTag, const_pointer_type input, MPI_Request rqst[4]) const
{
    if(!m_trivial)
    {
        unsigned size = buffer_size();
        #pragma omp parallel for
        for( unsigned i=0; i<4*size; i++)
            m_internal_buffer.data()[size+i] = input[m_gather_map_middle[i]];
    }
}
#endif
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
template<class I, class B, class V>
void KroneckerComm<I,B,V>::do_global_gather_init( CudaTag, const_pointer_type input, MPI_Request rqst[4]) const
{
    //gather values from input into sendbuffer
    if(!m_trivial)
    {
        unsigned size = buffer_size();
        thrust::gather( thrust::cuda::tag(), m_gather_map_middle.begin(), m_gather_map_middle.end(), input, m_internal_buffer.data().begin()+size);
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
void KroneckerComm<I,B,V>::sendrecv( const_pointer_type sb1_ptr, const_pointer_type sb2_ptr, pointer_type rb1_ptr, pointer_type rb2_ptr, MPI_Request rqst[4]) const
{
    unsigned size = buffer_size();
#ifdef _DG_CUDA_UNAWARE_MPI
    if( std::is_same< get_execution_policy<V>, CudaTag>::value ) //could be serial tag
    {
        cudaError_t code = cudaGetLastError( );
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_host_buffer.data()[1*size]),//dst
            sb1_ptr, size*sizeof(get_value_type<V>), cudaMemcpyDeviceToHost); //src
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_host_buffer.data()[4*size]),  //dst
            sb2_ptr, size*sizeof(get_value_type<V>), cudaMemcpyDeviceToHost); //src
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        sb1_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer.data()[1*size]);
        sb2_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer.data()[4*size]);
        rb1_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer.data()[0*size]);
        rb2_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer.data()[5*size]);
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
}
