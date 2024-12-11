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

namespace detail
{
// manage the case that everything is contiguous in memory
// This is a local class (no MPI calls involved)
template<class Vector>
struct MemoryManager
{
    MemoryManager(
        const std::map<int, thrust::host_vector<int>>& recvIdx,
        const std::map<int, thrust::host_vector<int>>& sendIdx,
        unsigned chunk_size)
    : m_chunk_size( chunk_size), m_sendIdx( sendIdx), m_recvIdx( recvIdx)
    {
        // Think about contiguous and non-contiguous separately
        // We need to find out the minimum amount of memory we need to allocate
        for( auto& idx : recvIdx)
        {
            // size of recvIdx w/o self-message
            if( idx.first != rank)
                m_buffer_size += idx.second.size();
        }
        // if cuda unaware we need to send messages through host
        for( auto& idx : sendIdx)
        {
            // size of sendIdx w/o self-message
            if( idx.first != rank)
                m_h_store_size += idx.second.size();
        }
    }

    // Also sync cuda if necessary
    template<class ContainerType>
    void make_sendrecv_ptrs( const ContainerType& input,
        std::map<int, thrust::host_vector<const dg::get_value_type<ContainerType>*>>& send_ptrs,
        std::map<int, thrust::host_vector<dg::get_value_type<ContainerType>*>>& recv_ptrs)
    {
        int rank;
        MPI_Comm_rank( comm, &rank);
        using value_type = dg::get_value_type<ContainerType>;
        //init pointers on host
        for( auto& idx : m_sendIdx)
        {
            send_ptrs[idx.first] = thrust::host_vector<const value_type*>>(idx.second.size());
            for( unsigned u=0; u<idx.second.size(); u++)
                send_ptrs[idx.first][u] = thrust::raw_pointer_cast(input.data())
                           + idx.second[u]*m_chunk_size;
        }

        // cuda - sync device
        if constexpr (std::is_same_v< dg::get_execution_policy<ContainerType>,
            dg::CudaTag>)
        {
#ifdef __CUDACC__ // g++ does not know cuda code
            cudaError_t code = cudaGetLastError( );
            if( code != cudaSuccess)
                throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
            if constexpr ( not dg::cuda_aware_mpi)
            {
                m_h_store.template set<value_type>( m_h_store_size*m_chunk_size);
                auto& h_store = m_h_store.template get<value_type>();
                unsigned start = 0;
                for( auto& idx : m_sendIdx)
                {
                    if( idx.first != rank) // only non-self messages need to go through host
                    {
                        for( unsigned u=0; u<idx.second.size(); u++)
                        {
                            code = cudaMemcpy( &h_store.data()[start], &send_ptrs[u],
                                size*sizeof(value_type), cudaMemcpyDeviceToHost);
                            send_ptrs[idx.first][u] = // re-point pointer
                                thrust::raw_pointer_cast(&h_store.data()[start]);
                            start+= chunk_size;
                        }
                    }
                }
                if( code != cudaSuccess)
                    throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
            }
            // We have to wait that all kernels are finished and values are
            // ready to be sent
            code = cudaDeviceSynchronize();
            if( code != cudaSuccess)
                throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
#else
            assert( false && "Something is wrong! This should never execute!");
#endif
        }

        m_buffer.template set<value_type>( m_buffer_size*m_chunk_size); // recv memory
        auto& buffer = m_bufer.template get<value_type>();
        unsigned recv_counter = 0;
        for( auto& idx : m_recvIdx)
        {
            recv_ptrs[idx.first] = thrust::host_vector<const value_type*>>(idx.second.size());
            for( unsigned u=0; u<idx.second.size(); u++)
            {
                if( idx.first != rank)
                {
                    if constexpr (std::is_same_v< dg::get_execution_policy<ContainerType>,
                        dg::CudaTag> and not dg::cuda_aware_mpi)
                    {
                        m_h_buffer.template set<value_type>( m_buffer_size*m_chunk_size);
                        recv_ptrs[idx.first][u] = thrust::raw_pointer_cast( h_buffer.data())
                           + recv_counter*m_chunk_size;
                    }
                    else
                        recv_ptrs[idx.first][u] = thrust::raw_pointer_cast( buffer.data())
                           + recv_counter*m_chunk_size;
                }
                else
                    recv_ptrs[idx.first][u] = send_ptrs[idx.first][u];
                recv_counter++;
            }
        }


    }
    void host2device_ptrs(
        std::map<int, thrust::host_vector<value_type*>>& recv_ptrs)
    {
        // in case of cuda unaware mpi some recv pointers pointers point to host
        for( auto& idx : recv_ptrs)
            if( idx.first != rank) // no self-recvs
                for( unsigned u=0; u<idx.second.size(); u++)

    }
    void sync_buffer( ) const
    {
        if constexpr (std::is_same_v< dg::get_execution_policy<ContainerType>,
            dg::CudaTag> and !dg::cuda_aware_mpi)
            buffer = m_h_buffer.template get<value_type>();
        //copy pointers to device
        thrust::device_vector<const value_type*> d_send_ptrs = send_ptrs;
    }
    private:
    unsigned m_chunk_size; // from constructor
    std::map<int,thrust::host_vector<int>> m_sendIdx; // from constructor
    std::map<int,thrust::host_vector<int>> m_recvIdx; // from constructor

    unsigned m_buffer_size = 0;
    mutable detail::AnyVector< Vector> m_buffer; // used for foreign message recv
    mutable detail::AnyVector<thrust::host_vector>  m_h_buffer; // same size as buffer

    unsigned m_h_store_size = 0;
    mutable detail::AnyVector<thrust::host_vector>  m_h_store;


};
} // namespace detail
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
        m_communicating = true;
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
        // The goal here is to minimize data movement especially when the data
        // is contiguous in memory, then the only write is to the local receive buffer
    // TODO comm is the communicator that the pids refer to
    //
    // not all participating pids need to have the same number of hyperblocks or have the same shape
    // even though left_size, n, num_cols, and right_size must be equal between any two communicating pids
    KroneckerComm( unsigned left_size, // (local) of SparseBlockMat
        const thrust::host_vector<std::array<int,2>>& gIdx,
        const thrust::host_vector<int>& bufferIdx,
        unsigned n, // block size
        unsigned num_cols, // (local) number of block columns
        unsigned right_size, // (local) of SparseBlockMat
        MPI_Comm comm_1d)
    : m_comm(comm_1d)
    {
        int ndims;
        MPI_Cartdim_get( comm_1d, &ndims);
        asssert( ndims == 1);
        int rank;
        MPI_Comm_rank( comm_1d, &rank);
        static_assert( std::is_base_of<SharedVectorTag,
                get_tensor_category<Vector<double>>>::value,
               "Only Shared vectors allowed");
        // Receive side (similar to MPI_Gather)
        auto recvIdx = detail::gIdx2unique_idx ( gIdx, bufferIdx);
        // bootstrap communication pattern
        auto sendIdx = detail::recvIdx2sendIdx ( recvIdx, comm_1d,
                m_communicating);
        unsigned rqst_size = 0;
        for( auto& idx : recvIdx)
            rqst_size += idx.second.size();
        for( auto& idx : sendIdx)
            rqst_size += idx.second.size();
        m_rqst.resize( rqst_size);
        // If not contiguous data we need to make it contiguous (gather into store)
        if( not m_contiguous)
        {
            // get unique send indices (to avoid duplicate message store)
            detail::Unique<int> uni = detail::find_unique_order_preserving(
                detail::flatten_map( sendIdx));
            m_unique_sendIdx = uni.unique;
            m_unique_storeIdx = detail::make_map(
                detail::combine_gather( uni.gather1, uni.gather2),
                detail::get_size_map( sendIdx)); // m_unique_storeIdx is sendIdx into unique messages
            m_store_size = uni.unique.size();
            // everything becomes a z - derivative ...
            // we gather all unique indices contiguously into send buffer
            thrust::host_vector<int> g2( uni.unique.size()*n*left*right);
            for( unsigned l=0; l<uni.unique.size(); l++)
            for( unsigned j=0; j<n; j++)
            for( unsigned i=0; i<left_size; i++)
            for( unsigned k=0; k<right_size; k++)
                g2[((l*n+j)*left_size+i)*right_size + k] =
                     ((i*num_cols + uni.unique[l])*n + j)*right_size + k;
            m_g2 = LocalGatherMatrix<Vector>(g2);
            m_memory = detail::MemoryManager( recvIdx, m_unique_storeIdx, n*left*right);
        }
        else
            m_memory = detail::MemoryManager( recvIdx, sendIdx, n*left*right);

    }
    template< template<class> class OtherVector>
    friend class KroneckerComm; // enable copy

    /**
    * @brief Construct from other Communicator
    *
    * Simply copies halo size, dimensions, communicator and direction and
    constructs a new object
    * @tparam OtherIndex other index type
    * @tparam OtherVector other container type
    * @param src source object
    */
    template< template<typename > typename OtherVector>
    KroneckerComm( const KroneckerComm<OtherVector>& src){
        ...
    }

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
    ///@copydoc aCommunicator::isCommunicating()
    bool isCommunicating() const{
        return m_communicating
    }

    /**
    * @brief Gather values from given Vector and initiate asynchronous MPI communication
    * @param input from which to gather data (it is @b unsafe to change values on return)
    * @param buffer (write only) pointers to the received data after \c global_gather_wait() was called (must be allocated by \c allocate_buffer())
    * @param rqst four request variables that can be used to call MPI_Waitall
    */
    template<class ContainerType>
    Vector<dg::get_value_type<ContainerType>*>
        global_gather_init( const ContainerType& input)const
    {
        using value_type = dg::get_value_type<ContainerType>;
        std::map<int, thrust::host_vector<const value_type*>> send_ptrs;
        std::map<int, thrust::host_vector<value_type*>> recv_ptrs;
        if( not m_contiguous)
        {
            m_store.template set<value_type>( m_store_size*m_chunk_size);
            auto& store = m_bufer.template get<value_type>();
            m_g2.gather( input, store);
            m_memory.make_sendrecv_ptrs( store, send_ptrs, recv_ptrs);
        }
        else
            m_memory.make_sendrecv_ptrs( input, send_ptrs, recv_ptrs);
        // now setup send
        int rqst_counter = 0;
        for( auto& idx : recv_ptrs)
        {
            if( idx.first != rank) // no self-recvs
                for( unsigned u=0; u<idx.second.size(); u++)
                {
                    MPI_Irecv( idx.second[u], m_chunk_size,
                           getMPIDataType<value_type>(),  //receiver
                           idx.first, u, comm, &m_rqst[rqst_counter]);  //source
                    rqst_counter ++;
                }
        }
        for( auto& idx : send_ptrs)
        {
            if( idx.first != rank) // no self-sends
                for( unsigned u=0; u<idx.second.size(); u++)
                {
                    MPI_Isend( idx.second[u], m_chunk_size,
                           getMPIDataType<value_type>(),  //sender
                           idx.first, u, comm, &m_rqst[rqst_counter]);  //destination
                    rqst_counter ++;
                }
        }
        Vector<value_type*> flat_ptrs = detail::flatten_map( recv_ptrs);
        return flat_ptrs;
    }
    /**
    * @brief Wait for asynchronous communication to finish and gather received data into buffer
    *
    * Calls MPI_Waitall on the \c rqst variables and may do additional cleanup. After this call returns it is safe to use data the buffer points to.
    * @param input from which to gather data (it is safe to change values on return since values to communicate are copied into \c buffer)
    * @param buffer (write only) where received data resides on return (must be allocated by \c allocate_buffer())
    * @param rqst the same four request variables that were used in global_gather_init
    */
    template<class value_type>
    Vector<const value_type*> global_gather_wait( ) const
    {
        // can be called on MPI_REQUEST_NULL
        MPI_Waitall( m_rqst.size(), &m_rqst[0], MPI_STATUSES_IGNORE );
    }
    private:
    unsigned m_slice_size;
    MPI_Comm m_comm; // typically 1d
    bool m_communicating, m_contiguous=false; //silent -> no comm, m_contiguous-> comm in last dim
    LocalGatherMatrix<Vector> m_g2;
    detail::MemoryManager<Vector> m_memory;

    unsigned m_store_size = 0;
    mutable detail::AnyVector< Vector> m_store; // used for gather operation
    mutable std::vector<MPI_Request> m_rqst;

};
}
