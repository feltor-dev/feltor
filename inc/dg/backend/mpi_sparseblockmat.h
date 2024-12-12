#pragma once

#include <cassert>
#include <thrust/host_vector.h>

#include "exceptions.h"
#include "config.h"
#include "mpi_datatype.h"
#include "tensor_traits.h"
#include "memory.h"
#include "gather.h"

namespace dg
{
    //TODO Docu of Vector and @ingroup
    //Also this discussion holds for all our "MPIGather" objects

/*!@brief A hand implemented \c MPI_Ialltoallv for a contiguous \c MPI_Type_contiguous/\c MPI_Type_vector
 *
 * What we are doing here is implementing an \c MPI_Ialltoallv for
 * a contiguous chunked \c MPI_Type_contiguous or \c MPI_Type_vector
 * i.e. it should be possible to describe the data layout here with
 * clever combinations of \c MPI_Type_contiguous and \c MPI_Type_vector
 *
 * We are doing this by hand in terms of \c MPI_Isend \c MPI_Irecv because
 * - we capture cuda unaware MPI and manage the associated host memory
 * - according to OpenMPI implementation \c MPI_Ialltoallv is not implemented for cuda
 * - we use more convenient map datatype for a more convenient setup
 * - we manage the associated \c MPI_Request handles
 * - the datatypes (double/complex) to send are only available when sending
 *   (not in the constructor), thus an MPI_Type can only be commited during the
 *   send process which may be expensive
 */
template<class Vector>
struct MPIContiguousGather
{
    ///@copydoc MPIGather<Vector>::MPIGather(comm)
    MPIContiguousGather( MPI_Comm comm = MPI_COMM_NULL) : m_comm(comm){ }
    /*!@brief Construct from indices in units of \c chunk_size
     *
     * @param recvIdx recvIdx[PID] consists of the local indices on PID
     * that the calling receives from that PID (indices in units of \c chunk_size)
     * @param chunk_size Size of contiguous block of data
     * @param comm
     */
    MPIContiguousGather(
        const std::map<int, thrust::host_vector<int>>& recvIdx,
        unsigned chunk_size, MPI_Comm comm)
    : m_chunk_size( chunk_size), m_recvIdx( recvIdx), m_comm(comm)
    {
        m_sendIdx = detail::recvIdx2sendIdx ( recvIdx, comm,
                m_communicating);
        // Think about contiguous and non-contiguous separately
        // We need to find out the minimum amount of memory we need to allocate
        for( auto& idx : recvIdx)
            m_buffer_size += idx.second.size();
        // if cuda unaware we need to send messages through host
        for( auto& idx : sendIdx)
            m_store_size += idx.second.size();
        m_rqst.resize( m_buffer_size + m_store_size);
    }
    template< template<class> class OtherVector>
    friend class MPIContiguousGather; // enable copy
    template< template<typename > typename OtherVector>
    MPIContiguousGather( const MPIContiguousGather<OtherVector>& src)
    {
        m_chunk_size = src.m_chunk_size;
        m_sendIdx = src.m_sendIdx;
        m_recvIdx = src.m_recvIdx;
        m_comm = src.m_comm;
        m_communicating = src.m_communicating;
        m_buffer_size = src.m_buffer_size;
        m_store_size = src.m_store_size;
        m_rqst.resize ( m_buffer_size + m_store_size);
    }
    MPI_Comm communicator() const{return m_comm;}
    unsigned buffer_size() const { return m_buffer_size;}
    bool isCommunicating() const{
        return m_communicating;
    }

    // Also sync cuda if necessary
    ///@copydoc MPIGather<Vector>::global_gather_init
    template<class ContainerType0, class ContainerType1>
    void global_gather_init( const ContainerType0& gatherFrom, ContainerType1& buffer) const
    {
        using value_type = dg::get_value_type<ContainerType0>;
        static_assert( std::is_same_v<value_type,
                get_value_type<ContainerType1>>);
        // Receives
        unsigned recv_counter = 0;
        unsigned rqst_counter = 0;
        for( auto& idx : m_recvIdx)
        for( unsigned u=0; u<idx.second.size(); u++)
        {
            void * recv_ptr;
            if constexpr (std::is_same_v< dg::get_execution_policy<ContainerType>,
                dg::CudaTag> and not dg::cuda_aware_mpi)
            {
                m_h_buffer.template set<value_type>( m_buffer_size*m_chunk_size);
                recv_ptr = thrust::raw_pointer_cast( h_buffer.data())
                   + recv_counter*m_chunk_size;
            }
            else
                recv_ptr = thrust::raw_pointer_cast( buffer.data())
                   + recv_counter*m_chunk_size;
            MPI_Irecv( recv_ptr, m_chunk_size,
                   getMPIDataType<value_type>(),  //receiver
                   idx.first, u, m_comm, &m_rqst[rqst_counter]);  //source
            rqst_counter ++;
            recv_counter ++;
        }

        // Send
        for( auto& idx : m_sendIdx)
        for( unsigned u=0; u<idx.second.size(); u++)
        {
            void * send_ptr = thrust::raw_pointer_cast(input.data())
                       + idx.second[u]*m_chunk_size;
            if constexpr (std::is_same_v< dg::get_execution_policy<ContainerType0>,
                dg::CudaTag>)
            {
#ifdef __CUDACC__ // g++ does not know cuda code
                // cuda - sync device
                cudaError_t code = cudaGetLastError( );
                if( code != cudaSuccess)
                    throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
                if constexpr ( not dg::cuda_aware_mpi)
                {
                    m_h_store.template set<value_type>( m_store_size*m_chunk_size);
                    auto& h_store = m_h_store.template get<value_type>();
                    unsigned start = 0;
                    for( auto& idx : m_sendIdx)
                    {
                        code = cudaMemcpy( &h_store.data()[start], send_ptr,
                            size*sizeof(value_type), cudaMemcpyDeviceToHost);
                        send_ptr = // re-point pointer
                            thrust::raw_pointer_cast(&h_store.data()[start]);
                        start+= chunk_size;
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
            MPI_Isend( send_ptr, m_chunk_size,
                   getMPIDataType<value_type>(),  //sender
                   idx.first, u, m_comm, &m_rqst[rqst_counter]);  //destination
            rqst_counter ++;
        }
    }
    ///@copydoc MPIGather<Vector>::global_gather_wait
    template<class ContainerType>
    void global_gather_wait( ContainerType& buffer) const
    {
        using value_type = dg::get_value_type<ContainerType>;
        MPI_Waitall( m_rqst.size(), &m_rqst[0], MPI_STATUSES_IGNORE );
        if constexpr (std::is_same_v< dg::get_execution_policy<ContainerType>,
            dg::CudaTag> and !dg::cuda_aware_mpi)
            buffer = m_h_buffer.template get<value_type>();
    }
    private:
    unsigned m_chunk_size; // from constructor
    std::map<int,thrust::host_vector<int>> m_sendIdx;
    std::map<int,thrust::host_vector<int>> m_recvIdx; // from constructor
    MPI_Comm m_comm; // from constructor
    bool m_communicating = false;

    unsigned m_buffer_size = 0;
    mutable detail::AnyVector<thrust::host_vector>  m_h_buffer;

    unsigned m_store_size = 0;
    mutable detail::AnyVector<thrust::host_vector>  m_h_store;

    mutable std::vector<MPI_Request> m_rqst;
};

/////////////////////////////communicator//////////////////////////
//TODO update this docu
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
struct MPIKroneckerGather
{
    ///@copydoc MPIGather<Vector>::MPIGather(comm)
    MPIKroneckerGather( MPI_Comm comm = MPI_COMM_NULL) : m_comm(comm){ }
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
    MPIKroneckerGather( unsigned left_size, // (local) of SparseBlockMat
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
        m_contiguous == (left_size==1);
        // If not contiguous data we need to make it contiguous (gather into store)
        if( not m_contiguous)
        {
            // In that case only gather unique messages into send store
            detail::Unique<int> uni = detail::find_unique_order_preserving(
                detail::flatten_map( sendIdx));
            sendIdx = detail::make_map(
                detail::combine_gather( uni.gather1, uni.gather2),
                detail::get_size_map( sendIdx)); // now sendIdx goes into unique messages
            // bootstrap communication pattern back to recvIdx
            // (so that recvIdx has index into same store)
            recvIdx = detail::recvIdx2sendIdx ( sendIdx, comm_1d,
                    m_communicating);
            m_mpi_gather = dg::MPIContiguousGather( recvIdx, n*left*right);
            // everything becomes a z - derivative ...
            // we gather all unique indices contiguously into send buffer
            // uni.unique == unique local indices into gatherFrom
            m_store_size = uni.unique.size()*n*left*right;
            thrust::host_vector<int> g2( uni.unique.size()*n*left*right);
            for( unsigned l=0; l<uni.unique.size(); l++)
            for( unsigned j=0; j<n; j++)
            for( unsigned i=0; i<left_size; i++)
            for( unsigned k=0; k<right_size; k++)
                g2[((l*n+j)*left_size+i)*right_size + k] =
                     ((i*num_cols + uni.unique[l])*n + j)*right_size + k;
            m_g2 = LocalGatherMatrix<Vector>(g2);
        }
        else
            m_mpi_gather = dg::MPIContiguousGather( recvIdx, n*left*right);

    }
    template< template<class> class OtherVector>
    friend class MPIKroneckerGather; // enable copy

    template< template<typename > typename OtherVector>
    MPIKroneckerGather( const MPIKroneckerGather<OtherVector>& src)
    :   m_comm( src.m_comm),
        m_contiguous( src.m_contiguous),
        m_g2( src.m_g2),
        m_mpi_gather( src.m_mpi_gather)
    {
    }

    ///@copydoc aCommunicator::communicator()
    MPI_Comm communicator() const{return m_comm;}

    unsigned buffer_size() const { return m_mpi_gather.buffer_size();}

    ///@copydoc aCommunicator::isCommunicating()
    bool isCommunicating() const{
        return m_mpi_gather.isCommunicating();
    }

    ///@copydoc MPIGather<Vector>::global_gather_init
    template<class ContainerType0, class ContainerType1>
    void global_gather_init( const ContainerType0& gatherFrom, ContainerType1& buffer) const
    {
        using value_type = dg::get_value_type<ContainerType>;
        if( not m_contiguous)
        {
            m_store.template set<value_type>( m_g2.index_map().size());
            auto& store = m_bufer.template get<value_type>();
            m_g2.gather( gatherFrom, store);
            m_mpi_gather.global_gather_init( store, buffer);
        }
        else
            m_mpi_gather.global_gather_init( gatherFrom, buffer);
    }
    ///@copydoc MPIGather<Vector>::global_gather_wait
    template<class ContainerType>
    void global_gather_wait( ContainerType& buffer) const
    {
        m_mpi_gather.global_gather_wait( buffer);
    }
    private:
    MPI_Comm m_comm; // typically 1d
    bool m_contiguous=false;
    LocalGatherMatrix<Vector> m_g2;
    dg::MPIContiguousGather<Vector> m_mpi_gather;

    mutable detail::AnyVector<Vector>  m_store;
};
}
