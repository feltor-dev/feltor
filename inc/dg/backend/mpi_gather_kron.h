#pragma once

#include <cassert>
#include <thrust/host_vector.h>

#include "exceptions.h"
#include "config.h"
#include "mpi_datatype.h"
#include "tensor_traits.h"
#include "memory.h"
#include "mpi_gather.h"

namespace dg
{
///@cond
namespace detail
{
// Gather into internal buffer and provide pointers from Contiguous Gather without self-messages
// i.e. this class is in charge of handling pointers
template<template< typename> typename Vector>
struct MPIContiguousKroneckerGather
{
    MPIContiguousKroneckerGather( MPI_Comm comm = MPI_COMM_NULL) : m_mpi_gather(comm){ }
    MPIContiguousKroneckerGather(
        const std::map<int, thrust::host_vector<int>>& recvIdx, // 1d block index in units of chunk size
        unsigned chunk_size,
        MPI_Comm comm_1d)
    : m_recvIdx(recvIdx), m_chunk_size(chunk_size)
    {
        int ndims;
        MPI_Cartdim_get( comm_1d, &ndims);
        assert( ndims == 1);
        int rank;
        MPI_Comm_rank( comm_1d, &rank);
        static_assert( dg::is_vector_v<Vector<double>, SharedVectorTag>,
                "Only Shared vectors allowed");
        auto recvChunks = detail::MPIContiguousGather::make_chunks(
            recvIdx, chunk_size);
        m_mpi_gather = dg::detail::MPIContiguousGather( recvChunks, comm_1d);
        m_buffer_size = m_mpi_gather.buffer_size(false);
    }
    template< template<class> class OtherVector>
    friend class MPIContiguousKroneckerGather; // enable copy

    template< template<typename > typename OtherVector>
    MPIContiguousKroneckerGather( const MPIContiguousKroneckerGather<OtherVector>& src)
    :   m_mpi_gather( src.m_mpi_gather),
        m_recvIdx(src.m_recvIdx),
        m_chunk_size(src.m_chunk_size),
        m_buffer_size(src.m_buffer_size)
    {
    }

    MPI_Comm communicator() const{return m_mpi_gather.communicator();}
    // Number of pointers in receive buffer is number of indices in m_recvIdx
    unsigned buffer_size() const { return detail::flatten_map( m_recvIdx).size(); }
    unsigned chunk_size() const { return m_chunk_size; }
    bool isCommunicating() const{ return m_mpi_gather.isCommunicating(); }

    template<class ContainerType>
    void global_gather_init( const ContainerType& gatherFrom) const
    {
        using value_type = dg::get_value_type<ContainerType>;
        m_buffer.template set<value_type>( m_buffer_size);
        auto& buffer = m_buffer.template get<value_type>();
        m_mpi_gather.global_gather_init( gatherFrom, buffer, false);
    }
    template<class ContainerType>
    void global_gather_wait( const ContainerType& gatherFrom,
        Vector<const dg::get_value_type<ContainerType>*>& buffer_ptrs) const
    {
        using value_type = dg::get_value_type<ContainerType>;
        auto& buffer = m_buffer.template get<value_type>();
        m_mpi_gather.global_gather_wait( buffer);

        int rank  = 0;
        MPI_Comm_rank( communicator(), &rank);
        unsigned start = 0, buffer_start = 0;
        for( auto& idx : m_recvIdx)
        for( unsigned u=0; u<idx.second.size(); u++)
        {
            if( rank != idx.first)
            {
                buffer_ptrs[start] = thrust::raw_pointer_cast(
                    buffer.data()) + buffer_start*m_chunk_size;
                buffer_start ++;
            }
            else
                buffer_ptrs[start] = thrust::raw_pointer_cast(
                    gatherFrom.data()) + idx.second[u]*m_chunk_size;
            start++;
        }

    }
    private:
    dg::detail::MPIContiguousGather m_mpi_gather;
    std::map<int,thrust::host_vector<int>> m_recvIdx;
    unsigned m_chunk_size = 0;
    unsigned m_buffer_size = 0;
    mutable detail::AnyVector<Vector> m_buffer;
};
}//namespace detail
///@endcond

/////////////////////////////communicator//////////////////////////
/**
* @brief Communicator for asynchronous communication of \c MPISparseBlockMat
*
* This is a version of \c MPIGather that is optimised for the Kronecker
* type communication pattern ("Hyperblocks") present in our \c EllSparseBlockMat
* It avoids data movement to the greatest possible extent and exposes send
* and receive buffers via pointers.
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
* @ingroup mpi_comm
* @tparam Vector a thrust Vector e.g. \c thrust::host_vector or \c thrust::device_vector
* Determines the internal buffer type of the \f$ G_2\f$ gather operation
* @sa dg::RowColDistMat
*/
template<template< typename> typename Vector>
struct MPIKroneckerGather
{
    ///@copydoc MPIGather<Vector>::MPIGather(MPI_Comm)
    MPIKroneckerGather( MPI_Comm comm = MPI_COMM_NULL) : m_mpi_gather(comm){ }
    /**
    * @brief Construct from communication pattern
    *
    * @param left_size (local) left size in \c EllSparseBlockMat (determines chunk size)
    * @param recvIdx 1d block index in units of chunk size.
    * <tt>recvIdx[PID]</tt> contains the block indices on rank PID (in \c comm_1d) that the
    * calling rank receives.  The \c global_gather_wait function returns one
    * pointer for each element of \c recvIdx pointing to a block of memory of
    * size \c chunk_size=n*left_size*right_size.
    * @param n block size of \c EllSparseBlockMat (determines chunk size)
    * @param num_cols local number of blocks columns in \c EllSparseBlockMat
    * @param right_size (local) right size of \c EllSparseBlockMat (determines chunk size)
    * @param comm_1d the one dimensional Cartesian communicator along which to exchange the hyperblocks
    * @note not all participating pids need to have the same number of hyperblocks or have the same shape
    * even though left_size, n, num_cols, and right_size must be equal between any two communicating pids
    * @sa \c dg::make_mpi_sparseblockmat
    */
    MPIKroneckerGather( unsigned left_size,
        const std::map<int, thrust::host_vector<int>>& recvIdx,
        unsigned n,
        unsigned num_cols,
        unsigned right_size,
        MPI_Comm comm_1d)
    {
        m_contiguous = (left_size==1);
        // If not contiguous data we need to make it contiguous (gather into store)
        if( not m_contiguous)
        {
            auto sendIdx = mpi_permute( recvIdx, comm_1d);
            // In that case only gather unique messages into send store
            detail::Unique<int> uni = detail::find_unique_order_preserving(
                detail::flatten_map( sendIdx));
            auto uni_sendIdx = detail::make_map_t<thrust::host_vector<int>>(
                detail::combine_gather( uni.gather1, uni.gather2),
                detail::get_size_map( sendIdx)); // now sendIdx goes into unique messages
            // bootstrap communication pattern back to recvIdx
            // (so that recvIdx has index into same store)
            auto uni_recvIdx = dg::mpi_permute ( uni_sendIdx, comm_1d);
            m_mpi_gather = detail::MPIContiguousKroneckerGather<Vector>(
                uni_recvIdx, n*left_size*right_size, comm_1d);
            // everything becomes a z - derivative ...
            // we gather all unique indices contiguously into send buffer
            // uni.unique == unique local indices into gatherFrom
            thrust::host_vector<int> g2( uni.unique.size()*n*left_size*
                right_size);
            for( unsigned l=0; l<uni.unique.size(); l++)
            for( unsigned j=0; j<n; j++)
            for( unsigned i=0; i<left_size; i++)
            for( unsigned k=0; k<right_size; k++)
                g2[((l*n+j)*left_size+i)*right_size + k] =
                     ((i*num_cols + uni.unique[l])*n + j)*right_size + k;
            m_g2 = g2;
        }
        else
            m_mpi_gather = detail::MPIContiguousKroneckerGather<Vector>(
                recvIdx, n*left_size*right_size, comm_1d);
    }

    /// Enable copy from different Vector type
    template< template<class> class OtherVector>
    friend class MPIKroneckerGather;

    ///@copydoc MPIGather::MPIGather(const MPIGather<OtherVector>&)
    template< template<typename > typename OtherVector>
    MPIKroneckerGather( const MPIKroneckerGather<OtherVector>& src)
    :   m_contiguous( src.m_contiguous),
        m_g2( src.m_g2),
        m_mpi_gather( src.m_mpi_gather)
    {
    }

    ///@copydoc MPIGather::communicator()
    MPI_Comm communicator() const{return m_mpi_gather.communicator();}
    /// @brief Number of pointers in receive buffer equals number of indices in \c recvIdx
    /// @copydetails MPIGather::buffer_size
    unsigned buffer_size() const { return m_mpi_gather.buffer_size(); }
    /// \c n*left_size*right_size
    unsigned chunk_size() const { return m_mpi_gather.chunk_size(); }
    ///@copydoc MPIGather::isCommunicating()
    bool isCommunicating() const{ return m_mpi_gather.isCommunicating(); }

    /**
     * @brief \f$ w' = P_{G,MPI} G_2 v\f$. Globally (across processes) asynchronously gather data into a buffer
     *
     * @tparam ContainerType Can be any shared vector container on host or device, e.g.
     *  - thrust::host_vector<double>
     *  - thrust::device_vector<double>
     *  - thrust::device_vector<thrust::complex<double>>
     *  .
     * @param gatherFrom source vector v; data is collected from this vector
     * @note If <tt> !isCommunicating() </tt> then this call will not involve
     * MPI communication but will still gather values according to the given
     * index map
     */
    template<class ContainerType>
    void global_gather_init( const ContainerType& gatherFrom) const
    {
        using value_type = dg::get_value_type<ContainerType>;
        if( not m_contiguous)
        {
            m_store.template set<value_type>( m_g2.size());
            auto& store = m_store.template get<value_type>();
            thrust::gather( m_g2.begin(), m_g2.end(), gatherFrom.begin(),
                store.begin());
            m_mpi_gather.global_gather_init( store);
        }
        else
            m_mpi_gather.global_gather_init( gatherFrom);
    }
    /**
    * @brief Wait for asynchronous communication to finish and gather received
    * data into buffer and return pointers to it
    *
    * Call \c MPI_Waitall on internal \c MPI_Request variables and manage host
    * memory in case of cuda-unaware MPI.  After this call returns it is safe
    * to use the buffer and the \c gatherFrom variable from the corresponding
    * \c global_gather_init call
    * @param gatherFrom source vector v; data is collected from this vector
    * @param buffer_ptrs (write only) pointers coresponding to \c recvIdx
    * from the constructor
    */
    template<class ContainerType>
    void global_gather_wait( const ContainerType& gatherFrom,
        Vector<const dg::get_value_type<ContainerType>*>& buffer_ptrs) const
    {
        if( not m_contiguous)
        {
            using value_type = dg::get_value_type<ContainerType>;
            auto& store = m_store.template get<value_type>();
            m_mpi_gather.global_gather_wait( store, buffer_ptrs);
        }
        else
            m_mpi_gather.global_gather_wait( gatherFrom, buffer_ptrs);

    }
    private:
    bool m_contiguous=false;
    Vector<int> m_g2;
    dg::detail::MPIContiguousKroneckerGather<Vector> m_mpi_gather;
    mutable detail::AnyVector<Vector>  m_store;
};
} // namespace dg
