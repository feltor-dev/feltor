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
        static_assert( std::is_base_of<SharedVectorTag,
                get_tensor_category<Vector<double>>>::value,
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
    MPIKroneckerGather( MPI_Comm comm = MPI_COMM_NULL) : m_mpi_gather(comm){ }
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
        const std::map<int, thrust::host_vector<int>>& recvIdx, // 1d block index in units of chunk size
        unsigned n, // block size
        unsigned num_cols, // (local) number of block columns (of gatherFrom)
        unsigned right_size, // (local) of SparseBlockMat
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
            m_g2 = LocalGatherMatrix<Vector>(g2);
        }
        else
            m_mpi_gather = detail::MPIContiguousKroneckerGather<Vector>(
                recvIdx, n*left_size*right_size, comm_1d);
    }
    template< template<class> class OtherVector>
    friend class MPIKroneckerGather; // enable copy

    template< template<typename > typename OtherVector>
    MPIKroneckerGather( const MPIKroneckerGather<OtherVector>& src)
    :   m_contiguous( src.m_contiguous),
        m_g2( src.m_g2),
        m_mpi_gather( src.m_mpi_gather)
    {
    }

    ///@copydoc aCommunicator::communicator()
    MPI_Comm communicator() const{return m_mpi_gather.communicator();}
    // Number of pointers in receive buffer is number of indices in m_recvIdx
    unsigned buffer_size() const { return m_mpi_gather.buffer_size(); }
    unsigned chunk_size() const { return m_mpi_gather.chunk_size(); }
    bool isCommunicating() const{ return m_mpi_gather.isCommunicating(); }

// TODO docu
    template<class ContainerType>
    void global_gather_init( const ContainerType& gatherFrom) const
    {
        using value_type = dg::get_value_type<ContainerType>;
        if( not m_contiguous)
        {
            m_store.template set<value_type>( m_g2.index_map().size());
            auto& store = m_store.template get<value_type>();
            m_g2.gather( gatherFrom, store);
            m_mpi_gather.global_gather_init( store);
        }
        else
            m_mpi_gather.global_gather_init( gatherFrom);
    }
// TODO docu
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
    LocalGatherMatrix<Vector> m_g2;
    dg::detail::MPIContiguousKroneckerGather<Vector> m_mpi_gather;
    mutable detail::AnyVector<Vector>  m_store;
};
} // namespace dg
