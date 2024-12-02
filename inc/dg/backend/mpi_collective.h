#pragma once
#include <cassert>
#include <exception>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "blas1_dispatch_shared.h"
#include "dg/blas1.h"
#include "memory.h"
#include "mpi_communicator.h"

namespace dg{

///@cond
namespace detail
{
    //TODO who handles isCommunicating?
    //TODO Handle other improvements like isSymmetric!


/**
 * @brief engine class for mpi permutating (i.e. bijective)
 *  gather and scatter operations
 *
 * We call the two vectors involved the "buffer" w  and the "store" array v.
 * The buffer array is the one that is associated with the index map
 * that we give in the constructor i.e
 * in the constructor you can specify how many values from each process
 * the buffer array receives or how many values to each process
 * the buffer array sends:
 * \f$ w = G v\f$
 *
 * In this way we gather values from the buffer array into the store array
 * while scatter inverts the operation
 * \f$ v = S w = G^T w = G^{-1} w\f$
 * @note The global size of buffer and store are equal. The local buffer
 * size is the accumulation of the \c connect values in the constructor
 * The store size is infered from the connection matrix.
 *
 * @b Scatter is the following: The first sendTo[0] elements of the buffer array
 * are sent to rank 0, The next sendTo[1] elements are sent to rank 1 and so
 * on; the first recvFrom[0] elements in the store array are the values (in
 * order) sent from rank 0, etc.
 * @b Gather: The first recvFrom[0] elements of the buffer array on the calling
 * rank are received from the "rank slot" in the store array on rank 0. etc.
 *
 * If the collaboration matrix is symmetric scatter and gather are the same
 * operation and "in-place" operation can be used, i.e. buffer and store
 * can be the same array
 */
struct MPIPermutation
{
    MPIPermutation(){
        m_comm = MPI_COMM_NULL;
    }
    /**
     * @brief Construct from an connect
     *
     * The size of connect must match the number of processes in the communicator
     * @param connect connect[PID] equals the number of points in the store array
     * that the calling process has to connect to the buffer array on rank PID
     * (can be zero, but not negative)
     * @param comm Communicator
     *
     * @note Constructor calls MPI_Allgather with given communicator
     * The idea here is that the communciation matrix can be analysed and
     * the most efficient MPI calls can be chosen based on it
     */
    MPIPermutation( const thrust::host_vector<int>& connect, MPI_Comm comm)
    {
        // interpret connect as "sendTo"
        const thrust::host_vector<int> sendTo = connect;
        thrust::host_vector<int> recvFrom(sendTo),
            accS(sendTo), accR(sendTo);
        m_comm=comm;
        int rank, size;
        MPI_Comm_rank( m_comm, &rank);
        MPI_Comm_size( m_comm, &size);
        assert( sendTo.size() == (unsigned)size);
        thrust::host_vector<unsigned> global( size*size);
        // everyone knows what everyone is sending
        MPI_Allgather( sendTo.data(), size, MPI_UNSIGNED,
                       global.data(), size, MPI_UNSIGNED,
                       m_comm);
        m_symmetric = true;

        for( unsigned i=0; i<(unsigned)size; i++)
        {
            recvFrom[i] = global[i*size+rank];
            if( recvFrom[i] != sendTo[i])
                m_symmetric = false;
            // check if global has entries off the diagonal
            for( unsigned k=0; k<(unsigned)size; k++)
                if( k != i and global[i*size+k] != 0)
                    m_communicating = true;
        }
        // exclusive_scan sums up all the elements up to (but excluding) index,
        // i.e. [0,sendTo[0], sendTo[0]+sendTo[1], ...]
        // These give the partition of the send and receive buffers in Alltoallv
        thrust::exclusive_scan( sendTo.begin(), sendTo.end(), accS.begin());
        thrust::exclusive_scan( recvFrom.begin(), recvFrom.end(), accR.begin());
        m_sendTo=sendTo, m_recvFrom=recvFrom, m_accS=accS, m_accR=accR;
        m_store_size = thrust::reduce( m_recvFrom.begin(), m_recvFrom.end() );
        m_buffer_size = thrust::reduce( m_sendTo.begin(), m_sendTo.end() );
    }

    unsigned store_size() const{
        return m_store_size;
    }
    unsigned buffer_size() const{
        return m_buffer_size;
    }

    MPI_Comm communicator() const{return m_comm;}

    bool isSymmetric() const { return m_symmetric;}

    bool isCommunicating() const{
        return m_communicating;
    }


    // synchronous gather
    template<class ContainerType0, class ContainerType1>
    void global_gather( const ContainerType0& store, ContainerType1& buffer)
    {
        MPI_Request rqst = global_gather_init( store, buffer);
        MPI_Wait( rqst, MPI_STATUS_IGNORE );
    }

    // synchronous scatter
    template<class ContainerType0, class ContainerType1>
    void global_scatter( const ContainerType0& buffer, ContainerType1& store)
    {
        MPI_Request rqst = global_scatter_init( buffer, store);
        MPI_Wait( rqst, MPI_STATUS_IGNORE );
    }

    // asynchronous scatter
    // user is responsible to catch CUDA-unawareness!
    template<class ContainerType0, class ContainerType1>
    MPI_Request global_scatter_init(
            const ContainerType0& buffer, ContainerType1& store)
    {
        using value_type = dg::get_value_type<ContainerType0>;
        static_assert( std::is_same_v<value_type,
                get_value_type<ContainerType1>);
        b_ptr = thrust::raw_pointer_cast( buffer.begin());
        s_ptr = thrust::raw_pointer_cast( store.begin());
        if constexpr ( std::is_same_v<
                dg::get_execution_policy<ContainerType1>,
                dg::CudaTag>)
        {
            // We have to wait that all kernels are finished and values are
            // ready to be sent
            dg::detail::CudaErrorHandle code = cudaGetLastError();
            code = cudaDeviceSynchronize();
        }
        assert( buffer.size() == m_buffer_size);
        assert( store.size() == m_store_size);
        return global_scatter_init( b_ptr, s_ptr);

    }
    // asynchronous gather
    template<class ContainerType0, class ContainerType1>
    MPI_Request global_gather_init(
            const ContainerType0& store, ContainerType1& buffer)
    {
        transpose();
        MPI_Request rqst = global_scatter_init( store, buffer);
        transpose();
        return rqst;
    }
    private:
    //surprisingly MPI_Alltoallv wants the integers to be on the host, only
    //the data is on the device
    unsigned m_store_size = 0;
    unsigned m_buffer_size = 0;
    thrust::host_vector<int> m_sendTo,   m_accS; //accumulated send
    thrust::host_vector<int> m_recvFrom, m_accR; //accumulated recv
    MPI_Comm m_comm = MPI_COMM_NULL;
    bool m_symmetric = false;
    bool m_communicating = false;
    // invert the scatter and gather operations (too confusing to be public...)
    // transpose is the same as invert
    void transpose()
    {
        m_sendTo.swap( m_recvFrom);
        m_accS.swap( m_accR);
        std::swap( m_store_size, m_buffer_size);
    }
    template<class value_type>
    MPI_Request global_scatter_init( const value_type* buffer,
            value_type* store) const
    {
        MPI_Request rqst;
        if ( buffer == store) // same address
        {
            assert( m_symmetric);
            MPI_IAlltoallv(
                MPI_IN_PLACE,
                thrust::raw_pointer_cast( m_sendTo.data()),
                thrust::raw_pointer_cast( m_accS.data()),
                getMPIDataType<value_type>(),
                store,
                thrust::raw_pointer_cast( m_recvFrom.data()),
                thrust::raw_pointer_cast( m_accR.data()),
                getMPIDataType<value_type>(),
                m_comm, &rqst);
        }
        else
        {
            MPI_IAlltoallv(
                buffer,
                thrust::raw_pointer_cast( m_sendTo.data()),
                thrust::raw_pointer_cast( m_accS.data()),
                getMPIDataType<value_type>(),
                store,
                thrust::raw_pointer_cast( m_recvFrom.data()),
                thrust::raw_pointer_cast( m_accR.data()),
                getMPIDataType<value_type>(),
                m_comm, &rqst);
         }
        return rqst;
    }


};

template<class T>
void find_same(
        const thrust::host_vector<T>& numbers, // numbers must be sorted
        thrust::host_vector<T>& unique_numbers,
        thrust::host_vector<int>& howmany_numbers // same size as uniqe_numbers
        )
{
    // Find unique numbers and how often they appear
    thrust::host_vector<T> unique_ids( numbers.size());
    thrust::host_vector<int> ones( numbers.size(), 1),
        howmany(ones);
    auto new_end = thrust::reduce_by_key( numbers.begin(), numbers.end(),
            //pids are the keys by which one is reduced
            ones.begin(), unique_ids.begin(), howmany.begin() );
    unique_numbers = thrust::host_vector<T>( unique_ids.begin(),
            new_end.first);
    howmany_numbers = thrust::host_vector<int>( howmany.begin(), new_end.second);
}

//given global indices -> make a sorted unique indices vector + a gather map
//into the unique vector:
//@param buffer_idx -> (gather map/ new column indices) same size as global_idx
//( can alias global_idx, index into unique_global_idx
//@param unique_global_idx -> (list of unique global indices to be used in a
//Collective Communication object, sorted)
static inline void global2bufferIdx(
    // 1st is pid, 2nd is local index
    const thrust::host_vector<std::array<int,2>>& global_idx,
    thrust::host_vector<int>& buffer_idx,
    thrust::host_vector<std::array<int,2>>& locally_unique_global_idx)
{
// sort_map is the gather map wrt to the sorted vector!
// To duplicate the sort:
// thrust::gather( sort_map.begin(), sort_map.end(), numbers.begin(), sorted.begin());
// To undo the sort:
// thrust::scatter( sorted.begin(), sorted.end(), sort_map.begin(), numbers.begin());
// To get the gather map wrt to unsorted vector (i.e the inverse index map)
// "Scatter the index"
// auto gather_map = sort_map;
// auto seq = sort_map;
// thrust::sequence( seq.begin(), seq.end());
// thrust::scatter( seq.begin(), seq.end(), sort_map.begin(), gather_map.begin());
// Now gather_map indicates where each of the numbers went in the sorted vector
    // 1. Sort pids with indices so we get associated gather map
    thrust::host_vector<int> sort_map, howmany;
    auto ids = global_idx;
    sort_map.resize( ids.size());
    thrust::sequence( sort_map.begin(), sort_map.end()); // 0,1,2,3,...
    thrust::stable_sort_by_key( ids.begin(), ids.end(),
            sort_map.begin()); // this changes both ids and sort_map

    find_same( ids, locally_unique_global_idx,
            howmany);

    // manually make gather map from sorted into locally_unique_global_idx
    thrust::host_vector<int> gather_map;
    for( unsigned i=0; i<locally_unique_global_idx.size(); i++)
        for( int j=0; j<howmany[i]; j++)
            gather_map.push_back(i);
    assert( gather_map.size() == global_idx.size());
    // buffer idx is the new index
    buffer_idx.resize( global_idx.size());
    thrust::scatter( gather_map.begin(), gather_map.end(), sort_map.begin(),
            buffer_idx.begin());
}


//create sendTo map for MPIPermutation
static inline thrust::host_vector<int> lugi2sendTo(
    thrust::host_vector<std::array<int,2>>& locally_unique_global_idx,
    MPI_Comm comm)
{
    int comm_size;
    MPI_Comm_size( comm, &comm_size);
    // get pids
    thrust::host_vector<int> pids(locally_unique_global_idx.size()),
        lIdx(pids);
    for( int i=0; i<pids.size(); i++)
    {
        pids[i] = locally_unique_global_idx[i][0];
        lIdx[i] = locally_unique_global_idx[i][1];
        // Sanity check
        assert( 0 <= pids[i] && pids[i] < comm_size);
    }
    thrust::host_vector<int> locally_unique_pids, howmany;
    detail::find_same( pids, locally_unique_pids, howmany);
    thrust::host_vector<int> sendTo( comm_size, 0 );
    for( unsigned i=0; i<locally_unique_pids.size(); i++)
        sendTo[locally_unique_pids[i]] = howmany[i];
    return sendTo;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace detail
///@endcond

/**
 * @ingroup mpi_structures
 * @brief Perform surjective gather and its transpose (scatter) operation across processes
 * on distributed vectors using mpi
 *
 * This Communicator performs surjective global gather and
 scatter operations, which means that the index map
 is @b surjective: If the index map idx[i] is surjective, each element of the source vector v
maps to at least one location in the buffer vector w. This means that the scatter matrix S
can have more than one 1's in each line. (see \c aCommunicator for more details)
 Compared to \c BijectiveComm in the \c global_gather function there is an additional
 gather and in the \c global_scatter_reduce function a reduction
 needs to be performed.
 * @tparam Index an integer thrust Vector (needs to be \c int due to MPI interface)
 * @tparam Vector a thrust Vector
 */
template< template <class> class Vector>
struct MPIGather
{
    ///@copydoc GeneralComm::GeneralComm()
    MPIGather(){
        m_buffer_size = m_store_size = 0;
    }
    /**
    * @brief Construct from local indices and PIDs index map
    *
    * The indices in the index map are written with respect to the buffer
    * vector.  Each location in the source vector is uniquely specified by a
    * local vector index and the process rank.
    * @param local_size local size of a \c dg::MPI_Vector (same for all
    * processes)
    * @param localIndexMap Each element <tt>localIndexMap[i]</tt> represents a
    * local vector index from (or to) where to take the value
    * <tt>buffer[i]</tt>. There are <tt>local_buffer_size =
    * localIndexMap.size()</tt> elements.
    * @param pidIndexMap Each element <tt>pidIndexMap[i]</tt> represents the
    * pid/rank to which the corresponding index <tt>localIndexMap[i]</tt> is
    * local.  Same size as \c localIndexMap.  The pid/rank needs to be element
    * of the given communicator.
    * @param comm The MPI communicator participating in the scatter/gather
    * operations
    */
    MPIGather( unsigned local_size,
            const thrust::host_vector<std::array<int,2>& gIdx,
            // TODO gIdx can be unsorted and contain duplicate entries
            // TODO idx 0 is pid, idx 1 is localIndex on that pid
            MPI_Comm comm)
    {
        thrust::host_vector<int> bufferIdx, locally_unique_gIdx;
        detail::global2bufferIdx(gIdx, bufferIdx, locally_unique_gIdx);
        auto sendTo = detail::lugi2sendTo( locally_unique_gIdx, comm);
        //Now construct the MPIPermutation object by getting the number of
        //elements to send
        m_permute = detail::MPIPermutation( sendTo, comm);
        assert( m_permute.buffer_size() == bufferIdx.size());

        m_g1 = bufferIdx; //

        // Finally, construct G2
        thrust::host_vector<int> storeIdx( m_permute.store_size());
        // Scatter local indices to the other processes so they may know
        // which values they need to provide:
        m_permute.global_scatter( lIdx, storeIdx);
        // All the indices we receive here are local to the current rank!
        m_g2 = storeIdx;
    }

    /**
     * @brief Construct from global indices index map
     *
     * Uses the \c global2localIdx() member of MPITopology to generate \c
     * localIndexMap and \c pidIndexMap
     * @param globalIndexMap Each element <tt> globalIndexMap[i] </tt>
     * represents a global vector index from (or to) where to take the value
     * <tt>buffer[i]</tt>. There are <tt> local_buffer_size =
     * globalIndexMap.size() </tt> elements.
     * @param p the conversion object
     * @tparam ConversionPolicy has to have the members:
     *  - <tt> bool global2localIdx(unsigned,unsigned&,unsigned&) const;</tt>
     *  where the first parameter is the global index and the other two are the
     *  output pair (localIdx, rank). return true if successful, false if
     *  global index is not part of the grid
     *  - <tt> MPI_Comm %communicator() const;</tt>  returns the communicator
     *  to use in the gather/scatter
     *  - <tt> local_size(); </tt> return the local vector size
     * @sa basictopology the MPI %grids defined in Level 3 can all be used as a
     * ConversionPolicy
     */
    template<class ConversionPolicy>
    MPIGather( const thrust::host_vector<int>& globalIndexMap,
            const ConversionPolicy& p)
    {
        // TODO update docu on local_size() ( if we don't scatter we don't need it)
        thrust::host_vector<std::array<int,2>> gIdx( globalIndexMap.size());
        bool success = true;
        for(unsigned i=0; i<gIdx.size(); i++)
            if( !p.global2localIdx(globalIndexMap[i],
                        gIdx[i][1], gIdx[i][0]) )
                success = false;

        assert( success);
        *this = MPIGather( gIdx, p.communicator());
    }

    template< template<typename > typename OtherVector>
    MPIGatherMatrix( const MPIGatherMatrix<OtherVector>& src)
    {
        *this = MPIGatherMatrix( src.local_size(), src.getLocalIndexMap(),
                src.communicator());
    }

    const thrust::host_vector<std::array<int,2>>& getLocalIndexMap() const
    {
        return m_gIdx;
    }

    MPI_Comm communicator()const
    {
        return m_permute.communicator();
    }
    unsigned buffer_size() const { m_permute.buffer_size();}

    // gather map from buffer to gIdx given in constructor
    const thrust::host_vector<int>& get_buffer_idx() const
    {
        return m_g1;
    }
    bool isCommunicating() const
    {
        return m_permute.isCommunicating();
    }
    template<class ContainerType>
    void global_gather_init( const ContainerType& vector)const
    {
        using value_type = dg::get_value_type<ContainerType>;
        m_store.set<value_type>( m_permute.store_size());
        m_buffer.set<value_type>( m_permute.buffer_size());

        //gather values to store
        m_g2.gather( vector, m_store.get<value_type>());
#ifdef _DG_CUDA_UNAWARE_MPI // we need to send through host
        if constexpr ( std::is_same_v<
                dg::get_execution_policy<ContainerType>,
                dg::CudaTag>)
        {
            m_h_buffer.set<value_type>( m_permute.buffer_size());
            m_h_store.set<value_type>( m_permute.store_size());
            // cudaMemcpy
            m_h_store.get<value_type>() = m_store.get<value_type>();
        }
        m_rqst.data() = m_permute.global_gather_init(
                m_h_store.get<value_type>(), m_h_buffer.get<value_type>());
#else
        m_rqst.data() = m_permute.global_gather_init(
                m_store.get<value_type>(), m_buffer.get<value_type>());
#endif// _DG_CUDA_UNAWARE_MPI
    }
    template<class value_type>
    const Vector<value_type>& get_buffer() const
    {
        // can be called on MPI_REQUEST_NULL
        MPI_Wait( rqst, MPI_STATUS_IGNORE );
#ifdef _DG_CUDA_UNAWARE_MPI // we need to copy from host
        if constexpr ( std::is_same_v<
                dg::get_execution_policy<Vector<value_type>>,
                dg::CudaTag>)
            return m_h_buffer.get<value_type>();
#endif// _DG_CUDA_UNAWARE_MPI
        return m_buffer.get<value_type>();
    }

    private:
    // stored to be able to copy construct
    thrust::host_vector<std::array<int,2>> m_gIdx;
    Vector<int> m_g1, m_g2;
    detail::AnyVector< Vector> m_buffer, m_store;
    dg::detail::MPIPermutation m_permute;
#ifdef _DG_CUDA_UNAWARE_MPI // nothing serious is cuda unaware ...
    detail::AnyVector<thrust::host_vector>  m_h_buffer, m_h_store;
#endif// _DG_CUDA_UNAWARE_MPI
};

}//namespace dg
