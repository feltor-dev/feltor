#pragma once
#include <cassert>
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

/**
 * @brief engine class for mpi permutating (i.e. bijective)
 *  gather and scatter operations
 *
 * The easiest way to think about it is that we define a matrix G
 * in the constructor and implement its application
 * as well as the application of its tranpose (scatter).
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
template<class value_type>
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
     */
    MPIPermutation( const thrust::host_vector<int>& connect, MPI_Comm comm)
    {
        //sollte schnell sein weil klein
        //
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
    }

    MPI_Comm comm() const {return m_comm;}

    MPI_Request global_scatter_init( CudaTag,      const value_type* buffer,
            value_type* store) const;
    MPI_Request global_scatter_init( AnyPolicyTag, const value_type* buffer,
            value_type* store) const;

    MPI_Request global_gather_init( CudaTag, const value_type* store,
            value_type* buffer) const
    {
        transpose();
        global_scatter_init( CudaTag, store, buffer);
        transpose();
    }
    MPI_Request global_gather_init( AnyPolicyTag, const value_type* store,
            value_type* buffer) const
    {
        transpose();
        global_scatter_init( AnyPolicyTag, store, buffer);
        transpose();
    }

    void global_wait( CudaTag,      const value_type* in, value_type* out,
            MPI_Request) const;
    void global_wait( AnyPolicyTag, const value_type* in, value_type* out,
            MPI_Request) const;

    template<class ContainerType0, class ContainerType1>
    MPI_Request global_scatter_init(
            const ContainerType0& buffer, ContainerType0& store)
    {
        // TODO static_assert value_types
        // assert( buffer.size() == buffer_size());
        // assert( store.size() == store_size());
        return rqst = global_scatter_init(dg::get_execution_policy<ContainerType0>(),
                thrust::raw_pointer_cast( buffer.begin())
                thrust::raw_pointer_cast( store.begin()));
    }
    template<class ContainerType0, class ContainerType1>
    MPI_Request global_gather_init(
            const ContainerType0& store, ContainerType0& buffer)
    {
        // TODO static_assert value_types
        // assert( buffer.size() == buffer_size());
        // assert( store.size() == store_size());
        return rqst = global_gather_init(dg::get_execution_policy<ContainerType0>(),
                thrust::raw_pointer_cast( store.begin())
                thrust::raw_pointer_cast( buffer.begin()));
    }
    template<class ContainerType0, class ContainerType1>
    MPI_Request global_wait(
            const ContainerType0& in, ContainerType0& out, MPI_Request rqst)
    {
        global_wait(dg::get_execution_policy<ContainerType0>(),
                thrust::raw_pointer_cast( in.begin())
                thrust::raw_pointer_cast( out.begin()), rqst);
    }

    template<class ContainerType0, class ContainerType1>
    void global_scatter( const ContainerType0& buffer, ContainerType0& store)
    {
        global_scatter_init( buffer, store);
        global_wait( buffer, store);
    }
    template<class ContainerType0, class ContainerType1>
    void global_gather( const ContainerType0& store, ContainerType0& buffer)
    {
        global_gather_init( store, buffer);
        global_wait( store, buffer);
    }

    bool isSymmetric() const { return m_symmetric;}
    unsigned store_size() const{
        if( m_recvFrom.empty())
            return 0;
        return thrust::reduce( m_recvFrom.begin(), m_recvFrom.end() );
    }
    unsigned buffer_size() const{
        if( m_sendTo.empty()) // this only happens after default construction
            return 0;
        return thrust::reduce( m_sendTo.begin(), m_sendTo.end() );
    }
    MPI_Comm communicator() const{return m_comm;}
    bool isCommunicating() const{
        return m_communicating;
    }
    private:
    //surprisingly MPI_Alltoallv wants the integers to be on the host, only
    //the data is on the device
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
    }
#ifdef _DG_CUDA_UNAWARE_MPI // nothing serious is cuda unaware ...
    dg::Buffer<thrust::host_vector<value_type>>  m_host_buffer;
#endif// _DG_CUDA_UNAWARE_MPI
};

#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
template<class value_type>
MPI_Request Collective::global_scatter_init( CudaTag, const value_type* buffer, value_type* store) const
{
    // We have to wait that all kernels are finished and values are ready to be sent
    cudaError_t code = cudaGetLastError( );
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    code = cudaDeviceSynchronize(); //needs to be called
    if( code != cudaSuccess)
        throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));

#ifdef _DG_CUDA_UNAWARE_MPI // nothing serious is cuda unaware ...
    thrust::device_vector<value_type> d_buffer( buffer, buffer+buffer_size());
    thrust::device_vector<value_type> h_buffer( d_buffer);
    m_host_buffer.data().resize( store_size());
    unsigned buffer_size = buffer_size();
    global_scatter_init( SerialTag, thrust::raw_pointer_cast( h_buffer.begin()),
        thrust::raw_pointer_cast( m_host_buffer.data().begin()));
#else
    global_scatter_init( SerialTag, buffer, store);
#endif// _DG_CUDA_UNAWARE_MPI

}
#endif //THRUST_DEVICE_SYSTEM
template<class value_type>
MPI_Request Collective::global_scatter_init( AnyPolicyTag,
            const value_type* buffer, value_type* store) const
{
    MPI_IAlltoallv(
            buffer,
            thrust::raw_pointer_cast( m_sendTo.data()),
            thrust::raw_pointer_cast( m_accS.data()), getMPIDataType<value_type>(),
            store,
            thrust::raw_pointer_cast( m_recvFrom.data()),
            thrust::raw_pointer_cast( m_accR.data()), getMPIDataType<value_type>(),
            m_comm);
}
template<class value_type>
void global_wait( AnyPolicyTag, const value_type* in,
        value_type* out, MPI_Request rqst)const
{
    MPI_Wait( rqst, MPI_STATUS_IGNORE );
}
template<class value_type>
void global_wait( CudaTag, const value_type* in,
        value_type* out, MPI_Request rqst)const
{
    MPI_Wait( rqst, MPI_STATUS_IGNORE );
#ifdef _DG_CUDA_UNAWARE_MPI
    thrust::device_vector<value_type> d_out = m_host_buffer.data();
    thrust::copy( d_out.begin(), d_out.end(), out);
#endif
}

//given global indices -> make a sorted unique indices vector + a gather map
//into the unique vector:
//@param buffer_idx -> (gather map/ new column indices) same size as global_idx
//( can alias global_idx, index into unique_global_idx
//@param unique_global_idx -> (list of unique global indices to be used in a
//Collective Communication object)
static void global2bufferIdx(
    const thrust::host_vector<int>& global_idx,
    thrust::host_vector<int>& buffer_idx,
    thrust::host_vector<int>& locally_unique_global_idx)
{
    // 1. Sort global_idx with indices so we get associated gather map
    thrust::host_vector<int> index(global_idx);
    thrust::sequence( index.begin(), index.end());
    thrust::stable_sort_by_key( global_idx.begin(), global_idx.end(),
            index.begin());//this changes both global_idx and index
    //2. now find unique indices and how often they appear
    thrust::host_vector<int> ones( index.size(), 1),
                             unique_global( one), howmany( one);
    typedef typename thrust::host_vector<int>::iterator iterator;
    thrust::pair<iterator, iterator> new_end;
    new_end = thrust::reduce_by_key( m_global_idx.begin(), m_global_idx.end(),
            ones.begin(), unique_global.begin(), howmany.begin());
    //3. copy unique indices
    locally_unique_global_idx.assign( unique_global.begin(), new_end.first);
    //4. manually make gather map into locally_unique_global_idx
    thrust::host_vector<int> gather_map;
    for( unsigned i=0; i<locally_unique_global_idx.size(); i++)
        for( int j=0; j<howmany[i]; j++)
            gather_map.push_back(i);
    assert( gather_map.size() == global_idx.size());
    //5. buffer idx is the new index
    buffer_idx.resize( global_idx.size());
    thrust::scatter( gather_map.begin(), gather_map.end(), index.begin(),
            buffer_idx.begin());
}
thrust::host_vector<int> find_numbers_of_pids( thrust::host_vector<int> pids,
        unsigned comm_size)
{
    // find unique pids and how often they appear
    thrust::host_vector<int> ones( pids.size(), 1), unique_ids(one),
        howmany(one);
    typedef thrust::host_vector<int>::iterator iterator;
    thrust::pair< iterator, iterator> new_end =
        thrust::reduce_by_key( pids.begin(), pids.end(),
            //pids are the keys by which one is reduced
            ones.begin(), unique_ids.begin(), howmany.begin() );
    thrust::host_vector<int> locally_unique_pids ( unique_ids.begin(),
            new_end.first);
    thrust::host_vector<int> sendTo( comm_size, 0 );
    for( unsigned i=0; i<locally_unique_pids.size(); i++)
        sendTo[locally_unique_pids[i]] = howmany[i]; //scatter
    return sendTo;
}
} // namespace detail


// Given an index map construct the associate gather matrix and its transpose
// Represents the matrix \f$ G_{ij} = \delta_{g[i]j}\f$ and its transpose
template<class ContainerType>
struct LocalGatherMatrix
{
    using value_type = dg::get_value_type<ContainerType>;
    using matrix_type = std::conditional_t< std::is_same_v<
        dg::get_execution_policy<ContainerType>, dg::SerialTag>,
         cusp::coo_matrix<int, int, cusp::host_memory>,
         cusp::coo_matrix<int, int, cusp::device_memory>>;

    // index_map gives num_rows of G, so we need num_cols as well
    LocalGatherMatrix( const thrust::host_vector<int>& index_map, unsigned num_cols) :
        m_idx( index_map), m_num_cols( num_cols){}

    const ContainerType& index_map() const{ return m_idx;}

    // w = Gv
    void gather( const ContainerType& store, ContainerType& buffer) const
    {
        assert( buffer.size() == m_num_cols);
        thrust::gather( m_idx.begin(), m_idx.end(), store.begin(), buffer.begin());
    }
    // v = v + S w
    void scatter_plus( const ContainerType& buffer, ContainerType& store)
    {
        if( not m_allocated)
        {
            cusp::coo_matrix<int, value_type, cusp::host_memory> A(
              m_idx.size(), m_num_cols, m_idx.size());

            thrust::host_vector<int> h_idx = m_idx;
            dg::blas1::copy( h_idx, A.row_indices);
            dg::blas1::copy( 1, A.column_indices);
            dg::blas1::copy( 1.0, A.values);
            A.sort_by_row_and_column();
            m_scatter = A;
            m_allocated = true;
        }
        cusp::array1d_view< typename ContainerType::const_iterator>
            cx( buffer.cbegin(), buffer.cend());
        cusp::array1d_view< typename ContainerType::iterator>
            cy( store.begin(), store.end());
        thrust::multiplies<value_type>  combine;
        thrust::plus<value_type>        reduce;
        cusp::generalized_spmv( m_scatter, cx, cy, cy, combine, reduce);
    }
    private:
    ContainerType m_idx;
    unsigned m_num_cols;
    bool m_allocated = false;
    matrix_type m_scatter; // only construct if needed
};

///@endcond

// TODO Merge with unique global indices conversion

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
template< template <class> class Vector, class value_type>
struct MPIGatherScatter : public aCommunicator<Vector<value_type>>
{
    ///@copydoc GeneralComm::GeneralComm()
    MPIGatherScatter(){
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
    MPIGatherScatter( unsigned local_size, const thrust::host_vector<int>&
            localIndexMap, const thrust::host_vector<int>& pidIndexMap,
            MPI_Comm comm)
    {
        construct( local_size, localIndexMap, pidIndexMap, comm);
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
    MPIGatherMatrix( const thrust::host_vector<int>& globalIndexMap, const
            ConversionPolicy& p)
    {
        thrust::host_vector<int> local(globalIndexMap.size()), pids(globalIndexMap.size());
        bool success = true;
        for(unsigned i=0; i<local.size(); i++)
            if( !p.global2localIdx(globalIndexMap[i], local[i], pids[i]) )
                success = false;

        assert( success);
        construct( p.local_size(), local, pids, p.communicator());
    }

    template< template<typename > typename OtherVector, class other_value_type>
    MPIGatherMatrix( const MPIGatherMatrix<OtherVector, other_value_type>& src)
    {
        construct( src.local_size(), src.getLocalIndexMap(),
                src.getPidIndexMap(), src.communicator());
    }

    const thrust::host_vector<int>& getLocalIndexMap() const
    {
        return m_lIdx;
    }
    const thrust::host_vector<int>& getPidIndexMap() const
    {
        return m_pids;
    }

    // buffer to matrix, indexMap.size() x buffer_size
    const LocalGatherMatrix<Vector<value_type>>& get_G1() const
    {
        return m_g1;
    }
    // vector to store, store_size x local_size
    const LocalGatherMatrix<Vector<value_type>>& get_G2() const
    {
        return m_g2;
    }
    bool isCommunicating() const
    {
        return m_p.isCommunicating();
    }
    template<class ContainerType0, class ContainerType1>
    MPI_Request global_gather_init( const ContainerType& vector, ContainerType1& buffer)const
    {
        //gather values to store
        m_g2.gather( vector, m_store.data());
        m_p.global_gather_init( m_store.data(), m_buffer.data());
    }
    template<class ContainerType0, class ContainerType1>
    void global_gather_wait( const ContainerType& vector, ContainerType1&
            buffer, MPI_Request rqst)const
    {
        m_p.global_gather_wait( m_store.data(), m_buffer.data(), rqst);
    }

    template<class ContainerType0, class ContainerType1>
    MPI_Request global_scatter_plus_init( const ContainerType0& buffer,
            ContainerType1& vector)const
    {
        m_p.global_scatter_init( buffer, m_store.data());
    }
    template<class ContainerType0, class ContainerType1>
    void global_scatter_plus_wait( const ContainerType& buffer, ContainerType1&
            vector, MPI_Request rqst)const
    {
        m_p.global_scatter_wait( buffer, m_store.data(), rqst);
        m_g2.scatter_plus( m_store.data(), vector);
    }
    Vector<value_type> allocate_buffer() const {
        return Vector<value_type>( m_buffer_size);
    }

    MPI_Comm communicator()const
    {
        return m_p.communicator();
    }
    private:
    virtual unsigned do_size() const override final {return m_buffer_size;}
    void construct( unsigned local_size, thrust::host_vector<int>
            lIdx, thrust::host_vector<int> pids, MPI_Comm comm)
    {
        m_local_size = local_size;
        m_pids = pids;
        // 0 Sanity check
        int size;
        MPI_Comm_size( comm, &size);
        for( unsigned i=0; i<pids.size(); i++)
            assert( 0 <= pids[i] && pids[i] < size);
        // 1. Sort pids with indices so we get associated gather map
        thrust::host_vector<int> index(pids);
        thrust::sequence( index.begin(), index.end());
        thrust::stable_sort_by_key( pids.begin(), pids.end(), index.begin());

        //Now construct the MPIPermutation object by getting the number of elements
        //to send
        m_p.construct( detail::find_numbers_of_pids( pids, size), comm);

        m_store_size = m_p.store_size();
        m_buffer_size = m_p.buffer_size();
        m_store.data().resize( m_store_size);
        m_buffer.data().resize( m_buffer_size);
        m_g1=LocalGatherMatrix(index, m_buffer_size);

        // Finally, construct G2
        thrust::host_vector<int> storeIdx( m_store_size),
            bufferIdx( m_buffer_size);
        // permute the localIdxMap
        thrust::gather( index.begin(), index.end(), lIdx.begin(),
                bufferIdx.begin());
        // and scatter/MPI permute to the other processes so they may know
        // which values they need to provide:
        dg::detail::MPIPermutation<int> permute( sendTo, comm);
        permute.global_scatter( bufferIdx, storeIdx);
        // All the indices we receive here are local to the current rank!
        m_g2 = LocalGatherMatrix( storeIdx, m_local_size);

    }
    thrust::host_vector<int> m_lIdx, m_pids; // stored to be able to copy construct
    unsigned m_buffer_size, m_store_size, m_local_size;

    LocalGatherMatrix< Vector<value_type>> m_g1, m_g2;
    dg::Buffer< Vector<value_type>> m_values, m_store;
    dg::detail::MPIPermutation<value_type> m_p;
};

}//namespace dg
