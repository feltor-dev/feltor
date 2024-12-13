#pragma once
#include <cassert>
#include <thrust/host_vector.h>

#include "exceptions.h"
#include "config.h"
#include "mpi_datatype.h"
#include "tensor_traits.h"
#include "memory.h"
#include "gather.h"

namespace dg{

    /**
     * @brief engine for mpi permutating (i.e. bijective)
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
// TODO MAybe make a group mpi_utilities
// TODO test the permutation quality
// I think this is fairly intuitive to understand which is good
/**
 * @brief Bootstrap irregular communication between processes
 *
 * @param elements (in) <tt> elements[PID] </tt> contains the message that the calling rank
 * sends to PID
 * @param comm Communicator
 * @return <tt> received elements[PID] </tt> contains the message that the calling rank
 *  receveived from PID
 *
 * @note Calls \c MPI_Allgather with given communicator to send the sizes
 * followed by \c MPI_Alltoall to send the actual data. This means all processes in comm
 * need to call this function
 * @note This can be used to bootstrap mpi gather operations if elements is an index map
 * "recvIdx" of local indices of elements to receive from PID, because it "tells" every
 * process which elements to send
 * @note Also can be used to invert a bijective mpi gather map
 *
 * @note This function is a permutation i.e.
 * @code{.cpp}
 * recvIdx == mpi_permutation( mpi_permutation(recvIdx, comm), comm);
 * @endcode
 * @tparam ContainerType Shared ContainerType.
 */
template<class ContainerType>
std::map<int,ContainerType> mpi_permutation(
    const std::map<int,ContainerType>& elements,
    MPI_Comm comm)
{
    int rank, comm_size;
    MPI_Comm_rank( comm, &rank);
    MPI_Comm_size( comm, &comm_size);
    thrust::host_vector<int> sendTo( comm_size, 0 ), recvFrom( comm_size, 0);
    thrust::host_vector<int> global( comm_size*comm_size);
    for( auto& send : elements)
        sendTo[send.first] = send.second.size();
    // everyone knows howmany elements everyone is sending
    MPI_Allgather( sendTo.data(), comm_size, MPI_INT,
                   global.data(), comm_size, MPI_INT,
                   comm);
    for( int i=0; i<comm_size; i++)
        recvFrom[i] = global[i*comm_size+rank];
    // Now we can use Alltoallv to send
    thrust::host_vector<int> accS( comm_size), accR(comm_size);
    thrust::exclusive_scan( sendTo.begin(), sendTo.end(), accS.begin());
    thrust::exclusive_scan( recvFrom.begin(), recvFrom.end(), accR.begin());
    thrust::host_vector<int> recv(
        thrust::reduce( recvFrom.begin(), recvFrom.end()));
    auto send = detail::flatten_map( elements);

    void * send_ptr          = thrust::raw_pointer_cast( send.data());
    const int * sendTo_ptr   = thrust::raw_pointer_cast( sendTo.data());
    const int * accS_ptr     = thrust::raw_pointer_cast( accS.data());
    void * recv_ptr          = thrust::raw_pointer_cast( recv.data());
    const int * recvFrom_ptr = thrust::raw_pointer_cast( recvFrom.data());
    const int * accR_ptr     = thrust::raw_pointer_cast( accR.data());
    MPI_Datatype type = dg::getMPIDataType<dg::get_value_type<ContainerType>>();
    MPI_Alltoallv( send_ptr, sendTo_ptr,   accS_ptr, type,
                   recv_ptr, recvFrom_ptr, accR_ptr, type,
                   comm);
    return detail::make_map( recv, detail::make_size_map( recvFrom) );
}
/*! @brief Check if communication map involves actual mpi communication

 * @param elements (in) elements[PID] is the message the calling rank sends to PID
 * @return false if no process in comm sends or receives any
 * message to another process, true else
 * @tparam M message type (\c M.size() must be callable)
 */
template<class ContainerType>
bool is_communicating(
    const std::map<int,ContainerType>& elements,
    MPI_Comm comm)
{
    int comm_size;
    MPI_Comm_size( comm, &comm_size);
    thrust::host_vector<int> sendTo( comm_size, 0 );
    thrust::host_vector<int> global( comm_size*comm_size);
    for( auto& send : elements)
        sendTo[send.first] = send.second.size();
    // everyone knows howmany elements everyone is sending
    MPI_Allgather( sendTo.data(), comm_size, MPI_INT,
                   global.data(), comm_size, MPI_INT,
                   comm);
    bool isCommunicating = false;
    for( int i=0; i<comm_size; i++)
        for( int k=0; k<comm_size; k++)
            if( k != i and global[i*comm_size+k] != 0)
                isCommunicating = true;
    return isCommunicating;
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
     *  to use in the gatherscatter
     *  - <tt> local_size(); </tt> return the local vector size
     * @sa basictopology the MPI %grids defined in Level 3 can all be used as a
     * ConversionPolicy
     */
template<class ConversionPolicy>
std::map<int, thrust::host_vector<int>> gIdx2unique_idx(
    const thrust::host_vector<int>& globalIndexMap,
    thrust::host_vector<int>& bufferIdx, // may alias globalIndexMap
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
    return gIdx2unique_idx( gIdx, bufferIdx);
}
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
template< template <class> class Vector>
struct MPIContiguousGather
{
    struct MsgChunk{
        int idx; /// !< starting index of message
        int size; /// !< size of message
    };
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
        const std::map<int, thrust::host_vector<MsgChunk>>& recvMsg,
        MPI_Comm comm)
    : m_comm(comm), m_recvMsg( recvMsg)
    {
        // pack messages
        std::map<int, thrust::host_vector<int>> recvIdx, recvSize;
        for( auto& idx : recvMsg)
        for( unsigned u=0; u<idx.second.size(); u++)
        {
            recvIdx[idx.first].push_back(idx.second[u].idx);
            recvSize[idx.first].push_back(idx.second[u].size);
        }
        auto sendIdx = mpi_permutation ( recvIdx, comm);
        auto sendSize = mpi_permutation ( recvSize, comm);
        m_communicating = is_communicating( recvIdx, comm);
        // un-pack
        for( auto& idx : sendIdx)
        for( unsigned u=0; u<idx.second.size(); u++)
            m_sendMsg[idx.first].push_back({idx.second[u],
                                            sendSize[idx.first][u]});
        // We need to find out the minimum amount of memory we need to allocate
        for( auto& idx : recvSize)
            m_buffer_size += thrust::reduce( idx.second.begin(), idx.second.end());
        // if cuda unaware we need to send messages through host
        for( auto& idx : sendSize)
            m_store_size += thrust::reduce( idx.second.begin(), idx.second.end());
        resize_rqst();
    }

    /// Concatenate neigboring indices to bulk messasge
    static const std::map<int, thrust::host_vector<MsgChunk>> make_chunks(
        const std::map<int, thrust::host_vector<int> > &recvIdx, int chunk_size)
    {
        std::map<int, thrust::host_vector<MsgChunk>> recvChunk;
        for( auto& idx: recvIdx)
        {
            auto range = detail::contiguous_range( idx.second);
            for( auto& r : range)
            {
                recvChunk[idx.first].push_back( {r.first*chunk_size,
                    r.second*chunk_size});
            }
        }
        return recvChunk;
    }
    template< template<class> class OtherVector>
    friend class MPIContiguousGather; // enable copy

    template< template<class> class OtherVector>
    MPIContiguousGather( const MPIContiguousGather<OtherVector>& src)
    {
        m_sendMsg = src.m_sendMsg;
        m_recvMsg = src.m_recvMsg;
        m_comm = src.m_comm;
        m_communicating = src.m_communicating;
        m_buffer_size = src.m_buffer_size;
        m_store_size = src.m_store_size;
        resize_rqst();
    }

    MPI_Comm communicator() const{return m_comm;}
    /// How many elements in the buffer in total
    unsigned buffer_size() const { return m_buffer_size;}

    bool isCommunicating() const{
        return m_communicating;
    }
    template<class ContainerType0, class ContainerType1>
    void global_gather_init( const ContainerType0& gatherFrom, ContainerType1& buffer) const
    {
        // TODO only works if m_recvMsg is non-overlapping
        using value_type = dg::get_value_type<ContainerType0>;
        static_assert( std::is_same_v<value_type,
                get_value_type<ContainerType1>>);
        // Receives
        unsigned rqst_counter = 0;
        for( auto& msg : m_recvMsg) // first is PID, second is vector of chunks
        for( unsigned u=0; u<msg.second.size(); u++)
        {
            auto chunk = msg.second[u];
            void * recv_ptr;
            if constexpr (std::is_same_v< dg::get_execution_policy<ContainerType0>,
                dg::CudaTag> and not dg::cuda_aware_mpi)
            {
                m_h_buffer.template set<value_type>( buffer_size);
                auto& h_buffer = m_h_buffer.template get<value_type>();
                recv_ptr = thrust::raw_pointer_cast( h_buffer.data())
                   + chunk.idx;
            }
            else
                recv_ptr = thrust::raw_pointer_cast( buffer.data())
                   + chunk.idx;
            MPI_Irecv( recv_ptr, chunk.size,
                   getMPIDataType<value_type>(),  //receiver
                   msg.first, u, m_comm, &m_rqst[rqst_counter]);  //source
            rqst_counter ++;
        }

        // Send
        unsigned start = 0;
        for( auto& msg : m_sendMsg) // first is PID, second is vector of chunks
        for( unsigned u=0; u<msg.second.size(); u++)
        {
            auto chunk = msg.second[u];
            const void * send_ptr = thrust::raw_pointer_cast(gatherFrom.data()) + chunk.idx;
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
                    m_h_store.template set<value_type>( store_size);
                    auto& h_store = m_h_store.template get<value_type>();
                    code = cudaMemcpy( &h_store[start], send_ptr,
                        chunk.size*sizeof(value_type), cudaMemcpyDeviceToHost);
                    send_ptr = // re-point pointer
                        thrust::raw_pointer_cast(&h_store[start]);
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
            MPI_Isend( send_ptr, chunk.size,
                   getMPIDataType<value_type>(),  //sender
                   msg.first, u, m_comm, &m_rqst[rqst_counter]);  //destination
            rqst_counter ++;
            start+= chunk.size;
        }
    }

    // Also sync cuda if necessary
    ///@copydoc MPIGather<Vector>::global_gather_init
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
    MPI_Comm m_comm; // from constructor
    bool m_communicating = false;
    std::map<int,thrust::host_vector<MsgChunk>> m_sendMsg;
    std::map<int,thrust::host_vector<MsgChunk>> m_recvMsg; // from constructor

    unsigned m_buffer_size = 0;
    mutable detail::AnyVector<thrust::host_vector>  m_h_buffer;

    unsigned m_store_size = 0;
    mutable detail::AnyVector<thrust::host_vector>  m_h_store;

    mutable std::vector<MPI_Request> m_rqst;
    void resize_rqst()
    {
        unsigned rqst_size = 0;
        // number of messages to send and receive
        for( auto& msg : m_recvMsg)
            rqst_size += msg.second.size();
        for( auto& msg : m_sendMsg)
            rqst_size += msg.second.size();
        m_rqst.resize( rqst_size);
    }
};

/**
 * @brief Perform MPI distributed gather and its transpose (scatter) operation across processes
 * on distributed vectors using MPI
 *
 * In order to understand what this class does you should first really(!) understand what
 gather and scatter operations are, so grab pen and paper:

 First, we note that gather and scatter are most often used in the context
 of memory buffers. The buffer needs to be filled wih values (gather) or these
 values need to be written back into the original place (scatter).

 Imagine a buffer vector w and an index map \f$ \text{g}[i]\f$
 that gives to every index \f$ i\f$ in this vector w
 an index \f$ \text{g}[i]\f$ into a source vector v.

We can now define:
 @b Gather values from v and put them into w according to
  \f$ w[i] = v[\text{g}[i]] \f$

 Loosely we think of @b Scatter as the reverse operation, i.e. take the values
 in w and write them back into v. However, simply writing
 \f$ v[\text{g}[j]] = w[j] \f$ is a very **bad** definition.
 What should happen if \f$ g[j] = g[k]\f$
 for some j and k? What if some indices \f$ v_i\f$ are not mapped at all?

It is more accurate to represent the gather and scatter operation
by a matrix.

@b Gather matrix: A matrix \f$ G\f$ of size \f$ m \times N\f$ is a gather
 matrix if it consists of only 1's and 0's and has exactly one "1" in each row.
 \f$ m\f$ is the buffer size, \f$ N \f$ is the vector size and \f$ N\f$ may be smaller,
 same or larger than \f$m\f$.
 If \f$ \text{g}[i]\f$ is the index map then \f[ G_{ij} := \delta_{\text{g}[i] j}\f]
 We have \f$ w = G v\f$

@b Scatter matrix: A matrix \f$ S \f$ is a
 scatter matrix if its transpose is a gather matrix.

 This means that \f$ S\f$ has size \f$ N \times m \f$
 consists of only 1's and 0's and has exactly one "1" in each column.
 If \f$ \text{g}[j]\f$ is the index map then \f[ S_{ij} := \delta_{i \text{g}[j]}\f]
 We have \f$ v = S w\f$

All of the following statements are true

- The transpose of a gather matrix is a scatter matrix \f$ S  = G^\mathrm{T}\f$.
    The associated index map of \f$ S\f$ is identical to the index map of \f$ G\f$.
- The transpose of a scatter matrix is a gather matrix \f$ G  = S^\mathrm{T}\f$.
    The associated index map of \f$ G\f$ is identical to the index map of \f$ S\f$.
- From a given index map we can construct two matrices (\f$ G \f$ and \f$ S\f$)
- A simple consistency test is given by \f$ (Gv)\cdot (Gv) = S(Gv)\cdot v\f$.
- A scatter matrix can have zero, one or more "1"s in each row.
- A gather matrix can have zero, one or more "1"s in each column.
- If v is filled with its indices i.e. \f$ v_i = i\f$ then \f$ m = Gv\f$ i.e. the gather operation
    reproduces the index map
- If the entries of w are \f$ w_j = j\f$ then \f$ m \neq Sw\f$ does **not**
    reproduce the index map
- In a "coo" formatted sparse matrix format the gather matrix is obtained:
    \f$ m \f$ rows, \f$ N\f$ columns and \f$ m\f$ non-zeroes,
    the values array would consist only of "1"s,
    the row array is just the index \f$i\f$
    and the column array is the map \f$ g[i]\f$.
- In a "coo" formatted sparse matrix format the scatter matrix is obtained:
    \f$ N \f$ rows, \f$ m\f$ columns and \f$ m\f$ non-zeroes,
    the values array would consist only of "1"s,
    the row array is the map \f$g[j]\f$
    and the column array is the index \f$ j\f$.
- \f$ G' = G_1 G_2 \f$, i.e. the multiplication of two gather matrices is again a gather
- \f$ S' = S_1 S_2 \f$, i.e. the multiplication of two scatter matrices is again a scatter

Of the scatter and gather matrices permutations are especially interesting
A matrix is a **permutation** if and only if it is both a scatter and a gather matrix.
    In such a case it is square \f$ m \times m\f$ and \f[ P^{-1} = P^T\f].
    The buffer \f$ w\f$ and vector \f$ v\f$ have the same size \f$m\f$.


The following statements are all true
- The index map of a permutation is bijective i.e. invertible i.e. each element
    of the source vector v maps to exactly one location in the buffer vector w.
- The scatter matrix \f$ S = G^T \equiv G'\neq G\f$ is a gather matrix (in
    general unequal \f$ G\f$) with the associate index map \f$ m^{-1}\f$.
    Since the index map is recovered by applying the gather operation to the vector
    containing its index as values, we have
    \f[ m^{-1} = G' \vec i = S \vec i\f]
- \f$ S' = P_1 S P_2 \f$, i.e. multiplication of a scatter matrix by a permutation is again a scatter matrix
- \f$ G' = P_1 G P_2 \f$, i.e. multiplication of a gather matrix by a permutation is again a gather matrix
- A Permutation is **symmetric** if and only if it has identical scatter and gather maps
- Symmetric permutations can be implemented "in-place" i.e. the source and buffer can be identical



This class performs these operations for the case that v and w are distributed
across processes.  Accordingly, the index map \f$ g\f$  is also distributed
across processes (in the same way w is).  The elements of \f$ g\f$ are
**global** indices into v that have to be transformed to pairs (local index
        into v, rank in communicator) according to a user provided function. Or
the user can directly provide the index map as vector of mentioned pairs.

Imagine now that we want to perform a globally distributed gather operation.
Then, the following steps are performed
 - From the given index array a MPI communication matrix (of size
 \f$ s \times s\f$ where \f$ s\f$ is the number of processes in the MPI
 communicator) can be inferred. Each row shows how many elements a
 given rank ( the row index) receives from each of the other ranks in the
 communicator (the column indices). Each column of this map describe the
 sending pattern, i.e. how many elements a given rank (the column index) has to
 send each of the other ranks in the communicator.  If the MPI communication
 matrix is symmetric we can  perform MPI communications **in-place**
 - The information from the communication matrix can be used to allocate
 appropriately sized MPI send and receive buffers. Furthermore, it is possible
 to define a **permutation** across different processes. It is important to
 note that the index map associated to that permutation is immplementation
 defined i.e.  the implementation analyses the communication matrix and chooses
 an optimal call of MPI Sends and Recvs. The implementation then provides two
 index maps. The first one must be used to gather values from v into the
 MPI send buffer and the second one can be used to gather values from the
 receive buffer into the target buffer. Notice that these two operations are
 **local** and require no MPI communication.

 In total we thus describe the global gather as
 \f[ w = G v = G_1 P_{G,MPI} G_2 v\f]

 The global scatter operation is then simply
 \f[ v = S w = G_2^T P^T_{G,MPI} G^T_1 w = S_2 P_{S,MPI} S_1 w \f]
 (The scatter operation is constructed the same way as the gather operation, it is just the execution that is different)

 @note If the scatter/gather operations are part of a matrix-vector multiplication
 then \f$ G_1\f$ or \f$ S_1\f$ can be absorbed into the matrix

 \f[ M v = R G v  = R G_1 P_{G,MPI} G_2 v = R' P_{G,MPI} G_2 v\f]. If R was a
 coo matrix the simple way to obtain R' is replacing the column indices with
 the map \f$ g_1\f$.
 @note To give the involved vectors unique names we call v the "vector", \f$ s = G_2 v\f$ is the "store" and, \f$ b = P s\f$ is the "buffer".

 For \f[ M v = S C v = S_2 P_{S,MPI} S_1 C v = S_2 P_{S,MPI} C' v\f]. Again, if
 C was a coo matrix the simple way to obtain C' is replacing the row indices
 with the map \f$ g_1\f$.

 Simplifications can be achieved if \f$ G_2 = S_2 = I\f$ is the identity
 or if \f$ P_{G,MPI} = P_{S,MPI} = P_{MPI}\f$ is symmetric, which means that
 in-place communication can be used.

 @note Locally, a gather operation is trivially parallel but a scatter operation
 is not in general (because of the possible reduction operation).
 @sa LocalGatherMatrix

 * @ingroup mpi_structures
 * @code
 int i = myrank;
 double values[8] = {i,i,i,i, 9,9,9,9};
 thrust::host_vector<double> hvalues( values, values+8);
 int pids[8] =      {0,1,2,3, 0,1,2,3};
 thrust::host_vector<int> hpids( pids, pids+8);
 BijectiveComm coll( hpids, MPI_COMM_WORLD);
 thrust::host_vector<double> hrecv = coll.global_gather( hvalues); //for e.g. process 0 hrecv is now {0,9,1,9,2,9,3,9}
 thrust::host_vector<double> hrecv2( hvalues.size());
 coll.global_scatter_reduce( hrecv, hrecv2); //hrecv2 now equals hvalues independent of process rank
 @endcode
 * @tparam Vector a thrust Vector
 */
template< template <class> class Vector>
struct MPIGather
{

    /*!@brief no communication
     *
     * @param comm optional MPI communicator: the purpose is to be able to store
     MPI communicator even if no communication is involved in order to
     construct MPI_Vector with it
     */
    MPIGather( MPI_Comm comm = MPI_COMM_NULL) : m_mpi_gather(comm){ }
    //TODO  update docu
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
    * @param comm The MPI communicator participating in the gather
    * operations
    */
    MPIGather(
        const std::map<int, thrust::host_vector<int>>& recvIdx,
        MPI_Comm comm) : MPIGather( recvIdx, 1, comm)
    { }
    MPIGather(
        const std::map<int, thrust::host_vector<int>>& recvIdx, // in units of chunk size
        unsigned chunk_size, // can be 1 (contiguous indices in recvIdx are concatenated)
        MPI_Comm comm)
    {
        // The idea is that recvIdx and sendIdx completely define the communication pattern
        // and we can choose an optimal implementation
        // Actually the MPI library should do this but general gather indices
        // don't seem to be supported
        // So for now let's just use MPI_Alltoallv
        // Possible optimization includes
        // - check for duplicate messages to save internal buffer memory
        // - check for contiguous messages to avoid internal buffer memory
        // - check if an MPI_Type could fit the index map
        //v -> store -> buffer -> w
        // G_1 P G_2 v
        // G_2^T P^T G_1^T w
        // We need some logic to determine if we should pre-gather messages into a store
        unsigned num_messages = 0;
        auto recvChunks = MPIContiguousGather<Vector>::make_chunks(
            recvIdx, chunk_size);
        for( auto& chunks : recvChunks)
            num_messages+= chunks.second.size();
        unsigned size = recvChunks.size();
        double avg_msg_per_pid = (double)num_messages/(double)size; // > 1
        MPI_Allreduce( MPI_IN_PLACE, &avg_msg_per_pid, 1, MPI_DOUBLE, MPI_MAX, comm);
        m_contiguous = ( avg_msg_per_pid < 10); // 10 is somewhat arbitrary
        if( not m_contiguous) // messages are too fractioned
        {
            // bootstrap communication pattern
            auto sendIdx = mpi_permutation ( recvIdx, comm);
            // In that case only gather unique messages into send store
            detail::Unique<int> uni = detail::find_unique_order_preserving(
                detail::flatten_map( sendIdx));
            sendIdx = detail::make_map(
                detail::combine_gather( uni.gather1, uni.gather2),
                detail::get_size_map( sendIdx)); // now sendIdx goes into unique messages
            // bootstrap communication pattern back to recvIdx
            // (so that recvIdx has index into same store)
            auto uni_recvIdx = mpi_permutation( sendIdx, comm);
            recvChunks = MPIContiguousGather<Vector>::make_chunks(
                uni_recvIdx, chunk_size);
            m_mpi_gather = dg::MPIContiguousGather<Vector>( recvChunks, comm);
            // everything becomes a z - derivative ...
            // we gather all unique indices contiguously into send buffer
            // uni.unique == unique local indices into gatherFrom
            thrust::host_vector<int> g2( uni.unique.size()*chunk_size);
            for( unsigned l=0; l<uni.unique.size(); l++)
            for( unsigned k=0; k<chunk_size; k++)
                g2[l*chunk_size + k] = uni.unique[l] + k;
            m_g2 = LocalGatherMatrix<Vector>(g2);
        }
        else
            m_mpi_gather = dg::MPIContiguousGather<Vector>( recvChunks, comm);

    }


    /// https://stackoverflow.com/questions/26147061/how-to-share-protected-members-between-c-template-classes
    template< template<class> class OtherVector>
    friend class MPIGather; // enable copy

    /**
    * @brief Construct from other execution policy
    *
    * This makes it possible to construct an object on the host
    * and then copy everything on to a device
    * @tparam OtherVector other container type
    * @param src source object
    */
    template< template<typename > typename OtherVector>
    MPIGather( const MPIGather<OtherVector>& src)
    :   m_contiguous( src.m_contiguous),
        m_g2( src.m_g2),
        m_mpi_gather( src.m_mpi_gather)
        // we don't need to copy memory buffers (they are just workspace) or the request
    {
    }


    /**
    * @brief The internal MPI communicator used
    *
    * @return MPI Communicator
    */
    MPI_Comm communicator() const{return m_mpi_gather.communicator();}

    bool isContiguous() const { return m_contiguous;}
    /**
    * @brief The local size of the buffer vector w = local map size
    *
    * Consider that both the source vector v and the buffer w are distributed across processes.
    * In Feltor the vector v is (usually) distributed equally among processes and the local size
    * of v is the same for all processes. However, the buffer size might be different for each process.
    * @return buffer size (may be different for each process)
    * @note may return 0
    * @attention it is NOT a good idea to check for zero buffer size if you
    * want to find out whether a given process needs to send MPI messages or
    * not. The first reason is that even if no communication is happening the
    * buffer_size is not zero as there may still be local gather/scatter
    * operations. The right way to do it is to call <tt> isCommunicating() </tt>
    * @sa local_size() isCommunicating()
    */
    unsigned buffer_size() const { return m_mpi_gather.buffer_size();}

    /**
     * @brief True if the gather/scatter operation involves actual MPI communication
     *
     * This is more than just a test for zero message size.  This is because
     * even if a process has zero message size indicating that it technically
     * does not need to send any data at all it might still need to participate
     * in an MPI communication (sending an empty message to indicate that a
     * certain point in execution has been reached). Only if NONE of the
     * processes in the process group has anything to send will this function
     * return false.  This test can be used to avoid the gather operation
     * alltogether in e.g. the construction of a MPI distributed matrix.
     * @note this check may involve MPI communication itself, because a process
     * needs to check if itself or any other process in its group is
     * communicating.
     *
     * @return False, if the global gather can be done without MPI
     * communication (i.e. the indices are all local to each calling process),
     * or if the communicator is \c MPI_COMM_NULL. True else.
     * @sa buffer_size()
     */
    bool isCommunicating() const
    {
        return m_mpi_gather.isCommunicating();
    }
    /**
     * @brief \f$ w = G v\f$. Globally (across processes) asynchronously gather data into a buffer
     *
     * @param values source vector v; data is collected from this vector
     * @note If <tt> !isCommunicating() </tt> then this call will not involve
     * MPI communication but will still gather values according to the given
     * index map
     * @attention It is @b unsafe to write values to \c gatherFrom
     * or to read values in \c buffer until \c global_gather_wait
     * has been called
     * @sa The transpose operation is <tt> global_scatter_reduce() </tt>
     */
    template<class ContainerType0, class ContainerType1>
    void global_gather_init( const ContainerType0& gatherFrom, ContainerType1& buffer) const
    {
        // TODO docu It is save to use and change vector immediately after this function
        using value_type = dg::get_value_type<ContainerType0>;
        if( not m_contiguous)
        {
            m_store.template set<value_type>( m_g2.index_map().size());
            auto& store = m_store.template get<value_type>();
            m_g2.gather( gatherFrom, store);
            m_mpi_gather.global_gather_init( store, buffer);
        }
        else
            m_mpi_gather.global_gather_init( gatherFrom, buffer);
    }
    /**
    * @brief Wait for asynchronous communication to finish and gather received data into buffer
    *
    * Call \c MPI_Waitall on internal \c MPI_Request variables
    * and manage host memory in case of cuda-unaware MPI.
    * After this call returns it is safe to use the buffer and the \c gatherFrom variable
    * from the corresponding \c global_gather_init call
    * @param buffer (write only) where received data resides on return; must be
    * identical to the one given in a previous call to \c global_gather_init()
    */
    template<class ContainerType>
    void global_gather_wait( ContainerType& buffer) const
    {
        m_mpi_gather.global_gather_wait( buffer);
    }

    private:

    bool m_contiguous = false;
    LocalGatherMatrix<Vector> m_g2;
    dg::MPIContiguousGather<Vector> m_mpi_gather;
    // These are mutable and we never expose them to the user
    //unsigned m_store_size;// not needed because g2.index_map.size()
    mutable detail::AnyVector< Vector> m_store;

};

}//namespace dg
