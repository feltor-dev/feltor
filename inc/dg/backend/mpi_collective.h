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
#include "tensor_traits.h"
#include "memory.h"
#include "mpi_communicator.h"

namespace dg{

///@cond
namespace detail
{
    //TODO who handles isCommunicating?
    //TODO Handle other improvements like isSymmetric!

//v -> store -> buffer -> w
// G_1 P G_2 v
// G_2^T P^T G_1^T w

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
        m_buffer_size = m_store_size = 0;
        m_comm = MPI_COMM_NULL;
    }
    /**
     * @brief Give number of elements to connect
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

    unsigned buffer_size() const{
        return m_buffer_size;
    }
    unsigned store_size() const{
        return m_store_size;
    }


    /**
    * @brief The internal MPI communicator used
    *
    * @return MPI Communicator
    */
    MPI_Comm communicator() const{return m_comm;}

    bool isSymmetric() const { return m_symmetric;}

    bool isCommunicating() const{
        return m_communicating;
    }


    // synchronous gather
    template<class ContainerType0, class ContainerType1>
    void global_gather( const ContainerType0& store, ContainerType1& buffer) const
    {
        MPI_Request rqst = global_gather_init( store, buffer);
        MPI_Wait( &rqst, MPI_STATUS_IGNORE );
    }

    // synchronous scatter
    template<class ContainerType0, class ContainerType1>
    void global_scatter( const ContainerType0& buffer, ContainerType1& store) const
    {
        MPI_Request rqst = global_scatter_init( buffer, store);
        MPI_Wait( &rqst, MPI_STATUS_IGNORE );
    }

    // asynchronous scatter
    // user is responsible to catch CUDA-unawareness!
    template<class ContainerType0, class ContainerType1>
    MPI_Request global_scatter_init(
            const ContainerType0& buffer, ContainerType1& store) const
    {
        return global_comm_init( buffer, store, true);
    }
    // asynchronous gather
    template<class ContainerType0, class ContainerType1>
    MPI_Request global_gather_init(
            const ContainerType0& store, ContainerType1& buffer)const
    {
        return global_comm_init( store, buffer, false);
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
    template<class ContainerType0, class ContainerType1>
    MPI_Request global_comm_init(
            const ContainerType0& send, ContainerType1& recv, bool scatter) const
    {
        using value_type = dg::get_value_type<ContainerType0>;
        static_assert( std::is_same_v<value_type,
                get_value_type<ContainerType1>>);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
        if constexpr ( std::is_same_v<
                dg::get_execution_policy<ContainerType1>,
                dg::CudaTag>)
        {
            // We have to wait that all kernels are finished and values are
            // ready to be sent
            cudaError_t code = cudaGetLastError( );
            if( code != cudaSuccess)
                throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
            code = cudaDeviceSynchronize();
            if( code != cudaSuccess)
                throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        }
#endif
        const value_type * s_ptr    = thrust::raw_pointer_cast( send.data());
        const int * sendTo_ptr      = thrust::raw_pointer_cast( m_sendTo.data());
        const int * accS_ptr        = thrust::raw_pointer_cast( m_accS.data());
        value_type * r_ptr          = thrust::raw_pointer_cast( recv.data());
        const int * recvFrom_ptr    = thrust::raw_pointer_cast( m_recvFrom.data());
        const int * accR_ptr        = thrust::raw_pointer_cast( m_accR.data());
        unsigned send_size    = m_buffer_size;
        unsigned recv_size    = m_store_size;
        if( !scatter)
        {
            // invert the scatter and gather operations
            std::swap( sendTo_ptr, recvFrom_ptr);
            std::swap( accS_ptr, accR_ptr);
            std::swap( send_size, recv_size);
        }
        assert( send.size()  == send_size);
        assert( recv.size()  == recv_size);
        return global_comm_init_v( s_ptr, sendTo_ptr,   accS_ptr,
                                   r_ptr, recvFrom_ptr, accR_ptr);

    }
    template<class value_type>
    MPI_Request global_comm_init_v(
        const value_type* send, const int* sendTo,   const int* accS,
              value_type* recv, const int* recvFrom, const int* accR) const
    {
        MPI_Request rqst;
        if ( send == recv) // same address
        {
            assert( m_symmetric);
            MPI_Ialltoallv(
                MPI_IN_PLACE,
                sendTo, accS,
                getMPIDataType<value_type>(),
                recv, recvFrom, accR,
                getMPIDataType<value_type>(),
                m_comm, &rqst);
        }
        else
        {
            MPI_Ialltoallv(
                send, sendTo, accS,
                getMPIDataType<value_type>(),
                recv, recvFrom, accR,
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
    const thrust::host_vector<std::array<int,2>>& locally_unique_global_idx,
    MPI_Comm comm)
{
    int comm_size;
    MPI_Comm_size( comm, &comm_size);
    // get pids
    thrust::host_vector<int> pids(locally_unique_global_idx.size());
    for( int i=0; i<(int)pids.size(); i++)
    {
        pids[i] = locally_unique_global_idx[i][0];
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
    MPIGather() = default;
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
    MPIGather( const thrust::host_vector<std::array<int,2>>& gIdx,
            // TODO gIdx can be unsorted and contain duplicate entries
            // TODO idx 0 is pid, idx 1 is localIndex on that pid
            MPI_Comm comm)
    {
        thrust::host_vector<int> bufferIdx;
        thrust::host_vector<std::array<int,2>> locally_unique_gIdx;
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
        thrust::host_vector<int> lIdx( m_permute.buffer_size());
        for( unsigned i=0; i<lIdx.size(); i++)
            lIdx[i] = locally_unique_gIdx[i][1]; // the local index
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
    MPIGather( const MPIGather<OtherVector>& src)
    {
        *this = MPIGatherMatrix( src.getGlobalIndexMap(),
                src.communicator());
    }

    const thrust::host_vector<std::array<int,2>>& getGlobalIndexMap() const
    {
        return m_gIdx;
    }

    MPI_Comm communicator()const
    {
        return m_permute.communicator();
    }
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
    unsigned buffer_size() const { return m_permute.buffer_size();}

    // gather map from buffer to gIdx given in constructor
    const thrust::host_vector<int>& get_buffer_idx() const
    {
        return m_g1;
    }
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
        return m_permute.isCommunicating();
    }
    /**
     * @brief \f$ w = G v\f$. Globally (across processes) gather data into a buffer
     *
     * The transpose operation is <tt> global_scatter_reduce() </tt>
     * @param values source vector v; data is collected from this vector
     * @note If <tt> !isCommunicating() </tt> then this call will not involve
     * MPI communication but will still gather values according to the given
     * index map
     */
    template<class ContainerType>
    void gather( const ContainerType& vector) const
    {
        using value_type = get_value_type<ContainerType>;

        m_store.template set<value_type>( m_permute.store_size());
        m_buffer.template set<value_type>( m_permute.buffer_size());

        //gather values to store
        thrust::gather( m_g2.begin(), m_g2.end(), vector.begin(),
            m_store.template get<value_type>().begin());
#ifdef _DG_CUDA_UNAWARE_MPI // we need to send through host
        if constexpr ( std::is_same_v<
                dg::get_execution_policy<ContainerType>,
                dg::CudaTag>)
        {
            m_h_buffer.set<value_type>( m_permute.buffer_size());
            m_h_store.set<value_type>( m_permute.store_size());
            // cudaMemcpy
            m_h_store.template get<value_type>() =
                m_store.template get<value_type>();
        }
        m_rqst.data() = m_permute.global_gather_init(
                m_h_store.template get<value_type>(),
                m_h_buffer.template get<value_type>());
#else
        m_rqst.data() = m_permute.global_gather_init(
                m_store.template get<value_type>(),
                m_buffer.template get<value_type>());
#endif// _DG_CUDA_UNAWARE_MPI
    }
    template<class value_type>
    const Vector<value_type>& read_buffer() const
    {
        // can be called on MPI_REQUEST_NULL
        MPI_Wait( &m_rqst.data(), MPI_STATUS_IGNORE );
#ifdef _DG_CUDA_UNAWARE_MPI // we need to copy from host
        if constexpr ( std::is_same_v<
                dg::get_execution_policy<Vector<value_type>>,
                dg::CudaTag>)
            return m_h_buffer.template get<value_type>();
#endif// _DG_CUDA_UNAWARE_MPI
        return m_buffer.template get<value_type>();
    }
    template<class value_type>
    Vector<value_type>& write_buffer() const // write access
    {
        return m_buffer.template get<value_type>();
    }

    // defined only if gIdx map is injective (one-to-one)
    // TODO infer this information in constructor
    template<class value_type>
    void scatter_init( )const
    {
        m_store.template set<value_type>( m_permute.store_size());
        // throw if buffer has wrong type
#ifdef _DG_CUDA_UNAWARE_MPI // we need to send through host
        if constexpr ( std::is_same_v<
                dg::get_execution_policy<ContainerType>,
                dg::CudaTag>)
        {
            m_h_buffer.template set<value_type>( m_permute.buffer_size());
            m_h_store.template set<value_type>( m_permute.store_size());
            // cudaMemcpy
            m_h_buffer.template get<value_type>() = m_buffer.get<value_type>();
        }
        m_rqst.data() = m_permute.global_scatter_init(
                m_h_buffer.template get<value_type>(),
                m_h_store.template get<value_type>());
#else
        m_rqst.data() = m_permute.global_gather_init(
                m_buffer.template get<value_type>(),
                m_store.template get<value_type>());
#endif// _DG_CUDA_UNAWARE_MPI
    }
    /**
     * @brief \f$ v = G^\mathrm{T} b\f$. Globally (across processes) scatter data inside buffer
     *
     * @attention no reduction on double indices is performed so this operation is
     * currently only defined if the gather map is injective
     * This is the transpose operation of <tt> global_gather() </tt>
     * @param v target vector v; on output contains values from other
     * processes sent back to the origin (must have <tt> local_size() </tt>
     * elements)
     * @note If <tt> !isCommunicating() </tt> then this call will not involve
     * MPI communication but will still scatter and reduce values according to
     * the given index map
     */
    template<class ContainerType>
    void scatter( ContainerType& v) const
    {
        using value_type = get_value_type<ContainerType>;
        // can be called on MPI_REQUEST_NULL
        MPI_Wait( &m_rqst.data(), MPI_STATUS_IGNORE );
        //scatter store to v
#ifdef _DG_CUDA_UNAWARE_MPI // we need to send through host
        if constexpr ( std::is_same_v<
                dg::get_execution_policy<ContainerType>,
                dg::CudaTag>)
            m_store.template get<value_type>() =
                m_h_store.template get<value_type>();
#endif// _DG_CUDA_UNAWARE_MPI
        thrust::scatter( m_store.template get<value_type>().begin(),
            m_store.template get<value_type>().end(), m_g2.begin(), v.begin());
    }

    private:
    // stored to be able to copy construct
    thrust::host_vector<std::array<int,2>> m_gIdx;
    Vector<int> m_g1, m_g2;
    detail::AnyVector< Vector> m_buffer, m_store;
    dg::detail::MPIPermutation m_permute;
    dg::detail::Buffer<MPI_Request> m_rqst;
#ifdef _DG_CUDA_UNAWARE_MPI // nothing serious is cuda unaware ...
    detail::AnyVector<thrust::host_vector>  m_h_buffer, m_h_store;
#endif// _DG_CUDA_UNAWARE_MPI
};

}//namespace dg
