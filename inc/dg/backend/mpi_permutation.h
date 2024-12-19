#pragma once
#include <map>
#include <thrust/host_vector.h>
#include "index.h"
#include "mpi_datatype.h"
#include "tensor_traits.h"
#include "tensor_traits_scalar.h"
#include "tensor_traits_thrust.h"
#include "tensor_traits_cusp.h"
#include "tensor_traits_std.h"


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
 * "recvIdx" of local indices of messages to receive from PID, because it "tells" every
 * process which messages to send
 * @note Also can be used to invert a bijective mpi gather map
 *
 * @note This function is a permutation i.e.
 * @code{.cpp}
 * recvIdx == mpi_permute( mpi_permute(recvIdx, comm), comm);
 * @endcode
 * @tparam ContainerType Shared ContainerType.
 */
template<class MessageType>
std::map<int,MessageType> mpi_permute(
    const std::map<int,MessageType>& messages,
    MPI_Comm comm)
{
    int rank, comm_size;
    MPI_Comm_rank( comm, &rank);
    MPI_Comm_size( comm, &comm_size);
    thrust::host_vector<int> sendTo( comm_size, 0 ), recvFrom( comm_size, 0);
    thrust::host_vector<int> global( comm_size*comm_size);
    auto flat_vals = detail::flatten_values( messages);
    for( auto& send : flat_vals)
    {
        sendTo[send.first] = send.second.size();
    }
    // everyone knows howmany messages everyone is sending
    MPI_Allgather( sendTo.data(), comm_size, MPI_INT,
                   global.data(), comm_size, MPI_INT,
                   comm);
    for( int i=0; i<comm_size; i++)
        recvFrom[i] = global[i*comm_size+rank];
    // Now we can use Alltoallv to send
    thrust::host_vector<int> accS( comm_size), accR(comm_size);
    thrust::exclusive_scan( sendTo.begin(), sendTo.end(), accS.begin());
    thrust::exclusive_scan( recvFrom.begin(), recvFrom.end(), accR.begin());

    auto send = detail::flatten_map( flat_vals);
    using value_type = dg::get_value_type<decltype(send)>;
    thrust::host_vector<value_type> recv(
        thrust::reduce( recvFrom.begin(), recvFrom.end()));
    void * send_ptr          = thrust::raw_pointer_cast( send.data());
    const int * sendTo_ptr   = thrust::raw_pointer_cast( sendTo.data());
    const int * accS_ptr     = thrust::raw_pointer_cast( accS.data());
    void * recv_ptr          = thrust::raw_pointer_cast( recv.data());
    const int * recvFrom_ptr = thrust::raw_pointer_cast( recvFrom.data());
    const int * accR_ptr     = thrust::raw_pointer_cast( accR.data());
    MPI_Datatype type = dg::getMPIDataType<value_type>();
    MPI_Alltoallv( send_ptr, sendTo_ptr,   accS_ptr, type,
                   recv_ptr, recvFrom_ptr, accR_ptr, type,
                   comm);
    return detail::make_map_t<MessageType>( recv, detail::make_size_map( recvFrom) );
}
/*! @brief Check if communication map involves actual mpi communication

 * @param messages (in) messages[PID] is the message the calling rank sends to PID
 * @return false if no process in comm sends or receives any
 * message to another process, true else
 * @tparam M message type (\c M.size() must be callable)
 */
template<class MessageType>
bool is_communicating(
    const std::map<int,MessageType>& messages,
    MPI_Comm comm)
{
    int comm_size;
    MPI_Comm_size( comm, &comm_size);
    thrust::host_vector<int> sendTo( comm_size, 0 );
    thrust::host_vector<int> global( comm_size*comm_size);
    auto flat_vals = detail::flatten_values( messages);
    for( auto& send : flat_vals)
        sendTo[send.first] = send.second.size();
    // everyone knows howmany messages everyone is sending
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

// un-optimized version (mainly used for testing)
template<class ContainerType>
void mpi_gather( const thrust::host_vector<std::array<int,2>>& gather_map,
    const ContainerType& gatherFrom, ContainerType& result, MPI_Comm comm)
{
    thrust::host_vector<int> bufferIdx;
    auto gather_m = gIdx2unique_idx( gather_map, bufferIdx);
    auto send_map = mpi_permute(gather_m, comm);
    auto gather = detail::flatten_map( send_map);
    auto size_map = detail::get_size_map( send_map);
    ContainerType res( gather.size());
    thrust::gather( gather.begin(), gather.end(), gatherFrom.begin(), res.begin());
    std::map<int,ContainerType> result_map = detail::make_map_t<ContainerType>( res, size_map);
    std::map<int,ContainerType> recv = mpi_permute( result_map, comm);
    ContainerType flat = detail::flatten_map( recv);
    result.resize( bufferIdx.size());
    thrust::gather( bufferIdx.begin(), bufferIdx.end(), flat.begin(), result.begin());
}
// un-optimized version (mainly used for testing), only works for host arrays (only works for injective)
template<class ContainerType>
void mpi_scatter( const thrust::host_vector<std::array<int,2>>& scatter_map,
    const ContainerType& toScatter, ContainerType& result, // result needs to have correct size!
    MPI_Comm comm, bool resize_result = false // if true we resize the result (needed for invert_permutation)
    )
{
    thrust::host_vector<int> bufferIdx;
    // must be injective
    auto scatter_m = gIdx2unique_idx( scatter_map, bufferIdx);
    auto recv_map = mpi_permute(scatter_m, comm);
    auto scatter = detail::flatten_map( recv_map);

    auto size_map = detail::get_size_map( scatter_m);

    ContainerType to_send( toScatter);
    thrust::scatter( toScatter.begin(), toScatter.end(), bufferIdx.begin(), to_send.begin());

    auto send = detail::make_map_t<ContainerType>( to_send, size_map);
    auto result_map = mpi_permute( send, comm);
    auto res = detail::flatten_map( result_map);
    if( resize_result)
        result.resize( res.size());
    thrust::scatter( res.begin(), res.end(), scatter.begin(), result.begin());
}

// TODO document ( only works on host)
template<class Integer>
thrust::host_vector<std::array<Integer,2>>
    mpi_invert_permutation( const thrust::host_vector<std::array<Integer,2>>& p, MPI_Comm comm)
{
    thrust::host_vector<Integer> seq( p.size());
    thrust::host_vector<std::array<Integer,2>> seq_arr( p.size());
    thrust::sequence( seq.begin(), seq.end());
    Integer rank;
    MPI_Comm_rank( comm, &rank);
    // package sequence
    for( unsigned u=0; u<seq.size(); u++)
        seq_arr[u] = {rank, seq[u]};
    thrust::host_vector<std::array<Integer,2>> sort_map;
    mpi_scatter( p, seq_arr, sort_map, comm, true);
    return sort_map;

}


} // namespace dg
