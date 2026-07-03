#pragma once
#include <map>
#include <thrust/host_vector.h>
#include "index.h"
#include "mpi_datatype.h"
#include "tensor_traits.h"
#include "tensor_traits_scalar.h"
#include "tensor_traits_thrust.h"
#include "tensor_traits_std.h"


namespace dg{
///@cond
/*! @brief Check if communication map involves actual mpi communication

 * @param messages (in) messages[rank] is the message the calling rank sends to rank
 * @return false if no process in comm sends or receives any
 * message to another process, true else
 * @tparam M message type (\c M.size() must be callable)
 * @note This involves MPI communication because all ranks need to know if they
 * themselves send message **and** if all other ranks also send any messages
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
    for( auto& send : messages)
        sendTo[send.first] = 1; // just indicate that a message is being sent
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
///@endcond

///@addtogroup mpi_comm
///@{

/**
 * @brief Exchange messages between processes in a communicator
 *
 * This happens in two communication phases (find more details in 
 * \ref mpi_dist_gather)
 *  -# Call \c MPI_Allgather with given communicator to setup the
 *  communication pattern among processes
 *  -# Call \c MPI_Alltoallv to send the actual messages.
 *
 * For example
 * @snippet{trimleft} mpi_permutation_mpit.cpp permute
 *
 * @tparam MessageType Can be one of the following
 *  -# A primitive type like \c int or \c double
 *  -# A (host) vector of primitive types like \c std::vector<int> or \c
 *  thrust::host_vector<double>
 *  -# A (host) vector of std::array of primitive types like \c
 *  thrust::host_vector<std::array<double,3>>
 *  .
 * @param messages (in) <tt>messages[rank]</tt> contains the message that the
 * calling process sends to the process rank within comm
 * @param comm The MPI communicator within which to exchange messages. All
 * processes in comm need to call this function.
 * @return <tt>received_messages[rank]</tt> contains the message that the
 * calling process receveived from the process with rank in comm
 *
 * @note This can be used to bootstrap mpi gather operations if elements is an
 * index map "recvIdx" of local indices of messages to receive from rank,
 * because it "tells" every process which messages to send
 *
 * @note This function is a permutation i.e.
 * @code{.cpp}
 * recvIdx == dg::mpi_permute( dg::mpi_permute(recvIdx, comm), comm);
 * @endcode
 * @tparam ContainerType Shared ContainerType.
 * @sa \ref mpigather 
 * @sa Also can be used to invert a bijective mpi gather map in \c
 * dg::mpi_invert_permutation
 */
template<class MessageType>
std::map<int,MessageType> mpi_permute(
    const std::map<int,MessageType>& messages,
    MPI_Comm comm)
{
    // I think this function is fairly intuitive to understand which is good
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

/**
 * @brief Un-optimized distributed gather operation
 *
 * @tparam ContainerType A (host) vector
 * @param gather_map Each element consists of <tt>{rank within comm, local
 * index on that rank}</tt> pairs, which is equivalent to the global address of
 * a vector element in \c gatherFrom
 * @param gatherFrom Local part of the vector from which the calling and other
 * ranks can gather indices
 * @param result (Same size as gather_map on output) On output contains the
 * elements that \c gather_map referenced
 * @param comm The MPI communicator within which to exchange elements
 * @sa \ref mpigather
 */
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
/**
 * @brief Un-optimized distributed scatter operation
 *
 * @tparam ContainerType A (host) vector
 * @param scatter_map Each element consists of <tt>{rank, local index on that
 * rank}</tt> pairs, which is equivalent to the global address of an element in
 * \c result
 *
 * @attention Must be injective i.e. globally distinct elements in \c toScatter
 * must map to distince elements in \c result
 * @param toScatter Same size as \c scatter_map. The \c scatter_map tells where
 * each element in this vector is sent to
 * @param result In principle we must know the size of \c result beforehand
 * (because how else did you come up with a \c scatter_map)
 * @param comm The MPI communicator within which to exchange elements
 * @param resize_result If true we resize the result to the correct size (mainly
 * needed for \c dg::mpi_invert_permutation)
 * @sa \ref mpigather
 */
template<class ContainerType>
void mpi_scatter( const thrust::host_vector<std::array<int,2>>& scatter_map,
    const ContainerType& toScatter, ContainerType& result,
    MPI_Comm comm,
    bool resize_result = false
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

/**
 * @brief Invert a globally bijective index map
 *
 * @param p Each element consists of <tt>{rank, local index on that
 * rank}</tt> pairs, which is equivalent to the global address of a vector element
 *
 * @attention Must be bijective i.e. globally distinct elements in \c toScatter must
 * map to distince elements in \c result and all elements in \c result must be mapped
 * @param comm The MPI communicator within which to exchange elements
 * @return inverse map
 * @sa \ref mpigather
 */
template<class Integer>
thrust::host_vector<std::array<Integer,2>>
    mpi_invert_permutation( const thrust::host_vector<std::array<Integer,2>>& p,
            MPI_Comm comm)
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
///@}


} // namespace dg
