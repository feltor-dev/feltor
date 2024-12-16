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
///@cond
namespace detail
{


// for every size != 0 in sizes add map[idx] = size;
static inline std::map<int,int> make_size_map( const thrust::host_vector<int>& sizes)
{
    std::map<int,int> out;
    for( unsigned u=0; u<sizes.size(); u++)
    {
        if ( sizes[u] != 0)
            out[u] = sizes[u];
    }
    return out;
}

////////////////// Functionality for packaging mpi messages
// We pack every message into a vector of primitive type
//
//forward declare
template<class T>
auto flatten ( const T& t);

static inline thrust::host_vector<int> flatten ( const MsgChunk& t)
{
    thrust::host_vector<int> flat(2);
    flat[0] = t.idx, flat[1] = t.size;
    return flat;
}

template<class T>
thrust::host_vector<T> flatten ( const T& t, dg::AnyScalarTag)
{
    return thrust::host_vector<T>(1, t);
}
template<class ContainerType>
auto flatten (
    const ContainerType& t, dg::AnyVectorTag)
{
    decltype( flatten(t[0])) flat;
    for( unsigned u=0; u<t.size(); u++)
    {
        auto value = flatten(t[u]);
        flat.insert(flat.end(), value.begin(), value.end());
    }
    return flat;
}

template<class T>
auto flatten ( const T& t) // return type is  thrust::host_vector<Type>
{
    using tensor_tag = dg::get_tensor_category<T>;
    return flatten( t, tensor_tag());
}

// 1. flatten values -> a map of flattened type
template<class Message>
auto flatten_values( const std::map<int,Message>& idx_map // map is sorted automatically
)
{
    using vector_type = decltype( flatten( idx_map.at(0) ));
    std::map<int,vector_type> flat;
    for( auto& idx : idx_map)
    {
        flat[idx.first]  = flatten(idx.second);
    }
    return flat;
}

// 2. flatten the map (keys get lost)
template<class T>
thrust::host_vector<T> flatten_map(
    const std::map<int,thrust::host_vector<T>>& idx_map // map is sorted automatically
    )
{
    thrust::host_vector<T> flat;
    for( auto& idx : idx_map)
        flat.insert(flat.end(), idx.second.begin(), idx.second.end());
    return flat;
}

////////////////// Functionality for unpacking mpi messages
// unpack a vector of primitive types into original data type

static inline void make_target(
    const thrust::host_vector<int>& src, MsgChunk& target)
{
    assert( src.size() == 2);
    target = {src[0], src[1]};
}
static inline void make_target(
    const thrust::host_vector<int>& src, thrust::host_vector<MsgChunk>& target)
{
    assert( src.size() % 2 == 0);
    target.clear();
    for( unsigned u=0; u<src.size()/2; u++)
        target.push_back( { src[2*u], src[2*u+1]});
}
template<class Target, class Source>
void make_target( const thrust::host_vector<Source>& src, Target& t, AnyScalarTag)
{
    assert( src.size() == 1);
    t = src[0];
}
template<class Target, class Source>
void make_target( const thrust::host_vector<Source>& src, Target& t, AnyVectorTag)
{
    t = src;
}
template<class Target, class Source>
void make_target( const thrust::host_vector<Source>& src, Target& t)
{
    using tensor_tag = dg::get_tensor_category<Target>;
    make_target(src, t, tensor_tag()  );
}

template<class Target, class T>
std::map<int, Target> make_map(
    const thrust::host_vector<T>& flat_map,
    const std::map<int,int>& size_map // key and chunk size of idx_map
    )
{
    // 1. unflatten vector
    std::map<int, Target> map;
    unsigned start = 0;
    for( auto& size : size_map)
    {
        if( size.second != 0)
        {
            thrust::host_vector<T> partial(
                flat_map.begin()+start, flat_map.begin() + start + size.second);
            start += size.second;
            make_target( partial, map[size.first]);
        }
    }
    // 2. Convert each message into target type
    return map;
}




///////////////////////////////////////////////Get sizes of flat message/////////////
template<class T>
std::map<int,int> get_size_map( const std::map<int, thrust::host_vector<T>>& idx_map)
{
    std::map<int,int> out;
    for( auto& idx : idx_map)
        out[idx.first] = idx.second.size();
    return out;
}

} // namespace detail
///@endcond


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
    thrust::host_vector<int> recv(
        thrust::reduce( recvFrom.begin(), recvFrom.end()));

    auto send = detail::flatten_map( flat_vals);
    void * send_ptr          = thrust::raw_pointer_cast( send.data());
    const int * sendTo_ptr   = thrust::raw_pointer_cast( sendTo.data());
    const int * accS_ptr     = thrust::raw_pointer_cast( accS.data());
    void * recv_ptr          = thrust::raw_pointer_cast( recv.data());
    const int * recvFrom_ptr = thrust::raw_pointer_cast( recvFrom.data());
    const int * accR_ptr     = thrust::raw_pointer_cast( accR.data());
    MPI_Datatype type = dg::getMPIDataType<dg::get_value_type<decltype(send)>>();
    MPI_Alltoallv( send_ptr, sendTo_ptr,   accS_ptr, type,
                   recv_ptr, recvFrom_ptr, accR_ptr, type,
                   comm);
    return detail::make_map<MessageType>( recv, detail::make_size_map( recvFrom) );
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


} // namespace dg
