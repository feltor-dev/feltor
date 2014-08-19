#pragma once

#include <cassert>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

#include "thrust/host_vector.h"

namespace dg{


///@cond

/**
 * @brief Stores the sendTo and the recvFrom maps
 */
struct Pattern
{
    Pattern(){}
    /**
     * @brief Construct from a map: PID -> howmanyToSend
     *
     * The number of points in sendTo must match the number of processes inthe communicator
     * @param sendTo howmany points to send 
     * @param comm Communicator
     */
    Pattern( const thrust::host_vector<int>& sendTo, MPI_Comm comm) { 
        construct( sendTo, comm);}
    void construct( const thrust::host_vector<int>& map, MPI_Comm comm){
        sendTo_=map, recvFrom_=sendTo_, comm_=comm;
        accS_ = sendTo_, accR_ = recvFrom_;
        int rank, size; 
        MPI_Comm_rank( comm_, &rank);
        MPI_Comm_size( comm_, &size);
        assert( sendTo_.size() == (unsigned)size);
        thrust::host_vector<unsigned> global_( size*size);
        MPI_Allgather( sendTo_.data(),  size, MPI_UNSIGNED,
                       global_.data(), size, MPI_UNSIGNED,
                       comm_);
        for( unsigned i=0; i<(unsigned)size; i++)
            recvFrom_[i] = global_[i*size+rank]; 
        thrust::exclusive_scan( sendTo_.begin(),   sendTo_.end(),   accS_.begin());
        thrust::exclusive_scan( recvFrom_.begin(), recvFrom_.end(), accR_.begin());
    }
    /**
     * @brief Number of processes in the communicator
     *
     * Same as a call to MPI_Comm_size(..)
     *
     * @return Total number of processes
     */
    unsigned size() const {return sendTo_.size();}
    MPI_Comm comm() const {return comm_;}
    /**
     * @brief Number of elements to send to process pid 
     *
     * @param pid Process ID
     *
     * @return Number
     */
    unsigned sendTo( unsigned pid) const {return sendTo_[pid];}
    /**
     * @brief Number of elements received from process pid
     *
     * @param pid Process ID
     *
     * @return Number
     */
    unsigned recvFrom( unsigned pid) const {return recvFrom_[pid];}
    /**
     * @brief swaps the send and receive maps 
     *
     * Now the pattern works backwards
     */
    void transpose(){ sendTo_.swap( recvFrom_);}
    thrust::host_vector<double> scatter( const thrust::host_vector<double>& values);
    void gather( const thrust::host_vector<double>& gatherFrom, thrust::host_vector<double>& values);
    unsigned recv_size() const{ return thrust::reduce( recvFrom_.begin(), recvFrom_.end() );}
    unsigned send_size() const{ return thrust::reduce( sendTo_.begin(), sendTo_.end() );}
    private:
    unsigned sum;
    thrust::host_vector<int> sendTo_,   accS_;
    thrust::host_vector<int> recvFrom_, accR_;
    MPI_Comm comm_;
};
thrust::host_vector<double> Pattern::scatter( const thrust::host_vector<double>& values)
{
    thrust::host_vector<double> received(thrust::reduce( recvFrom_.begin(), recvFrom_.end() ));
    MPI_Alltoallv( values.data(), sendTo_.data(), accS_.data(), MPI_DOUBLE,
                   received.data(), recvFrom_.data(), accR_.data(), MPI_DOUBLE, comm_);
    return received;
}
void Pattern::gather( const thrust::host_vector<double>& gatherFrom, thrust::host_vector<double>& values)
{
    assert( gatherFrom.size() == (unsigned)thrust::reduce( recvFrom_.begin(), recvFrom_.end()));
    values.resize( thrust::reduce( sendTo_.begin(), sendTo_.end()) );
    MPI_Alltoallv( 
            gatherFrom.data(), recvFrom_.data(), accR_.data(), MPI_DOUBLE, 
            values.data(), sendTo_.data(), accS_.data(), MPI_DOUBLE, comm_);
}
///@endcond
//
/**
 * @brief Struct that performs collective scatter and gather operations
 */
struct Collective
{
    Collective( ){}
    /**
     * @brief Construct from a given map 
     *
     * @param ranks Gives to every point of the values array the pid to which to send this data element. The pid needs to be element of the given communicator.
     * @param comm An MPI Communicator that contains the participants of the scatter/gather
     */
    Collective( thrust::host_vector<int> pids, MPI_Comm comm): idx_(pids)
    {
        int rank, size; 
        MPI_Comm_size( comm, &size);
        MPI_Comm_rank( comm, &rank);
        for( unsigned i=0; i<pids.size(); i++)
            assert( 0 <= pids[i] && pids[i] <= size);
        thrust::sequence( idx_.begin(), idx_.end());
        thrust::host_vector<int> one( pids.size(), 1), keys(one), number(one);
        thrust::stable_sort_by_key( pids.begin(), pids.end(), idx_.begin());

        typedef typename thrust::host_vector<int>::iterator iterator;
        thrust::pair< iterator, iterator> new_end = 
            thrust::reduce_by_key( pids.begin(), pids.end(), one.begin(), 
                                                     keys.begin(), number.begin() ); 
        unsigned distance = thrust::distance( keys.begin(), new_end.first);
        thrust::host_vector<int> sendTo( size, 0 );
        for( unsigned i=0; i<distance; i++)
            sendTo[keys[i]] = number[i];
        p_.construct( sendTo, comm);
    }
    /**
     * @brief Scatters data according to the map given in the Constructor
     *
     * @param values data to send (must have the size given 
     * by the map in the constructor)
     *
     * @return received data from other processes 
     * @note a scatter followed by a gather of the received values restores the original array
     */
    thrust::host_vector<double> scatter( const thrust::host_vector<double>& values)
    {
        assert( values.size() == idx_.size());
        thrust::host_vector<double> values_(values);
        thrust::gather( idx_.begin(), idx_.end(), values.begin(), values_.begin());
        thrust::host_vector<double> received = p_.scatter( values_);
        return received;
    }
    /**
     * @brief Gather data according to the map given in the constructor
     *
     * @param gatherFrom other processes collect data from this vector (has to be of size given by recv_size())
     * @param values contains values from other processes sent back to the origin (must have the size of the map given in the constructor)
     * @note a scatter followed by a gather of the received values restores the original array
     */
    void gather( const thrust::host_vector<double>& gatherFrom, thrust::host_vector<double>& values)
    {
        thrust::host_vector<double> values_;
        p_.gather( gatherFrom, values_);
        thrust::scatter( values_.begin(), values_.end(), idx_.begin(), values.begin());
    }
    unsigned recv_size() const {return p_.recv_size();}
    unsigned send_size() const {return p_.send_size();}
    private:
    thrust::host_vector<int> idx_;
    Pattern p_;
};

///@cond

/**
 * @brief Stores the send and receive buffers and manages the mapping to actual vectors
 */
//struct Buffer
//{
//    Buffer( ){}
//    Buffer( const Pattern& p){
//         p_=p;
//         construct( p);
//    }
//    void construct( const Pattern& p)
//    {
//        sendbuf_.resize( p_.size());
//        recvbuf_.resize( p_.size());
//        for( unsigned i=0; i<p_.size(); i++)
//        {
//            sendbuf_[i].resize( p_.sendTo(i));
//            recvbuf_[i].resize( p_.recvFrom(i));
//        }
//    }
//    /**
//     * @brief The total number of elements that this process spreads to other processes
//     *
//     * equals the sum of all send buffer sizes
//     * @return Total size
//     */
//    unsigned send_size() const;
//    /**
//     * @brief The total size of the vector be received 
//     *
//     * equals the sum of all recv buffer sizse, i.e. the total number of elements that are sent to this process
//     * @return Total size
//     */
//    unsigned recv_size() const;
//    /**
//     * @brief Sends data in the send buffers to the receive buffers
//     */
//    void send()
//    {
//        int rank;
//        MPI_Comm_rank( p_.comm(), &rank);
//        thrust::host_vector<MPI_Request > Sreq( p_.size());
//        thrust::host_vector<MPI_Request > Rreq( p_.size());
//        for( int i=0; i<(int)p_.size(); i++)
//        {
//            if( p_.sendTo(i) != 0)
//                MPI_Isend( sendbuf_[i].data(), p_.sendTo(i), MPI_DOUBLE, 
//                        i, rank, p_.comm(), &Sreq[i]);
//            if( p_.recvFrom(i) != 0)
//                MPI_Irecv( recvbuf_[i].data(), p_.recvFrom(i), MPI_DOUBLE,
//                        i, i, p_.comm(), &Rreq[i]);
//        }
//        for( int i=0; i<(int)p_.size(); i++)
//        {
//            if( p_.sendTo(i) != 0)
//                MPI_Wait( &Sreq[i], MPI_STATUS_IGNORE);
//            if( p_.recvFrom(i) != 0)
//                MPI_Wait( &Rreq[i], MPI_STATUS_IGNORE);
//        }
//    }
//    /**
//     * @brief Swaps send and receive buffers
//     *
//     * Now the sender will work backwards
//     */
//    void transpose() 
//    {
//        p_.transpose();
//        sendbuf_.swap( recvbuf_);
//    }
//    /**
//     * @brief Get the current data in the receive buffer
//     *
//     * @return Concatenated data of the receive buffers
//     */
//    thrust::host_vector<double> get_received( ) const;
//    /**
//     * @brief Set data to send
//     *
//     * @param v Contiguous array of data to send
//     */
//    void set_toSend( const thrust::host_vector<double>& v );
//    private:
//    Pattern p_;
//    thrust::host_vector<thrust::host_vector<double> > sendbuf_;
//    thrust::host_vector<thrust::host_vector<double> > recvbuf_;
//};
//unsigned Buffer::recv_size() const
//{
//    unsigned sum = 0;
//    for( unsigned i=0; i<p_.size(); i++)
//        sum += p_.recvFrom(i);
//    return sum;
//}
//unsigned Buffer::send_size() const
//{
//    unsigned sum = 0;
//    for( unsigned i=0; i<p_.size(); i++)
//        sum += p_.sendTo(i);
//    return sum;
//}
//
//thrust::host_vector<double> Buffer::get_received() const
//{
//    thrust::host_vector<double> v( recv_size());
//    typename thrust::host_vector<double>::iterator position = v.begin();
//    for( unsigned i=0; i<recvbuf_.size(); i++)
//    {
//        thrust::copy_n( recvbuf_[i].begin(), recvbuf_[i].size(), position);
//        position = position + recvbuf_[i].size();
//    }
//    return v;
//}
//
//void Buffer::set_toSend( const thrust::host_vector<double>& v)
//{
//    assert( v.size() == send_size());
//    typename thrust::host_vector<double>::const_iterator position = v.begin();
//    for( unsigned i=0; i<sendbuf_.size(); i++)
//    {
//        thrust::copy_n( position, sendbuf_[i].size(), sendbuf_[i].begin());
//        position = position + sendbuf_[i].size();
//    }
//}
//struct Sender
//{
//    /**
//     * @brief Construct from a given map 
//     *
//     * @param ranks Gives to every point of the values array the pid to which to send this data
//     * @param comm The Communicator 
//     */
//    Sender( const thrust::host_vector<int>& ranks, MPI_Comm comm): map_(ranks), idx_(ranks)
//    {
//        int size; 
//        MPI_Comm_size( comm, &size);
//        for( unsigned i=0; i<ranks.size(); i++)
//            assert( 0 <= ranks[i] && ranks[i] <= size);
//        thrust::sequence( idx_.begin(), idx_.end());
//        thrust::host_vector<int> one( ranks.size(), 1), keys(one), number(one);
//        thrust::stable_sort_by_key( map_.begin(), map_.end(), idx_.begin());
//        typedef typename thrust::host_vector<int>::iterator iterator;
//
//        thrust::pair< iterator, iterator> new_end = 
//            thrust::reduce_by_key( map_.begin(), map_.end(), one.begin(), 
//                                                     keys.begin(), number.begin() ); 
//        unsigned distance = 
//            thrust::distance( keys.begin(), new_end.first);
//        thrust::host_vector<int> sendTo( size, 0 );
//        for( unsigned i=0; i<distance; i++)
//            sendTo[keys[i]] = number[i];
//        Pattern p( sendTo, comm);
//        b_.construct(p);
//    }
//    unsigned recv_size() const;
//    /**
//     * @brief Scatters data according to the map given in the Constructor
//     *
//     * @param values data to send (must have the size given by the map in the constructor)
//     *
//     * @return received data from other processes 
//     * @note a scatter followed by a gather of the received values restores the original array
//     */
//    thrust::host_vector<double> scatter( const thrust::host_vector<double>& values);
//    /**
//     * @brief Gather data according to the map given in the constructor
//     *
//     * @param gatherFrom other processes collect data from this vector (has to be of size given by recv_size())
//     * @param values contains values from other processes sent back to the origin (must have the size of the map given in the constructor)
//     * @note a scatter followed by a gather of the received values restores the original array
//     */
//    void gather( const thrust::host_vector<double>& gatherFrom, thrust::host_vector<double>& values );
//    private:
//    thrust::host_vector<int> map_, idx_;
//    Buffer b_;
//};
//
//thrust::host_vector<double> Sender::scatter( const thrust::host_vector<double>& values)
//{
//    assert( values.size() == idx_.size());
//    thrust::host_vector<double> values_(values);
//    thrust::gather( values.begin(), values.end(), idx_.begin(), values_.begin());
//    b_.set_toSend( values_);
//    b_.send();
//    thrust::host_vector<double> received = b_.get_received();
//    return received;
//}
//void Sender::gather( const thrust::host_vector<double>& gatherFrom, thrust::host_vector<double>& values)
//{
//    b_.transpose(); //now b_ is in gather mode
//    assert( gatherFrom.size() == b_.send_size());
//    b_.set_toSend( gatherFrom);
//    b_.send();
//    values = b_.get_received();
//
//    thrust::host_vector<int> idx( idx_);
//    thrust::scatter( values.begin(), values.end(), idx_.begin(), values.begin());
//    b_.transpose(); //back to send mode
//}

///@endcond

}//namespace dg
