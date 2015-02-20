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
     * The size of sendTo must match the number of processes in the communicator
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
    MPI_Alltoallv( const_cast<double*>(values.data()), sendTo_.data(), accS_.data(), MPI_DOUBLE,
                   received.data(), recvFrom_.data(), accR_.data(), MPI_DOUBLE, comm_);
    return received;
}
void Pattern::gather( const thrust::host_vector<double>& gatherFrom, thrust::host_vector<double>& values)
{
    assert( gatherFrom.size() == (unsigned)thrust::reduce( recvFrom_.begin(), recvFrom_.end()));
    values.resize( thrust::reduce( sendTo_.begin(), sendTo_.end()) );
    MPI_Alltoallv( 
            const_cast<double*>(gatherFrom.data()), recvFrom_.data(), accR_.data(), MPI_DOUBLE, 
            values.data(), sendTo_.data(), accS_.data(), MPI_DOUBLE, comm_);
}
///@endcond
//
/**
 * @ingroup mpi_structures
 * @brief Struct that performs collective scatter and gather operations across processes
 * on distributed vectors using mpi
 * @code

 int i = myrank;
 double values[10] = {i,0,i,0,i, 0,i,0,i,0};
 thrust::host_vector<double> hvalues( values, values+10);
 int pids[10] = {1,2,3,4,5, 1, 2, 3, 4, 5};
 thrust::host_vector<int> hpids( pids, pids+10);
 Collective coll( hpids, MPI_Comm_world);
 thrust::host_vector<double> hrecv = coll.scatter( hvalues);
 //hrecv is now {}
 */
struct Collective
{
    /**
     * @brief Construct empty class
     */
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

        typedef thrust::host_vector<int>::iterator iterator;
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


}//namespace dg
