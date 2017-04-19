#pragma once

#include <cassert>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "thrust_vector_blas.cuh"

namespace dg{


///@cond

/**
 * @brief Stores the sendTo and the recvFrom maps
 */
struct Collective
{
    Collective(){}
    /**
     * @brief Construct from a map: PID -> howmanyToSend
     *
     * The size of sendTo must match the number of processes in the communicator
     * @param sendTo howmany points to send 
     * @param comm Communicator
     */
    Collective( const thrust::host_vector<int>& sendTo, MPI_Comm comm) { 
        construct( sendTo, comm);}

    void construct( const thrust::host_vector<int>& map, MPI_Comm comm){
        //sollte schnell sein
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
    void invert(){ sendTo_.swap( recvFrom_);}

    thrust::host_vector<double> scatter( const thrust::host_vector<double>& values)const;
    template<class Device>
    void scatter( const Device& values, Device& store) const;
    template<class Device>
    void gather( const Device& store, Device& values) const;
    unsigned store_size() const{ return thrust::reduce( recvFrom_.begin(), recvFrom_.end() );}
    unsigned values_size() const{ return thrust::reduce( sendTo_.begin(), sendTo_.end() );}
    MPI_Comm communicator() const{return comm_;}
    private:
    void scatter_( const thrust::host_vector<double>& values, thrust::host_vector<double>& store) const;
    void gather_( const thrust::host_vector<double>& store, thrust::host_vector<double>& values) const;
    unsigned sum;
    thrust::host_vector<int> sendTo_,   accS_;
    thrust::host_vector<int> recvFrom_, accR_;
    MPI_Comm comm_;
};

template<class Device>
void Collective::scatter( const Device& values, Device& store) const
{
    //transfer to host, then scatter and transfer result to device
    thrust::host_vector<double> hvalues, hstore(store.size());
    dg::blas1::detail::doTransfer( values, hvalues, typename VectorTraits<Device>::vector_category(), ThrustVectorTag()) ;
    scatter_( hvalues, hstore);
    dg::blas1::detail::doTransfer( hstore, store, ThrustVectorTag(), typename VectorTraits<Device>::vector_category()) ;
    thrust::copy( hstore.begin(), hstore.end(), store.begin());
}

void Collective::scatter_( const thrust::host_vector<double>& values, thrust::host_vector<double>& store) const
{
    assert( store.size() == store_size() );
    MPI_Alltoallv( const_cast<double*>(values.data()), 
                   const_cast<int*>(sendTo_.data()), 
                   const_cast<int*>(accS_.data()), MPI_DOUBLE,
                   store.data(), 
                   const_cast<int*>(recvFrom_.data()), 
                   const_cast<int*>(accR_.data()), MPI_DOUBLE, comm_);
                   //the const_cast shouldn't be necessary any more in MPI-3 standard
}

thrust::host_vector<double> Collective::scatter( const thrust::host_vector<double>& values) const 
{
    thrust::host_vector<double> received( store_size() );
    scatter_( values, received);
    return received;
}

template<class Device>
void Collective::gather( const Device& gatherFrom, Device& values) const 
{
    //transfer to host, then gather and transfer result to device
    thrust::host_vector<double> hvalues(values.size()), hgatherFrom;
    dg::blas1::detail::doTransfer( (gatherFrom), hgatherFrom, typename VectorTraits<Device>::vector_category(), ThrustVectorTag()) ;
    gather_( hgatherFrom, hvalues);
    dg::blas1::detail::doTransfer( hvalues, values, ThrustVectorTag(), typename VectorTraits<Device>::vector_category()) ;
}

void Collective::gather_( const thrust::host_vector<double>& gatherFrom, thrust::host_vector<double>& values) const 
{
    //std::cout << gatherFrom.size()<<" "<<store_size()<<std::endl;
    assert( gatherFrom.size() == store_size() );
    values.resize( values_size() );
    MPI_Alltoallv( 
            const_cast<double*>(gatherFrom.data()), 
            const_cast<int*>(recvFrom_.data()),
            const_cast<int*>(accR_.data()), MPI_DOUBLE, 
            values.data(), 
            const_cast<int*>(sendTo_.data()), 
            const_cast<int*>(accS_.data()), MPI_DOUBLE, comm_);
}
//BijectiveComm ist der Spezialfall, dass jedes Element nur ein einziges Mal gebraucht wird. 
///@endcond
//
/**
 * @ingroup mpi_structures
 * @brief Struct that performs collective scatter and gather operations across processes
 * on distributed vectors using mpi
 *
 * @code
 int i = myrank;
 double values[10] = {i,i,i,i, 9,9,9,9};
 thrust::host_vector<double> hvalues( values, values+10);
 int pids[10] =      {0,1,2,3, 0,1,2,3};
 thrust::host_vector<int> hpids( pids, pids+10);
 BijectiveComm coll( hpids, MPI_COMM_WORLD);
 thrust::host_vector<double> hrecv = coll.scatter( hvalues);
 //hrecv is now {0,9,1,9,2,9,3,9} e.g. for process 0 
 thrust::host_vector<double> hrecv2( coll.send_size());
 coll.gather( hrecv, hrecv2);
 //hrecv2 now equals hvalues independent of process rank
 @endcode
 @tparam Index an integer Vector
 @tparam Vector a Vector
 @note models aCommunicator
 */
template< class Index, class Vector>
struct BijectiveComm
{
    /**
     * @brief Construct empty class
     */
    BijectiveComm( ){ }
    /**
     * @brief Construct from a given map 
     *
     * @param pids Gives to every point of the values array the rank to which to send this data element. The rank needs to be element of the given communicator.
     * @param comm An MPI Communicator that contains the participants of the scatter/gather
     */
    BijectiveComm( thrust::host_vector<int> pids, MPI_Comm comm): idx_(pids)
    {
        int rank, size; 
        MPI_Comm_size( comm, &size);
        MPI_Comm_rank( comm, &rank);
        for( unsigned i=0; i<pids.size(); i++)
            assert( 0 <= pids[i] && pids[i] < size);
        thrust::host_vector<int> index(pids);
        thrust::sequence( index.begin(), index.end());
        thrust::host_vector<int> one( pids.size(), 1), keys(one), number(one);
        thrust::stable_sort_by_key( pids.begin(), pids.end(), index.begin());

        typedef thrust::host_vector<int>::iterator iterator;
        thrust::pair< iterator, iterator> new_end = 
            thrust::reduce_by_key( pids.begin(), pids.end(), one.begin(), 
                                                     keys.begin(), number.begin() ); 
        unsigned distance = thrust::distance( keys.begin(), new_end.first);
        thrust::host_vector<int> sendTo( size, 0 );
        for( unsigned i=0; i<distance; i++)
            sendTo[keys[i]] = number[i];
        p_.construct( sendTo, comm);
        idx_=index;
    }

    /**
     * @brief Scatters data according to the map given in the Constructor
     *
     * The order of the received elements is according to their original array index (i.e. a[0] appears before a[1]) and their process rank of origin ( i.e. values from rank 0 appear before values from rank 1)
     * @param values data to send (must have the size given 
     * by the map in the constructor, s.a. send_size())
     *
     * @return received data from other processes of size recv_size()
     * @note a scatter followed by a gather of the received values restores the original array
     */
     Vector collect( const Vector& values)const
    {
        assert( values.size() == idx_.size());
        Vector values_(values);
        //nach PID ordnen
        thrust::gather( idx_.begin(), idx_.end(), values.begin(), values_.begin());
        //senden
        Vector store( p_.store_size());
        p_.scatter( values_, store);
        return store;
    }

    /**
     * @brief Gather data according to the map given in the constructor 
     *
     * This method is the inverse of scatter 
     * @param gatherFrom other processes collect data from this vector (has to be of size given by recv_size())
     * @param values contains values from other processes sent back to the origin (must have the size of the map given in the constructor, or send_size())
     * @note a scatter followed by a gather of the received values restores the original array
     */
    void send_and_reduce( const Vector& gatherFrom, Vector& values) const
    {
        Vector values_(values.size());
        //sammeln
        p_.gather( gatherFrom, values_);
        //nach PID geordnete Werte wieder umsortieren
        thrust::scatter( values_.begin(), values_.end(), idx_.begin(), values.begin());
    }

    /**
     * @brief compute total # of elements the calling process receives in the scatter process (or sends in the gather process)
     *
     * (which might not equal the send size in each process)
     *
     * @return # of elements to receive
     */
    unsigned recv_size() const {
        return p_.store_size();}
    /**
     * @brief return # of elements the calling process has to send in a scatter process (or receive in the gather process)
     *
     * equals the size of the map given in the constructor
     * @return # of elements to send
     */
    unsigned send_size() const {
        return p_.values_size();}
    /**
    * @brief The size of the collected vector
    *
    * @return 
    */
    unsigned size() const {
        return p_.store_size();}
    /**
    * @brief The internal communicator used 
    *
    * @return MPI Communicator
    */
    MPI_Comm communicator() const {return p_.communicator();}
    private:
    Index idx_;
    Collective p_;
};


}//namespace dg
