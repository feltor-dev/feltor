#pragma once

#include <cassert>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "thrust_vector_blas.cuh"
#include "dg/blas1.h"
#include "mpi_vector.h"

namespace dg{


///@cond

/**
 * @brief engine class for mpi gather and scatter operations
 * 
 * This class takes a buffer with indices sorted according to the PID to which 
 * to send (or gather from) and connects it to an intermediate "store"
 * In this way gather and scatter are defined with respect to the buffer and 
 * the store is the vector.
 @note the data type of the Vector class has to be double
 */
template<class Index, class Vector>
struct Collective
{
    Collective(){}
    /**
     * @brief Construct from a map: PID -> howmanyToSend
     *
     * The size of sendTo must match the number of processes in the communicator
     * @param sendTo sendTo[PID] equals the number of points the calling process has to send to the given PID
     * @param comm Communicator 
     */
    Collective( const thrust::host_vector<int>& sendTo, MPI_Comm comm) { 
        construct( sendTo, comm);}

    void construct( const thrust::host_vector<int>& map, MPI_Comm comm){
        //sollte schnell sein
        thrust::host_vector<int> sendTo=map, recvFrom=sendTo; 
        comm_=comm;
        thrust::host_vector<int> accS = sendTo, accR = recvFrom;
        int rank, size; 
        MPI_Comm_rank( comm_, &rank);
        MPI_Comm_size( comm_, &size);
        assert( sendTo.size() == (unsigned)size);
        thrust::host_vector<unsigned> global( size*size);
        MPI_Allgather( sendTo.data(), size, MPI_UNSIGNED,
                       global.data(), size, MPI_UNSIGNED,
                       comm_);
        for( unsigned i=0; i<(unsigned)size; i++)
            recvFrom[i] = global[i*size+rank]; 
        thrust::exclusive_scan( sendTo.begin(),   sendTo.end(),   accS.begin());
        thrust::exclusive_scan( recvFrom.begin(), recvFrom.end(), accR.begin());
        sendTo_=sendTo, recvFrom_=recvFrom, accS_=accS, accR_=accR;
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

    void transpose(){ sendTo_.swap( recvFrom_);}
    void invert(){ sendTo_.swap( recvFrom_);}

    void scatter( const Vector& values, Vector& store) const;
    void gather( const Vector& store, Vector& values) const;
    unsigned store_size() const{ return thrust::reduce( recvFrom_.begin(), recvFrom_.end() );}
    unsigned values_size() const{ return thrust::reduce( sendTo_.begin(), sendTo_.end() );}
    MPI_Comm communicator() const{return comm_;}
    private:
    unsigned sendTo( unsigned pid) const {return sendTo_[pid];}
    unsigned recvFrom( unsigned pid) const {return recvFrom_[pid];}
    Index sendTo_,   accS_; //accumulated send
    Index recvFrom_, accR_; //accumulated recv
    MPI_Comm comm_;
};

template< class Index, class Device>
void Collective<Index, Device>::scatter( const Device& values, Device& store) const
{
    assert( store.size() == store_size() );
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    cudaDeviceSynchronize(); //needs to be called 
#endif //THRUST_DEVICE_SYSTEM
    MPI_Alltoallv( 
            thrust::raw_pointer_cast( values.data()), 
            thrust::raw_pointer_cast( sendTo_.data()), 
            thrust::raw_pointer_cast( accS_.data()), MPI_DOUBLE, 
            thrust::raw_pointer_cast( store.data()),
            thrust::raw_pointer_cast( recvFrom_.data()),
            thrust::raw_pointer_cast( accR_.data()), MPI_DOUBLE, comm_);
}

template< class Index, class Device>
void Collective<Index, Device>::gather( const Device& gatherFrom, Device& values) const 
{
    //std::cout << gatherFrom.size()<<" "<<store_size()<<std::endl;
    assert( gatherFrom.size() == store_size() );
    values.resize( values_size() );
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    cudaDeviceSynchronize(); //needs to be called 
#endif //THRUST_DEVICE_SYSTEM
    MPI_Alltoallv( 
            thrust::raw_pointer_cast( gatherFrom.data()), 
            thrust::raw_pointer_cast( recvFrom_.data()),
            thrust::raw_pointer_cast( accR_.data()), MPI_DOUBLE, 
            thrust::raw_pointer_cast( values.data()), 
            thrust::raw_pointer_cast( sendTo_.data()), 
            thrust::raw_pointer_cast( accS_.data()), MPI_DOUBLE, comm_);
}
//BijectiveComm ist der Spezialfall, dass jedes Element nur ein einziges Mal gebraucht wird. 
///@endcond
//
/**
 * @ingroup mpi_structures
 * @brief Struct that performs bijective collective scatter and gather operations across processes
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
 @tparam Vector a Vector (the data type of Vector must be double)
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
     * @brief Construct from a given map with respect to the data vector
     *
     * @param pids Gives to every point of the values/data vector (not the buffer vector!) 
     *   the rank to which to send this data element. 
     *   The rank needs to be element of the given communicator.
     * @param comm An MPI Communicator that contains the participants of the scatter/gather
     * @note The actual scatter/gather map is constructed from the given map so the result behaves as if pids was the actual scatter/gather map on the buffer
     */
    BijectiveComm( const thrust::host_vector<int>& pids, MPI_Comm comm)
    {
        construct( pids, comm);
    }

    template<class OtherIndex, class OtherVector>
    BijectiveComm( const BijectiveComm<OtherIndex, OtherVector>& src)
    {
        construct( src.get_pids(), src.communicator());
    }

    /**
     * @brief Globally gathers data into a buffer according to the map given in the Constructor
     *
     * The order of the received elements is according to their original array index 
     * (i.e. a[0] appears before a[1]) and their process rank of origin ( i.e. values from rank 0 appear before values from rank 1)
     * @param values data to send (must have the size given 
     * by the map in the constructor, s.a. size())
     *
     * @return received data from other processes of size size()
     * @note a scatter followed by a gather of the received values restores the original array
     */
    Vector global_gather( const Vector& values)const
    {
        //actually this is a scatter but we constructed it invertedly
        //we could maybe transpose the Collective object!?
        assert( values.size() == idx_.size());
        //nach PID ordnen
        thrust::gather( idx_.begin(), idx_.end(), values.begin(), values_.data()->begin());
        //senden
        Vector store( p_.store_size());
        p_.scatter( *values_.data(), store);
        return store;
    }

    /**
     * @brief Scatter data according to the map given in the constructor 
     *
     * This method is the inverse of gather
     * @param toScatter other processes collect data from this vector (has to be of size given by recv_size())
     * @param values contains values from other processes sent back to the origin (must have the size of the map given in the constructor, or send_size())
     * @note a scatter followed by a gather of the received values restores the original array
     */
    void global_scatter_reduce( const Vector& toScatter, Vector& values) const
    {
        //actually this is a gather but we constructed it invertedly
        p_.gather( toScatter, *values_.data());
        //nach PID geordnete Werte wieder umsortieren
        thrust::scatter( values_.data()->begin(), values_.data()->end(), idx_.begin(), values.begin());
    }

    /**
     * @brief compute total # of elements the calling process receives in the scatter process (or sends in the gather process)
     *
     * (which might not equal the send size in each process)
     *
     * @return # of elements to receive
     */
    unsigned size() const { return p_.store_size();}
    /**
    * @brief The internal communicator used 
    *
    * @return MPI Communicator
    */
    MPI_Comm communicator() const {return p_.communicator();}
    /**
    * @brief These are the pids that was given in the Constructor
    *
    * @return the vector given in the constructor
    */
    const thrust::host_vector<int> get_pids()const{return pids_;}
    private:
    void construct( thrust::host_vector<int> pids, MPI_Comm comm)
    {
        pids_ = pids;
        idx_.resize( pids.size());
        dg::blas1::transfer( pids, idx_);
        int rank, size; 
        MPI_Comm_size( comm, &size);
        MPI_Comm_rank( comm, &rank);
        for( unsigned i=0; i<pids.size(); i++)
            assert( 0 <= pids[i] && pids[i] < size);
        thrust::host_vector<int> index(pids);
        thrust::sequence( index.begin(), index.end());
        thrust::stable_sort_by_key( pids.begin(), pids.end(), index.begin());//note: this also sorts the pids
        idx_=index;
        //now we can repeat/invert the sort by a gather/scatter operation with index as map 
        //i.e. we could sort pids by a gather 

        //Now construct the collective object by getting the number of elements to send
        thrust::host_vector<int> one( pids.size(), 1), keys(one), number(one);
        typedef thrust::host_vector<int>::iterator iterator;
        thrust::pair< iterator, iterator> new_end = 
            thrust::reduce_by_key( pids.begin(), pids.end(), //sorted!
                one.begin(), keys.begin(), number.begin() ); 
        unsigned distance = thrust::distance( keys.begin(), new_end.first);
        thrust::host_vector<int> sendTo( size, 0 );
        for( unsigned i=0; i<distance; i++)
            sendTo[keys[i]] = number[i];
        p_.construct( sendTo, comm);
        values_.data()->resize( idx_.size());
    }
    Buffer<Vector> values_;
    Index idx_;
    Collective<Index, Vector> p_;
    thrust::host_vector<int> pids_;
};

/**
 * @ingroup mpi_structures
 * @brief Struct that performs surjective collective scatter and gather operations across processes on distributed vectors using mpi
 *
 * This Communicator can perform general global gather and
 scatter operations. We only assume that the gather/scatter map
 is surjective, i.e. all elements in a vector get gathered. This
 is important only in the global_scatter_reduce function.
 @tparam Index an integer Vector
 @tparam Vector a Vector (the data type of Vector must be double)
 @note models aCommunicator
 */
template< class Index, class Vector>
struct SurjectiveComm
{
    /**
     * @brief Construct empty class
     */
    SurjectiveComm(){}
    /**
    * @brief Construct from local indices and PIDs
    *
    * @param localGatherMap The gather map containing local vector indices
    * @param pidGatherMap The gather map containing the pids from where to gather the local index.
    Same size as localGatherMap.
     *   The rank needs to be element of the given communicator.
    * @param comm The MPI communicator participating in the scatter/gather operations
    * @note we assume that the gather map is surjective
    */
    SurjectiveComm( const thrust::host_vector<int>& localGatherMap, const thrust::host_vector<int>& pidGatherMap, MPI_Comm comm): bijectiveComm_(pidGatherMap, comm)
    {
        construct( localGatherMap, pidGatherMap, comm);
    }

    template<class OtherIndex, class OtherVector>
    SurjectiveComm( const SurjectiveComm<OtherIndex, OtherVector>& src): bijectiveComm_(src.getPidGatherMap(), src.communicator())
    {
        construct( src.getLocalGatherMap(), src.getPidGatherMap(), src.communicator());
    }

    Vector global_gather( const Vector& values)const
    {
        //gather values to store
        thrust::gather( gatherMap_.begin(), gatherMap_.end(), values.begin(), store_.data()->begin());
        //now gather from store into buffer
        Vector buffer( buffer_size_);
        bijectiveComm_.global_scatter_reduce( *store_.data(), buffer);
        return buffer;
    }
    void global_scatter_reduce( const Vector& toScatter, Vector& values)
    {
        //first gather values into store
        Vector store_t = bijectiveComm_.global_gather( toScatter);
        //now perform a local sort, reduce and scatter operation
        thrust::gather( sortMap_.begin(), sortMap_.end(), store_t.begin(), store_.data()->begin());
        thrust::reduce_by_key( sortedGatherMap_.begin(), sortedGatherMap_.end(), store_.data()->begin(), keys_.data()->begin(), values.begin());
    }
    unsigned size() const {return buffer_size_;}
    const thrust::host_vector<int> getLocalGatherMap() const {return localGatherMap_;}
    const thrust::host_vector<int> getPidGatherMap() const {return pidGatherMap_;}
    MPI_Comm communicator()const{return bijectiveComm_.communicator();}
    const Index& getSortedGatherMap() const {return sortedGatherMap_;}
    private:
    void construct( thrust::host_vector<int> localGatherMap, thrust::host_vector<int> pidGatherMap, MPI_Comm comm)
    {
        localGatherMap_ = localGatherMap, pidGatherMap_ = pidGatherMap;
        buffer_size_ = localGatherMap.size();
        assert( buffer_size_ == pidGatherMap.size());
        //the bijectiveComm behaves as if we had given the gather map for the store
        //now gather the localGatherMap from the buffer to the store to get the final gather map 
        Vector localGatherMap_d;
        dg::blas1::transfer( localGatherMap, localGatherMap_d);
        Index gatherMap = bijectiveComm_.global_gather( localGatherMap_d);
        dg::blas1::transfer(gatherMap, gatherMap_);
        store_size_ = gatherMap_.size();
        store_.data()->resize( store_size_);
        keys_.data()->resize( store_size_);

        //now prepare a reduction map and a scatter map
        thrust::host_vector<int> sortMap(gatherMap);
        thrust::sequence( sortMap.begin(), sortMap.end());
        thrust::stable_sort_by_key( gatherMap.begin(), gatherMap.end(), sortMap.begin());//note: this also sorts the gatherMap
        dg::blas1::transfer( sortMap, sortMap_);
        dg::blas1::transfer( gatherMap, sortedGatherMap_);
        //now we can repeat/invert the sort by a gather/scatter operation with sortMap as map 
    }
    unsigned buffer_size_, store_size_;
    BijectiveComm<Index, Vector> bijectiveComm_;
    Index gatherMap_; 
    Index sortMap_, sortedGatherMap_;
    Buffer<Index> keys_;
    Buffer<Vector> store_;
    thrust::host_vector<int> localGatherMap_, pidGatherMap_;
};

/**
 * @ingroup mpi_structures
 * @brief Struct that performs general collective scatter and gather operations across processes on distributed vectors using mpi
 *
 * This Communicator can perform general global gather and
 scatter operations. 
 @tparam Index an integer Vector
 @tparam Vector a Vector (the data type of Vector must be double)
 @note models aCommunicator
 */
template< class Index, class Vector>
struct GeneralComm
{
    GeneralComm(){}
    GeneralComm( const thrust::host_vector<int>& localGatherMap, const thrust::host_vector<int>& pidGatherMap, MPI_Comm comm): surjectiveComm_(localGatherMap, pidGatherMap, comm)
    {
        construct( localGatherMap, pidGatherMap, comm);
    }
    template<class OtherIndex, class OtherVector>
    GeneralComm( const GeneralComm<OtherIndex, OtherVector>& src): surjectiveComm_(src.getLocalGatherMap(), src.getPidGatherMap(), src.communicator())
    {
        construct( src.getLocalGatherMap(), src.getPidGatherMap(), src.communicator());
    }
    Vector global_gather( const Vector& values)const
    {
        return surjectiveComm_.global_gather( values);
    }
    void global_scatter_reduce( const Vector& toScatter, Vector& values)
    {
        surjectiveComm_.global_scatter_reduce( toScatter, *store_.data());
        thrust::scatter( store_.data()->begin(), store_.data()->end(), scatterMap_.begin(), values.begin());
    }

    unsigned size() const{return surjectiveComm_.size();}
    const thrust::host_vector<int> getLocalGatherMap() const {return surjectiveComm_.getLocalGatherMap();}
    const thrust::host_vector<int> getPidGatherMap() const {return surjectiveComm_.getPidGatherMap();}
    MPI_Comm communicator()const{return surjectiveComm_.communicator();}
    private:
    void construct( const thrust::host_vector<int>& localGatherMap, const thrust::host_vector<int>& pidGatherMap, MPI_Comm comm)
    {
        const Index& sortedGatherMap_ = surjectiveComm_.getSortedGatherMap();
        thrust::host_vector<int> gatherMap;
        dg::blas1::transfer( sortedGatherMap_, gatherMap);
        thrust::host_vector<int> one( gatherMap.size(), 1), keys(one), number(one);
        typedef thrust::host_vector<int>::iterator iterator;
        thrust::pair< iterator, iterator> new_end = 
            thrust::reduce_by_key( gatherMap.begin(), gatherMap.end(), //sorted!
                one.begin(), keys.begin(), number.begin() ); 
        unsigned distance = thrust::distance( keys.begin(), new_end.first);
        store_.data()->resize( distance);
        scatterMap_.resize(distance);
        thrust::copy( keys.begin(), keys.begin() + distance, scatterMap_.begin());
    }
    SurjectiveComm<Index, Vector> surjectiveComm_;
    Buffer<Vector> store_;
    Index scatterMap_;
};
}//namespace dg
