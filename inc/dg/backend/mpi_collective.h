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

/**
 * @brief engine class for mpi gather and scatter operations
 *
 * This class takes a buffer with indices sorted according to the PID to which
 * to send (or gather from) and connects it to an intermediate "store"
 * In this way gather and scatter are defined with respect to the buffer and
 * the store is the vector.
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
    unsigned size() const {return values_size();}
    MPI_Comm comm() const {return comm_;}

    void transpose(){ sendTo_.swap( recvFrom_);}
    void invert(){ sendTo_.swap( recvFrom_);}

    void scatter( const Vector& values, Vector& store) const;
    void gather( const Vector& store, Vector& values) const;
    unsigned store_size() const{
        if( recvFrom_.empty()) return 0;
        return thrust::reduce( recvFrom_.begin(), recvFrom_.end() );}
    unsigned values_size() const{
        if( sendTo_.empty()) return 0;
        return thrust::reduce( sendTo_.begin(), sendTo_.end() );}
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
            thrust::raw_pointer_cast( accS_.data()), getMPIDataType<get_value_type<Device> >(),
            thrust::raw_pointer_cast( store.data()),
            thrust::raw_pointer_cast( recvFrom_.data()),
            thrust::raw_pointer_cast( accR_.data()), getMPIDataType<get_value_type<Device> >(), comm_);
}

template< class Index, class Device>
void Collective<Index, Device>::gather( const Device& gatherFrom, Device& values) const
{
    assert( gatherFrom.size() == store_size() );
    values.resize( values_size() );
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    cudaDeviceSynchronize(); //needs to be called
#endif //THRUST_DEVICE_SYSTEM
    MPI_Alltoallv(
            thrust::raw_pointer_cast( gatherFrom.data()),
            thrust::raw_pointer_cast( recvFrom_.data()),
            thrust::raw_pointer_cast( accR_.data()), getMPIDataType<get_value_type<Device> >(),
            thrust::raw_pointer_cast( values.data()),
            thrust::raw_pointer_cast( sendTo_.data()),
            thrust::raw_pointer_cast( accS_.data()), getMPIDataType<get_value_type<Device> >(), comm_);
}
//BijectiveComm ist der Spezialfall, dass jedes Element nur ein einziges Mal gebraucht wird.
///@endcond

/**
 * @ingroup mpi_structures
 * @brief Perform bijective gather and its transpose (scatter) operation across processes
 * on distributed vectors using mpi
 *
If the gather map idx[i] is bijective, each element of the source vector v maps
to exactly one location in the buffer vector w. In this case the scatter matrix S
is the inverse of G. (see \c aCommunicator for more details)
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
 * @tparam Index an integer thrust Vector
 * @tparam Vector a thrust Vector
 * @note a scatter followed by a gather of the received values restores the original array
 * @note The order of the received elements is according to their original array index
 *   (i.e. a[0] appears before a[1]) and their process rank of origin ( i.e. values from rank 0 appear before values from rank 1)
 */
template< class Index, class Vector>
struct BijectiveComm : public aCommunicator<Vector>
{
    ///@copydoc GeneralComm::GeneralComm()
    BijectiveComm( ){ }
    /**
     * @brief Construct from a given scatter map with respect to the source/data vector
     *
     * @param pids Gives to every index i of the values/data vector (not the buffer vector!)
     *   the rank pids[i] to which to send the data element data[i].
     *   The rank pids[i] needs to be element of the given communicator.
     * @param comm An MPI Communicator that contains the participants of the scatter/gather
     * @note The actual scatter/gather map is constructed from the given map so the result behaves as if pids was the actual scatter/gather map on the buffer
     */
    BijectiveComm( const thrust::host_vector<int>& pids, MPI_Comm comm) {
        construct( pids, comm);
    }
    ///@copydoc GeneralComm::GeneralComm(unsigned,const thrust::host_vector<int>&,const thrust::host_vector<int>&,MPI_Comm)
    ///@note we assume that the gather map is bijective
    BijectiveComm( unsigned local_size, thrust::host_vector<int> localGatherMap, thrust::host_vector<int> pidGatherMap, MPI_Comm comm)
    {
        construct( pidGatherMap, comm);
        p_.transpose();
    }
    ///@copydoc GeneralComm::GeneralComm(const thrust::host_vector<int>&,const ConversionPolicy&)
    ///@note we assume that the gather map is bijective
    template<class ConversionPolicy>
    BijectiveComm( const thrust::host_vector<int>& globalGatherMap, const ConversionPolicy& p)
    {
        thrust::host_vector<int> local(globalGatherMap.size()), pids(globalGatherMap.size());
        bool success = true;
        for(unsigned i=0; i<local.size(); i++)
            if( !p.global2localIdx(globalGatherMap[i], local[i], pids[i]) ) success = false;
        assert( success);
        construct( pids, p.communicator());
        p_.transpose();
    }

    ///@copydoc GeneralComm::GeneralComm(const GeneralComm<OtherIndex,OtherVector>&)
    template<class OtherIndex, class OtherVector>
    BijectiveComm( const BijectiveComm<OtherIndex, OtherVector>& src) {
        construct( src.get_pids(), src.communicator());
    }

    /**
    * @brief These are the pids that were given in the Constructor
    * @return the vector given in the constructor
    */
    const thrust::host_vector<int>& get_pids()const{return pids_;}
    virtual BijectiveComm* clone() const override final {return new BijectiveComm(*this);}
    private:
    virtual bool do_isCommunicating() const override final{
        int rank;
        MPI_Comm_rank( do_communicator(), &rank);
        bool communicating = false;
        for( unsigned i=0; i<pids_.size(); i++)
            if( pids_[i] != rank)
                communicating = true;
        return communicating;
    }
    virtual MPI_Comm do_communicator() const override final {return p_.communicator();}
    virtual unsigned do_size() const override final { return p_.store_size();}
    virtual Vector do_make_buffer()const override final{
        Vector tmp( do_size() );
        return tmp;
    }
    void construct( thrust::host_vector<int> pids, MPI_Comm comm)
    {
        this->set_local_size( pids.size());
        pids_ = pids;
        dg::assign( pids, idx_);
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
        values_.data().resize( idx_.size());
    }
    virtual void do_global_gather( const get_value_type<Vector>* values, Vector& store)const override final
    {
        //actually this is a scatter but we constructed it invertedly
        //we could maybe transpose the Collective object!?
        //assert( values.size() == idx_.size());
        //nach PID ordnen
        typename Vector::const_pointer values_ptr(values);
        thrust::gather( idx_.begin(), idx_.end(), values_ptr, values_.data().begin());
        //senden
        p_.scatter( values_.data(), store);
    }

    virtual void do_global_scatter_reduce( const Vector& toScatter, get_value_type<Vector>* values) const override final
    {
        //actually this is a gather but we constructed it invertedly
        p_.gather( toScatter, values_.data());
        typename Vector::pointer values_ptr(values);
        //nach PID geordnete Werte wieder umsortieren
        thrust::scatter( values_.data().begin(), values_.data().end(), idx_.begin(), values_ptr);
    }
    Buffer<Vector> values_;
    Index idx_;
    Collective<Index, Vector> p_;
    thrust::host_vector<int> pids_;
};

/**
 * @ingroup mpi_structures
 * @brief Perform surjective gather and its transpose (scatter) operation across processes
 * on distributed vectors using mpi
 *
 * This Communicator performs surjective global gather and
 scatter operations, which means that the gather map
 is @b surjective: If the gather map idx[i] is surjective, each element of the source vector v
maps to at least one location in the buffer vector w. This means that the scatter matrix S
can have more than one 1's in each line. (see \c aCommunicator for more details)
 Compared to \c BijectiveComm in the \c global_gather function there is an additional
 gather and in the \c global_scatter_reduce function a reduction
 needs to be performed.
 * @tparam Index an integer thrust Vector
 * @tparam Vector a thrust Vector
 */
template< class Index, class Vector>
struct SurjectiveComm : public aCommunicator<Vector>
{
    ///@copydoc GeneralComm::GeneralComm()
    SurjectiveComm(){
        buffer_size_ = store_size_ = 0;
    }
    ///@copydoc GeneralComm::GeneralComm(unsigned,const thrust::host_vector<int>&,const thrust::host_vector<int>&,MPI_Comm)
    ///@note we assume that the gather map is surjective
    SurjectiveComm( unsigned local_size, const thrust::host_vector<int>& localGatherMap, const thrust::host_vector<int>& pidGatherMap, MPI_Comm comm)
    {
        construct( local_size, localGatherMap, pidGatherMap, comm);
    }

    ///@copydoc GeneralComm::GeneralComm(const thrust::host_vector<int>&,const ConversionPolicy&)
    ///@note we assume that the gather map is surjective
    template<class ConversionPolicy>
    SurjectiveComm( const thrust::host_vector<int>& globalGatherMap, const ConversionPolicy& p)
    {
        thrust::host_vector<int> local(globalGatherMap.size()), pids(globalGatherMap.size());
        bool success = true;
        for(unsigned i=0; i<local.size(); i++)
            if( !p.global2localIdx(globalGatherMap[i], local[i], pids[i]) ) success = false;

        assert( success);
        construct( p.local_size(), local, pids, p.communicator());
    }

    ///@copydoc GeneralComm::GeneralComm(const GeneralComm<OtherIndex,OtherVector>&)
    template<class OtherIndex, class OtherVector>
    SurjectiveComm( const SurjectiveComm<OtherIndex, OtherVector>& src)
    {
        construct( src.local_size(), src.getLocalGatherMap(), src.getPidGatherMap(), src.communicator());
    }

    ///@copydoc GeneralComm::getLocalGatherMap
    const thrust::host_vector<int>& getLocalGatherMap() const {return localGatherMap_;}
    ///@copydoc GeneralComm::getPidGatherMap
    const thrust::host_vector<int>& getPidGatherMap() const {return pidGatherMap_;}
    const Index& getSortedGatherMap() const {return sortedGatherMap_;}
    virtual SurjectiveComm* clone() const override final {return new SurjectiveComm(*this);}
    private:
    virtual bool do_isCommunicating() const override final{ return bijectiveComm_.isCommunicating();}
    virtual Vector do_make_buffer()const override final{
        Vector tmp(do_size());
        return tmp;
    }
    virtual void do_global_gather( const get_value_type<Vector>* values, Vector& buffer)const override final
    {
        //gather values to store
        typename Vector::const_pointer values_ptr(values);
        thrust::gather( gatherMap_.begin(), gatherMap_.end(), values_ptr, store_.data().begin());
        bijectiveComm_.global_scatter_reduce( store_.data(), thrust::raw_pointer_cast(buffer.data()));
    }
    virtual void do_global_scatter_reduce( const Vector& toScatter, get_value_type<Vector>* values)const override final
    {
        //first gather values into store
        Vector store_t = bijectiveComm_.global_gather( thrust::raw_pointer_cast(toScatter.data()));
        //now perform a local sort, reduce and scatter operation
        thrust::gather( sortMap_.begin(), sortMap_.end(), store_t.begin(), store_.data().begin());
        typename Vector::pointer values_ptr(values);
        thrust::reduce_by_key( sortedGatherMap_.begin(), sortedGatherMap_.end(), store_.data().begin(), keys_.data().begin(), values_ptr);
    }
    virtual MPI_Comm do_communicator()const override final{return bijectiveComm_.communicator();}
    virtual unsigned do_size() const override final {return buffer_size_;}
    void construct( unsigned local_size, thrust::host_vector<int> localGatherMap, thrust::host_vector<int> pidGatherMap, MPI_Comm comm)
    {
        this->set_local_size(local_size);
        bijectiveComm_ = BijectiveComm<Index, Vector>( pidGatherMap, comm);
        localGatherMap_ = localGatherMap, pidGatherMap_ = pidGatherMap;
        buffer_size_ = localGatherMap.size();
        assert( buffer_size_ == pidGatherMap.size());
        //the bijectiveComm behaves as if we had given the gather map for the store
        //now gather the localGatherMap from the buffer to the store to get the final gather map
        Vector localGatherMap_d = dg::construct<Vector>( localGatherMap);
        const typename aCommunicator<Vector>::value_type * v_ptr = thrust::raw_pointer_cast(localGatherMap_d.data());
        Vector gatherMap_V = bijectiveComm_.global_gather( v_ptr);
        Index gatherMap_I = dg::construct<Index>(gatherMap_V);
        gatherMap_ = gatherMap_I;
        store_size_ = gatherMap_.size();
        store_.data().resize( store_size_);
        keys_.data().resize( store_size_);

        //now prepare a reduction map and a scatter map
        sortMap_ = gatherMap_I;
        thrust::sequence( sortMap_.begin(), sortMap_.end());
        thrust::stable_sort_by_key( gatherMap_I.begin(), gatherMap_I.end(), sortMap_.begin());//note: this also sorts the gatherMap
        dg::assign( gatherMap_I, sortedGatherMap_);
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
 * @brief Perform general gather and its transpose (scatter) operation across processes
 * on distributed vectors using mpi
 *
 * This Communicator can perform general global gather and
 scatter operations.
In general the gather map idx[i] might or might not map an element of
the source vector v. This means that the scatter matrix S can have one or more
empty lines. (see \c aCommunicator for more details)
 Compared to \c SurjectiveComm the \c global_scatter_reduce function needs
 to perform an additional scatter as some elements of the source vector might be left empty
 * @tparam Index an integer thrust Vector
 * @tparam Vector a thrust Vector
 */
template< class Index, class Vector>
struct GeneralComm : public aCommunicator<Vector>
{
    /// no memory allocation; size 0
    GeneralComm(){}
    /**
    * @brief Construct from local indices and PIDs gather map
    *
    * The indices in the gather map are written with respect to the buffer vector.
    * Each location in the source vector is uniquely specified by a local vector index and the process rank.
    * @param local_size local size of a \c dg::MPI_Vector (same for all processes)
    * @param localGatherMap Each element \c localGatherMap[i] represents a local vector index from where to gather the value. There are "local buffer size" elements.
    * @param pidGatherMap Each element \c pidGatherMap[i] represents the pid/rank from where to gather the corresponding local index \c localGatherMap[i].  Same size as localGatherMap.
     *   The pid/rank needs to be element of the given communicator.
    * @param comm The MPI communicator participating in the scatter/gather operations
    */
    GeneralComm( unsigned local_size, const thrust::host_vector<int>& localGatherMap, const thrust::host_vector<int>& pidGatherMap, MPI_Comm comm) {
        construct(local_size, localGatherMap, pidGatherMap, comm);
    }
    ///@brief reconstruct from another type; if src is empty, same as default constructor
    template<class OtherIndex, class OtherVector>
    GeneralComm( const GeneralComm<OtherIndex, OtherVector>& src){
        if( src.buffer_size() > 0)
            construct( src.local_size(), src.getLocalGatherMap(), src.getPidGatherMap(), src.communicator());
    }

    /**
     * @brief Construct from global indices gather map
     *
     * Uses the \c global2localIdx() member of MPITopology to generate localGatherMap and pidGatherMap
     * @tparam ConversionPolicy has to have the members:
     *  - \c bool\c global2localIdx(unsigned,unsigned&,unsigned&) \c const;
     * where the first parameter is the global index and the
     * other two are the output pair (localIdx, rank).
       return true if successful, false if global index is not part of the grid
     *  - \c MPI_Comm \c %communicator() \c const;  returns the communicator to use in the gather/scatter
     *  - \c local_size(); return the local vector size
     * @param globalGatherMap Each element \c globalGatherMap[i] represents a global vector index from where to take the value. There are "local buffer size == size()" elements.
     * @param p the conversion object
     * @sa basictopology the MPI %grids defined in Level 3 can all be used as a ConversionPolicy
     */
    template<class ConversionPolicy>
    GeneralComm( const thrust::host_vector<int>& globalGatherMap, const ConversionPolicy& p)
    {
        thrust::host_vector<int> local(globalGatherMap.size()), pids(globalGatherMap.size());
        bool success = true;
        for(unsigned i=0; i<local.size(); i++)
            if( !p.global2localIdx(globalGatherMap[i], local[i], pids[i]) ) success = false;
        assert( success);
        construct(p.local_size(), local, pids, p.communicator());
    }

    ///@brief read access to the local index gather map
    const thrust::host_vector<int>& getLocalGatherMap() const {return surjectiveComm_.getLocalGatherMap();}
    ///@brief read access to the pid gather map
    const thrust::host_vector<int>& getPidGatherMap() const {return surjectiveComm_.getPidGatherMap();}
    virtual GeneralComm* clone() const override final {return new GeneralComm(*this);}
    private:
    virtual bool do_isCommunicating() const override final{ return surjectiveComm_.isCommunicating();}
    virtual Vector do_make_buffer() const override final{
        Vector tmp(do_size());
        return tmp;
    }
    virtual MPI_Comm do_communicator()const override final{return surjectiveComm_.communicator();}
    virtual void do_global_gather( const get_value_type<Vector>* values, Vector& sink)const override final {
        surjectiveComm_.global_gather( values, sink);
    }
    virtual void do_global_scatter_reduce( const Vector& toScatter, get_value_type<Vector>* values)const override final {
        surjectiveComm_.global_scatter_reduce( toScatter, thrust::raw_pointer_cast(store_.data().data()));
        typename Vector::pointer values_ptr(values);
        dg::blas1::detail::doSubroutine_dispatch(
            get_execution_policy<Vector>(),
            this->local_size(),
            dg::equals(),
            values,
            0
        );
        thrust::scatter( store_.data().begin(), store_.data().end(), scatterMap_.begin(), values_ptr);
    }

    virtual unsigned do_size() const override final{return surjectiveComm_.buffer_size();}
    void construct( unsigned local_size, const thrust::host_vector<int>& localGatherMap, const thrust::host_vector<int>& pidGatherMap, MPI_Comm comm)
    {
        this->set_local_size( local_size);
        surjectiveComm_ = SurjectiveComm<Index,Vector>(local_size, localGatherMap, pidGatherMap, comm);

        const Index& sortedGatherMap_ = surjectiveComm_.getSortedGatherMap();
        thrust::host_vector<int> gatherMap = dg::construct<thrust::host_vector<int>>( sortedGatherMap_);
        thrust::host_vector<int> one( gatherMap.size(), 1), keys(one), number(one);
        typedef thrust::host_vector<int>::iterator iterator;
        thrust::pair< iterator, iterator> new_end =
            thrust::reduce_by_key( gatherMap.begin(), gatherMap.end(), //sorted!
                one.begin(), keys.begin(), number.begin() );
        unsigned distance = thrust::distance( keys.begin(), new_end.first);
        store_.data().resize( distance);
        scatterMap_.resize(distance);
        thrust::copy( keys.begin(), keys.begin() + distance, scatterMap_.begin());
    }
    SurjectiveComm<Index, Vector> surjectiveComm_;
    Buffer<Vector> store_;
    Index scatterMap_;
};

}//namespace dg
