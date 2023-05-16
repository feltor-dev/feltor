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
    Collective(){
        m_comm = MPI_COMM_NULL;
    }
    /**
     * @brief Construct from a map: PID -> howmanyToSend
     *
     * The size of sendTo must match the number of processes in the communicator
     * @param sendTo sendTo[PID] equals the number of points the calling process has to send to the given PID
     * @param comm Communicator
     */
    Collective( const thrust::host_vector<int>& sendTo, MPI_Comm comm) {
        construct( sendTo, comm);
    }

    void construct( thrust::host_vector<int> sendTo, MPI_Comm comm){
        //sollte schnell sein
        thrust::host_vector<int> recvFrom(sendTo), accS(sendTo), accR(sendTo);
        m_comm=comm;
        int rank, size;
        MPI_Comm_rank( m_comm, &rank);
        MPI_Comm_size( m_comm, &size);
        assert( sendTo.size() == (unsigned)size);
        thrust::host_vector<unsigned> global( size*size);
        MPI_Allgather( sendTo.data(), size, MPI_UNSIGNED,
                       global.data(), size, MPI_UNSIGNED,
                       m_comm);
        for( unsigned i=0; i<(unsigned)size; i++)
            recvFrom[i] = global[i*size+rank];
        thrust::exclusive_scan( sendTo.begin(),   sendTo.end(),   accS.begin());
        thrust::exclusive_scan( recvFrom.begin(), recvFrom.end(), accR.begin());
        m_sendTo=sendTo, m_recvFrom=recvFrom, m_accS=accS, m_accR=accR;
    }
    /**
     * @brief Number of processes in the communicator
     *
     * Same as a call to MPI_Comm_size(..)
     *
     * @return Total number of processes
     */
    unsigned size() const {return values_size();}
    MPI_Comm comm() const {return m_comm;}

    void transpose(){ m_sendTo.swap( m_recvFrom);}
    void invert(){ m_sendTo.swap( m_recvFrom);}

    void scatter( const Vector& values, Vector& store) const;
    void gather( const Vector& store, Vector& values) const;
    unsigned store_size() const{
        if( m_recvFrom.empty())
            return 0;
        return thrust::reduce( m_recvFrom.begin(), m_recvFrom.end() );
    }
    unsigned values_size() const{
        if( m_sendTo.empty())
            return 0;
        return thrust::reduce( m_sendTo.begin(), m_sendTo.end() );
    }
    MPI_Comm communicator() const{return m_comm;}
    private:
    unsigned sendTo( unsigned pid) const {return m_sendTo[pid];}
    unsigned recvFrom( unsigned pid) const {return m_recvFrom[pid];}
#ifdef _DG_CUDA_UNAWARE_MPI
    thrust::host_vector<int> m_sendTo,   m_accS;
    thrust::host_vector<int> m_recvFrom, m_accR;
    dg::Buffer<thrust::host_vector<get_value_type<Vector> >> m_values, m_store;
#else
//surprisingly MPI_Alltoallv wants the integers to be on the host, only
//the data is on the device (does this question the necessity of Index?)
    thrust::host_vector<int> m_sendTo,   m_accS; //accumulated send
    thrust::host_vector<int> m_recvFrom, m_accR; //accumulated recv
#endif // _DG_CUDA_UNAWARE_MPI
    MPI_Comm m_comm;
};

template< class Index, class Device>
void Collective<Index, Device>::scatter( const Device& values, Device& store) const
{
    //assert( store.size() == store_size() );
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    if( std::is_same< get_execution_policy<Device>, CudaTag>::value ) //could be serial tag
    {
        cudaError_t code = cudaGetLastError( );
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        code = cudaDeviceSynchronize(); //needs to be called
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    }
#endif //THRUST_DEVICE_SYSTEM
#ifdef _DG_CUDA_UNAWARE_MPI
    m_values.data() = values;
    m_store.data().resize( store.size());
    MPI_Alltoallv(
            thrust::raw_pointer_cast( m_values.data().data()),
            thrust::raw_pointer_cast( m_sendTo.data()),
            thrust::raw_pointer_cast( m_accS.data()), getMPIDataType<get_value_type<Device> >(),
            thrust::raw_pointer_cast( m_store.data().data()),
            thrust::raw_pointer_cast( m_recvFrom.data()),
            thrust::raw_pointer_cast( m_accR.data()), getMPIDataType<get_value_type<Device> >(), m_comm);
    store = m_store.data();
#else
    MPI_Alltoallv(
            thrust::raw_pointer_cast( values.data()),
            thrust::raw_pointer_cast( m_sendTo.data()),
            thrust::raw_pointer_cast( m_accS.data()), getMPIDataType<get_value_type<Device> >(),
            thrust::raw_pointer_cast( store.data()),
            thrust::raw_pointer_cast( m_recvFrom.data()),
            thrust::raw_pointer_cast( m_accR.data()), getMPIDataType<get_value_type<Device> >(), m_comm);
#endif //_DG_CUDA_UNAWARE_MPI
}

template< class Index, class Device>
void Collective<Index, Device>::gather( const Device& gatherFrom, Device& values) const
{
    //assert( gatherFrom.size() == store_size() );
    values.resize( values_size() );
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    if( std::is_same< get_execution_policy<Device>, CudaTag>::value ) //could be serial tag
    {
        cudaError_t code = cudaGetLastError( );
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        code = cudaDeviceSynchronize(); //needs to be called
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
    }
#endif //THRUST_DEVICE_SYSTEM
#ifdef _DG_CUDA_UNAWARE_MPI
    m_store.data() = gatherFrom;
    m_values.data().resize( values.size());
    MPI_Alltoallv(
            thrust::raw_pointer_cast( m_store.data().data()),
            thrust::raw_pointer_cast( m_recvFrom.data()),
            thrust::raw_pointer_cast( m_accR.data()), getMPIDataType<get_value_type<Device> >(),
            thrust::raw_pointer_cast( m_values.data().data()),
            thrust::raw_pointer_cast( m_sendTo.data()),
            thrust::raw_pointer_cast( m_accS.data()), getMPIDataType<get_value_type<Device> >(), m_comm);
    values = m_values.data();
#else
    MPI_Alltoallv(
            thrust::raw_pointer_cast( gatherFrom.data()),
            thrust::raw_pointer_cast( m_recvFrom.data()),
            thrust::raw_pointer_cast( m_accR.data()), getMPIDataType<get_value_type<Device> >(),
            thrust::raw_pointer_cast( values.data()),
            thrust::raw_pointer_cast( m_sendTo.data()),
            thrust::raw_pointer_cast( m_accS.data()), getMPIDataType<get_value_type<Device> >(), m_comm);
#endif //_DG_CUDA_UNAWARE_MPI
}
//BijectiveComm ist der Spezialfall, dass jedes Element nur ein einziges Mal gebraucht wird.
///@endcond

/**
 * @ingroup mpi_structures
 * @brief Perform bijective gather and its transpose (scatter) operation across processes
 * on distributed vectors using mpi
 *
If the index map idx[i] is bijective, each element of the source vector v maps
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
 * @tparam Index an integer thrust Vector (needs to be \c int due to MPI interface)
 * @tparam Vector a thrust Vector
 * @note a scatter followed by a gather of the received values restores the original array
 * @note The order of the received elements is according to their original array index
 *   (i.e. a[0] appears before a[1]) and their process rank of origin ( i.e. values from rank 0 appear before values from rank 1)
 */
template< class Index, class Vector>
struct BijectiveComm : public aCommunicator<Vector>
{
    ///@copydoc GeneralComm::GeneralComm()
    BijectiveComm( ) = default;
    /**
     * @brief Construct from a given scatter map (inverse index map) with respect to the source/data vector
     *
     * Implicitly construct a bijective index map into the buffer vector. With only the pid map available
     * there exist in general several index maps that can fulfill the required scatter/gather of pids.
     * Which one is selected is undefined, but can be determined a posteriori through the \c getLocalIndexMap function
     * @note This constructor is useful if the only thing you care about is to which PID elements are sent, not
     * necessarily in which order the elements arrive there. This operation is then by construction bijective
     * with the size of the buffer determined to fit all elements.
     * @param pids Gives to every index i of the values/data vector (not the buffer vector!)
     *   the rank pids[i] to which to send the data element data[i].
     *   The rank pids[i] needs to be element of the given communicator.
     * @param comm An MPI Communicator that contains the participants of the scatter/gather
     * @note The actual scatter/gather map is constructed/inverted from the given map so the result behaves with scatter/gather defined wrt the buffer
     */
    BijectiveComm( const thrust::host_vector<int>& pids, MPI_Comm comm) {
        construct( pids, comm);
    }
    ///@copydoc GeneralComm::GeneralComm(unsigned,const thrust::host_vector<int>&,const thrust::host_vector<int>&,MPI_Comm)
    ///@note we assume that the index map is bijective and is given wrt the buffer
    ///@attention In fact, \c localIndexMap is ignored, only \c pidIndexMap is used. If the order of values is important, use \c SurjectiveComm
    BijectiveComm( unsigned local_size, thrust::host_vector<int> localIndexMap, thrust::host_vector<int> pidIndexMap, MPI_Comm comm)
    {
        construct( pidIndexMap, comm);
        m_p.transpose();
    }
    ///@copydoc GeneralComm::GeneralComm(const thrust::host_vector<int>&,const ConversionPolicy&)
    ///@note we assume that the index map is bijective and is given wrt the buffer
    ///@attention In fact, \c globalIndexMap is just used to produce a \c pidIndexMap, the resulting \c localIndexMap is ignored, only \c pidIndexMap is used. If the order of values is important, use \c SurjectiveComm
    template<class ConversionPolicy>
    BijectiveComm( const thrust::host_vector<int>& globalIndexMap, const ConversionPolicy& p)
    {
        thrust::host_vector<int> local(globalIndexMap.size()), pids(globalIndexMap.size());
        bool success = true;
        for(unsigned i=0; i<local.size(); i++)
            if( !p.global2localIdx(globalIndexMap[i], local[i], pids[i]) ) success = false;
        assert( success);
        construct( pids, p.communicator());
        m_p.transpose();
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
    const thrust::host_vector<int>& get_pids()const{return m_pids;}
    virtual BijectiveComm* clone() const override final {return new BijectiveComm(*this);}
    private:
    void compute_global_comm(){
        if( m_p.communicator()  == MPI_COMM_NULL){
            m_global_comm = false;
            return;
        }
        int rank;
        MPI_Comm_rank( m_p.communicator(), &rank);
        bool local_communicating = false, global_communicating=false;
        for( unsigned i=0; i<m_pids.size(); i++)
            if( m_pids[i] != rank)
                local_communicating = true;
        MPI_Allreduce( &local_communicating, &global_communicating, 1,
                       MPI_C_BOOL, MPI_LOR, m_p.communicator());
        m_global_comm = global_communicating;
    }
    virtual bool do_isCommunicating() const override final{ return m_global_comm;}
    virtual MPI_Comm do_communicator() const override final {return m_p.communicator();}
    virtual unsigned do_size() const override final { return m_p.store_size();}
    virtual Vector do_make_buffer()const override final{
        Vector tmp( do_size() );
        return tmp;
    }
    void construct( thrust::host_vector<int> pids, MPI_Comm comm)
    {
        this->set_local_size( pids.size());
        m_pids = pids;
        dg::assign( pids, m_idx);
        int size;
        MPI_Comm_size( comm, &size);
        for( unsigned i=0; i<pids.size(); i++)
            assert( 0 <= pids[i] && pids[i] < size);
        thrust::host_vector<int> index(pids);
        thrust::sequence( index.begin(), index.end());
        thrust::stable_sort_by_key( pids.begin(), pids.end(), index.begin());//note: this also sorts the pids
        m_idx=index;
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
        m_p.construct( sendTo, comm);
        m_values.data().resize( m_idx.size());
        compute_global_comm();
    }
    virtual void do_global_gather( const get_value_type<Vector>* values, Vector& store)const override final
    {
        //actually this is a scatter but we constructed it invertedly
        //we could maybe transpose the Collective object!?
        //assert( values.size() == m_idx.size());
        //nach PID ordnen
        typename Vector::const_pointer values_ptr(values);
        //senden
        if( m_global_comm)
        {
            thrust::gather( m_idx.begin(), m_idx.end(), values_ptr, m_values.data().begin());
            m_p.scatter( m_values.data(), store);
        }
        else
            thrust::gather( m_idx.begin(), m_idx.end(), values_ptr, store.begin());
    }

    virtual void do_global_scatter_reduce( const Vector& toScatter, get_value_type<Vector>* values) const override final
    {
        //actually this is a gather but we constructed it invertedly
        typename Vector::pointer values_ptr(values);
        if( m_global_comm)
        {
            m_p.gather( toScatter, m_values.data());
            //nach PID geordnete Werte wieder umsortieren
            thrust::scatter( m_values.data().begin(), m_values.data().end(), m_idx.begin(), values_ptr);
        }
        else
        {
            thrust::scatter( toScatter.begin(), toScatter.end(), m_idx.begin(), values_ptr);
        }
    }
    Buffer<Vector> m_values;
    Index m_idx;
    Collective<Index, Vector> m_p;
    thrust::host_vector<int> m_pids;
    bool m_global_comm = false;
};

/**
 * @ingroup mpi_structures
 * @brief Perform surjective gather and its transpose (scatter) operation across processes
 * on distributed vectors using mpi
 *
 * This Communicator performs surjective global gather and
 scatter operations, which means that the index map
 is @b surjective: If the index map idx[i] is surjective, each element of the source vector v
maps to at least one location in the buffer vector w. This means that the scatter matrix S
can have more than one 1's in each line. (see \c aCommunicator for more details)
 Compared to \c BijectiveComm in the \c global_gather function there is an additional
 gather and in the \c global_scatter_reduce function a reduction
 needs to be performed.
 * @tparam Index an integer thrust Vector (needs to be \c int due to MPI interface)
 * @tparam Vector a thrust Vector
 */
template< class Index, class Vector>
struct SurjectiveComm : public aCommunicator<Vector>
{
    ///@copydoc GeneralComm::GeneralComm()
    SurjectiveComm(){
        m_buffer_size = m_store_size = 0;
    }
    ///@copydoc GeneralComm::GeneralComm(unsigned,const thrust::host_vector<int>&,const thrust::host_vector<int>&,MPI_Comm)
    ///@note we assume that the index map is surjective
    SurjectiveComm( unsigned local_size, const thrust::host_vector<int>& localIndexMap, const thrust::host_vector<int>& pidIndexMap, MPI_Comm comm)
    {
        construct( local_size, localIndexMap, pidIndexMap, comm);
    }

    ///@copydoc GeneralComm::GeneralComm(const thrust::host_vector<int>&,const ConversionPolicy&)
    ///@note we assume that the index map is surjective
    template<class ConversionPolicy>
    SurjectiveComm( const thrust::host_vector<int>& globalIndexMap, const ConversionPolicy& p)
    {
        thrust::host_vector<int> local(globalIndexMap.size()), pids(globalIndexMap.size());
        bool success = true;
        for(unsigned i=0; i<local.size(); i++)
            if( !p.global2localIdx(globalIndexMap[i], local[i], pids[i]) ) success = false;

        assert( success);
        construct( p.local_size(), local, pids, p.communicator());
    }

    ///@copydoc GeneralComm::GeneralComm(const GeneralComm<OtherIndex,OtherVector>&)
    template<class OtherIndex, class OtherVector>
    SurjectiveComm( const SurjectiveComm<OtherIndex, OtherVector>& src)
    {
        construct( src.local_size(), src.getLocalIndexMap(), src.getPidIndexMap(), src.communicator());
    }

    ///@copydoc GeneralComm::getLocalIndexMap
    const thrust::host_vector<int>& getLocalIndexMap() const {return m_localIndexMap;}
    ///@copydoc GeneralComm::getPidIndexMap
    const thrust::host_vector<int>& getPidIndexMap() const {return m_pidIndexMap;}
    const Index& getSortedIndexMap() const {return m_sortedIndexMap;}
    virtual SurjectiveComm* clone() const override final {return new SurjectiveComm(*this);}
    ///No reduction on this process? True: no reduction,  False: need to reduce
    bool isLocalBijective() const {return !m_reduction;}
    private:
    virtual bool do_isCommunicating() const override final{
        return m_bijectiveComm.isCommunicating();
    }
    virtual Vector do_make_buffer()const override final{
        Vector tmp(do_size());
        return tmp;
    }
    virtual void do_global_gather( const get_value_type<Vector>* values, Vector& buffer)const override final
    {
        //gather values to store
        typename Vector::const_pointer values_ptr(values);
        thrust::gather( m_IndexMap.begin(), m_IndexMap.end(), values_ptr, m_store.data().begin());
        m_bijectiveComm.global_scatter_reduce( m_store.data(), thrust::raw_pointer_cast(buffer.data()));
    }
    virtual void do_global_scatter_reduce( const Vector& toScatter, get_value_type<Vector>* values)const override final
    {
        //now perform a local sort, reduce and scatter operation
        typename Vector::pointer values_ptr(values);
        if( m_reduction)
        {
            //first gather values into temporary store
            Vector storet = m_bijectiveComm.global_gather( thrust::raw_pointer_cast(toScatter.data()));
            thrust::gather( m_sortMap.begin(), m_sortMap.end(), storet.begin(), m_store.data().begin());
            thrust::reduce_by_key( m_sortedIndexMap.begin(), m_sortedIndexMap.end(), m_store.data().begin(), m_keys.data().begin(), values_ptr);
        }
        else
        {
            m_bijectiveComm.global_gather( thrust::raw_pointer_cast(toScatter.data()), m_store.data());
            thrust::gather( m_sortMap.begin(), m_sortMap.end(), m_store.data().begin(), values_ptr);
        }

    }
    virtual MPI_Comm do_communicator()const override final{return m_bijectiveComm.communicator();}
    virtual unsigned do_size() const override final {return m_buffer_size;}
    void construct( unsigned local_size, thrust::host_vector<int> localIndexMap, thrust::host_vector<int> pidIndexMap, MPI_Comm comm)
    {
        this->set_local_size(local_size);
        m_bijectiveComm = BijectiveComm<Index, Vector>( pidIndexMap, comm);
        m_localIndexMap = localIndexMap, m_pidIndexMap = pidIndexMap;
        m_buffer_size = localIndexMap.size();
        assert( m_buffer_size == pidIndexMap.size());
        //the bijectiveComm behaves as if we had given the index map for the store
        //now gather the localIndexMap from the buffer to the store to get the final index map
        Vector m_localIndexMapd = dg::construct<Vector>( localIndexMap);
        const typename aCommunicator<Vector>::value_type * v_ptr = thrust::raw_pointer_cast(m_localIndexMapd.data());
        Vector gatherMapV = m_bijectiveComm.global_gather( v_ptr); // a scatter wrt to the buffer
        m_sortMap = m_sortedIndexMap = m_IndexMap = dg::construct<Index>(gatherMapV);
        //now prepare a reduction map and a scatter map
        thrust::sequence( m_sortMap.begin(), m_sortMap.end());
        thrust::stable_sort_by_key( m_sortedIndexMap.begin(), m_sortedIndexMap.end(), m_sortMap.begin());//note: this sorts both keys and values (m_sortedIndexMap, m_sortMap)
        //now we can repeat/invert the sort by a gather/scatter operation with sortMap as map
        // if bijective, sortMap is the inverse of IndexMap
        m_store_size = m_IndexMap.size();
        m_store.data().resize( m_store_size);
        m_keys.data().resize( m_store_size);
        // Check if reduction is necessary
        Vector temp( m_store_size);
        auto new_end = thrust::reduce_by_key( m_sortedIndexMap.begin(), m_sortedIndexMap.end(), m_store.data().begin(), m_keys.data().begin(), temp.begin());
        if( new_end.second == temp.end())
            m_reduction = false;

    }
    unsigned m_buffer_size, m_store_size;
    BijectiveComm<Index, Vector> m_bijectiveComm;
    Index m_IndexMap, m_sortMap, m_sortedIndexMap;
    Buffer<Index> m_keys;
    Buffer<Vector> m_store;
    thrust::host_vector<int> m_localIndexMap, m_pidIndexMap;
    bool m_reduction = true;
};

/**
 * @ingroup mpi_structures
 * @brief Perform general gather and its transpose (scatter) operation across processes
 * on distributed vectors using mpi
 *
 * This Communicator can perform general global gather and
 scatter operations.
In general the index map idx[i] might or might not map an element of
the source vector v. This means that the scatter matrix S can have one or more
empty lines. (see \c aCommunicator for more details)
 Compared to \c SurjectiveComm the \c global_scatter_reduce function needs
 to perform an additional scatter as some elements of the source vector might be left empty
 * @tparam Index an integer thrust Vector (needs to be \c int due to MPI interface)
 * @tparam Vector a thrust Vector
 */
template< class Index, class Vector>
struct GeneralComm : public aCommunicator<Vector>
{
    /// no memory allocation; size 0
    GeneralComm() = default;
    /**
    * @brief Construct from local indices and PIDs index map
    *
    * The indices in the index map are written with respect to the buffer vector.
    * Each location in the source vector is uniquely specified by a local vector index and the process rank.
    * @param local_size local size of a \c dg::MPI_Vector (same for all processes)
    * @param localIndexMap Each element <tt>localIndexMap[i]</tt> represents a local vector index from (or to) where to take the value <tt>buffer[i]</tt>. There are <tt>local_buffer_size = localIndexMap.size()</tt> elements.
    * @param pidIndexMap Each element <tt>pidIndexMap[i]</tt> represents the pid/rank to which the corresponding index <tt>localIndexMap[i]</tt> is local.  Same size as \c localIndexMap.
     *   The pid/rank needs to be element of the given communicator.
    * @param comm The MPI communicator participating in the scatter/gather operations
    */
    GeneralComm( unsigned local_size, const thrust::host_vector<int>& localIndexMap, const thrust::host_vector<int>& pidIndexMap, MPI_Comm comm) {
        construct(local_size, localIndexMap, pidIndexMap, comm);
    }

    /**
     * @brief Construct from global indices index map
     *
     * Uses the \c global2localIdx() member of MPITopology to generate \c localIndexMap and \c pidIndexMap
     * @param globalIndexMap Each element <tt> globalIndexMap[i] </tt> represents a global vector index from (or to) where to take the value <tt>buffer[i]</tt>. There are <tt> local_buffer_size = globalIndexMap.size() </tt> elements.
     * @param p the conversion object
     * @tparam ConversionPolicy has to have the members:
     *  - <tt> bool global2localIdx(unsigned,unsigned&,unsigned&) const;</tt>
     * where the first parameter is the global index and the
     * other two are the output pair (localIdx, rank).
       return true if successful, false if global index is not part of the grid
     *  - <tt> MPI_Comm %communicator() const;</tt>  returns the communicator to use in the gather/scatter
     *  - <tt> local_size(); </tt> return the local vector size
     * @sa basictopology the MPI %grids defined in Level 3 can all be used as a ConversionPolicy
     */
    template<class ConversionPolicy>
    GeneralComm( const thrust::host_vector<int>& globalIndexMap, const ConversionPolicy& p)
    {
        thrust::host_vector<int> local(globalIndexMap.size()), pids(globalIndexMap.size());
        bool success = true;
        for(unsigned i=0; i<local.size(); i++)
            if( !p.global2localIdx(globalIndexMap[i], local[i], pids[i]) ) success = false;
        assert( success);
        construct(p.local_size(), local, pids, p.communicator());
    }

    ///@brief reconstruct from another type; if src is empty, same as default constructor
    template<class OtherIndex, class OtherVector>
    GeneralComm( const GeneralComm<OtherIndex, OtherVector>& src){
        if( src.buffer_size() > 0)
            construct( src.local_size(), src.getLocalIndexMap(), src.getPidIndexMap(), src.communicator());
    }

    ///@brief read access to the local index index map given in constructor
    const thrust::host_vector<int>& getLocalIndexMap() const {return m_surjectiveComm.getLocalIndexMap();}
    ///@brief read access to the pid index map given in constructor
    const thrust::host_vector<int>& getPidIndexMap() const {return m_surjectiveComm.getPidIndexMap();}
    virtual GeneralComm* clone() const override final {return new GeneralComm(*this);}
    private:
    virtual bool do_isCommunicating() const override final{
        return m_surjectiveComm.isCommunicating();
    }
    virtual Vector do_make_buffer() const override final{
        Vector tmp(do_size());
        return tmp;
    }
    virtual MPI_Comm do_communicator()const override final{return m_surjectiveComm.communicator();}
    virtual void do_global_gather( const get_value_type<Vector>* values, Vector& sink)const override final {
        m_surjectiveComm.global_gather( values, sink);
    }
    virtual void do_global_scatter_reduce( const Vector& toScatter, get_value_type<Vector>* values)const override final {
        m_surjectiveComm.global_scatter_reduce( toScatter, thrust::raw_pointer_cast(m_store.data().data()));
        typename Vector::pointer values_ptr(values);
        dg::blas1::detail::doSubroutine_dispatch(
            get_execution_policy<Vector>(),
            this->local_size(),
            dg::equals(),
            0,
            values
        );
        thrust::scatter( m_store.data().begin(), m_store.data().end(), m_scatterMap.begin(), values_ptr);
    }

    virtual unsigned do_size() const override final{return m_surjectiveComm.buffer_size();}
    void construct( unsigned local_size, const thrust::host_vector<int>& localIndexMap, const thrust::host_vector<int>& pidIndexMap, MPI_Comm comm)
    {
        this->set_local_size( local_size);
        m_surjectiveComm = SurjectiveComm<Index,Vector>(local_size, localIndexMap, pidIndexMap, comm);

        const Index& m_sortedIndexMap = m_surjectiveComm.getSortedIndexMap();
        thrust::host_vector<int> gatherMap = dg::construct<thrust::host_vector<int>>( m_sortedIndexMap);
        thrust::host_vector<int> one( gatherMap.size(), 1), keys(one), number(one);
        typedef thrust::host_vector<int>::iterator iterator;
        thrust::pair< iterator, iterator> new_end =
            thrust::reduce_by_key( gatherMap.begin(), gatherMap.end(), //sorted!
                one.begin(), keys.begin(), number.begin() );
        unsigned distance = thrust::distance( keys.begin(), new_end.first);
        m_store.data().resize( distance);
        m_scatterMap.resize(distance);
        thrust::copy( keys.begin(), keys.begin() + distance, m_scatterMap.begin());
    }
    SurjectiveComm<Index, Vector> m_surjectiveComm;
    Buffer<Vector> m_store;
    Index m_scatterMap;
};

}//namespace dg
