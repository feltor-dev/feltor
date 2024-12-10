#pragma once

#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include "exceptions.h"
#include "exblas/mpi_accumulate.h"
#include "tensor_traits.h"
#include "blas1_dispatch_shared.h"
#include "mpi_communicator.h"
#include "memory.h"
#include "config.h"

namespace dg
{
/////////////////////////////communicator//////////////////////////
/**
* @brief Communicator for asynchronous communication of MPISparseBlockMat
*
* Imagine a communicator with Cartesian topology and further imagine that the
* grid topology is also Cartesian (vectors form a box) in Nd dimensions.  A
* Sparseblockmat typically requires to gather slices of given index from other
* processes in a 1d communicator.  This class provides pointers to these other
* indices.  The pointers either reference data in an
* internal communication buffer (since it involves communciation to get the
* layers from neighboring processes) another buffer (if mpi communication
* requires to reorder input data) or the input vector itself (if the
* communication goes along the last dimension there is no need to reorder,
* in fact, here is the main gain we get from the pointer approach, we save
* on unnecessary data copies, which might be significant in cases where
* the communication to computation ratio is high).
* The size of the data each pointer references is the halo size, \c buffer_size()
*
* The communication is done asynchronously i.e. the user can initiate
* the communication and signal when the results are needed at a later stage.
*
* @note If the number of neighboring processes in the given direction is 1,
* the buffer size is 0 and all members return immediately.
* @note the pointers may alias each other (if the input contains less than 4 layers)
*
* @note the corresponding gather map is of general type and the communication
*  can also be modeled in \c MPIGather
* @ingroup mpi_structures
* @tparam Index the type of index container (must be either thrust::host_vector<int> or thrust::device_vector<int>)
* @tparam Buffer the container for the pointers to the buffer arrays
* @tparam Vector the vector container type must have a resize() function and work
* in the thrust library functions ( i.e. must a thrust::host_vector or thrust::device_vector)
* @sa dg::RowColDistMat
*/
template<template< typename> typename Vector>
struct KroneckerComm
{
    ///@brief no communication
    ///@param comm optional MPI communicator: the purpose is to be able to store MPI communicator even if no communication is involved in order to construct MPI_Vector with it
    KroneckerComm( MPI_Comm comm = MPI_COMM_NULL){
        m_n = 0;
        m_dim[0] = m_dim[1] = m_dim[2] = 0;
        m_comm = comm;
        m_communicating = true;
    }
    /**
    * @brief Construct
    *
    * @tparam Nd the dimensionality of the vector and the MPI Communicator
    * @param n depth of the halo
    * @param shape local # of vector elements in each direction
    * @param comm the (cartesian) communicator (must be of dimension Nd)
    * @param direction coordinate along which to exchange halo e.g. 0 is x, 1 is y, 2 is z
    */
    // TODO comm is the communicator that the pids refer to
    //
    // hyperblocks contains 1d indices {start,end,pid}
    // collapsed shape contains {left_vector_size, direction_vector_size, right_vector_size} in the Sparseblockmat sense
    // not all participating pids need to have the same number of hyperblocks or have the same shape
    // even though shape[0] and shape[2] must be equal between any two communicating pids
    // Index map must be injective i.e. vector element cannot be sent to more than one other rank
    KroneckerComm( unsigned left_size, // (local) of SparseBlockMat
        const thrust::host_vector<std::array<int,2>>& gIdx,
        const thrust::host_vector<int>& bufferIdx,
        unsigned n, // block size
        unsigned num_cols, // (local) number of block columns
        unsigned right_size, // (local) of SparseBlockMat
        MPI_Comm comm_1d)
    : m_slice_size(n*left*right), m_comm(comm_1d)
    {
        int ndims;
        MPI_Cartdim_get( comm_1d, &ndims);
        asssert( ndims == 1);
        int rank;
        MPI_Comm_rank( comm_1d, &rank);
        static_assert( std::is_base_of<SharedVectorTag,
                get_tensor_category<Vector<double>>>::value,
               "Only Shared vectors allowed");
        // Receive side
        thrust::host_vector<int> unique_gIdx, unique_pids, howmany;
        detail::gIdx2unique_idx( gIdx, bufferIdx, unique_gIdx, unique_pids,
                howmany);
        auto recvIdx = detail::make_map ( unique_gIdx, unique_pids,
                howmany);
        // receive only foreign messages
        for( unsigned u=0; u<unique_pids.size(); u++)
            if( unique_pids[u] != rank)
                m_buffer_size += howmany[u];
        // bootstrap communication pattern
        auto sendIdx = detail::recvIdx2sendIdx ( recvIdx, comm_1d,
                m_communicating);
        // now get unique send indices
        auto flat_idx = detail::flatten_map( sendIdx);
        thrust::host_vector<int> unique_lIdx, gather_map1, gather_map2;
        detail::find_unique_order_preserving( flat_idx, gather_map1,
            gather_map2, unique_lIdx, howmany);
        thrust::host_vector<int> storeIdx( flat_idx.size());
        for( unsigned u=0; u<storeIdx.size(); u++)
            storeIdx[u] = gather_map2[gather_map1[u]];

        m_contiguous = (left_size==1);
        if( not m_contiguous)
        {
            // everything becomes a z - derivative ...
            // we gather all unique indices into send buffer
            thrust::host_vector<int> g2( unique_lIdx.size()*n*left*right);
            for( unsigned l=0; l<unique_lIdx.size(); l++)
            for( unsigned j=0; j<n; j++)
            for( unsigned i=0; i<left_size; i++)
            for( unsigned k=0; k<right_size; k++)
                g2[((l*n+j)*left_size+i)*right_size + k] =
                     ((i*num_cols + unique_lIdx[l])*n + j)*right_size + k;
            m_g2 = LocalGatherMatrix<Vector>(g2);
            m_store_size = unique_lIdx.size();
            m_storeIdx = unique_lIdx;
        }
    }
    template< template<class> class OtherVector>
    friend class KroneckerComm; // enable copy

    /**
    * @brief Construct from other Communicator
    *
    * Simply copies halo size, dimensions, communicator and direction and
    constructs a new object
    * @tparam OtherIndex other index type
    * @tparam OtherVector other container type
    * @param src source object
    */
    template< template<typename > typename OtherVector>
    KroneckerComm( const KroneckerComm<OtherVector>& src){
        ...
    }

    ///@copydoc aCommunicator::communicator()
    MPI_Comm communicator() const{return m_comm;}

    /**
     * @brief Allocate a buffer object
     *
     * The buffer object is only a colletion of pointers to the actual data
     * @return a buffer object on the stack
     * @note if \c buffer_size()==0 the default constructor of \c Buffer is called
     */
    Buffer allocate_buffer( )const{
        if( buffer_size() == 0 ) return Buffer();
        return Buffer(6);
    }
    ///@copydoc aCommunicator::isCommunicating()
    bool isCommunicating() const{
        return m_communicating
    }

    /**
    * @brief Gather values from given Vector and initiate asynchronous MPI communication
    * @param input from which to gather data (it is @b unsafe to change values on return)
    * @param buffer (write only) pointers to the received data after \c global_gather_wait() was called (must be allocated by \c allocate_buffer())
    * @param rqst four request variables that can be used to call MPI_Waitall
    */
    template<class ContainerType>
    void global_gather_init( const ContainerType& input)const
    {
        using value_type = dg::get_value_type<ContainerType>;
        unsigned size = m_slice_size;
        //init pointers on host
        thrust::host_vector<const value_type*> send_ptrs(m_store_size);
        m_store.template set<value_type>( m_store_size*size);
        auto& store = m_store.template get<value_type>();
        if(m_contiguous)
        {
            for( unsigned u=0; u<send_ptrs.size(); u++)
                send_ptrs[u] = thrust::raw_pointer_cast(input.data())
                               + m_unique_lIdx[u]*size;
        }
        else
        {
            for( unsigned u=0; u<send_ptrs.size(); u++)
                send_ptrs[u] =
                    thrust::raw_pointer_cast(&store.data()[u*size]);
        }
        //copy pointers to device
        thrust::device_vector<const value_type*> d_send_ptrs = send_ptrs;
        //fill internal_buffer if not contiguous
        if( not m_contiguous)
            m_g2.gather( input, store);
        // continue here
        if constexpr (std::is_same_v< dg::get_execution_policy<ContainerType0>,
            dg::CudaTag> and !dg::cuda_aware_mpi)
        {
            m_h_buffer.template set<value_type>( m_buffer_size*size);
            m_h_store.template set<value_type>( m_store_size*size);
            // cudaMemcpy
            m_h_store.template get<value_type>() = store;
            m_rqst = global_comm_init(
                m_h_store.template get<value_type>(),
                m_h_buffer.template get<value_type>(), true);
        }
        else
            m_rqst = global_comm_init(store, buffer, true);
        sendrecv( host_ptr[1], host_ptr[4],
                  thrust::raw_pointer_cast(&m_internal_buffer.data()[0*size]), //host_ptr is const!
                  thrust::raw_pointer_cast(&m_internal_buffer.data()[5*size]), //host_ptr is const!
                  rqst);
    }
    /**
    * @brief Wait for asynchronous communication to finish and gather received data into buffer
    *
    * Calls MPI_Waitall on the \c rqst variables and may do additional cleanup. After this call returns it is safe to use data the buffer points to.
    * @param input from which to gather data (it is safe to change values on return since values to communicate are copied into \c buffer)
    * @param buffer (write only) where received data resides on return (must be allocated by \c allocate_buffer())
    * @param rqst the same four request variables that were used in global_gather_init
    */
    template<class value_type>
    Vector<const value_type*> global_gather_wait( ) const
    {
        // can be called on MPI_REQUEST_NULL
        MPI_Waitall( m_rqst.size(), &m_rqst[0], MPI_STATUSES_IGNORE );
        if constexpr (std::is_same_v< dg::get_execution_policy<ContainerType>,
            dg::CudaTag> and !dg::cuda_aware_mpi)
            buffer = m_h_buffer.template get<value_type>();
    }
    private:
    unsigned m_slice_size;
    MPI_Comm m_comm; // typically 1d
    bool m_communicating, m_contiguous=false; //silent -> no comm, m_contiguous-> comm in last dim
    LocalGatherMatrix<Vector> m_g2;

    unsigned m_store_size = 0; // in units of m_slice_size
    unsigned m_buffer_size = 0; // in units of m_slice_size
    thrust::host_vector<int> m_unique_lIdx; // store size (for contiguous)
    mutable detail::AnyVector< Vector> m_buffer, m_store;
    mutable detail::AnyVector<thrust::host_vector>  m_h_buffer, m_h_store;
    mutable std::vector<MPI_Request> m_rqst;

    void sendrecv(const_pointer_type, const_pointer_type, pointer_type, pointer_type, MPI_Request rqst[4])const;
};

///@cond



template<class V>
void KroneckerComm<V>::sendrecv( const_pointer_type sb1_ptr, const_pointer_type sb2_ptr, pointer_type rb1_ptr, pointer_type rb2_ptr, MPI_Request rqst[4]) const
{
    unsigned size = buffer_size();
#ifdef _DG_CUDA_UNAWARE_MPI
    if( std::is_same< get_execution_policy<V>, CudaTag>::value ) //could be serial tag
    {
        cudaError_t code = cudaGetLastError( );
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_host_buffer.data()[1*size]),//dst
            sb1_ptr, size*sizeof(get_value_type<V>), cudaMemcpyDeviceToHost); //src
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        code = cudaMemcpy( thrust::raw_pointer_cast(&m_internal_host_buffer.data()[4*size]),  //dst
            sb2_ptr, size*sizeof(get_value_type<V>), cudaMemcpyDeviceToHost); //src
        if( code != cudaSuccess)
            throw dg::Error(dg::Message(_ping_)<<cudaGetErrorString(code));
        sb1_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer.data()[1*size]);
        sb2_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer.data()[4*size]);
        rb1_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer.data()[0*size]);
        rb2_ptr = thrust::raw_pointer_cast(&m_internal_host_buffer.data()[5*size]);
    }
//This is a mistake if called with a host_vector
#endif
    MPI_Isend( sb1_ptr, size,
               getMPIDataType<get_value_type<V>>(),  //sender
               m_dest[0], 3, m_comm, &rqst[0]); //destination
    MPI_Irecv( rb2_ptr, size,
               getMPIDataType<get_value_type<V>>(), //receiver
               m_source[0], 3, m_comm, &rqst[1]); //source

    MPI_Isend( sb2_ptr, size,
               getMPIDataType<get_value_type<V>>(),  //sender
               m_dest[1], 9, m_comm, &rqst[2]);  //destination
    MPI_Irecv( rb1_ptr, size,
               getMPIDataType<get_value_type<V>>(), //receiver
               m_source[1], 9, m_comm, &rqst[3]); //source
}


///@endcond
}
