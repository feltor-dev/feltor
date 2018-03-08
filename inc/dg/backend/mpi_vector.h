#pragma once

#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include "exblas/mpi_accumulate.h"
#include "vector_traits.h"
#include "thrust_vector_blas.cuh"
#include "mpi_communicator.h"
#include "memory.h"

namespace dg
{

/**
 * @brief mpi Vector class
 *
 * @ingroup mpi_structures
 *
 * This class is a simple wrapper around a container object and an MPI_Comm.
 * The blas1 and blas2 functionality is available iff it is available for the container type.
 * We use mpi to communicate (e.g. boundary points in matrix-vector multiplications)
 * and use the existing blas functions for the local computations.
 * (At the blas level 1 communication is needed only for scalar products)
 * @tparam container local container type. Must have a \c size() and a \c swap() member function and a specialization of the \c VectorTraits class.
 */
template<class container>
struct MPI_Vector
{
    typedef container container_type;//!< typedef to acces underlying container
    ///no data is allocated, communicator is MPI_COMM_WORLD
    MPI_Vector(){ set_communicator(MPI_COMM_WORLD);}
    /**
     * @brief construct a vector
     * @param data internal data copy
     * @param comm MPI communicator
     */
    MPI_Vector( const container& data, MPI_Comm comm): data_( data) {
        set_communicator( comm);
    }

    /**
    * @brief Conversion operator
    *
    * uses conversion between compatible containers
    * @tparam OtherContainer another container class (container must be copy constructible from OtherContainer)
    * @param src the source
    */
    template<class OtherContainer>
    MPI_Vector( const MPI_Vector<OtherContainer>& src){
        data_ = src.data();
        set_communicator( src.communicator());
    }

    ///@brief Get underlying data
    ///@return read access to data
    const container& data() const {return data_;}
    ///@brief Set underlying data
    ///@return write access to data
    container& data() {return data_;}

    ///@brief Get the communicator to which this vector belongs
    ///@return read access to MPI communicator
    MPI_Comm communicator() const{return comm_;}
    ///@brief Returns a communicator of fixed size 128
    MPI_Comm communicator_mod() const{return comm128_;}

    /**
     * @brief Returns a communicator consisting of all processes with rank 0 in \c communicator_mod()
     *
     * @return returns MPI_COMM_NULL to processes not part of that group
     */
    MPI_Comm communicator_mod_reduce() const{return comm128Reduce_;}
    ///@brief Set the communicator to which this vector belongs
    void set_communicator(MPI_Comm comm){
        comm_ = comm;
        exblas::mpi_reduce_communicator( comm_, &comm128_, &comm128Reduce_);
    }

    ///@brief Return the size of the data object
    ///@return local size
    unsigned size() const{return data_.size();}

    ///@brief Swap data  and communicator
    ///@param src communicator and data is swapped
    void swap( MPI_Vector& src){
        data_.swap(src.data_);
        std::swap( comm_ , src.comm_);
        std::swap( comm128_ , src.comm128_);
        std::swap( comm128Reduce_ , src.comm128Reduce_);
    }
  private:
    container data_;
    MPI_Comm comm_, comm128_, comm128Reduce_;

};

///@addtogroup vec_list
///@{
template<class container>
struct VectorTraits<MPI_Vector<container> > {
    using value_type = typename container::value_type;
    using vector_category = MPIVectorTag;
    using execution_policy = get_execution_policy<container>;
};
template<class container>
struct VectorTraits<const MPI_Vector<container> > {
    using value_type = typename container::value_type;
    using vector_category = MPIVectorTag;
};
///@}

/////////////////////////////communicator exchanging columns//////////////////
/**
* @brief Communicator for asynchronous nearest neighbor communication
*
* exchanges a halo of given depth among neighboring processes in a given direction
* (the corresponding gather map is of general type and the communication
*  can also be modeled in \c GeneralComm, but not \c BijectiveComm or \c SurjectiveComm )
* @ingroup mpi_structures
* @tparam Index the type of index container (must be either thrust::host_vector<int> or thrust::device_vector<int>)
* @tparam Vector the vector container type must have a resize() function and work
* in the thrust library functions ( i.e. must a thrust::host_vector or thrust::device_vector)
*/
template<class Index, class Vector>
struct NearestNeighborComm
{
    typedef Vector container_type; //!< reveal local container type
    ///@brief no communication
    NearestNeighborComm(){
        silent_ = true;
    }
    /**
    * @brief Construct
    *
    * @param n size of the halo
    * @param vector_dimensions {x, y, z} dimension (total number of points)
    * @param comm the (cartesian) communicator
    * @param direction 0 is x, 1 is y, 2 is z
    */
    NearestNeighborComm( unsigned n, const unsigned vector_dimensions[3], MPI_Comm comm, unsigned direction)
    {
        construct( n, vector_dimensions, comm, direction);
    }

    /**
    * @brief Construct from other Communicator
    *
    * Simply copies halo size, dimensions, communicator and direction and
    constructs a new object
    * @tparam OtherIndex other index type
    * @tparam OtherVector other container type
    * @param src source object
    */
    template< class OtherIndex, class OtherVector>
    NearestNeighborComm( const NearestNeighborComm<OtherIndex, OtherVector>& src){
        if( src.size() == 0)  silent_=true;
        else
            construct( src.n(), src.dims(), src.communicator(), src.direction());
    }

    /**
    * @brief halo size
    * @return  halo size
    */
    unsigned n() const{return n_;}
    /**
    * @brief  The dimensionality of the input vector
    * @return dimensions ( 3)
    */
    const unsigned* dims() const{return dim_;}
    /**
    * @brief The direction of communication
    *
    * @return direction
    */
    unsigned direction() const {return direction_;}

    /**
     * @brief Allocate a buffer object of size \c size()
     * @return a buffer object on the stack
     * @note if \c size()==0 the default constructor of \c Vector is called
     */
    Vector allocate_buffer( )const{
        if( do_size() == 0 ) return Vector();
        return do_make_buffer();
    }

    /**
    * @brief Gather values from given Vector and initiate asynchronous MPI communication
    * @param values from which to gather data (it is safe to change values on return since values to communicate are copied into an internal buffer)
    * @param rqst four request variables that can be used to call MPI_Waitall
    */
    template<class container>
    void global_gather_init( const container& values, MPI_Request rqst[4])const
    {
        static_assert( std::is_base_of<SharedVectorTag, get_vector_category<container>>::value ,
                   "Only Shared vectors allowed");
        static_assert( std::is_same<get_execution_policy<container>, get_execution_policy<Vector>>::value, "Vector and container must have same execution policy!");
        static_assert( std::is_same<get_value_type<container>, get_value_type<Vector>>::value, "Vector and container must have same value type!");
        const get_value_type<container>* ptr = thrust::raw_pointer_cast( values.data());
        do_global_gather_init( get_execution_policy<container>(),  ptr, rqst);
    }
    /**
    * @brief Wait for asynchronous communication to finish and gather received data into buffer
    *
    * @param buffer (write only) where received data resides on return (must be of size \c size())
    * @param rqst the same four request variables that were used in global_gather_init
    */
    template<class container>
    void global_gather_wait(const container& input, container& buffer, MPI_Request rqst[4])const
    {
        static_assert( std::is_base_of<SharedVectorTag, get_vector_category<container>>::value ,
                   "Only Shared vectors allowed");
        static_assert( std::is_same<get_execution_policy<container>, get_execution_policy<Vector>>::value, "Vector and container must have same execution policy!");
        static_assert( std::is_same<get_value_type<container>, get_value_type<Vector>>::value, "Vector and container must have same value type!");
        get_value_type<container>* ptr = thrust::raw_pointer_cast( buffer.data());
        const get_value_type<container>* i_ptr = thrust::raw_pointer_cast( input.data());
        do_global_gather_wait( get_execution_policy<container>(), i_ptr, ptr, rqst);

    }
    ///@copydoc aCommunicator::size()
    unsigned size() const{return do_size();}
    ///@copydoc aCommunicator::isCommunicating()
    bool isCommunicating() const{
        if( do_size() == 0) return false;
        return true;
    }
    ///@copydoc aCommunicator::isCommunicating()
    MPI_Comm communicator() const{return comm_;}
    private:
    using value_type = get_value_type<Vector>;
    void do_global_gather_init( OmpTag, const value_type*, MPI_Request rqst[4])const;
    void do_global_gather_wait( OmpTag, const value_type*, value_type*, MPI_Request rqst[4])const;
    void do_global_gather_init( SerialTag, const value_type*, MPI_Request rqst[4])const;
    void do_global_gather_wait( SerialTag, const value_type*, value_type*, MPI_Request rqst[4])const;
    void do_global_gather_init( CudaTag, const value_type*, MPI_Request rqst[4])const;
    void do_global_gather_wait( CudaTag, const value_type*, value_type*, MPI_Request rqst[4])const;
    unsigned do_size()const; //size of values is size of input plus ghostcells
    Vector do_make_buffer( )const{
        Vector tmp( do_size());
        return tmp;
    }
    void construct( unsigned n, const unsigned vector_dimensions[3], MPI_Comm comm, unsigned direction);

    unsigned n_, dim_[3]; //deepness, dimensions
    MPI_Comm comm_;
    unsigned direction_;
    bool silent_;
    Index gather_map1, gather_map2, scatter_map1, scatter_map2; //buffer_size
    Index gather_map_middle, scatter_map_middle;
    Buffer<Vector> sb1, sb2, rb1, rb2;  //buffer_size
    Buffer<Vector> buffer_middle;

    void sendrecv(MPI_Request rqst[4])const;
    unsigned buffer_size() const;
    int m_source[2], m_dest[2];
};

typedef NearestNeighborComm<thrust::host_vector<int>, thrust::host_vector<double> > NNCH; //!< host Communicator for the use in an mpi matrix for derivatives

///@cond

template<class I, class V>
void NearestNeighborComm<I,V>::construct( unsigned n, const unsigned dimensions[3], MPI_Comm comm, unsigned direction)
{
    silent_=false;
    n_=n;
    dim_[0] = dimensions[0], dim_[1] = dimensions[1], dim_[2] = dimensions[2];
    comm_ = comm;
    direction_ = direction;
    {
        int ndims;
        MPI_Cartdim_get( comm, &ndims);
        int dims[ndims], periods[ndims], coords[ndims];
        MPI_Cart_get( comm, ndims, dims, periods, coords);
        if( dims[direction] == 1) silent_ = true;
    }
    //mpi_cart_shift may return MPI_PROC_NULL then the receive buffer is not modified
    MPI_Cart_shift( comm_, direction_, -1, &m_source[0], &m_dest[0]);
    MPI_Cart_shift( comm_, direction_, +1, &m_source[1], &m_dest[1]);
    assert( direction <3);
    thrust::host_vector<int> hbgather1(buffer_size()), hbgather2(hbgather1), hbscattr1(buffer_size()), hbscattr2(hbscattr1);
    thrust::host_vector<int> mid_gather( 4*buffer_size()), mid_scatter( 4*buffer_size());
    switch( direction)
    {
        case( 0):
        for( unsigned i=0; i<dim_[2]*dim_[1]; i++)
            for( unsigned j=0; j<n; j++)
            {
                hbgather1[i*n+j]        = i*dim_[0]               + j;
                mid_gather[i*4*n+0*n+j] = i*dim_[0]               + j;
                mid_gather[i*4*n+1*n+j] = i*dim_[0] + n           + j;
                mid_gather[i*4*n+2*n+j] = i*dim_[0] + dim_[0]-2*n + j;
                mid_gather[i*4*n+3*n+j] = i*dim_[0] + dim_[0]-  n + j;
                hbgather2[i*n+j]        = i*dim_[0] + dim_[0]-  n + j;
                hbscattr1[i*n+j]         = i*(6*n) + 0*n + j;
                mid_scatter[i*4*n+0*n+j] = i*(6*n) + 1*n + j;
                mid_scatter[i*4*n+1*n+j] = i*(6*n) + 2*n + j;
                mid_scatter[i*4*n+2*n+j] = i*(6*n) + 3*n + j;
                mid_scatter[i*4*n+3*n+j] = i*(6*n) + 4*n + j;
                hbscattr2[i*n+j]         = i*(6*n) + 5*n + j;
            }
        break;
        case( 1):
        for( unsigned i=0; i<dim_[2]; i++)
            for( unsigned j=0; j<n; j++)
                for( unsigned k=0; k<dim_[0]; k++)
                {
                    hbgather1[(i*n+j)*dim_[0]+k]        = (i*dim_[1] +               j)*dim_[0] + k;
                    mid_gather[(i*4*n+0*n+j)*dim_[0]+k] = (i*dim_[1]               + j)*dim_[0] + k;
                    mid_gather[(i*4*n+1*n+j)*dim_[0]+k] = (i*dim_[1] + n           + j)*dim_[0] + k;
                    mid_gather[(i*4*n+2*n+j)*dim_[0]+k] = (i*dim_[1] + dim_[1]-2*n + j)*dim_[0] + k;
                    mid_gather[(i*4*n+3*n+j)*dim_[0]+k] = (i*dim_[1] + dim_[1]-  n + j)*dim_[0] + k;
                    hbgather2[(i*n+j)*dim_[0]+k]        = (i*dim_[1] + dim_[1] - n + j)*dim_[0] + k;
                    hbscattr1[(i*n+j)*dim_[0]+k]         = (i*(6*n) + 0*n + j)*dim_[0] + k;
                    mid_scatter[(i*4*n+0*n+j)*dim_[0]+k] = (i*(6*n) + 1*n + j)*dim_[0] + k;
                    mid_scatter[(i*4*n+1*n+j)*dim_[0]+k] = (i*(6*n) + 2*n + j)*dim_[0] + k;
                    mid_scatter[(i*4*n+2*n+j)*dim_[0]+k] = (i*(6*n) + 3*n + j)*dim_[0] + k;
                    mid_scatter[(i*4*n+3*n+j)*dim_[0]+k] = (i*(6*n) + 4*n + j)*dim_[0] + k;
                    hbscattr2[(i*n+j)*dim_[0]+k]         = (i*(6*n) + 5*n + j)*dim_[0] + k;
                }
        break;
        case( 2):
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<dim_[0]*dim_[1]; j++)
            {
                hbgather1[i*dim_[0]*dim_[1]+j]            = (i               )*dim_[0]*dim_[1] + j;
                mid_gather[(i*4*n+0*n)*dim_[0]*dim_[1]+j] = (i               )*dim_[0]*dim_[1] + j;
                mid_gather[(i*4*n+1*n)*dim_[0]*dim_[1]+j] = (i + n           )*dim_[0]*dim_[1] + j;
                mid_gather[(i*4*n+2*n)*dim_[0]*dim_[1]+j] = (i + dim_[2]-2*n )*dim_[0]*dim_[1] + j;
                mid_gather[(i*4*n+3*n)*dim_[0]*dim_[1]+j] = (i + dim_[2]-  n )*dim_[0]*dim_[1] + j;
                hbgather2[i*dim_[0]*dim_[1]+j]            = (i + dim_[2]-  n )*dim_[0]*dim_[1] + j;

                hbscattr1[i*dim_[0]*dim_[1]+j]             = (i*(6*n) + 0*n)*dim_[0]*dim_[1] + j;
                mid_scatter[(i*4*n+0*n)*dim_[0]*dim_[1]+j] = (i*(6*n) + 1*n)*dim_[0]*dim_[1] + j;
                mid_scatter[(i*4*n+1*n)*dim_[0]*dim_[1]+j] = (i*(6*n) + 2*n)*dim_[0]*dim_[1] + j;
                mid_scatter[(i*4*n+2*n)*dim_[0]*dim_[1]+j] = (i*(6*n) + 3*n)*dim_[0]*dim_[1] + j;
                mid_scatter[(i*4*n+3*n)*dim_[0]*dim_[1]+j] = (i*(6*n) + 4*n)*dim_[0]*dim_[1] + j;
                hbscattr2[i*dim_[0]*dim_[1]+j]             = (i*(6*n) + 5*n)*dim_[0]*dim_[1] + j;
            }
        break;
    }
    gather_map1 =hbgather1, gather_map2 =hbgather2;
    scatter_map1=hbscattr1, scatter_map2=hbscattr2;
    gather_map_middle = mid_gather, scatter_map_middle = mid_scatter;
    sb1.data().resize( buffer_size()), sb2.data().resize( buffer_size());
    buffer_middle.data().resize( 4*buffer_size());
    rb1.data().resize( buffer_size()), rb2.data().resize( buffer_size());
}

template<class I, class V>
unsigned NearestNeighborComm<I,V>::do_size() const
{
    if( silent_) return 0;
    return 6*buffer_size(); //3 buffers on each side
}

template<class I, class V>
unsigned NearestNeighborComm<I,V>::buffer_size() const
{
    switch( direction_)
    {
        case( 0): //x-direction
            return n_*dim_[1]*dim_[2];
        case( 1): //y-direction
            return n_*dim_[0]*dim_[2];
        case( 2): //z-direction
            return n_*dim_[0]*dim_[1]; //no further n_ (hide in dim_)
        default:
            return 0;
    }
}

#ifdef _OPENMP
template<class I, class V>
void NearestNeighborComm<I,V>::do_global_gather_init( OmpTag, const value_type* input, MPI_Request rqst[4]) const
{
    unsigned size = buffer_size();
#pragma omp parallel for
    for( unsigned i=0; i<size; i++)
    {
        sb1.data()[i] = input[gather_map1[i]];
        sb2.data()[i] = input[gather_map2[i]];
    }
    //mpi sendrecv
    sendrecv( rqst);
}
template<class I, class V>
void NearestNeighborComm<I,V>::do_global_gather_wait(OmpTag, const value_type* input, value_type* values, MPI_Request rqst[4]) const
{
    unsigned size = buffer_size();
#pragma omp parallel for
    for( unsigned i=0; i<4*size; i++)
        values[scatter_map_middle[i]] = input[gather_map_middle[i]];
    MPI_Waitall( 4, rqst, MPI_STATUSES_IGNORE );
#pragma omp parallel for
    for( unsigned i=0; i<size; i++)
    {
        values[scatter_map1[i]] = rb1.data()[i];
        values[scatter_map2[i]] = rb2.data()[i];
    }
}
#endif
template<class I, class V>
void NearestNeighborComm<I,V>::do_global_gather_init( SerialTag, const value_type* input, MPI_Request rqst[4]) const
{
    unsigned size = buffer_size();
    for( unsigned i=0; i<size; i++)
    {
        sb1.data()[i] = input[gather_map1[i]];
        sb2.data()[i] = input[gather_map2[i]];
    }
    sendrecv( rqst);
}
template<class I, class V>
void NearestNeighborComm<I,V>::do_global_gather_wait( SerialTag, const value_type* input, value_type * values, MPI_Request rqst[4]) const
{
    unsigned size = buffer_size();
    for( unsigned i=0; i<4*size; i++)
        values[scatter_map_middle[i]] = input[gather_map_middle[i]];
    MPI_Waitall( 4, rqst, MPI_STATUSES_IGNORE );
    for( unsigned i=0; i<size; i++)
    {
        values[scatter_map1[i]] = rb1.data()[i];
        values[scatter_map2[i]] = rb2.data()[i];
    }
}
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
template<class I, class V>
void NearestNeighborComm<I,V>::do_global_gather_init( CudaTag, const value_type* input, MPI_Request rqst[4]) const
{
    //gather values from input into sendbuffer
    thrust::gather( thrust::cuda::tag(), gather_map1.begin(), gather_map1.end(), input, sb1.data().begin());
    thrust::gather( thrust::cuda::tag(), gather_map2.begin(), gather_map2.end(), input, sb2.data().begin());
    cudaDeviceSynchronize(); //wait until device functions are finished before sending data
    sendrecv( rqst);
}
template<class I, class V>
void NearestNeighborComm<I,V>::do_global_gather_wait( CudaTag, const value_type* input, value_type * values, MPI_Request rqst[4]) const
{
    thrust::gather( thrust::cuda::tag(), gather_map_middle.begin(), gather_map_middle.end(), input, buffer_middle.data().begin());
    thrust::scatter( thrust::cuda::tag(), buffer_middle.data().begin(), buffer_middle.data().end(), scatter_map_middle.begin(), values);
    MPI_Waitall( 4, rqst, MPI_STATUSES_IGNORE );
    //scatter received values into values array
    thrust::scatter( thrust::cuda::tag(), rb1.data().begin(), rb1.data().end(), scatter_map1.begin(), values);
    thrust::scatter( thrust::cuda::tag(), rb2.data().begin(), rb2.data().end(), scatter_map2.begin(), values);
}
#endif


template<class I, class V>
void NearestNeighborComm<I,V>::sendrecv( MPI_Request rqst[4]) const
{
    MPI_Isend( thrust::raw_pointer_cast(sb1.data().data()), buffer_size(), getMPIDataType<get_value_type<V>>(),  //sender
               m_dest[0], 3, comm_, &rqst[0]); //destination
    MPI_Irecv( thrust::raw_pointer_cast(rb2.data().data()), buffer_size(), getMPIDataType<get_value_type<V>>(), //receiver
               m_source[0], 3, comm_, &rqst[1]); //source

    MPI_Isend( thrust::raw_pointer_cast(sb2.data().data()), buffer_size(), getMPIDataType<get_value_type<V>>(),  //sender
               m_dest[1], 9, comm_, &rqst[2]);  //destination
    MPI_Irecv( thrust::raw_pointer_cast(rb1.data().data()), buffer_size(), getMPIDataType<get_value_type<V>>(), //receiver
               m_source[1], 9, comm_, &rqst[3]); //source
}


///@endcond
}//namespace dg
