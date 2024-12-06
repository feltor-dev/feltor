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

//TODO: should we catch the cases where outer_size \in {1,2,3} in NearestNeighborComm?
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
 * @tparam container local container type. Must have a \c size() and a \c swap() member function and a specialization of the \c TensorTraits class.
 */
template<class container>
struct MPI_Vector
{
    typedef container container_type;//!< typedef to acces underlying container
    ///no data is allocated, communicators are \c MPI_COMM_NULL
    MPI_Vector(){
        m_comm = m_comm128 = m_comm128Reduce = MPI_COMM_NULL;
    }
    /**
     * @brief construct a vector
     *
     * calls \c dg::exblas::mpi_reduce_communicator() (collective call)
     * @param data internal data copy
     * @param comm MPI communicator (may not be \c MPI_COMM_NULL)
     */
    MPI_Vector( const container& data, MPI_Comm comm): m_data( data), m_comm(comm) {
        exblas::mpi_reduce_communicator( comm, &m_comm128, &m_comm128Reduce);
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
        m_data = src.data();
        m_comm = src.communicator();
        m_comm128 = src.communicator_mod();
        m_comm128Reduce = src.communicator_mod_reduce();
    }

    ///@brief Get underlying data
    ///@return read access to data
    const container& data() const {return m_data;}
    ///@brief Set underlying data
    ///@return write access to data
    container& data() {return m_data;}

    ///@brief Get the communicator to which this vector belongs
    ///@return read access to MPI communicator
    MPI_Comm communicator() const{return m_comm;}
    ///@brief Returns a communicator of fixed size 128
    MPI_Comm communicator_mod() const{return m_comm128;}

    /**
     * @brief Returns a communicator consisting of all processes with rank 0 in \c communicator_mod()
     *
     * @return returns MPI_COMM_NULL to processes not part of that group
     */
    MPI_Comm communicator_mod_reduce() const{return m_comm128Reduce;}

    /**
    * @brief Set the communicators with \c dg::exblas::mpi_reduce_communicator
    *
    * The reason why you can't just set the comm and need three parameters is
    * that generating communicators involves communication, which you might want to
    * avoid when you do it many times. So you have to call the function as
    * @code
    * MPI_Comm comm = MPI_COMM_WORLD, comm_mod, comm_mod_reduce;
    * dg::exblas::mpi_reduce_communicator( comm, &comm_mod, &comm_mod_reduce);
    * mpi_vector.set_communicator( comm, comm_mod, comm_mod_reduce);
    * @endcode
    */
    void set_communicator(MPI_Comm comm, MPI_Comm comm_mod, MPI_Comm comm_mod_reduce)
    {
        m_comm = comm;
        m_comm128 = comm_mod;
        m_comm128Reduce = comm_mod_reduce;
    }

    ///@brief Return the size of the data object
    ///@return \c data.size()
    unsigned size() const{return m_data.size();}

    ///@brief Swap data  and communicator
    ///@param src communicator and data is swapped
    void swap( MPI_Vector& src)
    {
        m_data.swap(src.m_data);
        std::swap( m_comm , src.m_comm);
        std::swap( m_comm128 , src.m_comm128);
        std::swap( m_comm128Reduce , src.m_comm128Reduce);
    }
    ///@copydetails dg::ClonePtr::swap
    friend void swap( MPI_Vector<container>& first, MPI_Vector<container>& second)
    {
        //free function as required by the std to be swappable
        //https://en.cppreference.com/w/cpp/named_req/Swappable
        //even though with move assignments std::swap also works as fast
        first.swap(second);
    }
  private:
    container m_data;
    MPI_Comm m_comm, m_comm128, m_comm128Reduce;

};

///@addtogroup traits
///@{

///@brief prototypical MPI vector
template<class container>
struct TensorTraits<MPI_Vector<container> > {
    using value_type = get_value_type<container>;
    using tensor_category = MPIVectorTag;
    using execution_policy = get_execution_policy<container>;
};
///@}

}//namespace dg
