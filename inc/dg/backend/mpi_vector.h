#pragma once

#include <cassert>
#include <thrust/host_vector.h>
#include <thrust/gather.h>
#include "exceptions.h"
#include "tensor_traits.h"
#include "blas1_dispatch_shared.h"
#include "mpi_datatype.h"
#include "memory.h"
#include "config.h"

namespace dg
{

/**
 * @brief mpi Vector class
 *
 * @ingroup mpi_structures
 *
 * This class is a simple wrapper around a container object and an MPI_Comm.
 * The blas1 and blas2 functionality is available iff it is available for the
 * container type.  We use mpi to communicate (e.g. boundary points in
 * matrix-vector multiplications) and use the existing blas functions for the
 * local computations.  (At the blas level 1 communication is needed only for
 * scalar products)
 * @tparam container local container type. Must have a \c size() and a \c
 * swap() member function and a specialization of the \c TensorTraits class.
 */
template<class container>
struct MPI_Vector
{
    using container_type = container;//!< typedef to acces underlying container

    ///no data is allocated, communicator is \c MPI_COMM_NULL
    MPI_Vector()
    : m_comm ( MPI_COMM_NULL)
    {
    }
    /**
     * @brief construct a vector
     *
     * @param data internal data copy
     * @param comm MPI communicator (may not be \c MPI_COMM_NULL)
     */
    MPI_Vector( container data, MPI_Comm comm)
    : m_data( data), m_comm(comm)
    {
    }

    /**
    * @brief Conversion operator
    *
    * uses conversion between compatible containers
    * @tparam OtherContainer another container class (container must be copy
    * constructible from OtherContainer)
    * @param src the source
    */
    template<class OtherContainer>
    MPI_Vector( const MPI_Vector<OtherContainer>& src)
    : m_data( src.data()), m_comm(src.communicator())
    {
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

    /**
    * @brief Set the communicator
    *
    * @param comm new communicator to set
    */
    void set_communicator(MPI_Comm comm)
    {
        m_comm = comm;
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
    MPI_Comm m_comm;

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
