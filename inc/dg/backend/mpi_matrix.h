#pragma once

#include "mpi_vector.h"
#include "memory.h"
#include "timer.h"

/*!@file

@brief MPI matrix classes

@note the corresponding blas file for the Local matrix must be included before this file
*/
namespace dg {
namespace blas2 {
//forward declare blas2 symv functions
template< class MatrixType, class ContainerType1, class ContainerType2>
void symv( MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y);
template< class FunctorType, class MatrixType, class ContainerType1, class ContainerType2>
void stencil( FunctorType f, MatrixType&& M,
                  const ContainerType1& x,
                  ContainerType2& y);
template< class MatrixType, class ContainerType1, class ContainerType2>
void symv( get_value_type<ContainerType1> alpha,
                  MatrixType&& M,
                  const ContainerType1& x,
                  get_value_type<ContainerType1> beta,
                  ContainerType2& y);
}//namespace blas2

///@addtogroup mpi_structures
///@{

/**
* @brief Distributed memory matrix class, asynchronous communication
*
* The idea of this mpi matrix is to separate communication and computation in
* order to reuse existing optimized matrix formats for the computation. It can
* be expected that this works particularly well for cases in which the
* communication to computation ratio is low. This class assumes that the matrix
* and vector elements are distributed either rowwise or column-wise among mpi processes.
* The matrix elements are then further separated into rows that do not require communication
* and the ones that do, i.e.
* \f[
 M=M_i+M_o
 \f]
 where \f$ M_i\f$ is the inner matrix which requires no communication, while
 \f$ M_o\f$ is the outer matrix containing all elements which require
 communication from the Collective object.
* @tparam LocalMatrixInner The class of the matrix for local computations of the inner points.
 symv(m,x,y) needs to be callable on the container class of the MPI_Vector
* @tparam LocalMatrixOuter The class of the matrix for local computations of the outer points.
 symv(1,m,x,1,y) needs to be callable on the container class of the MPI_Vector
* @tparam Vector The storage class for internal buffers must match the execution policy
* of the containers in the symv functions
@note This class overlaps communication with computation of the inner matrix
*/
template<template<class > class Vector, class LocalMatrixInner, class LocalMatrixOuter = LocalMatrixInner>
struct MPIDistMat
{
    ///Type of distribution of MPI distributed matrices
    enum dist_type
    {
        row_dist=0, //!< Row distributed
        col_dist=1, //!< Column distributed
        allreduce=2 //!< special distribution for partial reduction (Average)
    };
    ///@brief no memory allocation
    MPIDistMat(){}

    /**
    * @brief Constructor
    *
    * @param inside The local matrix for the inner elements
    * @param outside A local matrix for the elements from other processes
    * @param c The communication object
    * needs to gather values across processes. The \c global_gather_init(),
    * \c global_gather_wait(), \c buffer_size() and \c isCommunicating() member
    * functions are called for row distributed matrices.
    * The \c global_scatter_plus_init() and \c global_scatter_plus_wait() are called
    * for column distributed matrices.
    * If \c !isCommunicating() the gather/scatters functions won't be called and
    * only the inner matrix is applied.
    */
    MPIDistMat( const LocalMatrixInner& inside,
                const LocalMatrixOuter& outside,
                const MPIGather<Vector>& mpi_gather,
                enum dist_type dist = row_dist
                )
    : m_i(inside), m_o(outside), m_c(mpi_gather),
      m_dist( dist), m_comm(mpi_gather.communicator()) { }

    /**
     * @brief Allreduce dist type
     *
     * In this mode the mpi matrix locally aplies the inner matrix and
     * calls \c MPI_Allreduce on the result
     * (outer matrix remains empty)
     */
    MPIDistMat( const LocalMatrixInner& m, MPI_Comm comm)
    : m_i(m), m_dist(allreduce), m_comm(comm){}

    /**
    * @brief Copy constructor

    * The idea is that a device matrix can be constructed by copying a host matrix.
    *
    * @tparam OtherVector Vector must be copy-constructible from OtherVector
    * @tparam OtherMatrixInner LocalMatrixInner must be copy-constructible from
    * OtherMatrixInner
    * @tparam OtherMatrixOuter LocalMatrixOuter must be copy-constructible from
    * OtherMatrixOuter
    * @param src another Matrix
    */
    template<template<class> class OtherVector, class OtherMatrixInner, class OtherMatrixOuter>
    MPIDistMat( const MPIDistMat<OtherMatrixInner, OtherMatrixOuter, OtherCollective>& src)
    : m_i(src.inner_matrix()), m_o( src.outer_matrix()), m_c(src.collective())
    m_dist( src.dist()), m_comm(src.communicator())
    { }
    ///@brief Read access to the inner matrix
    const LocalMatrixInner& inner_matrix() const{return m_i;}
    ///@brief Write access to the inner matrix
    LocalMatrixInner& inner_matrix(){return m_i;}
    ///@brief Read access to the outer matrix
    const LocalMatrixOuter& outer_matrix() const{return m_o;}
    ///@brief Write access to the outer matrix
    LocalMatrixOuter& outer_matrix(){return m_o;}
    ///@brief Read access to the communication object
    const MPIGather<Vector>& collective() const{return m_c;}

    ///@brief Read access to the MPI Communicator
    MPI_Comm communicator() const{ return m_comm;}

    ///@brief Read access to distribution
    enum dist_type dist() const {return m_dist;}
    ///@brief Write access to distribution
    enum dist_type& dist() {return m_dist;}

    /**
    * @brief Matrix Vector product
    *
    * @attention This version of symv only works for \c row_dist matrices
    *
    * First the inner elements are computed with a call to doSymv then
    * the global_gather function of the communication object is called.
    * Finally the outer elements are added with a call to doSymv for the outer matrix
    * @tparam ContainerType container class of the vector elements
    * @param alpha scalar
    * @param x input
    * @param beta scalar
    * @param y output
    */
    template<class ContainerType1, class ContainerType2>
    void symv( value_type alpha, const ContainerType1& x, value_type beta, ContainerType2& y) const
    {
        // ContainerType is MPI_Vector here...
        //the blas2 functions should make enough static assertions on tpyes
        if( !m_c.isCommunicating()) //no communication needed
        {
            dg::blas2::symv( alpha, m_i, x.data(), beta, y.data());
            return;

        }
        int result;
        MPI_Comm_compare( x.communicator(), y.communicator(), &result);
        assert( result == MPI_CONGRUENT || result == MPI_IDENT);

        auto& buffer = m_buffer.typename get<dg::get_value_type<ContainerType1>>( m_c.buffer_size());
        if( m_dist == row_dist)
        {
            // 1 initiate communication
            m_c.global_gather_init( x.data(), buffer);
            // 2 compute inner points
            dg::blas2::symv( alpha, m_i, x.data(), beta, y.data());
            // 3 wait for communication to finish
            m_c.global_gather_wait( buffer);
            // 4 compute and add outer points
            dg::blas2::symv( alpha, m_o, buffer, 1, y.data());
            // TODO do we need to do sth special to avoid accessing all y!? In Sparseblockmat?
        }
        else
            throw Error( Message(_ping_)<<"symv(a,x,b,y) can only be used with a row distributed mpi matrix!");
        // We theoretically could allow this for col_dist if the scatter_plus function in LocalGatherMatrix
        // accepted an inconsistency in "v = a S w + b v"
    }

    /**
    * @brief Matrix Vector product
    *
    * First the inner elements are computed with a call to doSymv then
    * the collect function of the communication object is called.
    * Finally the outer elements are added with a call to doSymv for the outer matrix
    * @tparam ContainerType container class of the vector elements
    * @param x input
    * @param y output
    */
    template<class ContainerType1, class ContainerType2>
    void symv( const ContainerType1& x, ContainerType2& y) const
    {
        //the blas2 functions should make enough static assertions on tpyes
        if( !m_c.isCommunicating()) //no communication needed
        {
            dg::blas2::symv( m_i, x.data(), y.data());
            if( m_dist == allreduce)
            {
                if constexpr (std::is_same_v< dg::get_execution_policy<ContainerType1>,
                    dg::CudaTag> and !dg::cuda_aware_mpi)
                {
                    using value_type = dg::get_value_type<ContainerType1>;
                    m_h_buffer.template set<value_type>( y.size());
                    m_h_buffer.template get<value_type>() = y.data();
                    MPI_Allreduce(
                        MPI_IN_PLACE,
                        thrust::raw_pointer_cast( m_h_buffer.template get<value_type>().data()),
                        y.data().size(),
                        getMPIDataType<dg::get_value_type<ContainerType2>>(),
                        MPI_SUM,
                        m_comm);
                    y.data() = m_h_buffer.template get<value_type>();
                }
                else
                {
                    MPI_Allreduce(
                        MPI_IN_PLACE,
                        thrust::raw_pointer_cast( y.data().data()),
                        y.data().size(),
                        getMPIDataType<dg::get_value_type<ContainerType2>>(),
                        MPI_SUM,
                        m_comm);
                }
            }
            return;

        }
        int result;
        MPI_Comm_compare( x.communicator(), y.communicator(), &result);
        assert( result == MPI_CONGRUENT || result == MPI_IDENT);

        auto& buffer = m_buffer.typename get<dg::get_value_type<ContainerType1>>( m_c.buffer_size());

        if( m_dist == row_dist)
        {
            // 1 initiate communication
            m_c.global_gather_init( x.data(), buffer);
            // 2 compute inner points
            dg::blas2::symv( m_i, x.data(), y.data());
            // 3 wait for communication to finish
            m_c.global_gather_wait( buffer);
            // 4 compute and add outer points
            dg::blas2::symv( m_o, buffer, y.data());
            // TODO do we need to do sth special to avoid accessing all y!? In Sparseblockmat?
        }
        else if( m_dist == col_dist)
        {
            // 1. compute outer points
            dg::blas2::symv( m_o, x.data(), buffer);
            // 2 initiate communication
            m_c.global_scatter_plus_init( buffer, y.data());
            // 3 compute inner points
            dg::blas2::symv( m_i, x.data(), y.data());
            // 4 wait for communication to finish
            m_c.global_scatter_plus_wait( y.data());
        }
    }

    /// Stencil computations
    template<class Functor, class ContainerType1, class ContainerType2>
    void stencil( const Functor f, const ContainerType1& x, ContainerType2& y) const
    {
        //the blas2 functions should make enough static assertions on tpyes
        if( !m_c->isCommunicating()) //no communication needed
        {
            dg::blas2::stencil( f, m_i, x.data(), y.data());
            return;

        }
        int result;
        MPI_Comm_compare( x.communicator(), y.communicator(), &result);
        assert( result == MPI_CONGRUENT || result == MPI_IDENT);
        auto& buffer = m_buffer.typename get<dg::get_value_type<ContainerType1>>( m_c.buffer_size());
        if( m_dist == row_dist)
        {
            const value_type * x_ptr = thrust::raw_pointer_cast(x.data().data());
            m_c->global_gather( x_ptr, m_buffer.data());
            dg::blas2::stencil( f, m_m, m_buffer.data(), y.data());

            // 1 initiate communication
            m_c.global_gather_init( x, buffer);
            // 2 compute inner points
            dg::blas2::stencil( f, m_i, x.data(), y.data());
            // 3 wait for communication to finish
            m_c.global_gather_wait( buffer);
            // 4 compute and add outer points
            dg::blas2::stencil( f, m_o, buffer, y.data());
        }
        if( m_dist == col_dist){
            throw Error( Message(_ping_)<<"stencil cannot be used with a column distributed mpi matrix!");
        }
    }

    private:
    LocalMatrixInner m_i;
    LocalMatrixOuter m_o;
    mutable MPIGather<Vector> m_gather;
    mutable dg::detail::AnyVector<Vector>  m_buffer;
#ifdef _DG_CUDA_UNAWARE_MPI // nothing serious is cuda unaware ...
    mutable detail::AnyVector<thrust::host_vector>  m_h_buffer;
#endif// _DG_CUDA_UNAWARE_MPI
};

///@}

///@addtogroup traits
///@{
template<template<class>class V, class LI, class LO>
struct TensorTraits<MPIDistMat<V, LI,LO> >
{
    using value_type = get_value_type<LI>;//!< value type
    using tensor_category = MPIMatrixTag;
};

///@}

} //namespace dg
