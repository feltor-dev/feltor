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

//TODO should this be public?
// grid is needed for local2globalIdx and global2localIdx
// communicators need to be the same
void mpi_split( // not actually communicating
    const EllSparseBlockMat<real_type>& src,
    // src left_size and src right_size are the correct local sizes
    // src has global rows and global cols
    const ConversionPolicy& g_rows,
    const ConversionPolicy& g_cols
    EllSparseBlockMat<real_type>& inner, // local rows, local cols, correct data
    CooSparseBlockMat<real_type>& outer  // local rows, global cols, correct data
)
{
    // Indices in src are BLOCK INDICES!
    int result;
    MPI_Comm_compare( g_rows.communicator(), g_cols.communicator(), &result);
    assert( result == MPI_CONGRUENT or result == MPI_IDENT);
    int rank;
    MPI_Comm_rank( g_cols.communicator(), &rank);
    int dim, period, coord;
    MPI_Cart_get( g_cols.communicator(), 1, &dim, &period, &coord); // coord of the calling process
    if( dim == 1)
    {
        m_i = src;
        return;
    }
    int bpl = src.blocks_per_line;
    EllSparseBlockMat<real_type> inner(g_rows.local().N(), g_cols.local().N(), // number of blocks
        bpl, src.data.size()/(src.n*src.n), src.n);
    inner.set_left_size( src.left_size);
    inner.set_right_size( src.right_size);
    CooSparseBlockMat<real_type> outer(inner.num_rows, inner.num_cols, // number of blocks
        src.n, src.left_size, src.right_size);
    //first copy data elements (even though not all might be needed it doesn't slow down things either)
    inner.data = src.cata;
    outer.data = src.data;
    std::vector<bool> local_row( inner.num_rows, true);
    //now grab the right rows of the cols and data indices
    for( unsigned i=0; i<(unsigned)inner.num_rows; i++)
    for( unsigned k=0; k<(unsigned)bpl; k++)
    {
        int gIdx=0;
        assert( g_rows.local2globalIdx( i*src.n, outerrd, gIdx)); // convert from idx to block idx !
        inner.data_idx[i*bpl + k] = src.data_idx[ gIdx/src.n*bpl + k];
        inner.cols_idx[i*bpl + k] = src.cols_idx[ gIdx/src.n*bpl + k];
        // find out if row is communicating
        int gIdx = inner.cols_idx[i*bpl + k], lIdx, pid;
        assert( g_cols.global2localIdx( gIdx*src.n, lIdx, pid)); // convert from idx to block idx !
        if ( pid != rank)
            local_row = false;
    }
    // Now split into local matrix and communication matrix
    // we need to grab the entire row to ensure reproducibility (which is guaranteed by order of computations)
    thrust::host_vector<std::array<int,2>> gColIdx;
    for( unsigned i=0; i<(unsigned)inner.num_rows; i++)
    for( unsigned k=0; k<(unsigned)bpl; k++)
    {
        int gIdx = inner.cols_idx[i*bpl + k], lIdx, pid;
        assert( g_cols.global2localIdx( gIdx*src.n, lIdx, pid));
        if ( !local_row[i])
        {
            outer.add_value( i, gIdx, inner.data_idx[i*bpl + k]);
            gColIdx.push_back( lIdx/src.n, pid);
            inner.cols_idx[i*bpl + k] = EllSparseBlockMat<real_type>::invalid_idx;
        }
        else
            inner.cols_idx[i*bpl + k] = lIdx/src.n; // convert from idx to block idx !
    }
    // now, inner is ready, and outer is a Coo matrix with global columns
}
///@addtogroup mpi_structures
///@{

template<template<class > class Vector, class LocalMatrixInner, class LocalMatrixOuter = LocalMatrixInner>
struct MPISparseBlockMat
{
    MPISparseBlockMat() = default;
    MPISparseBlockMat( const LocalMatrixInner& inside,
                const LocalMatrixOuter& outside,
                const MPIKroneckerGather<Vector>& mpi_gather // only gather foreign messages
                )
    : m_i(inside), m_o(outside), m_gather(mpi_gather)
    {
    }

    template<class real_type, class ConversionPolicy>
    MPISparseBlockMat(const EllSparseBlockMat<real_type>& src,
        const ConversionPolicy& g_rows, const ConversionPolicy& g_cols)
    {
        EllSparseBlockMat<real_type> inner;
        CooSparseBlockMat<real_type> outer;
        mpi_split( src, g_rows, g_cols, inner, outer);
        // gColIdx are all the blocks we need
        thrust::host_vector<int> outer_col_idx;
        MPIKroneckerGather<Vector> kron( outer.left_size, gColIdx, outer_col_idx,
            outer.n, outer.num_cols, outer.right_size);
        outer.cols_idx = outer_col_idx;
        m_o = outer;
        // If we are contiguous we can do an additional trick which is to save
        // memory movement for self-communications
        m_self = std::vector<bool>( gColIdx.size(), false);
        if( outer.left_size == 1)
        {
            thrust::host_vector<int> gColIdx_self;
            thrust::host_vector<std::array<int,2>> gColIdx_foreign;
            for( unsigned u=0; u<gColIdx.size(); u++)
            {
                m_self.push_back( gColIdx[u][0] == rank);
                if( *self.back())
                    gColIdx_self.push_back( gColIdx[u][1]);
                else
                    gColIdx_foreign.push_back( gColIdx[u]);
            }
            thrust::host_vector<int> outer_col_idx;
            MPIKroneckerGather<Vector> kron_foreign( outer.left_size, gColIdx_foreign,
                outer_col_idx, outer.n, outer.num_cols, outer.right_size);
            m_g = kron_foreign;
        }
        else
            m_g = kron;
    }

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
        if( !m_g.isCommunicating()) //no communication needed
        {
            dg::blas2::symv( alpha, m_i, x.data(), beta, y.data());
            return;

        }
        int result;
        MPI_Comm_compare( x.communicator(), y.communicator(), &result);
        assert( result == MPI_CONGRUENT || result == MPI_IDENT);
        using value_type = dg::get_value_type<ContainerType1>;

        auto& buffer = m_buffer.typename get<value_type>( m_g.buffer_size()*m_g.chunk_size());
        auto& buffer_ptrs = m_buffer_ptrs.typename get<const value_type*>(
            m_self.size());
        auto& h_buffer_ptrs = m_h_buffer_ptrs.typename get<const value_type*>(
            m_self.size());
        if( m_dist == row_dist)
        {
            // 1 initiate communication
            m_g.global_gather_init( x.data(), buffer);
            // 2 compute inner points
            dg::blas2::symv( alpha, m_i, x.data(), beta, y.data());
            // 3 wait for communication to finish
            m_g.global_gather_wait( buffer);
            // 4.0 construct pointers into buffer and possibly x
            for( unsigned u=0; u<m_self.size(); u++)
            {
                if( m_self[u])
                {
                    h_buffer_ptrs[u] = 
                }
            }
            // 4 compute and add outer points
            dg::blas2::symv( alpha, m_o, buffer, 1, y.data());
            // TODO do we need to do sth special to avoid accessing all y!? In Sparseblockmat?
        }
        else
            throw Error( Message(_ping_)<<"symv(a,x,b,y) can only be used with a row distributed mpi matrix!");
    }
    private:
    LocalMatrixInner m_i;
    LocalMatrixOuter m_o;
    MPIKroneckerGather<Vector> m_g;
    mutable detail::AnyVector<Vector>  m_buffer, m_buffer_ptrs;
    mutable detail::AnyVector<thrust::host_vector>  m_h_buffer_ptrs;
    std::vector<bool> m_self;
};


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
        allreduce=2 //!< special distribution for partial reduction (Average)
    };
    ///@brief no memory allocation
    MPIDistMat() = default;

    /**
     * @brief Only local computations
     *
     * No communications, only the given local inner matrix will be applied
     */
    MPIDistMat( const LocalMatrixInner& m)
    : m_i(m), m_dist(row_dist), m_comm(MPI_COMM_NULL){}

    /**
    * @brief Constructor
    *
    * @param inside The local matrix for the inner elements
    * @param outside A local matrix for the elements from other processes
    * @param c The communication object
    * needs to gather values across processes. The \c global_gather_init(),
    * \c global_gather_wait(), \c buffer_size() and \c isCommunicating() member
    * functions are called for row distributed matrices.
    * If \c !isCommunicating() the gather functions won't be called and
    * only the inner matrix is applied.
    */
    MPIDistMat( const LocalMatrixInner& inside,
                const LocalMatrixOuter& outside,
                const MPIGather<Vector>& mpi_gather,
                enum dist_type dist = row_dist
                )
    : m_i(inside), m_o(outside), m_g(mpi_gather),
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
    : m_i(src.inner_matrix()), m_o( src.outer_matrix()), m_g(src.collective())
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
    const MPIGather<Vector>& mpi_gather() const{return m_g;}

    ///@brief Read access to the MPI Communicator
    MPI_Comm communicator() const{ return m_comm;}

    ///@brief Read access to distribution
    enum dist_type dist() const {return m_dist;}
    ///@brief Write access to distribution
    enum dist_type& dist() {return m_dist;}

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
        if( !m_g.isCommunicating()) //no communication needed
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

        auto& buffer = m_buffer.typename get<dg::get_value_type<ContainerType1>>( m_g.buffer_size());

        if( m_dist == row_dist)
        {
            // 1 initiate communication
            m_g.global_gather_init( x.data(), buffer);
            // 2 compute inner points
            dg::blas2::symv( m_i, x.data(), y.data());
            // 3 wait for communication to finish
            m_g.global_gather_wait( buffer);
            // 4 compute and add outer points
            dg::blas2::symv( m_o, buffer, y.data());
            // TODO do we need to do sth special to avoid accessing all y!? In Sparseblockmat?
        }
    }

    /// Stencil computations
    template<class Functor, class ContainerType1, class ContainerType2>
    void stencil( const Functor f, const ContainerType1& x, ContainerType2& y) const
    {
        //the blas2 functions should make enough static assertions on tpyes
        if( !m_g->isCommunicating()) //no communication needed
        {
            dg::blas2::stencil( f, m_i, x.data(), y.data());
            return;

        }
        int result;
        MPI_Comm_compare( x.communicator(), y.communicator(), &result);
        assert( result == MPI_CONGRUENT || result == MPI_IDENT);
        auto& buffer = m_buffer.typename get<dg::get_value_type<ContainerType1>>( m_g.buffer_size());
        if( m_dist == row_dist)
        {
            const value_type * x_ptr = thrust::raw_pointer_cast(x.data().data());
            m_g->global_gather( x_ptr, m_buffer.data());
            dg::blas2::stencil( f, m_m, m_buffer.data(), y.data());

            // 1 initiate communication
            m_g.global_gather_init( x, buffer);
            // 2 compute inner points
            dg::blas2::stencil( f, m_i, x.data(), y.data());
            // 3 wait for communication to finish
            m_g.global_gather_wait( buffer);
            // 4 compute and add outer points
            dg::blas2::stencil( f, m_o, buffer, y.data());
        }
    }

    private:
    LocalMatrixInner m_i;
    LocalMatrixOuter m_o;
    MPIGather<Vector> m_g;
    mutable detail::AnyVector<Vector>  m_buffer;
    mutable detail::AnyVector<thrust::host_vector>  m_h_buffer;
};

///@}

///@addtogroup traits
///@{
template<template<class>class V, class LI, class LO>
struct TensorTraits<MPISparseBlockMat<V, LI,LO> >
{
    using value_type = get_value_type<LI>;//!< value type
    using tensor_category = MPIMatrixTag;
};
template<template<class>class V, class LI, class LO>
struct TensorTraits<MPIDistMat<V, LI,LO> >
{
    using value_type = get_value_type<LI>;//!< value type
    using tensor_category = MPIMatrixTag;
};
///@}

} //namespace dg
