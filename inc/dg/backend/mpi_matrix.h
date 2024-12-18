#pragma once

#include "mpi_gather_kron.h"
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

template<template<class > class Vector, class LocalMatrixInner, class LocalMatrixOuter = LocalMatrixInner>
struct MPISparseBlockMat
{
    MPISparseBlockMat() = default;
    MPISparseBlockMat( const LocalMatrixInner& inside,
                const LocalMatrixOuter& outside,
                const MPIKroneckerGather<Vector>& mpi_gather // only gather foreign messages
                )
    : m_i(inside), m_o(outside), m_g(mpi_gather)
    {
    }
    template< template<class> class V, class LI, class LO>
    friend class MPISparseBlockMat; // enable copy

    template< template<class> class V, class LI, class LO>
    MPISparseBlockMat( const MPISparseBlockMat<V,LI,LO>& src)
    : m_i(src.m_i), m_o(src.m_o), m_g(src.m_g)
    {}

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
    void symv( dg::get_value_type<ContainerType1> alpha, const ContainerType1&
        x, dg::get_value_type<ContainerType1> beta, ContainerType2& y) const
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

        m_buffer_ptrs.template set<const value_type*>( m_o.num_cols);
        auto& buffer_ptrs = m_buffer_ptrs.template get<const value_type*>();
        // 1 initiate communication
        m_g.global_gather_init( x.data());
        // 2 compute inner points
        dg::blas2::symv( alpha, m_i, x.data(), beta, y.data());
        // 3 wait for communication to finish
        m_g.global_gather_wait( x.data(), buffer_ptrs);
        // 4 compute and add outer points
        const value_type** b_ptrs = thrust::raw_pointer_cast( buffer_ptrs.data());
              value_type*  y_ptr  = thrust::raw_pointer_cast( y.data().data());
        m_o.symv( SharedVectorTag(), dg::get_execution_policy<ContainerType1>(),
            alpha, b_ptrs, 1., y_ptr);
    }
    template<class ContainerType1, class ContainerType2>
    void symv(const ContainerType1& x, ContainerType2& y) const
    {
        symv( 1, x, 0, y);
    }
    private:
    LocalMatrixInner m_i;
    LocalMatrixOuter m_o;
    MPIKroneckerGather<Vector> m_g;
    mutable detail::AnyVector<Vector>  m_buffer_ptrs;
};


//TODO should this be public? prob. not
// grid is needed for local2globalIdx and global2localIdx
// communicators need to be the same
template<class real_type, class ConversionPolicyRows, class ConversionPolicyCols>
MPISparseBlockMat<thrust::host_vector, EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>>
    make_mpi_sparseblockmat( // not actually communicating
    const EllSparseBlockMat<real_type>& src,
    // src left_size and src right_size are the correct local sizes
    // src has global rows and global cols
    const ConversionPolicyRows& g_rows, // must be 1d
    const ConversionPolicyCols& g_cols
)
{
    // Indices in src are BLOCK INDICES!
    // Our approach here is:
    // 1. some preliminary MPI tests
    // 2. grab the correct rows of src and mark rows that are communicating
    // we need to grab the entire row to ensure reproducibility (which is guaranteed by order of computations)
    // 3. Convert inner rows to local and move outer rows to vectors (invalidating them in inner)
    // 4. Use global columns vector to construct MPI Kronecker Gather
    // 5. Use buffer index to construct local CooSparseBlockMat
    constexpr int invalid_idx = EllSparseBlockMat<real_type>::invalid_index;
    int result;
    MPI_Comm_compare( g_rows.communicator(), g_cols.communicator(), &result);
    assert( result == MPI_CONGRUENT or result == MPI_IDENT);
    int rank;
    MPI_Comm_rank( g_cols.communicator(), &rank);
    int ndim;
    MPI_Cartdim_get( g_cols.communicator(), &ndim);
    assert( ndim == 1);
    int dim, period, coord;
    MPI_Cart_get( g_cols.communicator(), 1, &dim, &period, &coord); // coord of the calling process
    if( dim == 1)
    {
        return { src, {}, {}};
    }
    int bpl = src.blocks_per_line;
    EllSparseBlockMat<real_type> inner(g_rows.local().N(), g_cols.local().N(), // number of blocks
        bpl, src.data.size()/(src.n*src.n), src.n);
    inner.set_left_size( src.left_size);
    inner.set_right_size( src.right_size);
    //first copy data elements (even though not all might be needed it doesn't slow down things either)
    inner.data = src.data;
    std::vector<bool> local_row( inner.num_rows, true);
    //2. now grab the right rows of the cols and data indices (and mark communicating rows)
    for( int i=0; i<inner.num_rows; i++)
    for( int k=0; k<bpl; k++)
    {
        int gIdx=0;
        assert( g_rows.local2globalIdx( i*src.n, rank, gIdx)); // convert from idx to block idx !
        inner.data_idx[i*bpl + k] = src.data_idx[ gIdx/src.n*bpl + k];
        inner.cols_idx[i*bpl + k] = src.cols_idx[ gIdx/src.n*bpl + k];
        if( inner.cols_idx[i*bpl+k] == invalid_idx)
            continue;
        // find out if row is communicating
        int lIdx = 0, pid = 0;
        gIdx = inner.cols_idx[i*bpl + k]*src.n; // convert from block idx to idx !
        assert( g_cols.global2localIdx( gIdx, lIdx, pid));
        if ( pid != rank)
            local_row[i] = false;
    }
    thrust::host_vector<std::array<int,2>> gColIdx;
    thrust::host_vector<int> rowIdx;
    thrust::host_vector<int> dataIdx;
    for( int i=0; i<inner.num_rows; i++)
    {
    for( int k=0; k<bpl; k++)
    {
        int gIdx = inner.cols_idx[i*bpl + k];
        int lIdx = 0, pid = 0;
        if( gIdx != invalid_idx)
        {
            assert( g_cols.global2localIdx( gIdx*src.n, lIdx, pid));
            lIdx = lIdx/src.n; // convert from idx to block idx !
        }
        else
            lIdx = invalid_idx;
        if ( !local_row[i] )
        {
            assert( lIdx != invalid_idx);
            rowIdx.push_back( i);
            dataIdx.push_back( inner.data_idx[i*bpl+k]);
            gColIdx.push_back( {pid, lIdx});
            inner.cols_idx[i*bpl + k] = invalid_idx;
        }
        else
        {
            inner.cols_idx[i*bpl + k] = lIdx;
        }
    }
    }
    // Now make MPI Gather object
    thrust::host_vector<int> lColIdx;
    auto gather_map = dg::gIdx2unique_idx( gColIdx, lColIdx);

    MPIKroneckerGather<thrust::host_vector> mpi_gather( inner.left_size, gather_map, inner.n,
        inner.num_cols, inner.right_size, g_cols.communicator());
    CooSparseBlockMat<real_type> outer(inner.num_rows, mpi_gather.buffer_size(),
        src.n, src.left_size, src.right_size);
    outer.data     = src.data; outer.cols_idx = lColIdx;
    outer.rows_idx = rowIdx;   outer.data_idx = dataIdx;
    outer.num_entries = rowIdx.size();

    return {inner, outer, mpi_gather};
}


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
    : m_i(m), m_dist(row_dist){}

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
                Vector<int>& scatter, // TODO scatter comp -> y from symv(outside, recv, comp);
                enum dist_type dist = row_dist
                )
    : m_i(inside), m_o(outside), m_g(mpi_gather),
      m_dist( dist), m_reduce(mpi_gather.communicator()) { }

    /**
     * @brief Allreduce dist type
     *
     * In this mode the mpi matrix locally aplies the inner matrix and
     * calls \c MPI_Allreduce on the result
     * (outer matrix remains empty)
     */
    MPIDistMat( const LocalMatrixInner& m, MPI_Comm comm)
    : m_i(m), m_dist(allreduce), m_reduce(comm){}

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
    MPIDistMat( const MPIDistMat<OtherVector, OtherMatrixInner, OtherMatrixOuter>& src)
    : m_i(src.inner_matrix()), m_o( src.outer_matrix()), m_g(src.mpi_gather()),
    m_dist( src.dist()), m_reduce(src.communicator())
    { }
    ///@brief Read access to the inner matrix
    const LocalMatrixInner& inner_matrix() const{return m_i;}
    ///@brief Read access to the outer matrix
    const LocalMatrixOuter& outer_matrix() const{return m_o;}
    ///@brief Read access to the communication object
    const MPIGather<Vector>& mpi_gather() const{return m_g;}

    ///@brief Read access to the MPI Communicator
    MPI_Comm communicator() const{ return m_reduce.communicator();}

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
                m_reduce.reduce( y.data());
            }
            return;

        }
        assert( m_dist != row_dist);
        int result;
        MPI_Comm_compare( x.communicator(), y.communicator(), &result);
        assert( result == MPI_CONGRUENT || result == MPI_IDENT);

        using value_type = dg::get_value_type<ContainerType1>;
        m_recv_buffer.template set<value_type>( m_g.buffer_size());
        m_comp_buffer.template set<value_type>( m_o.num_rows);
        auto& recv_buffer = m_recv_buffer.template get<value_type>();
        auto& comp_buffer = m_comp_buffer.template get<value_type>();

        // 1 initiate communication
        m_g.global_gather_init( x.data(), recv_buffer);
        // 2 compute inner points
        dg::blas2::symv( m_i, x.data(), y.data());
        // 3 wait for communication to finish
        m_g.global_gather_wait( recv_buffer);
        // 4 compute and add outer points
        dg::blas2::stencil( m_o, recv_buffer, comp_buffer);
        unsigned size = comp_buffer.size();
        dg::blas2::detail::doParallelFor( SharedVectorTag(),
            [size]DG_DEVICE( unsigned i, const value_type* x,
                const int* idx, value_type* y) {
                y[idx[i]] += x[i];
            }, size, y.data(), m_scatter, comp_buffer);
        // What we really need here is a sparse vector format
        // so we can fuse the symv with the scatter
        // OR y += Ax
    }

    /// Stencil computations
    template<class Functor, class ContainerType1, class ContainerType2>
    void stencil( const Functor f, const ContainerType1& x, ContainerType2& y) const
    {
        //the blas2 functions should make enough static assertions on tpyes
        if( !m_g->isCommunicating()) //no communication needed
        {
            dg::blas2::stencil( f, m_i, x.data(), y.data());
            if( m_dist == allreduce)
            {
                m_reduce.reduce( y.data());
            }
            return;
        }
        int result;
        MPI_Comm_compare( x.communicator(), y.communicator(), &result);
        assert( result == MPI_CONGRUENT || result == MPI_IDENT);

        using value_type = dg::get_value_type<ContainerType1>;
        m_recv_buffer.template set<value_type>( m_g.buffer_size());
        m_comp_buffer.template set<value_type>( m_o.num_rows);
        auto& recv_buffer = m_recv_buffer.template get<value_type>();
        auto& comp_buffer = m_comp_buffer.template get<value_type>();

        // 1 initiate communication
        m_g.global_gather_init( x.data(), recv_buffer);
        // 2 compute inner points
        dg::blas2::stencil( f, m_i, x.data(), y.data());
        // 3 wait for communication to finish
        m_g.global_gather_wait( recv_buffer);
        // 4 compute and add outer points
        dg::blas2::stencil( f, m_o, recv_buffer, comp_buffer);
        unsigned size = comp_buffer.size();
        dg::blas2::detail::doParallelFor( SharedVectorTag(),
            [size]DG_DEVICE( unsigned i, const value_type* x,
                const int* idx, value_type* y) {
                y[idx[i]] += x[i];
            }, size, y.data(), m_scatter, comp_buffer);
        // What we really need here is a sparse vector format
        // so we can fuse the symv with the scatter
        // OR y += Ax
    }

    private:
    LocalMatrixInner m_i;
    LocalMatrixOuter m_o;
    MPIGather<Vector> m_g;
    detail::MPIAllreduce m_reduce;
    enum dist_type m_dist;
    mutable detail::AnyVector<Vector>  m_recv_buffer, m_comp_buffer ;
    Vector<int> m_scatter;
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
