#pragma once

#include "mpi_gather_kron.h"
#include "sparseblockmat.h"
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
namespace detail
{
template< class Stencil, class ContainerType, class ...ContainerTypes>
void doParallelFor( SharedVectorTag, Stencil f, unsigned N, ContainerType&& x, ContainerTypes&&... xs);
}
}//namespace blas2

/**
 * @addtogroup mpi_matvec
@section mpi_matrix MPI Matrices and the symv function

Contrary to a vector a matrix can be distributed among processes in two ways:
\a row-wise and \a column-wise. When we implement a matrix-vector
multiplication the order of communication and computation depends on the
distribution of the matrix.
@note The dg library only implements row distributed matrices. The reason is
that the associated matrix-vector product can be made binary reproducible.
Support for column distributed matrices was dropped.

\subsection mpi_row Row distributed matrices

In a row-distributed matrix each process holds the rows of the matrix that
correspond to the portion of the \c MPI_Vector it holds.  When we implement a
matrix-vector multiplication each process first has to gather all the elements
of the input vector it needs to be able to compute the elements of its output.
In general this requires MPI communication.  (s.a. \ref mpigather for more info
of how global scatter/gather operations work).  After the elements have been
gathered into a buffer the local matrix-vector multiplications can be executed.
Formally, the gather operation can be written as a matrix \f$G\f$ of \f$1'\f$s
and \f$0'\f$s and we write.
\f[
M v = R\cdot G v
\f]
where \f$R\f$ is the row-distributed matrix with modified indices into a buffer
vector and \f$G\f$ is the gather matrix, in which the MPI-communication takes
place.  In this way we achieve a simple split between communication \f$ w=Gv\f$
and computation \f$ Rw\f$. Since the computation of \f$ R w\f$ is entirely
local we can reuse the existing implementation for shared memory systems.

\subsection mpi_row_col Separation of communication and computation
We can go one step further on a row distributed matrix and
separate the matrix \f$ M \f$ into
\f[
M v = (M_i + M_o) v = (M_{i} + R_o\cdot G_o) v
\f]
 where \f$ M_i\f$ is the inner matrix which requires no communication, while
 \f$ M_o\f$ is the outer matrix containing all elements which require MPI
 communication.
This enables the implementation of overlapping communication and computation
which is done in the \c dg::MPIDistMat and \c dg::MPISparseBlockMat classes.
\subsection mpi_create Creation
You can create a row-distributed MPI matrix given its local parts on each
process with local row and global column indices by our \c dg::make_mpi_matrix
function.  If you have a column distributed matrix with its local parts on each
process with global row and local columns indices, you can use a combination of
\c dg::convertLocal2GlobalCols and \c dg::convertGlobal2LocalRows to bring it
to a row-distributed form. The latter can then be used in \c dg::make_mpi_matrix again.

\subsection mpi_column Column distributed matrices

In a column-distributed matrix each process holds the columns of the matrix
that correspond to the portion of the \c MPI_Vector it holds.  In a column
distributed matrix the local matrix-vector multiplication can be executed first
because each processor already has all vector elements it needs.  However the
resulting elements have to be communicated back to the process they belong to.
Furthermore, a process has to sum all elements it receives from other processes
on the same index. This is a scatter and reduce operation and it can be written
as a scatter matrix \f$S\f$
\f[
M v= S\cdot C v
\f]
where \f$S\f$ is the scatter matrix and \f$C\f$ is the column distributed
matrix with modified indices.  Again, we can reuse our shared memory algorithms
to implement the local matrix-vector operation \f$ w=Cv\f$ before the
communication step \f$ S w\f$.
\subsection mpi_transpose Transposition
It turns out that a row-distributed matrix can be transposed by transposition
of both the local matrix and the gather matrix: \f[
M^\mathrm{T} = G^\mathrm{T} R^\mathrm{T} = S C\f] The result is then a column
distributed matrix.  Analogously, the transpose of a column distributed matrix
is a row-distributed matrix. It is also possible to convert a column distributed
mpi matrix to a row distributed mpi matrix. In code
@code{.cpp}
    // Tranpose a row distributed matrix to another row distributed matrix
    dg::IHMatrix matrix;
    //...
    // Suppose we have row distributed matrix with local rows and global cols
    dg::IHMatrix matrixT;
    cusp::transpose( matrix, matrixT);
    // matrixT is column distributed
    // matrixT has global rows and local column indices
    dg::convertLocal2GlobalCols( matrixT, grid);
    // now matrixT has global rows and global column indices
    auto mat = dg::convertGlobal2LocalRows( matrixT, grid);
    // now mat is row distributed with global column indices
    auto mpi_mat = dg::make_mpi_matrix(  mat, grid);
@endcode

*/
///@addtogroup mpi_matvec
///@{

/**
* @brief Distributed memory Sparse block matrix class, asynchronous communication
*
* This is a specialisation of \c MPIDiatMat for our \c dg::EllSparseBlockMat
*
* @copydetails MPIDistMat
*/
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

    ///@brief Read access to the inner matrix
    const LocalMatrixInner& inner_matrix() const{return m_i;}
    ///@brief Read access to the outer matrix
    const LocalMatrixOuter& outer_matrix() const{return m_o;}
    ///@brief Write access to the inner matrix
    LocalMatrixInner& inner_matrix() {return m_i;}
    ///@brief Write access to the outer matrix
    LocalMatrixOuter& outer_matrix() {return m_o;}

    MPI_Comm communicator() const { return m_g.communicator();}

    /**
    * @brief Matrix Vector product
    *
    * First the inner elements are computed with a call to symv then
    * the global_gather function of the communication object is called.
    * Finally the outer elements are added with a call to symv for the outer matrix
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
        // Local computation may be unnecessary
        if( buffer_ptrs.size() > 0)
        {
            // 4 compute and add outer points
            const value_type** b_ptrs = thrust::raw_pointer_cast( buffer_ptrs.data());
                  value_type*  y_ptr  = thrust::raw_pointer_cast( y.data().data());
            m_o.symv( SharedVectorTag(), dg::get_execution_policy<ContainerType1>(),
                alpha, b_ptrs, 1., y_ptr);
        }
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


/*!
 * @brief Split given \c EllSparseBlockMat into computation and communication part
 *
 * @param src  global rows and global cols, right_size and left_size must have correct local sizes
 * @param g_rows 1 dimensional grid for the rows
 * Is needed for \c local2globalIdx and \c global2localIdx
 * @param g_cols 1 dimensional grid for the columns
 * @attention communicators in \c g_rows and \c g_cols need to be at least \c MPI_CONGRUENT
 * @return MPI distributed spares block matrix of type <tt>dg::MHMatrix_t<real_type></tt>
 */
template<class real_type, class ConversionPolicyRows, class ConversionPolicyCols>
auto make_mpi_sparseblockmat( // not actually communicating
    const EllSparseBlockMat<real_type>& src,
    const ConversionPolicyRows& g_rows, // must be 1d
    const ConversionPolicyCols& g_cols
)
{
    // Indices in src are BLOCK INDICES!
    // g_rows and g_cols can have different n!!
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
        return MPISparseBlockMat<thrust::host_vector, EllSparseBlockMat<real_type>,
            CooSparseBlockMat<real_type>>{ src, {}, {}};
    }
    unsigned n = src.n, rn = g_rows.n()/n, cn = g_cols.n()/n;
    int bpl = src.blocks_per_line;
    EllSparseBlockMat<real_type> inner(g_rows.local().N()*rn, g_cols.local().N()*cn, // number of blocks
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
        assert( g_rows.local2globalIdx( i*src.n, rank, gIdx)); // convert from block idx to idx !
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

    return MPISparseBlockMat<thrust::host_vector, EllSparseBlockMat<real_type>,
        CooSparseBlockMat<real_type>>{ inner, outer, mpi_gather};
}

/*
// ///@cond
// namespace detail
// {
// // A combined Symv with a scatter of rows
// // Not good for cuda if number of columns is high so can be removed
// struct CSRSymvScatterFilter
// {
//     template<class real_type>
//     DG_DEVICE
//     void operator()( unsigned i, const int* scatter, const int* row_offsets,
//             const int* column_indices, const real_type* values,
//             const real_type* x, real_type* y)
//     {
//         for( int k=row_offsets[i]; k<row_offsets[i+1]; k++)
//             y[scatter[i]] += x[column_indices[k]]*values[k];
//     }
// };
// }// namespace detail
// /// @endcond
*/
///@cond
///Type of distribution of MPI distributed matrices
enum dist_type
{
    row_dist=0, //!< Row distributed
    allreduce=2 //!< special distribution for partial reduction (Average)
};
///@endcond



/**
* @brief Distributed memory matrix class, asynchronous communication
*
* See \ref mpi_row_col
* @tparam LocalMatrixInner The class of the matrix for local computations of
* the inner points.  symv(m,x,y) needs to be callable on the container class of
* the MPI_Vector
* @tparam LocalMatrixOuter The class of the matrix for local computations of
* the outer points.  symv(1,m,x,1,y) needs to be callable on the container
* class of the MPI_Vector
* @tparam Vector The storage class for internal buffers must match the
* execution policy of the containers in the symv functions
* @note This class overlaps communication with computation of the inner matrix
*/
template<template<class > class Vector, class LocalMatrixInner, class LocalMatrixOuter = LocalMatrixInner>
struct MPIDistMat
{
    ///@brief no memory allocation
    MPIDistMat() = default;

    /**
     * @brief Only local computations
     *
     * No communications, only the given local inner matrix will be applied
     */
    MPIDistMat( const LocalMatrixInner& m)
    : m_i(m), m_dist(row_dist){}

    template< template<class> class V, class LI, class LO>
    friend class MPIDistMat; // enable copy
    /**
    * @brief Constructor
    *
    * @param inside The local matrix for the inner elements
    * @param outside A local matrix for the elements from other processes
    * @param mpi_gather The communication object needs to gather values across
    * processes. The \c global_gather_init(), \c global_gather_wait(), \c
    * buffer_size() and \c isCommunicating() member functions are called If \c
    * !isCommunicating() the gather functions won't be called and only the
    * inner matrix is applied.
    * @param scatter Scattering of the rows from row buffer back to result vector
    * @note This is needed is because <tt>blas2::symv( m_o, buffer, y);</tt>
    * currently initializes all elements in \c y with zero and therefore would
    * overwrite the result from <tt>blas2::symv( m_i, x, y)</tt>. We therefore
    * need to allocate a small computational buffer and compute
    * <tt>blas2::symv( m_o, buffer, result_buffer)</tt> followed by a
    * scattering of values inside \c result_buffer into \c y
    */
    MPIDistMat( const LocalMatrixInner& inside,
                const LocalMatrixOuter& outside,
                const MPIGather<Vector>& mpi_gather,
                const Vector<int>& scatter // TODO scatter comp -> y from symv(outside, recv, comp);
                )
    : m_i(inside), m_o(outside), m_g(mpi_gather), m_scatter(scatter),
      m_dist( row_dist), m_reduce(mpi_gather.communicator()) { }

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
    m_scatter( src.m_scatter), m_dist( src.m_dist), m_reduce(src.communicator())
    { }
    ///@brief Read access to the inner matrix
    const LocalMatrixInner& inner_matrix() const{return m_i;}
    ///@brief Read access to the outer matrix
    const LocalMatrixOuter& outer_matrix() const{return m_o;}
    ///@brief Read access to the communication object
    const MPIGather<Vector>& mpi_gather() const{return m_g;}

    ///@brief Read access to the MPI Communicator
    MPI_Comm communicator() const{ return m_reduce.communicator();}

    const Vector<int>& scatter() const {return m_scatter;}

    /**
    * @brief Matrix Vector product
    *
    * First the inner elements are computed with a call to symv then
    * the \c global_gather_init function of the communication object is called.
    * Finally the outer elements are added with a call to symv for the outer matrix
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
        assert( m_dist == row_dist);
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
        if( comp_buffer.size() > 0)
        {
            // 4 compute and add outer points
            dg::blas2::symv( m_o, recv_buffer, comp_buffer);
            unsigned size = comp_buffer.size();
            dg::blas2::detail::doParallelFor( SharedVectorTag(),
                []DG_DEVICE( unsigned i, const value_type* x,
                    const int* idx, value_type* y) {
                    y[idx[i]] += x[i];
                }, size, comp_buffer, m_scatter, y.data());
            // What we need is a fused symv + scatter:
            // Something like
            // dg::blas2::parallel_for( detail::CSRSymvScatterFilter(),
            //        m_o.num_rows, m_scatter, m_o.row_offsets,
            //        m_o.column_indices, m_o.values, recv_buffer, y.data());
            // parallel_for is quite slow for ds_mpit because 50 entries per row
            // TODO Find out if cuSparse can do that and how fast
        }
    }

    /// Stencil computations
    template<class Functor, class ContainerType1, class ContainerType2>
    void stencil( const Functor f, const ContainerType1& x, ContainerType2& y) const
    {
        //the blas2 functions should make enough static assertions on tpyes
        if( !m_g.isCommunicating()) //no communication needed
        {
            dg::blas2::stencil( f, m_i, x.data(), y.data());
            if( m_dist == allreduce)
            {
                m_reduce.reduce( y.data());
            }
            return;
        }
        assert( m_dist == row_dist);
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
            }, size, comp_buffer, m_scatter, y.data());
        // What we really need here is a sparse vector format
        // so we can fuse the symv with the scatter
        // OR y += Ax
    }

    private:
    LocalMatrixInner m_i;
    LocalMatrixOuter m_o;
    MPIGather<Vector> m_g;
    Vector<int> m_scatter;
    enum dist_type m_dist;
    detail::MPIAllreduce m_reduce;
    mutable detail::AnyVector<Vector>  m_recv_buffer, m_comp_buffer ;
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
