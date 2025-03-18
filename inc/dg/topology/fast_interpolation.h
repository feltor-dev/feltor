#pragma once

#include <thrust/host_vector.h>
#include "dg/backend/memory.h"
#include "dg/backend/typedefs.h"
#include "dg/enums.h"
#include "dg/blas.h"
#include "grid.h"
#include "derivatives.h" // for update_left_right
#include "interpolation.h"
#include "projection.h"
#ifdef MPI_VERSION
#include "mpi_grid.h"
#include "mpi_derivatives.h" // for make_mpi_sparseblockmat
#endif //MPI_VERSION


/**@file
* @brief A matrix type for fast interpolations/projections
*/

namespace dg
{

/**
 * @brief Struct that applies given matrices one after the other
 *
 * \f[ y = M_{N-1}(...M_1(M_0x))\f]
 * where \f$ M_i\f$ is the i-th matrix
 * @sa mainly used by dg::create::fast_interpolation and dg::ModalFilter
 * @copydoc hide_matrix
 * @copydoc hide_ContainerType
 * @ingroup fast_interpolation
 */
template <class MatrixType, class ContainerType>
struct MultiMatrix
{
    using real_type = get_value_type<ContainerType>;
    MultiMatrix(){}
    /**
    * @brief reserve space for dimension matrices  and dimension-1 ContainerTypes
    * @param dimension # of matrices to store
    * @attention it is the user's reponsibility to allocate memory for the intermediate "temp" vectors
    */
    MultiMatrix( int dimension): m_inter(dimension), m_temp(dimension-1 > 0 ? dimension-1 : 0 ){}
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = MultiMatrix( std::forward<Params>( ps)...);
    }

    //https://stackoverflow.com/questions/26147061/how-to-share-protected-members-between-c-template-classes
    template< class OtherMatrix, class OtherContainer>
    friend class MultiMatrix; // enable copy
    template<class OtherMatrix, class OtherContainer, class ... Params>
    MultiMatrix( const MultiMatrix<OtherMatrix, OtherContainer>& src, Params&& ... ps)
    {
        unsigned dimsM = src.m_inter.size();
        unsigned dimsT = src.m_temp.size();
        m_inter.resize(dimsM);
        m_temp.resize(dimsT);
        for( unsigned i=0; i<dimsM; i++)
            m_inter[i] = src.m_inter[i];
        for( unsigned i=0; i<dimsT; i++)
            dg::assign( src.m_temp[i], m_temp[i], std::forward<Params>(ps)...);
    }
    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y) const{ symv( 1., x,0,y);}
    template<class ContainerType0, class ContainerType1>
    void symv(real_type alpha, const ContainerType0& x, real_type beta, ContainerType1& y) const
    {
        int dims = m_inter.size();
        if( dims == 1)
        {
            dg::blas2::symv( alpha, m_inter[0], x, beta, y);
            return;
        }
        dg::blas2::symv( m_inter[0], x,m_temp[0]);
        for( int i=1; i<dims-1; i++)
            dg::blas2::symv( m_inter[i], m_temp[i-1], m_temp[i]);
        dg::blas2::symv( alpha, m_inter[dims-1], m_temp[dims-2], beta, y);
    }
    std::vector<ContainerType >& get_temp(){ return m_temp;}
    const std::vector<ContainerType >& get_temp()const{ return m_temp;}
    std::vector<MatrixType>& get_matrices(){ return m_inter;}
    const std::vector<MatrixType>& get_matrices()const{ return m_inter;}
    private:
    std::vector<MatrixType > m_inter;
    mutable std::vector<ContainerType > m_temp; // OMG mutable exits !? write even in const
};

///@cond
template <class M, class V>
struct TensorTraits<MultiMatrix<M, V> >
{
    using value_type  = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};

namespace detail
{
//pay attention that left and right must have correct sizes
template<class real_type, class Topology>
MultiMatrix< dg::HMatrix_t<real_type>, dg::HVec_t<real_type> > multiply( const dg::HMatrix_t<real_type>& left, const dg::HMatrix_t<real_type>& right, const Topology& t)
{
    MultiMatrix< dg::HMatrix_t<real_type>, dg::HVec_t<real_type> > matrix(2);
    if( right.total_num_rows() != left.total_num_cols())
        throw Error( Message(_ping_)<< "left and right cannot be multiplied due to wrong sizes" << left.total_num_cols() << " vs "<<right.total_num_rows());
    matrix.get_matrices()[0] = right;
    matrix.get_matrices()[1] = left;
    thrust::host_vector<real_type> vec( right.total_num_rows());
    matrix.get_temp()[0] = vec;
    return matrix;
}
template<class real_type, size_t Nd>
RealGrid<real_type, Nd> set_right_grid( const aRealTopology<real_type, Nd>& t, unsigned new_n , unsigned new_N)
{
    // We need to explicitly convert to grid
    RealGrid<real_type,Nd> tnew (t);
    tnew.set_axis( 0, new_n, new_N);
    return tnew;
}
#ifdef MPI_VERSION
template<class real_type, class MPITopology>
MultiMatrix< dg::MHMatrix_t<real_type>, dg::MHVec_t<real_type> > multiply( const dg::MHMatrix_t<real_type>& left, const dg::MHMatrix_t<real_type>& right, const MPITopology& t)
{
    MultiMatrix< dg::MHMatrix_t<real_type>, dg::MHVec_t<real_type> > matrix(2);
    if( right.inner_matrix().total_num_rows() != left.inner_matrix().total_num_cols())
        throw Error( Message(_ping_)<< "left and right cannot be multiplied due to wrong sizes" << left.inner_matrix().total_num_cols() << " vs "<<right.inner_matrix().total_num_rows());
    matrix.get_matrices()[0] = right;
    matrix.get_matrices()[1] = left;
    thrust::host_vector<real_type> vec( right.inner_matrix().total_num_rows());
    matrix.get_temp()[0] = dg::MHVec_t<real_type>({vec, t.communicator()});

    return matrix;
}
template<class real_type, size_t Nd>
RealMPIGrid<real_type, Nd> set_right_grid( const aRealMPITopology<real_type, Nd>& t, unsigned new_n , unsigned new_N)
{
    // We need to explicitly convert to grid
    RealMPIGrid<real_type,Nd> tnew (t);
    tnew.set_axis( 0, new_n, new_N);
    return tnew;
}
#endif
} //namespace detail

///@endcond

namespace create
{
/*!
 * @class hide_coord_param
 * @param coord The direction inside the structured grid to which to apply
 * the sparse block matrix.
 */
///@addtogroup fast_interpolation
///@{

/**
 * @brief Create interpolation matrix for integer multipliers
 *
 * When creating an interpolation from a given dg grid to one that has an
 * integer multiple of cells and/or polynomial coefficients, the resulting
 * interpolation matrix fits into our \c dg::EllSparseBlockMat format, which is
 * much faster to apply than the full sparse matrix format from the general
 * purpose interpolation function, especially since it requires no
 * communication from neighboring cells
 * @sa dg::create::interpolation
 * @sa For a derivation of the coefficients consult the %dg manual <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @tparam real_type a floating point type
 * @return a matrix that when applied to vectors on the old grid produces a
 * vector on the new grid
 * @param t The existing (old/coarse) grid
 * @param multiplyn integer multiplier, the new grid has \c n*multiplyn
 * polynomial coefficients
 * @param multiplyNx integer multiplier, the new grid has \c Nx*multiplyNx
 * points
 */
template<class real_type>
dg::HMatrix_t<real_type> fast_interpolation1d( const RealGrid1d<real_type>& t, unsigned multiplyn, unsigned multiplyNx)
{
    unsigned n=t.n();
    dg::RealGrid1d<real_type> g_old( -1., 1., n, 1);
    dg::RealGrid1d<real_type> g_new( -1., 1., n*multiplyn, multiplyNx);
    // Does not generate explicit zeros ...
    cusp::coo_matrix<int, real_type, cusp::host_memory> interpolX =
        dg::create::interpolation( g_new, g_old);
    unsigned size = multiplyn*multiplyNx;
    EllSparseBlockMat<real_type, thrust::host_vector> iX( size*t.N(), t.N(), 1, size, t.n());
    dg::blas1::copy( 0., iX.data);
    for( unsigned l=0; l<interpolX.num_entries; l++)
    {
        int row = interpolX.row_indices[l];
        int col = interpolX.column_indices[l];
        real_type val = interpolX.values[l];
        iX.data[row*interpolX.num_cols + col] = val;
    }
    for( unsigned i=0; i<size*t.N(); i++)
    {
        iX.cols_idx[i] = i/(size);
        iX.data_idx[i] = i%(size);
    }
    return iX;
}

/**
 * @brief Create projecton matrix for integer dividers
 *
 * When creating a projection from a given dg grid to one that has an integer
 * division of cells and/or polynomial coefficients, the resulting projection
 * matrix fits into our \c dg::EllSparseBlockMat format, which is much faster
 * to apply than the full sparse matrix format from the general purpose
 * projection function, especially since it requires no communication from
 * neighboring cells
 * @sa dg::create::projection
 * @sa For a derivation of the coefficients consult the %dg manual <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @tparam real_type a floating point type
 * @return a matrix that when applied to vectors on the old grid produces a
 * vector on the new grid
 * @param t The existing (old/fine) grid
 * @param dividen integer divisor, the new grid has \c n/multiplyn polynomial
 * coefficients
 * @param divideNx integer divisor, the new grid has \c Nx/multiplyNx points
 */
template<class real_type>
dg::HMatrix_t<real_type> fast_projection1d( const RealGrid1d<real_type>& t, unsigned dividen, unsigned divideNx)
{
    if( t.N()%divideNx != 0)
        throw Error( Message(_ping_)<< "Nx and divideNx don't match: Nx: "
                    << t.N()<< " divideNx "<< (unsigned)divideNx);
    if( t.n()%dividen != 0)
        throw Error( Message(_ping_)<< "n and dividen don't match: n: "
                    << t.n()<< " dividen "<< (unsigned)dividen);
    unsigned n=t.n()/dividen;
    dg::RealGrid1d<real_type> g_old( -1., 1., n*dividen, divideNx);
    dg::RealGrid1d<real_type> g_new( -1., 1., n, 1);
    // Does not generate explicit zeros ...
    cusp::coo_matrix<int, real_type, cusp::host_memory> projectX = dg::create::projection( g_new, g_old);
    unsigned size = dividen*divideNx;
    EllSparseBlockMat<real_type, thrust::host_vector> pX( t.N()/divideNx, t.N()*dividen, size, size, n);
    dg::blas1::copy( 0., pX.data);
    for( unsigned ll=0; ll<projectX.num_entries; ll++)
    {
        int row = projectX.row_indices[ll];
        int col = projectX.column_indices[ll];
        real_type val = projectX.values[ll];
        int k = col/(n*dividen), l = (col/n)%dividen, i = row, j = col%n;
        pX.data[((k*dividen+l)*n+i)*n+j] = val;
    }
    for( unsigned i=0; i<t.N()/divideNx; i++)
        for( unsigned d=0; d<size; d++)
        {
            pX.cols_idx[i*size+d] = i*size+d;
            pX.data_idx[i*size+d] = d;
        }
    return pX;
}

/**
 * @brief Create a block-diagonal matrix
 *
 * i.e. a matrix that has an \f$ n \times n \f$ block on its diagonal
\f[ M = \begin{pmatrix}
B &   &   &   &   & \\
  & B &   &   &   & \\
  &   & B &   &   & \\
  &   &   & B &   & \\
  &   &   &...&   &
  \end{pmatrix}
  \f]
 * Block diagonal matrices fit into our \c dg::EllSparseBlockMat format, which
 * is much faster to apply than a general sparse matrix, especially since it
 * requires no communication from neighboring cells
 * @note We use this function in combination with \c dg::DLT::forward() to
 * create a forward transformation from configuration to Legendre space (or
 * from nodal to modal values) in dg::forward_transform
 * @tparam real_type a floating point type
 * @return a block diagonal matrix
 * @param opx the block B
 * @param t The grid determines the number of rows and columns
 */
template<class real_type>
dg::HMatrix_t<real_type> fast_transform1d( const dg::SquareMatrix<real_type>& opx, const RealGrid1d<real_type>& t)
{
    EllSparseBlockMat<real_type, thrust::host_vector> A( t.N(), t.N(), 1, 1, t.n());
    if( opx.size() != t.n())
        throw Error( Message(_ping_)<< "SquareMatrix must have same n as grid!");
    dg::assign( opx.data(), A.data);
    for( unsigned i=0; i<t.N(); i++)
    {
        A.cols_idx[i] = i;
        A.data_idx[i] = 0;
    }
    return A;
}

///@copydoc fast_interpolation1d(const RealGrid1d<real_type>&,unsigned,unsigned)
///@copydoc hide_coord_param
template<class real_type, size_t Nd>
EllSparseBlockMat<real_type, thrust::host_vector> fast_interpolation( unsigned coord,
    const aRealTopology<real_type, Nd>& t, unsigned multiplyn, unsigned multiplyNx)
{
    auto trafo = dg::create::fast_interpolation1d( t.axis(coord),
        multiplyn, multiplyNx);
    detail::update_left_right( coord, trafo, t);
    return trafo;
}

///@copydoc fast_projection1d(const RealGrid1d<real_type>&,unsigned,unsigned)
///@copydoc hide_coord_param
template<class real_type, size_t Nd>
EllSparseBlockMat<real_type, thrust::host_vector> fast_projection( unsigned coord,
    const aRealTopology<real_type, Nd>& t, unsigned dividen, unsigned divideNx)
{
    auto trafo = dg::create::fast_projection1d( t.axis(coord),
        dividen, divideNx);
    detail::update_left_right( coord, trafo, t);
    return trafo;
}

///@copydoc fast_transform1d(const dg::SquareMatrix<real_type>&,const RealGrid1d<real_type>&)
///@copydoc hide_coord_param
template<class real_type, size_t Nd>
EllSparseBlockMat<real_type, thrust::host_vector> fast_transform( unsigned coord, const dg::SquareMatrix<real_type>& opx,
    const aRealTopology<real_type, Nd>& t)
{
    auto trafo = dg::create::fast_transform1d( opx, t.axis(coord));
    detail::update_left_right( coord, trafo, t);
    return trafo;
}

#ifdef MPI_VERSION

///@copydoc fast_interpolation1d(const RealGrid1d<real_type>&,unsigned,unsigned)
///@copydoc hide_coord_param
template<class real_type, size_t Nd>
dg::MHMatrix_t<real_type> fast_interpolation( unsigned coord,
    const aRealMPITopology<real_type, Nd>& t, unsigned multiplyn, unsigned multiplyNx)
{
    RealMPIGrid<real_type,1> t_rows = t.axis(coord);
    t_rows.set( t.n(coord)*multiplyn, t.N(coord)*multiplyNx);
    return make_mpi_sparseblockmat( dg::create::fast_interpolation( coord,
        detail::local_global_grid(coord, t), multiplyn, multiplyNx),
            t_rows, t.axis(coord));
}
///@copydoc fast_projection1d(const RealGrid1d<real_type>&,unsigned,unsigned)
///@copydoc hide_coord_param
template<class real_type, size_t Nd>
dg::MHMatrix_t<real_type> fast_projection( unsigned coord,
    const aRealMPITopology<real_type, Nd>& t, unsigned dividen, unsigned divideNx)
{
    RealMPIGrid<real_type,1> t_rows = t.axis(coord);
    t_rows.set( t.n(coord)/dividen, t.N(coord)/divideNx);
    return make_mpi_sparseblockmat( dg::create::fast_projection( coord,
        detail::local_global_grid(coord, t), dividen, divideNx),
            t_rows, t.axis(coord));
}
///@copydoc fast_transform1d(const dg::SquareMatrix<real_type>&,const RealGrid1d<real_type>&)
///@copydoc hide_coord_param
template<class real_type, size_t Nd>
MHMatrix_t<real_type> fast_transform( unsigned coord, dg::SquareMatrix<real_type> opx,
    const aRealMPITopology<real_type, Nd>& t)
{
    return make_mpi_sparseblockmat( dg::create::fast_transform( coord, opx,
        detail::local_global_grid(coord, t)), t.axis(coord), t.axis(coord));
}

#endif //MPI_VERSION

// Product interpolations/projections/transforms
// TODO We may want to generalize this

///@brief Create interpolation matrix for integer multipliers in the first 2
///grid dimensions
///@copydetails fast_interpolation1d(const RealGrid1d<real_type>&,unsigned,unsigned)
///@param multiplyNy integer multiplier, the new grid has \c Ny*multiplyNy points
template<class Topology>
auto fast_interpolation( const Topology& t, unsigned multiplyn, unsigned multiplyNx, unsigned multiplyNy)
{
    auto interX = dg::create::fast_interpolation( 0, t, multiplyn,multiplyNx);
    auto tX = dg::detail::set_right_grid( t, t.n()*multiplyn, t.Nx()*multiplyNx);
    auto interY = dg::create::fast_interpolation( 1, tX, multiplyn,multiplyNy);
    return dg::detail::multiply( interY, interX, t);
}

///@brief Create projection matrix for integer dividers in the first 2
///grid dimensions
///@copydetails fast_projection1d(const RealGrid1d<real_type>&,unsigned,unsigned)
///@param divideNy integer multiplier, the new grid has \c Ny/divideNy points
template<class Topology>
auto fast_projection( const Topology& t, unsigned dividen, unsigned divideNx, unsigned divideNy)
{
    auto interX = dg::create::fast_projection( 0, t, dividen, divideNx);
    auto tX = dg::detail::set_right_grid( t, t.n()/dividen, t.Nx()/divideNx);
    auto interY = dg::create::fast_projection( 1, tX, dividen, divideNy);
    return dg::detail::multiply( interY, interX, t);
}
///@brief Create transform matrix in the first 2 grid dimensions
///@copydoc fast_transform1d(const dg::SquareMatrix<real_type>&,const RealGrid1d<real_type>&)
///@param opy the block B for the y transform
template<class Topology>
auto fast_transform( const dg::SquareMatrix<typename Topology::value_type>&
        opx, const dg::SquareMatrix<typename Topology::value_type>& opy, const
        Topology& t)
{
    auto interX = dg::create::fast_transform( 0, opx, t);
    auto interY = dg::create::fast_transform( 1, opy, t);
    return dg::detail::multiply( interY, interX, t);
}
///@}

}//namespace create

/**
 * @brief Transform a vector from dg::xspace (nodal values) to dg::lspace
 * (modal values)
 *
 * This can speedup the dg::interpolate function
 * @param in input
 * @param g A grid
 *
 * @ingroup interpolation
 * @return the vector in dg::lspace
 * @sa dg::interpolate
 */
template<class Topology>
typename Topology::host_vector forward_transform( const
        typename Topology::host_vector& in, const Topology& g)
{
    typename Topology::host_vector out(in), tmp(in);
    for( unsigned u=0; u<Topology::ndim(); u++)
    {
        auto forward = create::fast_transform( u, SquareMatrix{dg::DLT<typename
                Topology::value_type>::forward(g.n(u))}, g);
        dg::blas2::symv( forward, out, tmp);
        tmp.swap( out);
    }
    return out;
}

}//namespace dg
