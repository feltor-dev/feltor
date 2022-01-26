#pragma once
#include "dg/functors.h"
#include "fast_interpolation.h"

/**@file
* @brief Modal Filtering
*/

namespace dg
{

///@cond
namespace create
{
    //op is evaluated at eta and returns a double
template<class UnaryOp, class real_type>
MultiMatrix< EllSparseBlockMat<real_type>, thrust::host_vector<real_type> >
modal_filter( UnaryOp op, const RealGrid1d<real_type>& g )
{
    Operator<real_type> backward=g.dlt().backward();
    Operator<real_type> forward=g.dlt().forward();
    Operator<real_type> filter( g.n(), 0);
    for( unsigned i=0; i<g.n(); i++)
        filter(i,i) = op( i);
    filter = backward*filter*forward;
    //Assemble the matrix
    EllSparseBlockMat<real_type> A(g.N(), g.N(), 1, 1, g.n());
    A.data = filter.data();
    for( unsigned i=0; i<g.N(); i++)
    {
        A.data_idx[i] = 0;
        A.cols_idx[i] = i;
    }
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> >
        filter_matrix(1);
    filter_matrix.get_matrices()[0] = A;
    return filter_matrix;
}
template<class UnaryOp, class real_type>
MultiMatrix< EllSparseBlockMat<real_type>, thrust::host_vector<real_type> >
modal_filter( UnaryOp op, const aRealTopology2d<real_type>& t)
{
    dg::RealGrid1d<real_type> gx(t.x0(), t.x1(), t.nx(), t.Nx());
    dg::RealGrid1d<real_type> gy(t.y0(), t.y1(), t.ny(), t.Ny());
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> >
        filterX = dg::create::modal_filter( op, gx);
    filterX.get_matrices()[0].left_size = t.ny()*t.Ny();
    filterX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> >
        filterY = dg::create::modal_filter( op, gy);
    filterY.get_matrices()[0].right_size = t.nx()*t.Nx();
    filterY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > filter(2);
    filter.get_matrices()[0] = filterX.get_matrices()[0];
    filter.get_matrices()[1] = filterY.get_matrices()[0];
    thrust::host_vector<real_type> vec( t.size());
    filter.get_temp()[0] = Buffer<thrust::host_vector<real_type > >(vec);
    return filter;
}
template<class UnaryOp, class real_type>
MultiMatrix< EllSparseBlockMat<real_type>, thrust::host_vector<real_type> >
modal_filter( UnaryOp op, const aRealTopology3d<real_type>& t)
{
    dg::RealGrid1d<real_type> gx(t.x0(), t.x1(), t.nx(), t.Nx());
    dg::RealGrid1d<real_type> gy(t.y0(), t.y1(), t.ny(), t.Ny());
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> >
        filterX = dg::create::modal_filter( op, gx);
    filterX.get_matrices()[0].left_size = t.ny()*t.Ny()*t.nz()*t.Nz();
    filterX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> >
        filterY = dg::create::modal_filter( op, gy);
    filterY.get_matrices()[0].right_size = t.nx()*t.Nx();
    filterY.get_matrices()[0].left_size = t.nz()*t.Nz();
    filterY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > filter(2);
    filter.get_matrices()[0] = filterX.get_matrices()[0];
    filter.get_matrices()[1] = filterY.get_matrices()[0];
    thrust::host_vector<real_type> vec( t.size());
    filter.get_temp()[0] = Buffer<thrust::host_vector<real_type > >(vec);
    return filter;
}

#ifdef MPI_VERSION
//very elaborate way of telling the compiler to just apply the local matrix to the local vector
template<class UnaryOp, class real_type>
MultiMatrix< RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type> >, MPI_Vector<thrust::host_vector<real_type> > >
modal_filter( UnaryOp op, const aRealMPITopology2d<real_type>& t)
{
    typedef RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> Matrix;
    typedef MPI_Vector<thrust::host_vector<real_type> > Vector;
    MultiMatrix<EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > temp = dg::create::modal_filter( op, t.local());
    MultiMatrix< Matrix, Vector > filter(2);
    filter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    filter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    filter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return filter;
}

template<class UnaryOp, class real_type>
MultiMatrix< RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type> >, MPI_Vector<thrust::host_vector<real_type> > >
modal_filter( UnaryOp op, const aRealMPITopology3d<real_type>& t)
{
    typedef RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> Matrix;
    typedef MPI_Vector<thrust::host_vector<real_type> > Vector;
    MultiMatrix<EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > temp = dg::create::modal_filter( op, t.local());
    MultiMatrix< Matrix, Vector > filter(2);
    filter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    filter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    filter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return filter;
}

#endif //MPI_VERSION

} //namespace create
///@endcond

/**
 * @brief Struct that applies a given modal filter to a vector
 *
 * \f[ y = V D V^{-1}\f]
 * where \f$ V\f$ is the Vandermonde matrix (the backward transformation matrix)
 * and \f$ D \f$ is a diagonal matrix with \f$ D_{ii} = \sigma(i)\f$
 * @sa A discussion of the effects of the modal filter on advection schemes can be found here https://mwiesenberger.github.io/advection
 * @note basically the result is that it is usually not advantageous to use a modal filter
 * @copydoc hide_matrix
 * @copydoc hide_ContainerType
 * @ingroup misc
 */
template<class MatrixType, class ContainerType>
struct ModalFilter
{
    using real_type = get_value_type<ContainerType>;
    ModalFilter(){}
    /**
     * @brief Create arbitrary filter
     *
     * @tparam Topology Any grid
     * @tparam UnaryOp Model of Unary Function \c real_type \c sigma(unsigned) The input will be the modal number \c i where \f$ i=0,...,n-1\f$ and \c n is the number of polynomial coefficients in use. The output is the filter strength for the given mode number
     * @param sigma The filter to evaluate on the normalized modal coefficients
     * @param t The topology to apply the modal filter on
     * @param ps parameters that are forwarded to the creation of a ContainerType (e.g. when a std::vector is to be created it is the vector size)
     */
    template<class UnaryOp, class Topology, class ...Params>
    ModalFilter( UnaryOp sigma, const Topology& t, Params&& ...ps) : m_filter (
            dg::create::modal_filter( sigma, t), std::forward<Params>(ps)...) { }

    /**
    * @brief Perfect forward parameters to one of the constructors
    *
    * @tparam Params deduced by the compiler
    * @param ps parameters forwarded to constructors
    */
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = ModalFilter( std::forward<Params>( ps)...);
    }

    void apply( const ContainerType& x, ContainerType& y) const{ symv( 1., x, 0., y);}
    void symv( const ContainerType& x, ContainerType& y) const{ symv( 1., x,0,y);}
    void symv(real_type alpha, const ContainerType& x, real_type beta, ContainerType& y) const
    {
        m_filter.symv( alpha, x, beta, y);
    }
    private:
    MultiMatrix<MatrixType, ContainerType> m_filter;
};

///@cond
template <class M, class V>
struct TensorTraits<ModalFilter<M, V> >
{
    using value_type  = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
///@endcond
}//namespace dg
