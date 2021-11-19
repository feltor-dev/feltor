#pragma once

#include <thrust/host_vector.h>
#include "dg/backend/memory.h"
#include "dg/backend/typedefs.h"
#include "dg/enums.h"
#include "dg/blas.h"
#include "grid.h"
#include "interpolation.h"
#include "projection.h"
#ifdef MPI_VERSION
#include "mpi_grid.h"
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
 * @ingroup misc
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
    MultiMatrix( int dimension): inter_(dimension), temp_(dimension-1 > 0 ? dimension-1 : 0 ){}

    template<class OtherMatrix, class OtherContainer, class ... Params>
    MultiMatrix( const MultiMatrix<OtherMatrix, OtherContainer>& src, Params&& ... ps){
        unsigned dimsM = src.get_matrices().size();
        unsigned dimsT = src.get_temp().size();
        inter_.resize( dimsM);
        temp_.resize(  dimsT);
        for( unsigned i=0; i<dimsM; i++)
            inter_[i] = src.get_matrices()[i];
        for( unsigned i=0; i<dimsT; i++)
            dg::assign( src.get_temp()[i].data(), temp_[i].data(), std::forward<Params>(ps)...);

    }
    template<class ...Params>
    void construct( Params&& ...ps){
        *this = MultiMatrix( std::forward<Params>(ps)...);
    }


    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y) const{ symv( 1., x,0,y);}
    template<class ContainerType0, class ContainerType1>
    void symv(real_type alpha, const ContainerType0& x, real_type beta, ContainerType1& y) const
    {
        int dims = inter_.size();
        if( dims == 1)
        {
            dg::blas2::symv( alpha, inter_[0], x, beta, y);
            return;
        }
        dg::blas2::symv( inter_[0], x,temp_[0].data());
        for( int i=1; i<dims-1; i++)
            dg::blas2::symv( inter_[i], temp_[i-1].data(), temp_[i].data());
        dg::blas2::symv( alpha, inter_[dims-1], temp_[dims-2].data(), beta, y);
    }
    std::vector<Buffer<ContainerType> >& get_temp(){ return temp_;}
    const std::vector<Buffer<ContainerType> >& get_temp()const{ return temp_;}
    std::vector<MatrixType>& get_matrices(){ return inter_;}
    const std::vector<MatrixType>& get_matrices()const{ return inter_;}
    private:
    std::vector<MatrixType > inter_;
    std::vector<Buffer<ContainerType> > temp_;
};

///@cond
template <class M, class V>
struct TensorTraits<MultiMatrix<M, V> >
{
    using value_type  = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
///@endcond


namespace create
{
///@addtogroup interpolation
///@{

/**
 * @brief Create interpolation matrix for integer multipliers
 *
 * When creating an interpolation from a given dg grid to one that has
 * an integer multiple of cells and/or polynomial coefficients, the
 * resulting interpolation matrix fits into our \c dg::EllSparseBlockMat format,
 * which is much faster to apply than the full sparse matrix format from
 * the general purpose interpolation function, especially since it requires
 * no communication from neighboring cells
 * @sa dg::create::interpolation
 * @sa For a derivation of the coefficients consult the %dg manual <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
 * @tparam real_type a floating point type
 * @return a matrix that when applied to vectors on the old grid produces a vector on the new grid
 * @param t The existing (old/coarse) grid
 * @param multiplyn integer multiplier, the new grid has \c n*multiplyn polynomial coefficients
 * @param multiplyNx integer multiplier, the new grid has \c Nx*multiplyNx points
 */
template<class real_type>
MultiMatrix< dg::HMatrix_t<real_type>, dg::HVec_t<real_type> > fast_interpolation( const RealGrid1d<real_type>& t, unsigned multiplyn, unsigned multiplyNx)
{
    unsigned n=t.n();
    dg::RealGrid1d<real_type> g_old( -1., 1., n, 1);
    dg::RealGrid1d<real_type> g_new( -1., 1., n*multiplyn, multiplyNx);
    cusp::coo_matrix<int, real_type, cusp::host_memory> interpolX = dg::create::interpolation( g_new, g_old);
    EllSparseBlockMat<real_type> iX( multiplyn*multiplyNx*t.N(), t.N(), 1, multiplyNx*multiplyn, t.n());
    for( unsigned  k=0; k<multiplyNx*multiplyn; k++)
    for( unsigned  i=0; i<n; i++)
    for( unsigned  j=0; j<n; j++)
        iX.data[(k*n+i)*n+j] = interpolX.values[(k*n+i)*n+j];
    for( unsigned i=0; i<multiplyNx*multiplyn*t.N(); i++)
    {
        iX.cols_idx[i] = i/(multiplyNx*multiplyn);
        iX.data_idx[i] = i%(multiplyNx*multiplyn);
    }
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > inter(1);
    inter.get_matrices()[0] = iX;
    return inter;
}

/**
 * @brief Create projecton matrix for integer dividers
 *
 * When creating a projection from a given dg grid to one that has
 * an integer division of cells and/or polynomial coefficients, the
 * resulting projection matrix fits into our \c dg::EllSparseBlockMat format,
 * which is much faster to apply than the full sparse matrix format from
 * the general purpose projection function, especially since it requires
 * no communication from neighboring cells
 * @sa dg::create::projection dg::create::interpolationT
 * @sa For a derivation of the coefficients consult the %dg manual <a href="./dg_introduction.pdf" target="_blank">Introduction to dg methods</a>
 * @tparam real_type a floating point type
 * @return a matrix that when applied to vectors on the old grid produces a vector on the new grid
 * @param t The existing (old/fine) grid
 * @param dividen integer divisor, the new grid has \c n/multiplyn polynomial coefficients
 * @param divideNx integer divisor, the new grid has \c Nx/multiplyNx points
 */
template<class real_type>
MultiMatrix< dg::HMatrix_t<real_type>, dg::HVec_t<real_type> > fast_projection( const RealGrid1d<real_type>& t, unsigned dividen, unsigned divideNx)
{
    if( t.N()%divideNx != 0) throw Error( Message(_ping_)<< "Nx and divideNx don't match: Nx: " << t.N()<< " divideNx "<< (unsigned)divideNx);
    if( t.n()%dividen != 0) throw Error( Message(_ping_)<< "n and dividen don't match: n: " << t.n()<< " dividen "<< (unsigned)dividen);
    unsigned n=t.n()/dividen;
    dg::RealGrid1d<real_type> g_old( -1., 1., n*dividen, divideNx);
    dg::RealGrid1d<real_type> g_new( -1., 1., n, 1);
    dg::HVec w1d = dg::create::weights( g_old);
    dg::HVec v1d = dg::create::inv_weights( g_new);
    cusp::coo_matrix<int, real_type, cusp::host_memory> projectX;
    //Here, we cannot use create::projection because that would remove explicit zeros!!
    projectX = dg::create::interpolationT( g_new, g_old);
    EllSparseBlockMat<real_type> pX( t.N()/divideNx, t.N()*dividen, divideNx*dividen, divideNx*dividen, n);
    for( unsigned k=0; k<divideNx; k++)
    for( unsigned l=0; l<dividen; l++)
    for( unsigned i=0; i<n; i++)
    for( unsigned j=0; j<n; j++)
    {
        pX.data[((k*dividen+l)*n+i)*n+j] = projectX.values[((i*divideNx+k)*dividen + l)*n+j];
        pX.data[((k*dividen+l)*n+i)*n+j] *= v1d[i]*w1d[l*n+j];
    }
    for( unsigned i=0; i<t.N()/divideNx; i++)
        for( unsigned d=0; d<divideNx*dividen; d++)
        {
            pX.cols_idx[i*divideNx*dividen+d] = i*divideNx*dividen+d;
            pX.data_idx[i*divideNx*dividen+d] = d;
        }
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > inter(1);
    inter.get_matrices()[0] = pX;
    return inter;
}

///@copydoc fast_interpolation(const RealGrid1d<real_type>&,unsigned,unsigned)
///@param multiplyNy integer multiplier, the new grid has \c Ny*multiplyNy points
template<class real_type>
MultiMatrix< dg::HMatrix_t<real_type>, dg::HVec_t<real_type> > fast_interpolation( const aRealTopology2d<real_type>& t, unsigned multiplyn, unsigned multiplyNx, unsigned multiplyNy)
{
    dg::RealGrid1d<real_type> gx(t.x0(), t.x1(), t.n(), t.Nx());
    dg::RealGrid1d<real_type> gy(t.y0(), t.y1(), t.n(), t.Ny());
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interX = dg::create::fast_interpolation( gx, multiplyn,multiplyNx);
    interX.get_matrices()[0].left_size = t.n()*t.Ny();
    interX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interY = dg::create::fast_interpolation( gy, multiplyn,multiplyNy);
    interY.get_matrices()[0].right_size = t.n()*t.Nx()*multiplyNx*multiplyn;
    interY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > inter(2);
    inter.get_matrices()[0] = interX.get_matrices()[0];
    inter.get_matrices()[1] = interY.get_matrices()[0];
    thrust::host_vector<real_type> vec( t.size()*multiplyNx*multiplyn);
    inter.get_temp()[0] = Buffer<thrust::host_vector<real_type > >(vec);
    return inter;
}

///@copydoc fast_projection(const RealGrid1d<real_type>&,unsigned,unsigned)
///@param divideNy integer multiplier, the new grid has \c Ny/divideNy points
template<class real_type>
MultiMatrix< dg::HMatrix_t<real_type>, dg::HVec_t<real_type> > fast_projection( const aRealTopology2d<real_type>& t, unsigned dividen, unsigned divideNx, unsigned divideNy)
{
    dg::RealGrid1d<real_type> gx(t.x0(), t.x1(), t.n(), t.Nx());
    dg::RealGrid1d<real_type> gy(t.y0(), t.y1(), t.n(), t.Ny());
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interX = dg::create::fast_projection( gx, dividen, divideNx);
    interX.get_matrices()[0].left_size = t.n()*t.Ny();
    interX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interY = dg::create::fast_projection( gy, dividen, divideNy);
    interY.get_matrices()[0].right_size = t.n()*t.Nx()/divideNx/dividen;
    interY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > inter(2);
    inter.get_matrices()[0] = interX.get_matrices()[0];
    inter.get_matrices()[1] = interY.get_matrices()[0];
    thrust::host_vector<real_type> vec( t.size()/divideNx/dividen);
    inter.get_temp()[0] = Buffer<thrust::host_vector<real_type> >(vec);
    return inter;
}

///@copydoc fast_interpolation(const RealGrid1d<real_type>&,unsigned,unsigned)
///@param multiplyNy integer multiplier, the new grid has \c Ny*multiplyNy points
template<class real_type>
MultiMatrix< dg::HMatrix_t<real_type>, dg::HVec_t<real_type> > fast_interpolation( const aRealTopology3d<real_type>& t, unsigned multiplyn, unsigned multiplyNx, unsigned multiplyNy)
{
    dg::RealGrid1d<real_type> gx(t.x0(), t.x1(), t.n(), t.Nx());
    dg::RealGrid1d<real_type> gy(t.y0(), t.y1(), t.n(), t.Ny());
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interX = dg::create::fast_interpolation( gx, multiplyn, multiplyNx);
    interX.get_matrices()[0].left_size = t.n()*t.Ny()*t.Nz();
    interX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interY = dg::create::fast_interpolation( gy, multiplyn, multiplyNy);
    interY.get_matrices()[0].right_size = t.n()*t.Nx()*multiplyNx*multiplyn;
    interY.get_matrices()[0].left_size = t.Nz();
    interY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > inter(2);
    inter.get_matrices()[0] = interX.get_matrices()[0];
    inter.get_matrices()[1] = interY.get_matrices()[0];
    thrust::host_vector<real_type> vec( t.size()*multiplyNx*multiplyn);
    inter.get_temp()[0] = Buffer<thrust::host_vector<real_type > >(vec);
    return inter;
}

///@copydoc fast_projection(const RealGrid1d<real_type>&,unsigned,unsigned)
///@param divideNy integer multiplier, the new grid has \c Ny/divideNy points
template<class real_type>
MultiMatrix< dg::HMatrix_t<real_type>, dg::HVec_t<real_type> > fast_projection( const aRealTopology3d<real_type>& t, unsigned dividen, unsigned divideNx, unsigned divideNy)
{
    dg::RealGrid1d<real_type> gx(t.x0(), t.x1(), t.n(), t.Nx());
    dg::RealGrid1d<real_type> gy(t.y0(), t.y1(), t.n(), t.Ny());
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interX = dg::create::fast_projection( gx, dividen,divideNx);
    interX.get_matrices()[0].left_size = t.n()*t.Ny()*t.Nz();
    interX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interY = dg::create::fast_projection( gy, dividen, divideNy);
    interY.get_matrices()[0].right_size = t.n()*t.Nx()/divideNx/dividen;
    interY.get_matrices()[0].left_size = t.Nz();
    interY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > inter(2);
    inter.get_matrices()[0] = interX.get_matrices()[0];
    inter.get_matrices()[1] = interY.get_matrices()[0];
    thrust::host_vector<real_type> vec( t.size()/divideNx/dividen);
    inter.get_temp()[0] = Buffer<thrust::host_vector<real_type> >(vec);
    return inter;
}

#ifdef MPI_VERSION
//very elaborate way of telling the compiler to just apply the local matrix to the local vector
///@copydoc fast_interpolation(const RealGrid1d<real_type>&,unsigned,unsigned)
///@param multiplyNy integer multiplier, the new grid has \c Ny*multiplyNy points
template<class real_type>
MultiMatrix< MHMatrix_t<real_type>, MHVec_t<real_type> > fast_interpolation( const aRealMPITopology2d<real_type>& t, unsigned multiplyn, unsigned multiplyNx, unsigned multiplyNy)
{
    typedef RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> Matrix;
    typedef MPI_Vector<thrust::host_vector<real_type> > Vector;
    MultiMatrix<EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > temp = dg::create::fast_interpolation( t.local(), multiplyn,multiplyNx, multiplyNy);
    MultiMatrix< Matrix, Vector > inter(2);
    inter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return inter;
}

///@copydoc fast_projection(const RealGrid1d<real_type>&,unsigned,unsigned)
///@param divideNy integer multiplier, the new grid has \c Ny/divideNy points
template<class real_type>
MultiMatrix< MHMatrix_t<real_type>, MHVec_t<real_type> > fast_projection( const aRealMPITopology2d<real_type>& t, unsigned dividen, unsigned divideNx, unsigned divideNy)
{
    typedef RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> Matrix;
    typedef MPI_Vector<thrust::host_vector<real_type> > Vector;
    MultiMatrix<EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > temp = dg::create::fast_projection( t.local(), dividen,divideNx, divideNy);
    MultiMatrix< Matrix, Vector > inter(2);
    inter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return inter;
}

///@copydoc fast_interpolation(const RealGrid1d<real_type>&,unsigned,unsigned)
///@param multiplyNy integer multiplier, the new grid has \c Ny*multiplyNy points
template<class real_type>
MultiMatrix< MHMatrix_t<real_type>, MHVec_t<real_type> > fast_interpolation( const aRealMPITopology3d<real_type>& t, unsigned multiplyn, unsigned multiplyNx, unsigned multiplyNy)
{
    typedef RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> Matrix;
    typedef MPI_Vector<thrust::host_vector<real_type> > Vector;
    MultiMatrix<EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > temp = dg::create::fast_interpolation( t.local(), multiplyn,multiplyNx, multiplyNy);
    MultiMatrix< Matrix, Vector > inter(2);
    inter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return inter;
}

///@copydoc fast_projection(const RealGrid1d<real_type>&,unsigned,unsigned)
///@param divideNy integer multiplier, the new grid has \c Ny/divideNy points
template<class real_type>
MultiMatrix< MHMatrix_t<real_type>, MHVec_t<real_type> > fast_projection( const aRealMPITopology3d<real_type>& t, unsigned dividen, unsigned divideNx, unsigned divideNy)
{
    typedef RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> Matrix;
    typedef MPI_Vector<thrust::host_vector<real_type> > Vector;
    MultiMatrix<EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > temp = dg::create::fast_projection( t.local(), dividen,divideNx, divideNy);
    MultiMatrix< Matrix, Vector > inter(2);
    inter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return inter;
}

#endif //MPI_VERSION
///@}

}//namespace create

}//namespace dg
