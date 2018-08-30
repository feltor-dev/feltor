#pragma once

#include <thrust/host_vector.h>
#include "dg/backend/memory.h"
#include "dg/enums.h"
#include "dg/blas.h"
#include "grid.h"
#include "interpolation.h"
#include "projection.h"
#ifdef MPI_VERSION
#include "mpi_grid.h"
#endif //MPI_VERSION



/**@file
* @brief contains a matrix type for fast interpolations/projections
*/

namespace dg
{

/**
 * @brief Struct that applies given matrices one after the other
 *
 * \f[ y = M_{N-1}(...M_1(M_0x))\f]
 * where \f$ M_i\f$ is the i-th matrix
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

    template<class OtherMatrix, class OtherContainer>
    MultiMatrix( const MultiMatrix<OtherMatrix, OtherContainer>& src){
        unsigned dimsM = src.get_matrices().size();
        unsigned dimsT = src.get_temp().size();
        inter_.resize( dimsM);
        temp_.resize(  dimsT);
        for( unsigned i=0; i<dimsM; i++)
            inter_[i] = src.get_matrices()[i];
        for( unsigned i=0; i<dimsT; i++)
            temp_[i].data() = dg::construct<ContainerType>(src.get_temp()[i].data());

    }
    template<class OtherMatrix, class OtherContainer, class ...Params>
    void construct( const MultiMatrix<OtherMatrix, OtherContainer>& src, Params&& ...ps){
        unsigned dimsM = src.get_matrices().size();
        unsigned dimsT = src.get_temp().size();
        inter_.resize( dimsM);
        temp_.resize(  dimsT);
        for( unsigned i=0; i<dimsM; i++)
            inter_[i] = src.get_matrices()[i];
        for( unsigned i=0; i<dimsT; i++)
            temp_[i].data() = dg::construct<ContainerType>(src.get_temp()[i].data(), std::forward<Params>(ps)...);
    }


    void symv( const ContainerType& x, ContainerType& y) const{ symv( 1., x,0,y);}
    void symv(real_type alpha, const ContainerType& x, real_type beta, ContainerType& y) const
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


namespace create
{
template<class real_type>
MultiMatrix< EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > fast_interpolation( const RealGrid1d<real_type>& t, unsigned multiply)
{
    unsigned n=t.n();
    dg::RealGrid1d<real_type> g_old( -1., 1., n, 1);
    dg::RealGrid1d<real_type> g_new( -1., 1., n, multiply);
    dg::IHMatrix interpolX = dg::create::interpolation( g_new, g_old);
    EllSparseBlockMat<real_type> iX( multiply*t.N(), t.N(), 1, multiply, t.n());
    for( unsigned  k=0; k<multiply; k++)
    for( unsigned  i=0; i<n; i++)
    for( unsigned  j=0; j<n; j++)
        iX.data[(k*n+i)*n+j] = interpolX.values[(k*n+i)*n+j];
    for( unsigned i=0; i<multiply*t.N(); i++)
    {
        iX.cols_idx[i] = i/multiply;
        iX.data_idx[i] = i%multiply;
    }
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > inter(1);
    inter.get_matrices()[0] = iX;
    return inter;
}

template<class real_type>
MultiMatrix< EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > fast_projection( const RealGrid1d<real_type>& t, unsigned divide, enum dg::norm no = normed)
{
    unsigned n=t.n();
    if( t.N()%divide != 0) throw Error( Message(_ping_)<< "Nx and divide don't match: Nx: " << t.N()<< " divide "<< (unsigned)divide);
    dg::RealGrid1d<real_type> g_oldX( -1., 1., n, divide);
    dg::RealGrid1d<real_type> g_new(  -1., 1., n, 1);
    dg::IHMatrix projectX;
    if(no == normed)
        projectX = dg::create::projection( g_new, g_oldX);
    else
        projectX = dg::create::interpolationT( g_new, g_oldX);
    EllSparseBlockMat<real_type> pX( t.N()/divide, t.N(), divide, divide, t.n());
    for( unsigned i=0; i<n; i++)
    for( unsigned j=0; j<n; j++)
        for( unsigned k=0; k<divide; k++)
            pX.data[(k*n+i)*n+j] = projectX.values[i*divide*n +k*n+j];
    for( unsigned i=0; i<t.N()/divide; i++)
        for( unsigned d=0; d<divide; d++)
        {
            pX.cols_idx[i*divide+d] = i*divide+d;
            pX.data_idx[i*divide+d] = d;
        }
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > inter(1);
    inter.get_matrices()[0] = pX;
    return inter;
}

template<class real_type>
MultiMatrix< EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > fast_interpolation( const aRealTopology2d<real_type>& t, unsigned multiplyX, unsigned multiplyY)
{
    dg::RealGrid1d<real_type> gx(t.x0(), t.x1(), t.n(), t.Nx());
    dg::RealGrid1d<real_type> gy(t.y0(), t.y1(), t.n(), t.Ny());
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interX = dg::create::fast_interpolation( gx, multiplyX);
    interX.get_matrices()[0].left_size = t.n()*t.Ny();
    interX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interY = dg::create::fast_interpolation( gy, multiplyY);
    interY.get_matrices()[0].right_size = t.n()*t.Nx()*multiplyX;
    interY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > inter(2);
    inter.get_matrices()[0] = interX.get_matrices()[0];
    inter.get_matrices()[1] = interY.get_matrices()[0];
    thrust::host_vector<real_type> vec( t.size()*multiplyX);
    inter.get_temp()[0] = Buffer<thrust::host_vector<real_type > >(vec);
    return inter;
}

template<class real_type>
MultiMatrix< EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > fast_projection( const aRealTopology2d<real_type>& t, unsigned divideX, unsigned divideY, enum dg::norm no = normed)
{
    dg::RealGrid1d<real_type> gx(t.x0(), t.x1(), t.n(), t.Nx());
    dg::RealGrid1d<real_type> gy(t.y0(), t.y1(), t.n(), t.Ny());
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interX = dg::create::fast_projection( gx, divideX, no);
    interX.get_matrices()[0].left_size = t.n()*t.Ny();
    interX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interY = dg::create::fast_projection( gy, divideY, no);
    interY.get_matrices()[0].right_size = t.n()*t.Nx()/divideX;
    interY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > inter(2);
    inter.get_matrices()[0] = interX.get_matrices()[0];
    inter.get_matrices()[1] = interY.get_matrices()[0];
    thrust::host_vector<real_type> vec( t.size()/divideX);
    inter.get_temp()[0] = Buffer<thrust::host_vector<real_type> >(vec);
    return inter;
}

template<class real_type>
MultiMatrix< EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > fast_interpolation( const aRealTopology3d<real_type>& t, unsigned multiplyX, unsigned multiplyY)
{
    dg::RealGrid1d<real_type> gx(t.x0(), t.x1(), t.n(), t.Nx());
    dg::RealGrid1d<real_type> gy(t.y0(), t.y1(), t.n(), t.Ny());
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interX = dg::create::fast_interpolation( gx, multiplyX);
    interX.get_matrices()[0].left_size = t.n()*t.Ny()*t.Nz();
    interX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interY = dg::create::fast_interpolation( gy, multiplyY);
    interY.get_matrices()[0].right_size = t.n()*t.Nx()*multiplyX;
    interY.get_matrices()[0].left_size = t.Nz();
    interY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > inter(2);
    inter.get_matrices()[0] = interX.get_matrices()[0];
    inter.get_matrices()[1] = interY.get_matrices()[0];
    thrust::host_vector<real_type> vec( t.size()*multiplyX);
    inter.get_temp()[0] = Buffer<thrust::host_vector<real_type > >(vec);
    return inter;
}

template<class real_type>
MultiMatrix< EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > fast_projection( const aRealTopology3d<real_type>& t, unsigned divideX, unsigned divideY, enum dg::norm no = normed)
{
    dg::RealGrid1d<real_type> gx(t.x0(), t.x1(), t.n(), t.Nx());
    dg::RealGrid1d<real_type> gy(t.y0(), t.y1(), t.n(), t.Ny());
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interX = dg::create::fast_projection( gx, divideX, no);
    interX.get_matrices()[0].left_size = t.n()*t.Ny()*t.Nz();
    interX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > interY = dg::create::fast_projection( gy, divideY, no);
    interY.get_matrices()[0].right_size = t.n()*t.Nx()/divideX;
    interY.get_matrices()[0].left_size = t.Nz();
    interY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > inter(2);
    inter.get_matrices()[0] = interX.get_matrices()[0];
    inter.get_matrices()[1] = interY.get_matrices()[0];
    thrust::host_vector<real_type> vec( t.size()/divideX);
    inter.get_temp()[0] = Buffer<thrust::host_vector<real_type> >(vec);
    return inter;
}

#ifdef MPI_VERSION
//very elaborate way of telling the compiler to just apply the local matrix to the local vector
template<class real_type>
MultiMatrix< RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type> >, MPI_Vector<thrust::host_vector<real_type> > > fast_interpolation( const aRealMPITopology2d<real_type>& t, unsigned divideX, unsigned divideY)
{
    typedef RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> Matrix;
    typedef MPI_Vector<thrust::host_vector<real_type> > Vector;
    MultiMatrix<EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > temp = dg::create::fast_interpolation( t.local(), divideX, divideY);
    MultiMatrix< Matrix, Vector > inter(2);
    inter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return inter;
}
template<class real_type>
MultiMatrix< RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type> >, MPI_Vector<thrust::host_vector<real_type> > > fast_projection( const aRealMPITopology2d<real_type>& t, unsigned divideX, unsigned divideY, enum dg::norm no = normed)
{
    typedef RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> Matrix;
    typedef MPI_Vector<thrust::host_vector<real_type> > Vector;
    MultiMatrix<EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > temp = dg::create::fast_projection( t.local(), divideX, divideY, no);
    MultiMatrix< Matrix, Vector > inter(2);
    inter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return inter;
}

template<class real_type>
MultiMatrix< RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type> >, MPI_Vector<thrust::host_vector<real_type> > > fast_interpolation( const aRealMPITopology3d<real_type>& t, unsigned divideX, unsigned divideY)
{
    typedef RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> Matrix;
    typedef MPI_Vector<thrust::host_vector<real_type> > Vector;
    MultiMatrix<EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > temp = dg::create::fast_interpolation( t.local(), divideX, divideY);
    MultiMatrix< Matrix, Vector > inter(2);
    inter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return inter;
}

template<class real_type>
MultiMatrix< RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type> >, MPI_Vector<thrust::host_vector<real_type> > > fast_projection( const aRealMPITopology3d<real_type>& t, unsigned divideX, unsigned divideY, enum dg::norm no = normed)
{
    typedef RowColDistMat<EllSparseBlockMat<real_type>, CooSparseBlockMat<real_type>, NNCH<real_type>> Matrix;
    typedef MPI_Vector<thrust::host_vector<real_type> > Vector;
    MultiMatrix<EllSparseBlockMat<real_type>, thrust::host_vector<real_type> > temp = dg::create::fast_projection( t.local(), divideX, divideY, no);
    MultiMatrix< Matrix, Vector > inter(2);
    inter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<real_type>(), NNCH<real_type>());
    inter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return inter;
}

#endif //MPI_VERSION
}//namespace create

///@endcond
}//namespace dg
