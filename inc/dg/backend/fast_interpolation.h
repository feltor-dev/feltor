#pragma once

#include <thrust/host_vector.h>
#include "../enums.h"
#include "grid.h"
#include "interpolation.cuh"
#include "projection.cuh"
#include "matrix_traits.h"
#include "sparseblockmat.h"
#include "memory.h"
#ifdef MPI_VERSION
#include "mpi_grid.h"
#include "mpi_vector.h"
#include "mpi_matrix.h"
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
 * @copydoc hide_container
 * @ingroup misc
 */
template <class Matrix, class container>
struct MultiMatrix
{
    MultiMatrix(){}
    /**
    * @brief reserve space for dimension matrices  and dimension-1 containers
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
            temp_[i].data() = src.get_temp()[i].data();

    }

    void symv( const container& x, container& y) const{ symv( 1., x,0,y);}
    void symv(double alpha, const container& x, double beta, container& y) const
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
    std::vector<Buffer<container> >& get_temp(){ return temp_;}
    const std::vector<Buffer<container> >& get_temp()const{ return temp_;}
    std::vector<Matrix>& get_matrices(){ return inter_;}
    const std::vector<Matrix>& get_matrices()const{ return inter_;}
    private:
    std::vector<Matrix > inter_;
    std::vector<Buffer<container> > temp_;
};

///@cond
template <class M, class V>
struct MatrixTraits<MultiMatrix<M, V> >
{
    typedef typename VectorTraits<V>::value_type value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template <class M, class V>
struct MatrixTraits<const MultiMatrix<M, V> >
{
    typedef typename VectorTraits<V>::value_type value_type;
    typedef SelfMadeMatrixTag matrix_category;
};


namespace create
{
MultiMatrix< EllSparseBlockMat<double>, thrust::host_vector<double> > fast_interpolation( const Grid1d& t, unsigned multiply)
{
    unsigned n=t.n();
    dg::Grid1d g_old( -1., 1., n, 1);
    dg::Grid1d g_new( -1., 1., n, multiply);
    dg::IHMatrix interpolX = dg::create::interpolation( g_new, g_old);
    EllSparseBlockMat<double> iX( multiply*t.N(), t.N(), 1, multiply, t.n()); 
    for( unsigned  k=0; k<multiply; k++)
    for( unsigned  i=0; i<n; i++)
    for( unsigned  j=0; j<n; j++)
        iX.data[(k*n+i)*n+j] = interpolX.values[(k*n+i)*n+j];
    for( unsigned i=0; i<multiply*t.N(); i++)
    {
        iX.cols_idx[i] = i/multiply;
        iX.data_idx[i] = i%multiply;
    }
    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > inter(1);
    inter.get_matrices()[0] = iX;
    return inter;
}

MultiMatrix< EllSparseBlockMat<double>, thrust::host_vector<double> > fast_projection( const Grid1d& t, unsigned divide, enum dg::norm no = normed)
{
    unsigned n=t.n();
    if( t.N()%divide != 0) throw Error( Message(_ping_)<< "Nx and divide don't match: Nx: " << t.N()<< " divide "<< divide);
    dg::Grid1d g_oldX( -1., 1., n, divide);
    dg::Grid1d g_new(  -1., 1., n, 1);
    dg::IHMatrix projectX;
    if(no == normed)
        projectX = dg::create::projection( g_new, g_oldX);
    else
        projectX = dg::create::interpolationT( g_new, g_oldX);
    EllSparseBlockMat<double> pX( t.N()/divide, t.N(), divide, divide, t.n()); 
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
    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > inter(1);
    inter.get_matrices()[0] = pX;
    return inter;
}

MultiMatrix< EllSparseBlockMat<double>, thrust::host_vector<double> > fast_interpolation( const aTopology2d& t, unsigned multiplyX, unsigned multiplyY)
{
    dg::Grid1d gx(t.x0(), t.x1(), t.n(), t.Nx());
    dg::Grid1d gy(t.y0(), t.y1(), t.n(), t.Ny());
    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > interX = dg::create::fast_interpolation( gx, multiplyX);
    interX.get_matrices()[0].left_size = t.n()*t.Ny();
    interX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > interY = dg::create::fast_interpolation( gy, multiplyY);
    interY.get_matrices()[0].right_size = t.n()*t.Nx()*multiplyX;
    interY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > inter(2);
    inter.get_matrices()[0] = interX.get_matrices()[0];
    inter.get_matrices()[1] = interY.get_matrices()[0];
    thrust::host_vector<double> vec( t.size()*multiplyX);
    inter.get_temp()[0] = Buffer<thrust::host_vector<double > >(vec);
    return inter;
}

MultiMatrix< EllSparseBlockMat<double>, thrust::host_vector<double> > fast_projection( const aTopology2d& t, unsigned divideX, unsigned divideY, enum dg::norm no = normed)
{
    dg::Grid1d gx(t.x0(), t.x1(), t.n(), t.Nx());
    dg::Grid1d gy(t.y0(), t.y1(), t.n(), t.Ny());
    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > interX = dg::create::fast_projection( gx, divideX, no);
    interX.get_matrices()[0].left_size = t.n()*t.Ny();
    interX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > interY = dg::create::fast_projection( gy, divideY, no);
    interY.get_matrices()[0].right_size = t.n()*t.Nx()/divideX;
    interY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > inter(2);
    inter.get_matrices()[0] = interX.get_matrices()[0];
    inter.get_matrices()[1] = interY.get_matrices()[0];
    thrust::host_vector<double> vec( t.size()/divideX);
    inter.get_temp()[0] = Buffer<thrust::host_vector<double> >(vec);
    return inter;
}

MultiMatrix< EllSparseBlockMat<double>, thrust::host_vector<double> > fast_interpolation( const aTopology3d& t, unsigned multiplyX, unsigned multiplyY)
{
    dg::Grid1d gx(t.x0(), t.x1(), t.n(), t.Nx());
    dg::Grid1d gy(t.y0(), t.y1(), t.n(), t.Ny());
    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > interX = dg::create::fast_interpolation( gx, multiplyX);
    interX.get_matrices()[0].left_size = t.n()*t.Ny()*t.Nz();
    interX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > interY = dg::create::fast_interpolation( gy, multiplyY);
    interY.get_matrices()[0].right_size = t.n()*t.Nx()*multiplyX;
    interY.get_matrices()[0].left_size = t.Nz();
    interY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > inter(2);
    inter.get_matrices()[0] = interX.get_matrices()[0];
    inter.get_matrices()[1] = interY.get_matrices()[0];
    thrust::host_vector<double> vec( t.size()*multiplyX);
    inter.get_temp()[0] = Buffer<thrust::host_vector<double > >(vec);
    return inter;
}

MultiMatrix< EllSparseBlockMat<double>, thrust::host_vector<double> > fast_projection( const aTopology3d& t, unsigned divideX, unsigned divideY, enum dg::norm no = normed)
{
    dg::Grid1d gx(t.x0(), t.x1(), t.n(), t.Nx());
    dg::Grid1d gy(t.y0(), t.y1(), t.n(), t.Ny());
    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > interX = dg::create::fast_projection( gx, divideX, no);
    interX.get_matrices()[0].left_size = t.n()*t.Ny()*t.Nz();
    interX.get_matrices()[0].set_default_range();
    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > interY = dg::create::fast_projection( gy, divideY, no);
    interY.get_matrices()[0].right_size = t.n()*t.Nx()/divideX;
    interY.get_matrices()[0].left_size = t.Nz();
    interY.get_matrices()[0].set_default_range();

    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > inter(2);
    inter.get_matrices()[0] = interX.get_matrices()[0];
    inter.get_matrices()[1] = interY.get_matrices()[0];
    thrust::host_vector<double> vec( t.size()/divideX);
    inter.get_temp()[0] = Buffer<thrust::host_vector<double> >(vec);
    return inter;
}

#ifdef MPI_VERSION
//very elaborate way of telling the compiler to just apply the local matrix to the local vector
MultiMatrix< RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH >, MPI_Vector<thrust::host_vector<double> > > fast_interpolation( const aMPITopology2d& t, unsigned divideX, unsigned divideY)
{
    typedef RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> Matrix; 
    typedef MPI_Vector<thrust::host_vector<double> > Vector; 
    MultiMatrix<EllSparseBlockMat<double>, thrust::host_vector<double> > temp = dg::create::fast_interpolation( t.local(), divideX, divideY);
    MultiMatrix< Matrix, Vector > inter(2); 
    inter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<double>(), NNCH());
    inter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<double>(), NNCH());
    inter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return inter;
}
MultiMatrix< RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH >, MPI_Vector<thrust::host_vector<double> > > fast_projection( const aMPITopology2d& t, unsigned divideX, unsigned divideY, enum dg::norm no = normed)
{
    typedef RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> Matrix; 
    typedef MPI_Vector<thrust::host_vector<double> > Vector; 
    MultiMatrix<EllSparseBlockMat<double>, thrust::host_vector<double> > temp = dg::create::fast_projection( t.local(), divideX, divideY, no);
    MultiMatrix< Matrix, Vector > inter(2); 
    inter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<double>(), NNCH());
    inter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<double>(), NNCH());
    inter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return inter;
}

MultiMatrix< RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH >, MPI_Vector<thrust::host_vector<double> > > fast_interpolation( const aMPITopology3d& t, unsigned divideX, unsigned divideY)
{
    typedef RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> Matrix; 
    typedef MPI_Vector<thrust::host_vector<double> > Vector; 
    MultiMatrix<EllSparseBlockMat<double>, thrust::host_vector<double> > temp = dg::create::fast_interpolation( t.local(), divideX, divideY);
    MultiMatrix< Matrix, Vector > inter(2); 
    inter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<double>(), NNCH());
    inter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<double>(), NNCH());
    inter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return inter;
}

MultiMatrix< RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH >, MPI_Vector<thrust::host_vector<double> > > fast_projection( const aMPITopology3d& t, unsigned divideX, unsigned divideY, enum dg::norm no = normed)
{
    typedef RowColDistMat<EllSparseBlockMat<double>, CooSparseBlockMat<double>, NNCH> Matrix; 
    typedef MPI_Vector<thrust::host_vector<double> > Vector; 
    MultiMatrix<EllSparseBlockMat<double>, thrust::host_vector<double> > temp = dg::create::fast_projection( t.local(), divideX, divideY, no);
    MultiMatrix< Matrix, Vector > inter(2); 
    inter.get_matrices()[0] = Matrix( temp.get_matrices()[0], CooSparseBlockMat<double>(), NNCH());
    inter.get_matrices()[1] = Matrix( temp.get_matrices()[1], CooSparseBlockMat<double>(), NNCH());
    inter.get_temp()[0] = Buffer<Vector> ( Vector( temp.get_temp()[0].data(), t.communicator())  );
    return inter;
}

#endif //MPI_VERSION
}//namespace create

///@endcond
}//namespace dg
