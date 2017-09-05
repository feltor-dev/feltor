#pragma once

#include <thrust/host_vector.h>
#include "grid.h"
#include "interpolation.cuh"
#include "projection.cuh"
#include "matrix_traits.h"
#include "sparseblockmat.h"
#include "memory.h"

namespace dg
{

template <class Matrix, class container>
struct MultiMatrix
{
    MultiMatrix(){}

    MultiMatrix( int dimension): inter_(dimension), temp_(dimension-1 > 0 ? dimension-1 : 0 ){}
    template<class OtherMatrix, class OtherContainer>
    MultiMatrix( const MultiMatrix<OtherMatrix, OtherContainer>& src){
            temp_.resize( src.get_temp().size());
            inter_.resize( src.get_matrices().size());
            for( int i=0; i<temp_.size(); i++)
            {
                temp_[i].data() = src.get_temp()[i].data();
                inter_[i] = src.get_matrices()[i];
            }
    }

    void symv( const container& x, container& y) const{gemv(x,y);}
    void gemv( const container& x, container& y) const
    {
        int dims = inter_.size();
        if( dims == 1) 
        {
            dg::blas2::symv( inter_[0], x, y);
            return;
        }
        dg::blas2::symv( inter_[0], x,temp_[0].data());
        for( int i=1; i<dims-1; i++)
            dg::blas2::symv( inter_[i], temp_[i-1].data(), temp_[i].data());
        dg::blas2::symv( inter_[dims-1], temp_[dims-2].data(), y);
    }
    std::vector<Buffer<container> >& get_temp(){ return temp_;}
    const std::vector<Buffer<container> >& get_temp()const{ return temp_;}
    std::vector<Matrix>& get_matrices(){ return inter_;}
    const std::vector<Matrix>& get_matrices()const{ return inter_;}
    private:
    std::vector<Matrix > inter_;
    std::vector<Buffer<container> > temp_;
};

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
MultiMatrix< EllSparseBlockMat<double>, thrust::host_vector<double> > fast_interpolation( const aTopology2d& t, unsigned multiplyX, unsigned multiplyY)
{
    unsigned n=t.n();
    dg::Grid1d g_old( -1., 1., n, 1);
    dg::Grid1d g_newX( -1., 1., n, multiplyX);
    dg::Grid1d g_newY( -1., 1., n, multiplyY);
    dg::IHMatrix interpolX = dg::create::interpolation( g_newX, g_old);
    dg::IHMatrix interpolY = dg::create::interpolation( g_newY, g_old);
    EllSparseBlockMat<double> iX( multiplyX*t.Nx(), t.Nx(), 1, multiplyX, t.n()); 
    EllSparseBlockMat<double> iY( multiplyY*t.Ny(), t.Ny(), 1, multiplyY, t.n()); 
    for( unsigned  i=0; i<n; i++)
    for( unsigned  j=0; j<n; j++)
    for( unsigned  k=0; k<multiplyX; k++)
        iX.data[(k*n+i)*n+j] = interpolX.values[(k*n+i)*n+j];
    for( unsigned  i=0; i<n; i++)
    for( unsigned  j=0; j<n; j++)
    for( unsigned  k=0; k<multiplyY; k++)
        iY.data[(k*n+i)*n+j] = interpolY.values[(k*n+i)*n+j];

    for( unsigned i=0; i<multiplyX*t.Nx(); i++)
    {
        iX.cols_idx[i] = i/multiplyX;
        iX.data_idx[i] = 0;
    }
    for( unsigned i=0; i<multiplyY*t.Ny(); i++)
    {
        iY.cols_idx[i] = i/multiplyY;
        iY.data_idx[i] = 0;
    }
    iX.left_size  = t.n()*t.Ny();
    iY.right_size = t.n()*t.Nx()*multiplyX;
    iX.set_default_range();
    iY.set_default_range();

    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > inter(2);
    inter.get_matrices()[0] = iX;
    inter.get_matrices()[1] = iY;
    thrust::host_vector<double> vec( t.size()*multiplyX);
    inter.get_temp()[0] = Buffer<thrust::host_vector<double > >(vec);
    return inter;
}

MultiMatrix< EllSparseBlockMat<double>, thrust::host_vector<double> > fast_projection( const aTopology2d& t, unsigned divideX, unsigned divideY)
{
    unsigned n=t.n();
    if( t.Nx()%divideX != 0) throw Error( Message(_ping_)<< "Nx and divideX don't match: Nx: " << t.Nx()<< " divideX "<< divideX);
    if( t.Ny()%divideY != 0) throw Error( Message(_ping_)<< "Ny and divideY don't match: Nx: " << t.Ny()<< " divideY "<< divideY);
    dg::Grid1d g_oldX( -1., 1., n, divideX);
    dg::Grid1d g_oldY( -1., 1., n, divideY);
    dg::Grid1d g_new(  -1., 1., n, 1);
    dg::IHMatrix projectX = dg::create::projection( g_new, g_oldX);
    dg::IHMatrix projectY = dg::create::projection( g_new, g_oldY);
    EllSparseBlockMat<double> pX( t.Nx()/divideX, t.Nx(), divideX, divideY, t.n()); 
    EllSparseBlockMat<double> pY( t.Ny()/divideY, t.Ny(), divideY, divideY, t.n()); 
    for( unsigned i=0; i<n; i++)
    for( unsigned j=0; j<n; j++)
    {
        for( unsigned k=0; k<divideX; k++)
            pX.data[(k*n+i)*n+j] = projectX.values[i*divideX*n +k*n+j];
        for( unsigned k=0; k<divideY; k++)
            pY.data[(k*n+i)*n+j] = projectY.values[i*divideX*n +k*n+j];
    }

    for( unsigned i=0; i<t.Nx()/divideX; i++)
        for( unsigned d=0; d<divideX; d++)
        {
            pX.cols_idx[i*divideX+d] = i*divideX+d;
            pX.data_idx[i*divideX+d] = d;
        }
    for( unsigned i=0; i<t.Ny()/divideY; i++)
        for( unsigned d=0; d<divideY; d++)
        {
            pY.cols_idx[i*divideY+d] = i*divideY+d;
            pY.data_idx[i*divideY+d] = d;
        }
    pX.left_size  = t.n()*t.Ny();
    pY.right_size = t.n()*t.Nx()/divideX;
    pX.set_default_range();
    pY.set_default_range();

    MultiMatrix < EllSparseBlockMat<double>, thrust::host_vector<double> > inter(2);
    inter.get_matrices()[0] = pX;
    inter.get_matrices()[1] = pY;
    thrust::host_vector<double> vec( t.size()/divideX);
    inter.get_temp()[0] = Buffer<thrust::host_vector<double> >(vec);
    return inter;
}

}//namespace create

}//namespace dg
