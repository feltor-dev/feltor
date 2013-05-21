#ifndef _DG_XSPACELIB_CUH_
#define _DG_XSPACELIB_CUH_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cusp/coo_matrix.h>
#include <cusp/ell_matrix.h>


//functions for evaluation
#include "grid.cuh"
#include "arrvec2d.cuh"
#include "dlt.h"
#include "evaluation.cuh"

//creational functions
#include "creation.cuh"
#include "dx.cuh"
#include "functions.h"
#include "functors.cuh"
#include "laplace.cuh"
#include "operator.cuh"
#include "operator_matrix.cuh"
#include "tensor.cuh"
#include "arakawa.cuh"


namespace dg
{

typedef thrust::device_vector<double> DVec;
typedef thrust::host_vector<double> HVec;

typedef cusp::coo_matrix<int, double, cusp::host_memory> CMatrix;

template< class Function, size_t n>
HVec evaluate( Function& f, const Grid<double,n>& g)
{
    return (evaluate<Function, n>( f, g.x0(), g.x1(), g.y0(), g.y1(), g.Nx(), g.Ny() )).data();
};
template< size_t n>
HVec evaluate( double(f)(double, double), const Grid<double,n>& g)
{
    //return evaluate<double(&)(double, double), n>( f, g );
    return (evaluate<double(&)(double, double), n>( *f, g.x0(), g.x1(), g.y0(), g.y1(), g.Nx(), g.Ny() )).data();
};

namespace create{


template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid<T, n>& g, bc bcx)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    int bound = ( bcx == PER )? -1 : 0; 
    Matrix dx = create::dx_symm<T,n>( g.Nx(), g.hx(), bound);
    Matrix bdxf = sandwich<T,n>( dx);

    return dgtensor<T,n>( tensor<T,n>( g.Ny(), delta), bdxf );
}
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid<T, n>& g) { return dx( g, g.bcx());}

template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid<T, n>& g, bc bcy)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    int bound = ( bcy == PER )? -1 : 0; 
    Matrix dy = create::dx_symm<T,n>( g.Ny(), g.hy(), bound);
    Matrix bdyf_ = sandwich<T,n>( dy);

    return dgtensor<T,n>( bdyf_, tensor<T,n>( g.Nx(), delta));
}
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid<T, n>& g){ return dy( g, g.bcy());}

template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> laplacian( const Grid<T, n>& g, bc bcx, bc bcy, bool normalized = true)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    Operator<T, n> forward1d( DLT<n>::forward);
    Operator<T, n> right( forward1d);
    Operator<T,n> weights_invx(0.), weights_invy(0.);
    for( unsigned i=0; i<n; i++)
    {
        weights_invx[i] = 2./g.hx()/DLT<n>::weight[i];
        weights_invy[i] = 2./g.hy()/DLT<n>::weight[i];
    }
    Operator<T,n> leftx( right.transpose() ), lefty( right.transpose());
    if( normalized) 
    {
        leftx = weights_invx*leftx;
        lefty = weights_invy*lefty;
    }

    Matrix ly;
    if( bcy == PER) 
        ly = create::laplace1d_per<double,  n>( g.Ny(), g.hy());
    else if( bcy == DIR) 
        ly = create::laplace1d_dir<double,  n>( g.Ny(), g.hy());
    Matrix lx;
    if( bcx == PER) 
        lx = create::laplace1d_per<double,  n>( g.Nx(), g.hx());
    else if( bcx == DIR) 
        lx = create::laplace1d_dir<double,  n>( g.Nx(), g.hx());

    Matrix flxf = sandwich<T,n>( leftx, lx, right);
    Matrix flyf = sandwich<T,n>( lefty, ly, right);

    Matrix ddyy = dgtensor<double, n>( flyf, tensor<double, n>( Nx, pipj));
    Matrix ddxx = dgtensor<double, n>( tensor<double, n>( Ny, pipj), flxf);
    Matrix laplace( ddxx);
    cusp::add( ddxx, ddyy, laplace);
    return laplace;
}

template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> laplacian( const Grid<T, n>& g, bool normalized = true)
{
    return laplacian( g, g.bcx(), b.bcy(), normalized);
}

} //namespace create

}//namespace dg

#endif // _DG_XSPACELIB_CUH_
