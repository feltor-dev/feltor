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
    //create forward dlt matrix
    Operator<T, n> forward1d( DLT<n>::forward);
    //create backward dlt matrix
    Operator<T, n> backward1d( DLT<n>::backward);
    //create derivatives
    Matrix dx = create::dx_symm<T,n>( g.Nx(), g.hx(), bound);
    Matrix fx = tensor( g.Nx(), forward1d);
    Matrix bx = tensor( g.Nx(), backward1d);
    Matrix dxf( dx), bdxf_(dx);

    cusp::multiply( dx, fx, dxf);
    cusp::multiply( bx, dxf, bdxf_);

    return dgtensor<T,n>( tensor<T,n>( g.Ny(), delta), bdxf_ );
}
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid<T, n>& g) { return dx( g, g.bcx());}

template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid<T, n>& g, bc bcy)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    int bound = ( bcy == PER )? -1 : 0; 
    //create forward dlt matrix
    Operator<T, n> forward1d( DLT<n>::forward);
    //create backward dlt matrix
    Operator<T, n> backward1d( DLT<n>::backward);
    //create derivatives
    Matrix dy = create::dx_symm<T,n>( g.Ny(), g.hy(), bound);
    Matrix fy = tensor( g.Ny(), forward1d);
    Matrix by = tensor( g.Ny(), backward1d);
    Matrix dyf( dy), bdyf_(dy);

    cusp::multiply( dy, fy, dyf);
    cusp::multiply( by, dyf, bdyf_);

    return dgtensor<T,n>( bdyf_, tensor<T,n>( g.Nx(), delta));
}
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid<T, n>& g){ return dy( g, g.bcy());}
} //namespace create

}//namespace dg

#endif // _DG_XSPACELIB_CUH_
