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
#include "polarisation.cuh"

//integral functions
#include "preconditioner.cuh"

namespace dg
{

typedef thrust::device_vector<double> DVec; //!< Device Vector
typedef thrust::host_vector<double> HVec; //!< Host Vector

typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix; //!< default matrix
typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix; //!< most efficient matrix format
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix; //!< most efficient matrix format

enum space {XSPACE, LSPACE};

namespace create{


template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid<T, n>& g, bc bcx, space s = XSPACE)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    int bound = ( bcx == PER )? -1 : 0; 
    Matrix dx = create::dx_symm<T,n>( g.Nx(), g.hx(), bound);
    Matrix bdxf( dx);
    if( s == XSPACE)
        bdxf = sandwich<T,n>( dx);

    return dgtensor<T,n>( tensor<T,n>( g.Ny(), delta), bdxf );
}
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dx( const Grid<T, n>& g, space s = XSPACE) { return dx( g, g.bcx(), s);}

template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid<T, n>& g, bc bcy, space s = XSPACE)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    int bound = ( bcy == PER )? -1 : 0; 
    Matrix dy = create::dx_symm<T,n>( g.Ny(), g.hy(), bound);
    Matrix bdyf_(dy);
    if( s == XSPACE)
        bdyf_ = sandwich<T,n>( dy);

    return dgtensor<T,n>( bdyf_, tensor<T,n>( g.Nx(), delta));
}
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> dy( const Grid<T, n>& g, space s = XSPACE){ return dy( g, g.bcy(), s);}

//the behaviour of CG is completely the same in xspace as in lspace
template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> laplacian( const Grid<T, n>& g, bc bcx, bc bcy, norm no = normed, space s = XSPACE)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;

    Matrix ly;
    if( bcy == PER) 
        ly = create::laplace1d_per<double,  n>( g.Ny(), g.hy(), no);
    else if( bcy == DIR) 
        ly = create::laplace1d_dir<double,  n>( g.Ny(), g.hy(), no);
    Matrix lx;
    if( bcx == PER) 
        lx = create::laplace1d_per<double,  n>( g.Nx(), g.hx(), no);
    else if( bcx == DIR) 
        lx = create::laplace1d_dir<double,  n>( g.Nx(), g.hx(), no);

    Matrix flxf(lx), flyf(ly);
    //sandwich with correctly normalized matrices
    if( s == XSPACE)
    {
        Operator<T, n> forward1d( DLT<n>::forward);
        Operator<T, n> backward1d( DLT<n>::backward);
        Operator<T,n> leftx( backward1d ), lefty( backward1d);
        if( no == not_normed)
            leftx = lefty = forward1d.transpose();

        flxf = sandwich<T,n>( leftx, lx, forward1d);
        flyf = sandwich<T,n>( lefty, ly, forward1d);
    }
    Operator<T,n> normx(0.), normy(0.);
    dg::W1D<T,n> w1dx( g.hx()), w1dy( g.hy());
    dg::S1D<T,n> s1dx( g.hx()), s1dy( g.hy());
    //generate norm
    for( unsigned i=0; i<n; i++)
    {
        if( no == not_normed) 
        {
            if( s==XSPACE)
            {
                normx(i,i) = w1dx(i);
                normy(i,i) = w1dy(i);
            } else {
                normx(i,i) = s1dx(i);
                normy(i,i) = s1dy(i);
            }
        }
        else
            normx(i,i) = normy(i,i) = 1.;
    }

    Matrix ddyy = dgtensor<double, n>( flyf, tensor( g.Nx(), normx));
    Matrix ddxx = dgtensor<double, n>( tensor(g.Ny(), normy), flxf);
    Matrix laplace;
    cusp::add( ddxx, ddyy, laplace); //cusp add does not sort output!!!!
    laplace.sort_by_row_and_column();
    //std::cout << "Is sorted? "<<laplace.is_sorted_by_row_and_column()<<"\n";
    return laplace;
}

template< class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> laplacian( const Grid<T, n>& g, norm no = normed, space s = XSPACE)
{
    return laplacian( g, g.bcx(), g.bcy(), no, s);
}

/**
 * @brief make a matrix that transforms values to an equidistant grid ready for visualisation
 *
 * @tparam T value type
 * @tparam n # of polynomial coefficients
 * @param g The grid on which to operate 
 * @param forward whether the vectors are given in XSPACE or not
 *
 * @return transformation matrix
 */
template < class T, size_t n>
cusp::coo_matrix<int, T, cusp::host_memory> backscatter( const Grid<T,n>& g, bool forward = true)
{
    typedef cusp::coo_matrix<int, T, cusp::host_memory> Matrix;
    //create equidistant backward transformation
    dg::Operator<double, n> backwardeq( dg::DLT<n>::backwardEQ);
    dg::Operator<double, n*n> backward2d = dg::tensor( backwardeq, backwardeq);

    if( forward){
        dg::Operator<double, n> forward( dg::DLT<n>::forward);
        dg::Operator<double, n*n> forward2d = dg::tensor( forward, forward);
        backward2d = backward2d*forward2d;
    }

    Matrix backward = dg::tensor( g.Nx()*g.Ny(), backward2d);

    thrust::host_vector<int> map = dg::makePermutationMap<n>( g.Nx(), g.Ny());
    Matrix permutation( map.size(), map.size(), map.size());
    cusp::array1d<int, cusp::host_memory> rows( thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(map.size()));
    cusp::array1d<int, cusp::host_memory> cols( map.begin(), map.end());
    cusp::array1d<T, cusp::host_memory> values(map.size(), 1.);
    permutation.row_indices = rows;
    permutation.column_indices = cols;
    permutation.values = values;
    Matrix scatter( permutation);

    cusp::multiply( permutation, backward, scatter);
    return scatter;

}
} //namespace create

//should there be a utility for W2D?
/*
//are these really necessary?
template< class Vector, size_t n>
typename Vector::value_type dot( const Vector& x, const Vector& y, const Grid<typename Vector::value_type, n>& g)
{
    return blas2::dot( x, W2D<typename Vector::value_type, n>(g.hx(), g.hy()), y);
}
template< class Vector, size_t n>
typename Vector::value_type nrml2( const Vector& x, const Grid<typename Vector::value_type, n>& g)
{
    return sqrt(blas2::dot( W2D<typename Vector::value_type, n>(g.hx(), g.hy()), x));
}
template< class Vector, size_t n>
typename Vector::value_type integ( const Vector& x, const Grid<typename Vector::value_type, n>& g)
{
    Vector one(x.size(), 1.);
    return dot( x, one, g);
}
*/


}//namespace dg

#endif // _DG_XSPACELIB_CUH_
