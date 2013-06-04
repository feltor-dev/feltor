#ifndef _DG_ARAKAWA_CUH
#define _DG_ARAKAWA_CUH

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dlt.h"
#include "vector_traits.h"

#include "derivatives.cuh"

/*! @file 
  
  objects for computation of Poisson bracket
  */

namespace dg
{


/**
 * @brief L-space generalized version of Arakawa's scheme
 *
 * @ingroup creation
 * @tparam T value-type
 * @tparam n # of polynomial coefficients
 * @tparam container The vector class on which to operate on
 */
template< class T, size_t n, class container=thrust::device_vector<T> >
struct Arakawa
{
    typedef T value_type;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, T, MemorySpace> Matrix;
    /**
     * @brief Create Arakawa on a grid
     *
     * @param g The 2D grid
     */
    Arakawa( const Grid<T,n>& g);
    /**
     * @brief Create Arakawa on a grid using different boundary conditions
     *
     * @param g The 2D grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    Arakawa( const Grid<T,n>& g, bc bcx, bc bcy);

    /**
     * @brief Compute poisson's bracket
     *
     * @param lhs left hand side in l-space
     * @param rhs rights hand side in l-space
     * @param result Poisson's bracket in l-space
     */
    void operator()( const container& lhs, const container& rhs, container& result);
    const Matrix& forward2d() {return forward;}
    const Matrix& backward2d() {return backward;}

  private:
    //typedef typename VectorTraits< Vector>::value_type value_type;
    Matrix bdxf, bdyf, forward, backward;
    container dxlhs, dylhs, dxrhs, dyrhs, blhs, brhs;
};

//idea: backward transform lhs and rhs and then use bdxf and bdyf , then forward transform
//needs less memory!! and is faster
template< class T, size_t n, class container>
Arakawa<T, n, container>::Arakawa( const Grid<T,n>& g, bc bcx, bc bcy): dxlhs( n*n*g.Nx()*g.Ny()), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), blhs( dxlhs), brhs( blhs)
{
    //create forward dlt matrix
    Operator<value_type, n> forward1d( DLT<n>::forward);
    Operator<value_type, n*n> forward2d = tensor( forward1d, forward1d);
    forward = tensor( g.Nx()*g.Ny(), forward2d);
    //create backward dlt matrix
    Operator<value_type, n> backward1d( DLT<n>::backward);
    Operator<value_type, n*n> backward2d = tensor( backward1d, backward1d);
    backward = tensor( g.Nx()*g.Ny(), backward2d);

    /*
    typedef cusp::coo_matrix<int, value_type, MemorySpace> HMatrix;
    //create derivatives
    HMatrix dx = create::dx_symm<T,n>( Nx, hx, bcx);
    HMatrix dy = create::dx_symm<T,n>( Ny, hy, bcy);
    HMatrix fx = tensor( Nx, forward1d);
    HMatrix fy = tensor( Ny, forward1d);
    HMatrix bx = tensor( Nx, backward1d);
    HMatrix by = tensor( Ny, backward1d);
    HMatrix dxf( dx), dyf( dy), bdxf_(dx), bdyf_(dy);

    cusp::multiply( dx, fx, dxf);
    cusp::multiply( bx, dxf, bdxf_);
    cusp::multiply( dy, fy, dyf);
    cusp::multiply( by, dyf, bdyf_);

    HMatrix bdxf__ = dgtensor<T,n>( tensor<T,n>( Ny, delta), bdxf_ );
    HMatrix bdyf__ = dgtensor<T,n>(  bdyf_, tensor<T,n>( Nx, delta));

    bdxf = bdxf__;
    bdyf = bdyf__;
    */
    bdxf = dg::create::dx( g, bcx, XSPACE);
    bdyf = dg::create::dy( g, bcy, XSPACE);
}
template< class T, size_t n, class container>
Arakawa<T, n, container>::Arakawa( const Grid<T,n>& g): dxlhs( n*n*g.Nx()*g.Ny()), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), blhs( dxlhs), brhs( blhs)
{
    //create forward dlt matrix
    Operator<value_type, n> forward1d( DLT<n>::forward);
    Operator<value_type, n*n> forward2d = tensor( forward1d, forward1d);
    forward = tensor( g.Nx()*g.Ny(), forward2d);
    //create backward dlt matrix
    Operator<value_type, n> backward1d( DLT<n>::backward);
    Operator<value_type, n*n> backward2d = tensor( backward1d, backward1d);
    backward = tensor( g.Nx()*g.Ny(), backward2d);

    bdxf = dg::create::dx( g, g.bcx(), XSPACE);
    bdyf = dg::create::dy( g, g.bcy(), XSPACE);
}

template< class T, size_t n, class container>
void Arakawa<T, n, container>::operator()( const container& lhs, const container& rhs, container& result)
{
    //transform to x-space
    blas2::symv( backward, lhs, blhs);
    blas2::symv( backward, rhs, brhs);
    cudaThreadSynchronize();
    //compute derivatives in x-space
    blas2::symv( bdxf, blhs, dxlhs);
    blas2::symv( bdyf, blhs, dylhs);
    blas2::symv( bdxf, brhs, dxrhs);
    blas2::symv( bdyf, brhs, dyrhs);

    // order is important now
    // +x (1) -> result und (2) -> blhs
    blas1::pointwiseDot( blhs, dyrhs, result);
    blas1::pointwiseDot( blhs, dxrhs, blhs);
    cudaThreadSynchronize();

    // ++ (1) -> dyrhs and (2) -> dxrhs
    blas1::pointwiseDot( dxlhs, dyrhs, dyrhs);
    blas1::pointwiseDot( dylhs, dxrhs, dxrhs);
    cudaThreadSynchronize();

    // x+ (1) -> dxlhs and (2) -> dylhs
    blas1::pointwiseDot( dxlhs, brhs, dxlhs);
    blas1::pointwiseDot( dylhs, brhs, dylhs);
    cudaThreadSynchronize();

    blas1::axpby( 1./3., dyrhs, -1./3., dxrhs);  //dxl*dyr - dyl*dxr -> dxrhs
    //everything which needs a dx 
    blas1::axpby( 1./3., dxlhs, -1./3., blhs);   //dxl*r - l*dxr     -> blhs 
    //everything which needs a dy
    blas1::axpby( 1./3., result, -1./3., dylhs); //l*dyr - dyl*r     -> dylhs

    //blas1::axpby( 1., dyrhs,  -1., dxrhs);
    ////for testing purposes (note that you need to set criss-cross)
    //blas1::axpby( 0., dxlhs,  -0., blhs);
    //blas1::axpby( 0., result, -0., dylhs);

    cudaThreadSynchronize();
    blas2::symv( bdyf, blhs, result);      //dy*(dxl*r - l*dxr) -> result
    blas2::symv( bdxf, dylhs, dxlhs);      //dx*(l*dyr - dyl*r) -> dxlhs
    //now sum everything up
    cudaThreadSynchronize();
    blas1::axpby( 1., result, 1., dxlhs); //result + dxlhs -> result
    cudaThreadSynchronize();
    blas1::axpby( 1., dxrhs, 1., dxlhs); //result + dyrhs -> result
    //transform to l-space
    blas2::symv( forward, dxlhs, result);
}


//saves about 20% time and needs less memory
/**
 * @brief X-space generalized version of Arakawa's scheme
 *
 * @ingroup creation
 * @tparam T value-type
 * @tparam n # of polynomial coefficients
 * @tparam container The vector class on which to operate on
 */
template< class T, size_t n, class container=thrust::device_vector<T> >
struct ArakawaX
{
    typedef T value_type;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    /**
     * @brief Create Arakawa on a grid
     *
     * @param g The 2D grid
     */
    ArakawaX( const Grid<T,n>& g);
    /**
     * @brief Create Arakawa on a grid using different boundary conditions
     *
     * @param g The 2D grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    ArakawaX( const Grid<T,n>& g, bc bcx, bc bcy);
    //ArakawaX( unsigned Nx, unsigned Ny, double hx, double hy, int bcx, int bcy); //deprecated

    /**
     * @brief Compute poisson's bracket
     *
     * @param lhs left hand side in x-space
     * @param rhs rights hand side in x-space
     * @param result Poisson's bracket in x-space
     */
    void operator()( const container& lhs, const container& rhs, container& result);

    /**
     * @brief Return internally used 2d - x - derivative in ell format in XSPACE
     *
     * The same as a call to 
     * dg::create::dx( g, bcx, XSPACE)
     * but the format is the fast ell_matrix format
     * @return derivative
     */
    const Matrix& dx() {return bdxf;}
    /**
     * @brief Return internally used 2d - y - derivative in ell format in XSPACE
     *
     * The same as a call to 
     * dg::create::dy( g, bcy, XSPACE)
     * but the format is the fast ell_matrix format
     * @return derivative
     */
    const Matrix& dy() {return bdyf;}

  private:
    //typedef typename VectorTraits< Vector>::value_type value_type;
    //void construct( unsigned Nx, unsigned Ny, double hx, double hy, int bcx, int bcy);
    Matrix bdxf, bdyf;
    container dxlhs, dylhs, dxrhs, dyrhs, helper;
};

//idea: backward transform lhs and rhs and then use bdxf and bdyf , then forward transform
//needs less memory!! and is faster
template< class T, size_t n, class container>
ArakawaX<T, n, container>::ArakawaX( const Grid<T,n>& g): dxlhs( n*n*g.Nx()*g.Ny()), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper( dxlhs)
{
    bdxf = dg::create::dx( g, g.bcx(), XSPACE);
    bdyf = dg::create::dy( g, g.bcy(), XSPACE);
}
template< class T, size_t n, class container>
ArakawaX<T, n, container>::ArakawaX( const Grid<T,n>& g, bc bcx, bc bcy): dxlhs( n*n*g.Nx()*g.Ny()), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper( dxlhs)
{
    bdxf = dg::create::dx( g, bcx, XSPACE);
    bdyf = dg::create::dy( g, bcy, XSPACE);
}
/*
template< class T, size_t n, class container>
ArakawaX<T, n, container>::ArakawaX( unsigned Nx, unsigned Ny, double hx, double hy, int bcx, int bcy): dxlhs( n*n*Nx*Ny), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper( dxlhs)
{
    construct( Nx, Ny, hx, hy, bcx, bcy);
}
template< class T, size_t n, class container>
void ArakawaX<T, n, container>::construct( unsigned Nx, unsigned Ny, double hx, double hy, int bcx, int bcy)
{
    typedef cusp::coo_matrix<int, value_type, MemorySpace> HMatrix;

    //create forward dlt matrix
    Operator<value_type, n> forward1d( DLT<n>::forward);
    //create backward dlt matrix
    Operator<value_type, n> backward1d( DLT<n>::backward);
    //create derivatives
    HMatrix dx = create::dx_symm<T,n>( Nx, hx, bcx);
    HMatrix dy = create::dx_symm<T,n>( Ny, hy, bcy);
    HMatrix fx = tensor( Nx, forward1d);
    HMatrix fy = tensor( Ny, forward1d);
    HMatrix bx = tensor( Nx, backward1d);
    HMatrix by = tensor( Ny, backward1d);
    HMatrix dxf( dx), dyf( dy), bdxf_(dx), bdyf_(dy);

    cusp::multiply( dx, fx, dxf);
    cusp::multiply( bx, dxf, bdxf_);
    cusp::multiply( dy, fy, dyf);
    cusp::multiply( by, dyf, bdyf_);

    HMatrix bdxf__ = dgtensor<T,n>( tensor<T,n>( Ny, delta), bdxf_ );
    HMatrix bdyf__ = dgtensor<T,n>(  bdyf_, tensor<T,n>( Nx, delta));

    bdxf = bdxf__;
    bdyf = bdyf__;
}
*/

template< class T, size_t n, class container>
void ArakawaX<T, n, container>::operator()( const container& lhs, const container& rhs, container& result)
{
    //compute derivatives in x-space
    blas2::symv( bdxf, lhs, dxlhs);
    blas2::symv( bdyf, lhs, dylhs);
    blas2::symv( bdxf, rhs, dxrhs);
    blas2::symv( bdyf, rhs, dyrhs);

    // order is important now
    // +x (1) -> result und (2) -> blhs
    blas1::pointwiseDot( lhs, dyrhs, result);
    blas1::pointwiseDot( lhs, dxrhs, helper);
    cudaThreadSynchronize();

    // ++ (1) -> dyrhs and (2) -> dxrhs
    blas1::pointwiseDot( dxlhs, dyrhs, dyrhs);
    blas1::pointwiseDot( dylhs, dxrhs, dxrhs);
    cudaThreadSynchronize();

    // x+ (1) -> dxlhs and (2) -> dylhs
    blas1::pointwiseDot( dxlhs, rhs, dxlhs);
    blas1::pointwiseDot( dylhs, rhs, dylhs);
    cudaThreadSynchronize();

    blas1::axpby( 1./3., dyrhs, -1./3., dxrhs);  //dxl*dyr - dyl*dxr -> dxrhs
    //everything which needs a dx 
    blas1::axpby( 1./3., dxlhs, -1./3., helper);   //dxl*r - l*dxr     -> helper 
    //everything which needs a dy
    blas1::axpby( 1./3., result, -1./3., dylhs); //l*dyr - dyl*r     -> dylhs

    //blas1::axpby( 1., dyrhs,  -1., dxrhs);
    ////for testing purposes (note that you need to set criss-cross)
    //blas1::axpby( 0., dxlhs,  -0., blhs);
    //blas1::axpby( 0., result, -0., dylhs);

    cudaThreadSynchronize();
    blas2::symv( bdyf, helper, result);      //dy*(dxl*r - l*dxr) -> result
    blas2::symv( bdxf, dylhs, dxlhs);      //dx*(l*dyr - dyl*r) -> dxlhs
    //now sum everything up
    cudaThreadSynchronize();
    blas1::axpby( 1., dxlhs, 1., result); //result + dxlhs -> result
    cudaThreadSynchronize();
    blas1::axpby( 1., dxrhs, 1., result); //result + dyrhs -> result
}

}//namespace dg

#endif //_DG_ARAKAWA_CUH
