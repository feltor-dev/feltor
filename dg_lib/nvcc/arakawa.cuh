#ifndef _DG_ARAKAWA_CUH
#define _DG_ARAKAWA_CUH

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dlt.h"
#include "vector_traits.h"

#include "grid.cuh"
#include "functions.h"
#include "tensor.cuh"
#include "operator_matrix.cuh"
#include "dx.cuh"

namespace dg
{

template< class T, size_t n, class container=thrust::device_vector<T> >
struct Arakawa
{
    typedef T value_type;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    Arakawa( unsigned Nx, unsigned Ny, double hx, double hy, int bcx, int bcy);

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
Arakawa<T, n, container>::Arakawa( unsigned Nx, unsigned Ny, double hx, double hy, int bcx, int bcy): dxlhs( n*n*Nx*Ny), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), blhs( n*n*Nx*Ny), brhs( blhs)
{
    typedef cusp::coo_matrix<int, value_type, MemorySpace> HMatrix;

    //create forward dlt matrix
    Operator<value_type, n> forward1d( DLT<n>::forward);
    Operator<value_type, n*n> forward2d = tensor( forward1d, forward1d);
    forward = tensor( Nx*Ny, forward2d);

    //create backward dlt matrix
    Operator<value_type, n> backward1d( DLT<n>::backward);
    Operator<value_type, n*n> backward2d = tensor( backward1d, backward1d);
    backward = tensor( Nx*Ny, backward2d);

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

    /*
    //test is really the same
    ArrVec2d<T, n, container> direct( test, Nx), indirect1(direct), indirect2(direct);
    blas2::symv( dy, test, direct.data());
    std::cout << "Direct\n"<<direct;
    blas2::symv( bdy, test, indirect1.data());
    blas2::symv( bdx, test, indirect2.data());
    std::cout << "InDirect1\n"<<indirect1;
    std::cout << "InDirect2\n"<<indirect2;
    blas1::pointwiseDot( indirect1.data(), indirect2.data(), indirect2.data());
    std::cout << "InDirect12\n"<<indirect2;
    blas2::symv( forward, indirect2.data(), test);
    indirect2.data() = test;
    std::cout << "InDirect\n"<<indirect2;
    */
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


//saves 20% time
template< class T, size_t n, class container=thrust::device_vector<T> >
struct ArakawaX
{
    typedef T value_type;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    ArakawaX( const Grid<T,n>& g);
    ArakawaX( const Grid<T,n>& g, bc bcx, bc bcy);
    ArakawaX( unsigned Nx, unsigned Ny, double hx, double hy, int bcx, int bcy);

    void operator()( const container& lhs, const container& rhs, container& result);
    const Matrix& dx2d() {return bdxf;}
    const Matrix& dy2d() {return bdyf;}

  private:
    //typedef typename VectorTraits< Vector>::value_type value_type;
    void construct( unsigned Nx, unsigned Ny, double hx, double hy, int bcx, int bcy);
    Matrix bdxf, bdyf;
    container dxlhs, dylhs, dxrhs, dyrhs, helper;
};

//idea: backward transform lhs and rhs and then use bdxf and bdyf , then forward transform
//needs less memory!! and is faster
template< class T, size_t n, class container>
ArakawaX<T, n, container>::ArakawaX( const Grid<T,n>& g): dxlhs( n*n*g.Nx()*g.Ny()), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper( dxlhs)
{
    int bx = (g.bcx() == PER)?-1:0;
    int by = (g.bcy() == PER)?-1:0;
    construct( g.Nx(), g.Ny(), g.hx(), g.hy(), bx, by);
}
template< class T, size_t n, class container>
ArakawaX<T, n, container>::ArakawaX( const Grid<T,n>& g, bc bcx, bc bcy): dxlhs( n*n*g.Nx()*g.Ny()), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper( dxlhs)
{
    int bx = (bcx == PER)?-1:0;
    int by = (bcy == PER)?-1:0;
    construct( g.Nx(), g.Ny(), g.hx(), g.hy(), bx, by);
}
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
