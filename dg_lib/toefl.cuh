#ifndef _DG_TOEFL_CUH
#define _DG_TOEFL_CUH

#include <cusp/ell_matrix.h>

#include "blas.h"
#include "dlt.h"
#include "vector_traits.h"

#include "cusp_eigen.h"
#include "arakawa.cuh"
#include "polarisation.cuh"
#include "laplace.cuh"
#include "functions.h"
#include "functors.cuh"
#include "tensor.cuh"
#include "operator_matrix.cuh"
#include "dx.cuh"
#include "cg.cuh"

namespace dg
{

struct Parameter
{
    double kappa;
    double a_i, a_z;
    double mu_i, mu_z;

    void check()
    {
        assert( fabs( a_i + a_z - 1.) > 1e-15 && "Background not neutral!");
    }
};

//Pb: what is DG expansion of ln(n)
template< class T, size_t n, class container=thrust::device_vector<T>, class MemorySpace = cusp::device_memory>
struct Toefl
{
    typedef std::vector<container> Vector;
    Toefl( unsigned Nx, unsigned Ny, double hx, double hy, Parameter p, double eps);

    void update_exponent( const std::vector<container>& y, std::vector<container>& target);
    void update_log( const std::vector<container>& y, std::vector<container>& target);
    const container& polarisation( const std::vector<container>& y);
    void operator()( const std::vector<container>& y, std::vector<container>& yp);
  private:
    typedef T value_type;
    //typedef typename VectorTraits< Vector>::value_type value_type;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;

    container omega, phi, chi, dyge, dyphi;
    cusp::array1d_view<typename container::iterator> chi_view;
    std::vector<container> expy;

    Matrix dy;
    Matrix A;
    Arakawa<T, n, container, MemorySpace> arakawa; 
    Polarisation2d<T, n, MemorySpace> pol;
    CG<Matrix, container, dg::T2D<T, n> > pcg;

    double hx, hy;
    Parameter p;
    double eps; 
};

template< class T, size_t n, class container, class MemorySpace>
Toefl<T, n, container, MemorySpace>::Toefl( unsigned Nx, unsigned Ny, double hx, double hy,
        Parameter p, double eps): 
    omega( n*n*Nx*Ny, 0.), phi(omega), chi(phi), dyge(omega), dyphi(omega), 
    chi_view( chi.begin(), chi.end()),
    expy( 3, omega),
    arakawa( Nx, Ny, hx, hy, 0, -1), 
    pol(     Nx, Ny, hx, hy, 0, -1),
    pcg( omega, n*n*Nx*Ny), 
    hx( hx), hy(hy), p(p), eps(eps)
{
    //create derivatives
    dy = dgtensor<T,n>( create::dx_symm<value_type,n>( Ny, hy, -1), tensor<T,n>(Nx, delta));

}

//how to set up a computation?
template< class T, size_t n, class container, class MemorySpace>
void Toefl<T, n, container, MemorySpace>::update_exponent( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::symv( arakawa.backward2d(), y[i], target[i]);
        thrust::transform( target[i].begin(), target[i].end(), target[i].begin(), dg::EXP<T>());
        blas2::symv( arakawa.forward2d(), target[i], target[i]);
    }
}
template< class T, size_t n, class container, class MemorySpace>
void Toefl<T, n, container, MemorySpace>::update_log( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::symv( arakawa.backward2d(), y[i], target[i]);
        thrust::transform( target[i].begin(), target[i].end(), target[i].begin(), dg::LN<T>());
        blas2::symv( arakawa.forward2d(), target[i], target[i]);
    }
}

template< class T, size_t n, class container, class MemorySpace>
const container& Toefl<T, n, container, MemorySpace>::polarisation( const std::vector<container>& y)
{

    //compute omega
    update_exponent( y, expy);
    blas1::axpby(     1., expy[0], 0., omega);
    blas1::axpby( -p.a_i, expy[1], 1., omega);
    blas1::axpby( -p.a_z, expy[2], 1., omega);
    //compute chi
    blas1::axpby( p.a_i*p.mu_i, expy[1], 0., chi);
    blas1::axpby( p.a_z*p.mu_z, expy[2], 1., chi);
    //compute S omega 
    blas2::symv( S2D<double, n>(hx, hy), omega, omega);
    cudaThreadSynchronize();
    A = pol.create( chi_view ); 
    unsigned number = pcg( A, phi, omega, T2D<double, n>(hx, hy), eps);
    std::cout << "Number of pcg iterations "<< number <<std::endl;
    return phi;

}

template< class T, size_t n, class container, class MemorySpace>
void Toefl<T, n, container, MemorySpace>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 3);
    assert( y.size() == yp.size());
    cudaThreadSynchronize();
    phi = polarisation( y);

    for( unsigned i=0; i<y.size(); i++)
        arakawa( y[i], phi, yp[i]);

    // curvature terms
    cudaThreadSynchronize();
    blas2::symv( dy, phi, dyphi);
    blas2::symv( dy, y[0], dyge);
    cudaThreadSynchronize();
    blas1::axpby( p.kappa, dyphi, 1., yp[0]);
    blas1::axpby( p.kappa, dyphi, 1., yp[1]);
    blas1::axpby( -p.kappa, dyge, 1., yp[0]);

}

}//namespace dg

#endif //_DG_TOEFL_CUH
