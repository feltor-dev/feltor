#ifndef _DG_TOEFL_CUH
#define _DG_TOEFL_CUH


#include "xspacelib.cuh"

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
template< class T, size_t n, class container=thrust::device_vector<T> >
struct Toefl
{
    typedef std::vector<container> Vector;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;
    Toefl( const Grid<T,n>& g Parameter p, double eps);

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

template< class T, size_t n, class container>
Toefl<T, n, container>::Toefl( const Grid<T,n>& g, Parameter p, double eps): 
    omega( n*n*Nx*Ny, 0.), phi(omega), chi(phi), dyge(omega), dyphi(omega), 
    chi_view( chi.begin(), chi.end()),
    expy( 3, omega),
    arakawa( grid, dg::DIR, dg::PER), 
    pol(     grid, dg::DIR, dg::PER), 
    pcg( omega, n*n*g.Nx()*g.Ny()), 
    hx( g.hx()), hy(g.hy()), p(p), eps(eps)
{
    //create derivatives
    dy = create::dx( grid, dg::PER);

}

//how to set up a computation?
template< class T, size_t n, class container>
void Toefl<T, n, container>::update_exponent( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::EXP<T>());
}
template< class T, size_t n, class container>
void Toefl<T, n, container>::update_log( const std::vector<container>& y, std::vector<container>& target)
{
    for( unsigned i=0; i<y.size(); i++)
        thrust::transform( y[i].begin(), y[i].end(), target[i].begin(), dg::LN<T>());
}

template< class T, size_t n, class container>
const container& Toefl<T, n, container>::polarisation( const std::vector<container>& y)
{

    //compute omega
    update_exponent( y, expy);
    blas1::axpby( 1., expy[0], -p.a_i, expy[1], omega);
    //compute chi
    blas1::axpby( p.a_i*p.mu_i, expy[1], 0., chi);
    //compute S omega 
    blas2::symv( W2D<double, n>(hx, hy), omega, omega);
    cudaThreadSynchronize();
    A = pol.create( chi_view ); 
    unsigned number = pcg( A, phi, omega, V2D<double, n>(hx, hy), eps);
    std::cout << "Number of pcg iterations "<< number <<std::endl;
    return phi;
}

template< class T, size_t n, class container, class MemorySpace>
void Toefl<T, n, container, MemorySpace>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
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
