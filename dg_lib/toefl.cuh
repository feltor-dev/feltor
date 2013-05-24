#ifndef _DG_TOEFL_CUH
#define _DG_TOEFL_CUH


#include "xspacelib.cuh"
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

template< class T, size_t n, class container=thrust::device_vector<T> >
struct Toefl
{
    typedef std::vector<container> Vector;
    typedef typename thrust::iterator_space<typename container::iterator>::type MemorySpace;

    Toefl( const Grid<T,n>& g, bool global, double eps, double, double);

    void update_exponent( const std::vector<container>& y, std::vector<container>& target);
    void update_log( const std::vector<container>& y, std::vector<container>& target);
    const container& polarisation( const std::vector<container>& y);
    void operator()( const std::vector<container>& y, std::vector<container>& yp);
  private:
    typedef T value_type;
    //typedef typename VectorTraits< Vector>::value_type value_type;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;

    container omega, phi, dyphi, chi;
    std::vector<container> expy, dxy, dyy, lapy;

    Matrix dx, dy;
    Matrix A; //contains unnormalized laplacian if local
    Matrix laplace; //contains normalized laplacian
    ArakawaX<T, n, container> arakawa; 
    Polarisation2dX<T, n, container> pol;
    CG<Matrix, container, dg::V2D<T, n> > pcg;

    double hx, hy;
    bool global;
    double eps; 
    double kappa, nu;
};

template< class T, size_t n, class container>
Toefl<T, n, container>::Toefl( const Grid<T,n>& g, bool global, double eps, double kappa, double nu): 
    omega( n*n*g.Nx()*g.Ny(), 0.), phi(omega), dyphi( phi), chi(phi),
    expy( 2, omega), dxy( expy), dyy( dxy), lapy( dyy),
    arakawa( g, dg::DIR, dg::PER), 
    pol(     g, dg::DIR, dg::PER), 
    pcg( omega, n*n*g.Nx()*g.Ny()), 
    hx( g.hx()), hy(g.hy()), global(global), eps(eps), kappa(kappa), nu(nu)
{
    //create derivatives
    dx = create::dx( g, dg::DIR);
    dy = create::dy( g, dg::PER);
    laplace = create::laplacian( g, dg::DIR, dg::PER);
    if( !global) 
        A = create::laplacian( g, dg::DIR, dg::PER, false);

}

//how to set up a computation?
template< class T, size_t n, class container>
const container& Toefl<T, n, container>::polarisation( const std::vector<container>& y)
{
    //compute omega
    if( global)
    {
        update_exponent( y, expy);
        blas1::axpby( -1., expy[0], 1., expy[1], omega); //omega = n_i - n_e
        //compute chi
        blas1::axpby( 1., expy[1], 0., chi);
    }
    else
    {
        blas1::axpby( -1, y[0], 1., y[1], omega);
    }
    //compute S omega 
    blas2::symv( W2D<double, n>(hx, hy), omega, omega);
    cudaThreadSynchronize();
    if( global)
    {
        A = pol.create( chi ); 
    }
    unsigned number = pcg( A, phi, omega, V2D<double, n>(hx, hy), eps);
    std::cout << "Number of pcg iterations "<< number <<std::endl;
    return phi;
}

template< class T, size_t n, class container>
void Toefl<T, n, container>::operator()( const std::vector<container>& y, std::vector<container>& yp)
{
    assert( y.size() == 2);
    assert( y.size() == yp.size());
    cudaThreadSynchronize();
    phi = polarisation( y);

    for( unsigned i=0; i<y.size(); i++)
        arakawa( y[i], phi, yp[i]);

    //compute derivatives
    cudaThreadSynchronize();
    blas2::gemv( dy, phi, dyphi);
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( dx, y[i], dxy[i]);
        blas2::gemv( dy, y[i], dyy[i]);
    }
    // curvature terms
    cudaThreadSynchronize();
    blas1::axpby( kappa, dyphi, 1., yp[0]);
    blas1::axpby( kappa, dyphi, 1., yp[1]);
    blas1::axpby( -kappa, dyy[0], 1., yp[0]);

    //add laplacians
    for( unsigned i=0; i<y.size(); i++)
    {
        blas2::gemv( laplace, y[i], lapy[i]);
        if( global)
        {
            blas1::pointwiseDot( dxy[i], dxy[i], dxy[i]);
            blas1::pointwiseDot( dyy[i], dyy[i], dyy[i]);
            //now sum all 3 terms up 
            blas1::axpby( 1., dyy[i], 1., lapy[i]);
            blas1::axpby( 1., dxy[i], 1., lapy[i]);
        }
        blas1::axpby( nu, lapy[i], 1., yp[i]); //rescale
    }


}

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

}//namespace dg

#endif //_DG_TOEFL_CUH
