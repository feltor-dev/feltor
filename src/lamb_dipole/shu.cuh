#ifndef _DG_SHU_CUH
#define _DG_SHU_CUH

#include <exception>
#include <cusp/ell_matrix.h>

#include "dg/blas.h"
#include "dg/arakawa.cuh"
#include "dg/derivatives.cuh"
#include "dg/cg.cuh"

namespace dg
{
template< class container>
struct Diffusion
{
    Diffusion( const dg::Grid2d<double>& g, double nu): nu_(nu),
        w2d( dg::create::w2d( g)), v2d( dg::create::v2d(g)) { 
        dg::Matrix Laplacian_ = dg::create::laplacianM( g, dg::normed, dg::XSPACE); 
        cusp::blas::scal( Laplacian_.values, -nu);
        Laplacian = Laplacian_;
        }
    void operator()( const container& x, container& y)
    {
        dg::blas2::gemv( Laplacian, x, y);
    }
    const container& weights(){return w2d;}
    const container& precond(){return v2d;}
  private:
    double nu_;
    const container w2d, v2d;
    dg::DMatrix Laplacian;
};

struct Fail : public std::exception
{

    Fail( double eps): eps( eps) {}
    double epsilon() const { return eps;}
    char const* what() const throw(){ return "Failed to converge";}
  private:
    double eps;
};

template< class container=thrust::device_vector<double> >
struct Shu 
{
    typedef typename container::value_type value_type;
    typedef container Vector;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;

    Shu( const Grid2d<value_type>& grid, double D, double eps);

    Matrix& lap() { return laplace;}
    /**
     * @brief Returns phi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return phi is the potential
     */
    const container& potential( ) {return phi;}
    void operator()( const Vector& y, Vector& yp);
  private:
    //typedef typename VectorTraits< Vector>::value_type value_type;
    Matrix laplace;
    container omega, phi, phi_old;
    ArakawaX< container> arakawa; 
    CG< container > pcg;
    //SimplicialCholesky cholesky;
    thrust::host_vector<double> x,b;
    container v2d, w2d;
    
    double D;
    double eps; 
};

template< class container>
Shu< container>::Shu( const Grid2d<value_type>& g, double D, double eps): 
    omega( g.size(), 0.), phi(omega), phi_old(phi),
    arakawa( g), 
    pcg( omega, g.size()), x( phi), b( x),
    v2d( create::v2d( g)), w2d( create::w2d(g)), D(D), eps(eps)
{
    typedef cusp::coo_matrix<int, value_type, MemorySpace> HMatrix;
    HMatrix A = dg::create::laplacianM( g, not_normed, XSPACE);
    laplace = A;
    //cholesky.compute( A);
}

template< class container>
void Shu<container>::operator()( const Vector& y, Vector& yp)
{
    //dg::blas2::symv( laplace, y, yp);
    //dg::blas2::symv( -D, v2d, yp, 0., yp); //laplace is unnormalized -laplace
    //compute S omega
    blas2::symv( w2d, y, omega);
    blas1::axpby( 2., phi, -1.,  phi_old);
    phi.swap( phi_old);
    unsigned number = pcg( laplace, phi, omega, v2d, eps);
    if( number == pcg.get_max())
        throw Fail( eps);
    //std::cout << "Number of pcg iterations "<< number<<"\n"; 
    //b = omega; //copy data to host
    //cholesky.solve( x.data(), b.data(), b.size()); //solve on host
    //phi = x; //copy data back to device
    arakawa( y, phi, omega); //A(y,phi)-> omega
    blas1::axpby( 1., omega, 0., yp);

}

}//namespace dg

#endif //_DG_SHU_CUH
