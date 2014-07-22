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
        w2d( dg::create::w2d( g) ), v2d( dg::create::v2d(g) ) 
    { 
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

template< class container=thrust::device_vector<double> >
struct Shu 
{
    typedef typename container::value_type value_type;
    typedef container Vector;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;

    Shu( const Grid2d<value_type>& grid, double eps);

    const Matrix& lap() const { return laplaceM;}
    ArakawaX<Matrix, container>& arakawa() {return arakawa_;}
    /**
     * @brief Returns psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return psi is the potential
     */
    const container& potential( ) {return psi;}
    void operator()( const Vector& y, Vector& yp);
  private:
    //typedef typename VectorTraits< Vector>::value_type value_type;
    container psi, w2d, v2d;
    Matrix laplaceM;
    ArakawaX< container> arakawa_; 
    Invert<container> invert;
};

template< class container>
Shu< container>::Shu( const Grid2d<value_type>& g, double eps): 
    psi( g.size()),
    w2d( create::w2d( g)), v2d( create::v2d(g)),  
    laplaceM( dg::create::laplacianM( g, not_normed, XSPACE)),
    arakawa_( g), 
    invert( psi, g.size(), eps)
{
}

template< class container>
void Shu<container>::operator()( const Vector& y, Vector& yp)
{
    invert( laplaceM, psi, y, w2d, v2d);
    arakawa_( y, psi, yp); //A(y,psi)-> yp
}

}//namespace dg

#endif //_DG_SHU_CUH
