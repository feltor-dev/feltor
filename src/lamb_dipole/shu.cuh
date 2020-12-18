#ifndef _DG_SHU_CUH
#define _DG_SHU_CUH

#include <exception>
#include <cusp/ell_matrix.h>

#include "dg/algorithm.h"

namespace shu
{
template< class Matrix, class container>
struct Diffusion
{
    Diffusion( const dg::Grid2d& g, double nu): nu_(nu),
        w2d( dg::create::weights( g) ), v2d( dg::create::inv_weights(g) ) ,
        LaplacianM( g, dg::normed)
    {
    }
    void operator()(double t, const container& x, container& y)
    {
        dg::blas2::gemv( LaplacianM, x, y);
        dg::blas1::scal( y, -nu_);
    }
    const container& weights(){return w2d;}
    const container& inv_weights(){return v2d;}
    const container& precond(){return v2d;}
  private:
    double nu_;
    const container w2d, v2d;
    dg::Elliptic<dg::CartesianGrid2d, Matrix,container> LaplacianM;
};

template< class Matrix, class container >
struct Shu
{
    using value_type = dg::get_value_type<container>;
    typedef container Vector;

    Shu( const dg::Grid2d& grid, double eps);

    const dg::Elliptic<Matrix, container, container>& lap() const { return m_multi_laplaceM[0];}
    dg::ArakawaX<dg::CartesianGrid2d, Matrix, container>& arakawa() {return m_arakawa;}
    /**
     * @brief Returns psi that belong to the last y in operator()
     *
     * In a multistep scheme this belongs to the point HEAD-1
     * @return psi is the potential
     */
    const container& potential( ) {return psi;}
    void operator()(double t, const Vector& y, Vector& yp);
  private:
    container psi, w2d, v2d;
    std::vector<dg::Elliptic<dg::CartesianGrid2d, Matrix, container>> m_multi_laplaceM;
    dg::ArakawaX<dg::CartesianGrid2d, Matrix, container> m_arakawa;
    dg::Extrapolation<container> m_old_psi;
    dg::MultigridCG2d<dg::CartesianGrid2d, Matrix, container> m_multigrid;
    double m_eps;
};

template<class Matrix, class container>
Shu< Matrix, container>::Shu( const dg::Grid2d& g, double eps):
    psi( g.size()),
    w2d( dg::create::weights( g)), v2d( dg::create::inv_weights(g)),
    m_arakawa( g),
    m_old_psi( 2, w2d),
    m_multigrid( g, 3),
    m_eps(eps)
{
    m_multi_laplaceM.resize(3);
    for( unsigned u=0; u<3; u++)
        m_multi_laplaceM[u].construct( m_multigrid.grid(u), dg::not_normed, dg::centered, 1);
}

template< class Matrix, class container>
void Shu<Matrix, container>::operator()(double t, const Vector& y, Vector& yp)
{
    m_old_psi.extrapolate( t, psi);
    std::vector<unsigned> number = m_multigrid.direct_solve( m_multi_laplaceM, psi, y, m_eps);
    m_old_psi.update( t, psi);
    if( number[0] == m_multigrid.max_iter())
        throw dg::Fail( m_eps);
    m_arakawa( y, psi, yp); //A(y,psi)-> yp
}

}//namespace shu

#endif //_DG_SHU_CUH
