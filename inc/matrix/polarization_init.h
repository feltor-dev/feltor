#pragma once

#include "dg/algorithm.h"

namespace dg {
namespace mat {

/**
 * @brief EXPERIMENTAL polarization solver class for N
 *
 * Try to solve \c N \f$ -\nabla \cdot ( N \nabla \phi) = \rho\f$ for given \c phi and \c rho
 * @attention Converges very slowly if at all
 * @ingroup matrixmatrixoperators
 *
 */
template <class Geometry, class Matrix, class Container>
class PolChargeN
{
    public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    PolChargeN(){}
    /**
     * @brief Construct from Grid
     *
     * @param g The Grid, boundary conditions are taken from here
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @note chi is assumed 1 per default
     */
    PolChargeN( const Geometry& g,
        direction dir = forward, value_type jfactor=1.):
        PolChargeN( g, g.bcx(), g.bcy(), dir, jfactor)
    {
    }

    /**
     * @brief Construct from grid and boundary conditions
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @note chi is assumed 1 per default
     */
    PolChargeN( const Geometry& g, bc bcx, bc bcy,
        direction dir = forward,
        value_type jfactor=1.):
        m_gamma(-0.5, {g, bcx, bcy, dir, jfactor})
    {
        m_ell.construct(g, bcx, bcy, dir, jfactor );
        dg::assign(dg::evaluate(dg::zero,g), m_phi);
        dg::assign(dg::evaluate(dg::one,g), m_temp);


        m_tempx = m_tempx2 = m_tempy = m_tempy2 = m_temp;
        dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
        dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);

        dg::assign( dg::create::inv_volume(g),    m_inv_weights);
        dg::assign( dg::create::volume(g),        m_weights);
        dg::assign( dg::create::inv_weights(g),   m_precond);
        m_temp = m_tempx = m_tempy = m_inv_weights;
        m_chi=g.metric();
        m_sigma = m_vol = dg::tensor::volume(m_chi);
        dg::assign( dg::create::weights(g), m_weights_wo_vol);
    }

    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = PolChargeN( std::forward<Params>( ps)...);
    }

    template<class ContainerType0>
    void set_phi( const ContainerType0& phi)
    {
      m_phi = phi;
    }
    template<class ContainerType0>
    void set_dxphi( const ContainerType0& dxphi)
    {
      m_dxphi = dxphi;
    }
    template<class ContainerType0>
    void set_dyphi( const ContainerType0& dyphi)
    {
      m_dyphi = dyphi;
    }
    template<class ContainerType0>
    void set_lapphi( const ContainerType0& lapphi)
    {
      m_lapphi = lapphi;
    }
    /**
     * @brief Return the vector making the matrix symmetric
     *
     * i.e. the volume form
     * @return volume form including weights
     */
    const Container& weights()const {
        return m_ell.weights();
    }
    /**
     * @brief Return the default preconditioner to use in conjugate gradient
     *
     * Currently returns the inverse weights without volume elment divided by the scalar part of \f$ \chi\f$.
     * This is especially good when \f$ \chi\f$ exhibits large amplitudes or variations
     * @return the inverse of \f$\chi\f$.
     */
    const Container& precond()const {
        return m_ell.precond();
    }
    /**
     * @brief Compute elliptic term and store in output
     *
     * i.e. \f$  y=f(x) = - \nabla \cdot (x \nabla_\perp \phi) \f$ or \f$  y=f(x) = x + (1+ \alpha \Delta_\perp )\nabla \cdot (x \nabla_\perp \phi) \f$
     * @param x left-hand-side
     * @param y result
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void operator()( const ContainerType0& x, ContainerType1& y){
        symv( 1., x, 0., y);
    }

    /**
     * @brief Compute elliptic term and store in output
     *
     * i.e. \f$ y=M(phi) x  \f$
     * @param x left-hand-side
     * @param y result
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y){
        //non-symmetric via analytical dx phi, dy phi and lap phi
        dg::blas1::copy(x, m_temp);
        dg::blas1::plus( m_temp, -1.);
        dg::blas2::gemv( m_rightx, m_temp, m_tempx2); //R_x*f
        dg::blas2::gemv( m_righty, m_temp, m_tempy2); //R_y*f

        dg::tensor::scalar_product2d(1., 1., m_dxphi, m_dyphi, m_chi, 1., m_tempx2, m_tempy2, 0., y); // y= nabla phi chi nabla (N-1)
        dg::blas1::pointwiseDot(m_lapphi, x, m_tempx);  // m_temp = N Lap phi

        dg::blas1::axpbypgz(1.0, m_tempx, 1.0, m_temp, 1.0, y);
        dg::blas1::scal(y,-1.0); //y = -nabla phi chi nabla (N-1) -N Lap phi - (N-1)

        //non-symmetric (only m_phi and x as input)
//         dg::blas2::gemv( m_rightx, m_phi, m_dxphi); //R_x*f
//         dg::blas2::gemv( m_righty, m_phi, m_dyphi); //R_y*f
//         dg::blas1::copy(x, m_temp);
//         dg::blas1::plus( m_temp, -1.); 
//         dg::blas2::gemv( m_rightx, m_temp, m_tempx2); //R_x*f
//         dg::blas2::gemv( m_righty, m_temp, m_tempy2); //R_y*f
//
//         dg::tensor::scalar_product2d(1., 1., m_dxphi, m_dyphi, m_chi, 1., m_tempx2, m_tempy2, 0., y); // y= nabla phi chi nabla (N-1)
//         dg::blas2::symv(m_ell, m_phi, m_lapphi);
//         dg::blas1::pointwiseDot(m_lapphi, x, m_tempx);  // m_tempx = -N Lap phi
//
//         dg::blas1::axpbypgz(-1.0, m_tempx, 1.0, m_temp, 0.0, y);;
//         dg::blas1::scal(y,-1.0);
//
//         non-symmetric mixed analyital (only m_phi, m_lapphi and x)
//         dg::blas2::gemv( m_rightx, m_phi, m_dxphi); //R_x*f
//         dg::blas2::gemv( m_righty, m_phi, m_dyphi); //R_y*f
//         dg::blas1::copy(x, m_temp);
//         dg::blas1::plus( m_temp, -1.);
//         dg::blas2::gemv( m_rightx, m_temp, m_tempx2); //R_x*f
//         dg::blas2::gemv( m_righty, m_temp, m_tempy2); //R_y*f
//
//         dg::tensor::scalar_product2d(1., 1., m_dxphi, m_dyphi, m_chi, 1., m_tempx2, m_tempy2, 0., y); // y= nabla phi chi nabla (N-1)
//         dg::blas1::pointwiseDot(m_lapphi, x, m_tempx);  // m_temp = N Lap phi
//
//         dg::blas1::axpbypgz(1.0, m_tempx, 1.0, m_temp, 1.0, y);
//         dg::blas1::scal(y,-1.0);
//
        //symmetric discr: only -lap term on rhs //TODO converges to non-physical solution
//         m_ell.set_chi(x);
//         m_ell.symv(1.0, m_phi, 0.0 , y);
//         dg::blas1::copy(x, m_temp);
//         dg::blas1::plus( m_temp, -1.);
//         dg::blas1::axpby(-1.0, m_temp,  1.0, y);
//
    }

    private:
    dg::Elliptic<Geometry,  Matrix, Container> m_ell;
    dg::Helmholtz<Geometry,  Matrix, Container> m_gamma;
    Container m_phi, m_dxphi,m_dyphi, m_lapphi, m_temp, m_tempx, m_tempx2, m_tempy, m_tempy2;

    SparseTensor<Container> m_chi, m_metric;
    Container m_sigma, m_vol;
    Container m_weights, m_inv_weights, m_precond, m_weights_wo_vol;

       Matrix m_rightx, m_righty;

};

}  //namespace mat

template< class G, class M, class V>
struct TensorTraits< mat::PolChargeN<G, M, V> >
{
    using value_type      = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};

}  //namespace dg
