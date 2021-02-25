#pragma once

#include "elliptic.h"
#include "helmholtz.h"

namespace dg
{
/**
 * @brief Various arbitary wavelength polarization charge operators of delta-f (df) and full-f (ff)
 *
 * @ingroup matrixoperators
 * 
 */
//polarization solver class for N
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
     * @param no choose \c dg::normed if you want to directly use the object,
     *  \c dg::not_normed if you want to invert the elliptic equation
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @note chi is assumed 1 per default
     */
    PolChargeN( const Geometry& g, norm no = not_normed,
        direction dir = forward, value_type jfactor=1., bool chi_weight_jump = false):
        PolChargeN( g, g.bcx(), g.bcy(), no, dir, jfactor, chi_weight_jump)
    {        
    }

    /**
     * @brief Construct from grid and boundary conditions
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param no choose \c dg::normed if you want to directly use the object,
     *  \c dg::not_normed if you want to invert the elliptic equation
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @param chi_weight_jump If true, the Jump terms are multiplied with the Chi matrix, else it is ignored
     * @note chi is assumed 1 per default
     */
    PolChargeN( const Geometry& g, bc bcx, bc bcy,
        norm no = not_normed, direction dir = forward,
        value_type jfactor=1., bool chi_weight_jump = false)
    {
        m_ell.construct(g, bcx, bcy, dg::normed, dir, jfactor, chi_weight_jump );
        m_gamma.construct(g, bcx, bcy, -0.5, dir, jfactor);
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
        
        m_no=no;
    }

    /**
    * @brief Perfect forward parameters to one of the constructors
    *
    * @tparam Params deduced by the compiler
    * @param ps parameters forwarded to constructors
    */
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
     * @brief Return the vector missing in the un-normed symmetric matrix
     *
     * i.e. the inverse of the weights() function
     * @return inverse volume form including inverse weights
     */
    const Container& inv_weights()const {
        return m_ell.inv_weights();
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
        symv( 1., x, 0., y);
    }
    /**
     * @brief Compute elliptic term and add to output
     *
     * i.e.  \f$ y=alpha*M(phi) x +beta*y \f$
     * @param alpha a scalar
     * @param x the chi term
     * @param beta a scalar
     * @param y result
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void symv( value_type alpha, const ContainerType0& x, value_type beta, ContainerType1& y)
    {
        //non-symmetric via analytical dx phi, dy phi and lap phi
        dg::blas1::copy(x, m_temp);
        dg::blas1::plus( m_temp, -1.); 
        dg::blas2::gemv( m_rightx, m_temp, m_tempx2); //R_x*f
        dg::blas2::gemv( m_righty, m_temp, m_tempy2); //R_y*f
        
        dg::tensor::scalar_product2d(1., 1., m_dxphi, m_dyphi, m_chi, 1., m_tempx2, m_tempy2, 0., y); // y= nabla phi chi nabla (N-1)
        dg::blas1::pointwiseDot(m_lapphi, x, m_tempx);  // m_temp = N Lap phi
        
        dg::blas1::axpbypgz(1.0, m_tempx, 1.0, m_temp, 1.0, y);
        dg::blas1::scal(y,-1.0); //y = -nabla phi chi nabla (N-1) -N Lap phi - (N-1)
        
        if( m_no == not_normed)
            dg::blas1::pointwiseDot(y, m_ell.weights(),  y);
//  
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
//         if( m_no == not_normed)
//             dg::blas1::pointwiseDot(y, m_ell.weights(),  y);
        
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
//         if( m_no == not_normed)
//             dg::blas1::pointwiseDot(y, m_ell.weights(),  y);
       
        //symmetric discr: only -lap term on rhs //TODO converges to non-physical solution
//         m_ell.set_chi(x);
//         m_ell.symv(1.0, m_phi, 0.0 , y);
//         dg::blas1::copy(x, m_temp);
//         dg::blas1::plus( m_temp, -1.); 
//         dg::blas1::axpby(-1.0, m_temp,  1.0, y);
//         
//         if( m_no == not_normed)
//             dg::blas1::pointwiseDot(y, m_ell.weights(),  y);
    }

    private:
    dg::Elliptic<Geometry,  Matrix, Container> m_ell;
    dg::Helmholtz<Geometry,  Matrix, Container> m_gamma;
    Container m_phi, m_dxphi,m_dyphi, m_lapphi, m_temp, m_tempx, m_tempx2, m_tempy, m_tempy2;
    
    SparseTensor<Container> m_chi, m_metric;
    Container m_sigma, m_vol;
    Container m_weights, m_inv_weights, m_precond, m_weights_wo_vol;

       Matrix m_rightx, m_righty;
           norm m_no;

};
    
template< class G, class M, class V>
struct TensorTraits< PolChargeN<G, M, V> >
{
    using value_type      = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
  
}  //namespace dg
