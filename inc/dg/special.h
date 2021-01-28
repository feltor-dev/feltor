#pragma once

#include <cassert>

#include "blas.h"
#include "elliptic.h"

/*!@file
 *
 * @brief contains special differential operators
 */
namespace dg{

/**
 * @brief Matrix class that represents the arbitrary polarization operator
 *
 * @ingroup matrixoperators
 *
 * Unnormed discretization of \f[ (-\nabla \cdot \chi \nabla - \Delta \iota \Delta +  \nabla \cdot\nabla \cdot 2\iota \nabla \nabla ) x \f]
 * where \f$ \chi\f$ is a function and \f$\alpha\f$ a scalar.
 */
 template< class Geometry, class Matrix, class Container>
 struct ArbPol
 {
     using container_type = Container;
     using geometry_type = Geometry;
     using matrix_type = Matrix;
     using value_type = get_value_type<Container>;
     ///@brief empty object ( no memory allocation)
     ArbPol() {}
     /**
     * @brief Construct \c ArbPol operator
     *
     * @param g The grid to use
     * @param dir Direction of the Laplace operator
     * @param jfactor The jfactor used in the Laplace operator (probably 1 is always the best factor but one never knows...)
     * @note The default value of \f$\chi\f$ and \f$\iota\f$ is one
     */
     ArbPol( const Geometry& g, direction dir = dg::forward, value_type jfactor=1.)
     {
         construct( g, dir, jfactor);
     }
     /**
     * @brief Construct \c ArbPol operator
     *
     * @param g The grid to use
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param dir Direction of the Laplace operator (Note: only dg::centered tested)
     * @param jfactor The jfactor used in the Laplace operator (probably 1 is always the best factor but one never knows...)
     * @note The default value of \f$\chi\f$ and \f$\iota\f$ is one
     */
     ArbPol( const Geometry& g, bc bcx, bc bcy,  direction dir = dg::centered, value_type jfactor=1.)
     {
         construct( g, bcx, bcy, dir, jfactor);
     }
     ///@copydoc ArbPol::ArbPol(const Geometry&,bc,bc,value_type,direction,value_type)
     void construct( const Geometry& g, bc bcx, bc bcy, direction dir = dg::centered, value_type jfactor = 1.)
     {
        m_jfactor=jfactor;
        m_laplaceM_chi.construct( g, bcx, bcy, dg::normed, dir, jfactor);
        m_laplaceM_iota.construct( g, bcx, bcy, dg::normed, dir, jfactor);
        dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), m_leftx);
        dg::blas2::transfer( dg::create::dy( g, inverse( bcy), inverse(dir)), m_lefty);
        dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
        dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);
        dg::blas2::transfer( dg::create::jumpX( g, bcx),   m_jumpX);
        dg::blas2::transfer( dg::create::jumpY( g, bcy),   m_jumpY);
        dg::blas2::transfer( dg::create::jumpX( g, bcx),   m_jumpX_sqrt);
        dg::blas2::transfer( dg::create::jumpY( g, bcy),   m_jumpY_sqrt);
        
        dg::assign( dg::evaluate( dg::one, g), m_temp);
        m_tempx = m_tempy = m_tempxy = m_tempyx = m_iota  = m_helper = m_temp;

        dg::assign( dg::create::inv_volume(g),    m_inv_weights);
        dg::assign( dg::create::volume(g),        m_weights);
        dg::assign( dg::create::inv_weights(g),   m_precond);
        m_chi=g.metric();
        m_metric=g.metric();
        m_vol=dg::tensor::volume(m_chi);
        dg::tensor::scal( m_chi, m_vol);
        dg::assign( dg::create::weights(g), m_weights_wo_vol);
        dg::assign( dg::evaluate(dg::one, g), m_sigma);
     }
     ///@copydoc ArbPol::ArbPo(const Geometry&,value_type,direction,value_type)
     void construct( const Geometry& g,  direction dir = dg::centered, value_type jfactor = 1.)
     {
         construct( g, g.bcx(), g.bcy(),  dir, jfactor);
     }
     ///@copydoc Elliptic::weights()const
     const Container& weights()const {return m_laplaceM_chi.weights();}
     ///@copydoc Elliptic::inv_weights()const
     const Container& inv_weights()const {return m_laplaceM_chi.inv_weights();} 
     /**
     * @brief Preconditioner to use in conjugate gradient solvers
     *
     * multiply result by these coefficients to get the normed result
     * @return the inverse weights without volume
     */
     const Container& precond()const {return m_laplaceM_chi.precond();}        
     /**
     * @brief Set Chi in the above formula
     *
     * @param chi new container
     */
     void set_chi( const Container& chi) {m_laplaceM_chi.set_chi(chi); }
     /**
     * @brief Set Iota in the above formula
     *
     * @param iota new container
     */
     void set_iota( const Container& iota) {m_iota=iota; }   
     /**
     * @brief compute the variational of the operator (psi_2 in gf theory): \f[ - \frac{chi}{2} \left\{|\nabla \phi|^2 + \alpha chi ( | \nabla^2 \phi | - (\Delta \phi)^2 / 2) \right\}\f]
     *
     * @param phi (e.g. Gamma phi)
     * @param alpha (e.g. tau/2)
     * @param chi (e.g. 1/B^2)
     * @param varphi equals psi_2 in gf theory if phi = gamma_phi
     */
     void variation(const Container& phi, const value_type& alpha, const Container& chi, Container& varphi)  
     {
        //tensor part
        dg::blas2::gemv( m_rightx, phi, m_tempx); //R_x phi          
        dg::blas2::gemv( m_righty , m_tempx, m_temp); //R_y R_x phi   
        //m_temp = tau/2/B^2
        dg::blas1::pointwiseDot(2.*alpha, chi, m_temp, m_temp, 0., varphi); //varphi = 2. * alpha * chi  (R_y R_x phi)^2
        
        dg::blas2::gemv( m_rightx, m_tempx, m_temp); //R_x R_x phi
        dg::blas1::pointwiseDot(alpha, chi, m_temp, m_temp, 1., varphi); //varphi+= alpha *chi (R_x R_x phi)^2
        
        dg::blas2::gemv( m_righty, phi, m_tempy); //R_y phi                
        dg::blas2::gemv( m_righty, m_tempy, m_temp); //R_y R_y phi  
        dg::blas1::pointwiseDot(alpha, chi, m_temp, m_temp, 1., varphi); //varphi+= alpha *chi (R_y R_y phi)^2
        
        //laplacian part
        dg::blas2::symv(m_laplaceM_iota, phi, m_temp); 
        dg::blas1::pointwiseDot(alpha/2, chi, m_temp, m_temp, -1., varphi); //varphi-= alpha *chi/2 (lap phi)^2
        
        //elliptic part
//         arakawa.variation(phi, phi[1]);   // (grad phi)^2
        tensor::multiply2d( m_metric, m_tempx, m_tempy, m_temp, m_helper);
        dg::blas1::pointwiseDot( 1., m_temp, m_tempx, 1., m_helper, m_tempy, 0., m_temp); //m_temp = |nabla phi|^2
        dg::blas1::axpby(-0.5, m_temp, -0.5, varphi);
        dg::blas1::pointwiseDot(chi, varphi, varphi);
     }
     /**
     * @brief apply operator
     *
     * Computes
     * \f[ y = W\left[-\nabla \cdot \chi \nabla - \Delta \iota \Delta +  \nabla \cdot\nabla \cdot 2\iota \nabla \nabla \right] x \f] to make the matrix symmetric
     * @param x lhs (is constant up to changes in ghost cells)
     * @param y rhs contains solution
     */
     void symv(const Container& x, Container& y)                                                                   
     {         
         //div div (iota nabla^2 f) term
        dg::blas2::symv( m_rightx, x, m_helper); //R_x*f        
        dg::blas2::symv(-1.0, m_leftx, m_helper, 0.0, m_tempx); //L_x R_x *f   
        dg::blas2::symv(-1.0, m_righty, m_helper, 0.0, m_tempyx); //L_y R_x*f //Ry Rx or Ly Rx?
        dg::blas2::symv( m_righty, x, m_helper); //R_y*f                
        dg::blas2::symv(-1.0, m_lefty, m_helper, 0.0, m_tempy); //L_y R_y *f   
        dg::blas2::symv(-1.0, m_rightx, m_helper, 0.0, m_tempxy); //L_x R_y *f//Rx Ry or Lx Ry?   
        
        dg::blas2::symv( m_jfactor, m_jumpX, x, 1., m_tempx);
        dg::blas2::symv( m_jfactor, m_jumpY, x, 1., m_tempy);
        
        dg::blas1::pointwiseDivide( 1., m_tempx,  m_vol, 0., m_tempx); //make normed 
        dg::blas1::pointwiseDivide( 1., m_tempyx, m_vol, 0., m_tempyx); //make normed 
        dg::blas1::pointwiseDivide( 1., m_tempy,  m_vol, 0., m_tempy); //make normed 
        dg::blas1::pointwiseDivide( 1., m_tempxy, m_vol, 0., m_tempxy); //make normed 
        
        dg::blas1::pointwiseDot(1.0, m_iota, m_tempx,  0.0, m_tempx); 
        dg::blas1::pointwiseDot(1.0, m_iota, m_tempyx, 0.0, m_tempyx); 
        dg::blas1::pointwiseDot(1.0, m_iota, m_tempy,  0.0, m_tempy); 
        dg::blas1::pointwiseDot(1.0, m_iota, m_tempxy, 0.0, m_tempxy); 
        
        dg::blas2::symv( m_rightx, m_tempx, m_helper);     
        dg::blas2::symv(-1.0, m_leftx, m_helper, 0.0, m_temp); 
        dg::blas2::symv( m_leftx, m_tempyx, m_helper);       
        dg::blas2::symv(-1.0, m_lefty, m_helper, 1.0, m_temp); // L L  or L R ? 
        dg::blas2::symv( m_righty, m_tempy, m_helper);       
        dg::blas2::symv(-1.0, m_lefty, m_helper, 1.0, m_temp); 
        dg::blas2::symv( m_lefty, m_tempxy, m_helper);       
        dg::blas2::symv(-1.0, m_leftx, m_helper, 1.0, m_temp);   // L L  or L R ? 
        
        dg::blas2::symv( m_jfactor, m_jumpX, m_tempx, 1., m_temp);
        dg::blas2::symv( m_jfactor, m_jumpY, m_tempy, 1., m_temp); 
        
        //-lap (iota lap f) term
        blas2::symv( m_laplaceM_iota, x, m_tempx);                                    
        blas1::pointwiseDot( m_iota, m_tempx, m_tempx);  
        blas2::symv(-1.0, m_laplaceM_iota, m_tempx,2.0,m_temp);    
        
        //elliptic term - div (chi nabla f)
        blas2::symv(1.0, m_laplaceM_chi, x, 1., m_temp);       
        
        //scale with weights to obtain not normed discr
        blas2::symv(1.0, m_weights, m_temp, 0., y);    
       
     }         
     
     private:
     bc inverse( bc bound)
     {
        if( bound == DIR) return NEU;
        if( bound == NEU) return DIR;
        if( bound == DIR_NEU) return NEU_DIR;
        if( bound == NEU_DIR) return DIR_NEU;
        return PER;
     }
     direction inverse( direction dir)
     {
        if( dir == forward) return backward;
        if( dir == backward) return forward;
        return centered;
     }
     Elliptic<Geometry, Matrix, Container> m_laplaceM_chi, m_laplaceM_iota;                                                             
     Matrix m_leftx, m_lefty, m_rightx, m_righty, m_jumpX, m_jumpY, m_jumpX_sqrt, m_jumpY_sqrt;
     Container m_temp, m_tempx, m_tempy, m_tempxy, m_tempyx, m_iota, m_helper;                                                                                  
     Container m_weights, m_inv_weights, m_precond, m_weights_wo_vol;
     SparseTensor<Container> m_chi;
     SparseTensor<Container> m_metric;
     Container m_sigma, m_vol;
     value_type m_jfactor;

};
///@cond

template< class G, class M, class V>
struct TensorTraits< ArbPol<G, M, V> >
{
    using value_type  = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
///@endcond


} //namespace dg

