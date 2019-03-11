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
 * Unnormed discretization of \f[ (-\nabla \cdot \chi + \Delta \iota \Delta) \f]
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
     * @param dir Direction of the Laplace operator
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
         m_laplaceM_chi.construct( g, bcx, bcy, dg::normed, dir, jfactor);
         m_laplaceM_iota.construct( g, bcx, bcy, dg::normed, dir, jfactor);
         dg::assign( dg::evaluate( dg::one, g), m_temp);
     }
     ///@copydoc ArbPol::ArbPo(const Geometry&,value_type,direction,value_type)
     void construct( const Geometry& g,  direction dir = dg::centered, value_type jfactor = 1.)
     {
         construct( g, g.bcx(), g.bcy(),  dir, jfactor);
     }
     /**
     * @brief apply operator
     *
     * Computes
     * \f[ y = W\left[ -\nabla \¢dot \chi \nabla + \Delta \iota \Delta)\right] x \f] to make the matrix symmetric
     * @param x lhs (is constant up to changes in ghost cells)
     * @param y rhs contains solution
     */
     void symv(const Container& x, Container& y)                                                                   
     {  
        blas2::symv( m_laplaceM_iota, x, y);    // y = -nabla_perp^2 x                                        
        blas1::pointwiseDot(y, m_iota, y);  //y = -iota*nabla_perp^2 x
        blas2::symv( m_laplaceM_iota, y, m_temp);    //m_temp = nabla_perp^2 (iota*nabla_perp^2 x)                      
         

        blas2::symv( m_laplaceM_chi, x,y);   // y = -nabla (chi nabl_perp x)                                
        blas1::axpby( 1., y, 1., m_temp, y);                                                                  
        blas2::symv( m_laplaceM_chi.weights(), y, y);                                                                  
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
                                                             
   private:
     Elliptic<Geometry, Matrix, Container> m_laplaceM_chi, m_laplaceM_iota;                                                             
     Container m_temp;                                                                                  
     Container m_iota; 
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

