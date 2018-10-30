#pragma once

#include <cassert>

#include "blas.h"
#include "elliptic.h"

/*!@file
 *
 * @brief contains Helmholtz and Maxwell operators
 */
namespace dg{

/**
 * @brief Matrix class that represents a Helmholtz-type operator
 *
 * @ingroup matrixoperators
 *
 * Unnormed discretization of \f[ (\chi+\alpha\Delta) \f]
 * where \f$ \chi\f$ is a function and \f$\alpha\f$ a scalar.
 * \f$ \Delta\f$ is a two- or three-dimensional elliptic operator.
 * Can be used by the CG class. The following example shows how the class can be used to act as a \c Helmholtz2 operator:
 @snippet helmholtzg2_b.cu doxygen
 * @attention The Laplacian in this formula is positive as opposed to the negative sign in the \c Elliptic operator
 * @tparam EllipticType Either dg::Elliptic or dg::Elliptic3d
 */
template<class EllipticType>
struct GeneralHelmholtz
{
    using container_type = typename EllipticType::container_type;
    using geometry_type = typename EllipticType::geometry_type;
    using value_type = typename EllipticType::value_type;
    ///@brief empty object ( no memory allocation)
    GeneralHelmholtz() {}
    GeneralHelmholtz( value_type alpha, EllipticType elliptic):
        m_alpha(alpha), m_laplaceM(elliptic),
        m_chi( m_laplaceM.weights())
    {
    }
    /**
     * @brief Construct
     *
     * @param g The grid to use (boundary conditions are taken from there)
     * @param alpha Scalar in the above formula
     * @param dir Direction of the Laplace operator
     * @param jfactor The jfactor used in the Laplace operator (probably 1 is always the best factor but one never knows...)
     * @note The default value of \f$\chi\f$ is one. \c Helmholtz is never normed
     */
    GeneralHelmholtz( const geometry_type& g, value_type alpha = 1., direction dir = dg::forward, value_type jfactor=1.):
        GeneralHelmholtz( g, g.bcx(), g.bcy(), alpha, dir, jfactor)
    {
    }
    /**
     * @brief Construct
     *
     * @param g The grid to use
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param alpha Scalar in the above formula
     * @param dir Direction of the Laplace operator
     * @param jfactor The jfactor used in the Laplace operator (probably 1 is always the best factor but one never knows...)
     * @note The default value of \f$\chi\f$ is one. \c Helmholtz is never normed
     */
    GeneralHelmholtz( const geometry_type& g, bc bcx, bc bcy, value_type alpha = 1., direction dir = dg::forward, value_type jfactor=1.):
        m_laplaceM( g, bcx, bcy, dg::not_normed, dir, jfactor),
        m_chi( m_laplaceM.weights()),
        m_alpha(alpha)
    {
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
        *this = GeneralHelmholtz( std::forward<Params>( ps)...);
    }
    /**
     * @brief apply operator
     *
     * Computes
     * \f[ y = W( \chi + \alpha\Delta) x \f] to make the matrix symmetric
     * @param x lhs (is constant up to changes in ghost cells)
     * @param y rhs contains solution
     * @tparam ContainerTypes must be usable with \c container_type in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y)
    {
        if( m_alpha != 0)
            blas2::symv( m_laplaceM, x, y);
        dg::blas1::pointwiseDot( 1., m_chi, x, -m_alpha, y);

    }
    ///@copydoc Elliptic::weights()const
    const container_type& weights()const {return m_laplaceM.weights();}
    ///@copydoc Elliptic::inv_weights()const
    const container_type& inv_weights()const {return m_laplaceM.inv_weights();}
    /**
     * @brief Preconditioner to use in conjugate gradient solvers
     *
     * @return inverse weights without volume
     */
    const container_type& precond()const {return m_laplaceM.precond();}
    ///Write access to the elliptic class
    EllipticType& elliptic(){
        return m_laplaceM;
    }
    /**
     * @brief Change alpha
     *
     * @return reference to alpha
     */
    value_type& alpha( ){  return m_alpha;}
    /**
     * @brief Access alpha
     *
     * @return alpha
     */
    value_type alpha( ) const  {return m_alpha;}
    /**
     * @brief Set Chi in the above formula
     *
     * @param chi new container
     * @tparam ContainerTypes must be usable with \c container in \ref dispatch
     */
    template<class ContainerType0>
    void set_chi( const ContainerType0& chi) {
        dg::blas1::pointwiseDot( m_laplaceM.weights(), chi, m_chi);
    }
    /**
     * @brief Access chi
     * @return chi
     */
    const container_type& chi() const{return m_chi;}
  private:
    EllipticType m_laplaceM;
    container_type m_chi;
    value_type m_alpha;
};


/**
 * @brief Matrix class that represents a Helmholtz-type operator
 *
 * @ingroup matrixoperators
 *
 * Unnormed discretization of \f[ (\chi+\alpha\Delta) \f]
 * where \f$ \chi\f$ is a function and \f$\alpha\f$ a scalar.
 * Can be used by the CG class. The following example shows how the class can be used to act as a \c Helmholtz2 operator:
 @snippet helmholtzg2_b.cu doxygen
 * @copydoc hide_geometry_matrix_container
 * @attention The Laplacian in this formula is positive as opposed to the negative sign in the \c Elliptic operator
 */
template<class Geometry, class Matrix, class Container>
using Helmholtz = GeneralHelmholtz<Elliptic<Geometry, Matrix, Container>>;
///@copydoc Helmholtz
///@ingroup matrixoperators
template<class Geometry, class Matrix, class Container>
using Helmholtz2d = GeneralHelmholtz<Elliptic<Geometry, Matrix, Container>>;
///@copydoc Helmholtz
///@ingroup matrixoperators
template<class Geometry, class Matrix, class Container>
using Helmholtz3d = GeneralHelmholtz<Elliptic3d<Geometry, Matrix, Container>>;

/**
 * @brief DEPRECATED, Matrix class that represents a more general Helmholtz-type operator
 *
 * @ingroup matrixoperators
 *
 * Unnormed discretization of
 * \f[ \left[ \chi +2 \alpha\Delta +  \alpha^2\Delta \left(\chi^{-1}\Delta \right)\right] \f]
 * where \f$ \chi\f$ is a function and \f$\alpha\f$ a scalar.
 * Can be used by the Invert class
 * @copydoc hide_geometry_matrix_container
 * @note The Laplacian in this formula is positive as opposed to the negative sign in the \c Elliptic operator
 * @attention It is MUCH better to solve the normal \c Helmholtz operator twice,
 * consecutively, than solving the \c Helmholtz2 operator once. The following code snippet shows how to do it:
 @snippet helmholtzg2_b.cu doxygen
 */
template< class Geometry, class Matrix, class Container>
struct Helmholtz2
{
    using container_type = Container;
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    Helmholtz2() {}
    /**
     * @brief Construct \c Helmholtz2 operator
     *
     * @param g The grid to use
     * @param alpha Scalar in the above formula
     * @param dir Direction of the Laplace operator
     * @param jfactor The jfactor used in the Laplace operator (probably 1 is always the best factor but one never knows...)
     * @note The default value of \f$\chi\f$ is one
     */
    Helmholtz2( const Geometry& g, value_type alpha = 1., direction dir = dg::forward, value_type jfactor=1.)
    {
        construct( g, alpha, dir, jfactor);
    }
    /**
     * @brief Construct \c Helmholtz2 operator
     *
     * @param g The grid to use
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param alpha Scalar in the above formula
     * @param dir Direction of the Laplace operator
     * @param jfactor The jfactor used in the Laplace operator (probably 1 is always the best factor but one never knows...)
     * @note The default value of \f$\chi\f$ is one
     */
    Helmholtz2( const Geometry& g, bc bcx, bc bcy, value_type alpha = 1., direction dir = dg::forward, value_type jfactor=1.)
    {
              construct( g, bcx, bcy, alpha, dir, jfactor);
    }
    ///@copydoc Helmholtz2::Helmholtz2(const Geometry&,bc,bc,value_type,direction,value_type)
    void construct( const Geometry& g, bc bcx, bc bcy, value_type alpha = 1, direction dir = dg::forward, value_type jfactor = 1.)
    {
        laplaceM_.construct( g, bcx, bcy, dg::normed, dir, jfactor);
        dg::assign( dg::evaluate( dg::one, g), temp1_);
        dg::assign( dg::evaluate( dg::one, g), temp2_);
        alpha_ = alpha;
    }
    ///@copydoc Helmholtz2::Helmholtz2(const Geometry&,value_type,direction,value_type)
    void construct( const Geometry& g, value_type alpha = 1, direction dir = dg::forward, value_type jfactor = 1.)
    {
        construct( g, g.bcx(), g.bcy(), alpha, dir, jfactor);
    }
    /**
     * @brief apply operator
     *
     * Computes
     * \f[ y = W\left[ \chi +2 \alpha\Delta +  \alpha^2\Delta \left(\chi^{-1}\Delta \right)\right] x \f] to make the matrix symmetric
     * @param x lhs (is constant up to changes in ghost cells)
     * @param y rhs contains solution
     * @note Takes care of sign in laplaceM and thus multiplies by -alpha
     */
    void symv(const Container& x, Container& y)
    {
        if( alpha_ != 0)
        {
            blas2::symv( laplaceM_, x, temp1_); // temp1_ = -nabla_perp^2 x
            blas1::pointwiseDivide(temp1_, chi_, y); //temp2_ = (chi^-1)*W*nabla_perp^2 x
            blas2::symv( laplaceM_, y, temp2_);//temp2_ = nabla_perp^2 *(chi^-1)*nabla_perp^2 x
        }
        blas1::pointwiseDot( chi_, x, y); //y = chi*x
        blas1::axpby( 1., y, -2.*alpha_, temp1_, y);
        blas1::axpby( alpha_*alpha_, temp2_, 1., y, y);
        blas2::symv( laplaceM_.weights(), y, y);//Helmholtz is never normed
    }
    ///@copydoc Elliptic::weights()const
    const Container& weights()const {return laplaceM_.weights();}
    ///@copydoc Elliptic::inv_weights()const
    const Container& inv_weights()const {return laplaceM_.inv_weights();}
    /**
     * @brief Preconditioner to use in conjugate gradient solvers
     *
     * multiply result by these coefficients to get the normed result
     * @return the inverse weights without volume
     */
    const Container& precond()const {return laplaceM_.precond();}
    /**
     * @brief Change alpha
     *
     * @return reference to alpha
     */
    value_type& alpha( ){  return alpha_;}
    /**
     * @brief Access alpha
     *
     * @return alpha
     */
    value_type alpha( ) const  {return alpha_;}
    /**
     * @brief Set Chi in the above formula
     *
     * @param chi new container
     */
    void set_chi( const Container& chi) {chi_=chi; }
    /**
     * @brief Access chi
     *
     * @return chi
     */
    const Container& chi()const {return chi_;}
  private:
    Elliptic<Geometry, Matrix, Container> laplaceM_;
    Container temp1_, temp2_;
    Container chi_;
    value_type alpha_;
};
///@cond
template< class E>
struct TensorTraits< GeneralHelmholtz<E> >
{
    using value_type  = get_value_type<typename E::value_type>;
    using tensor_category = SelfMadeMatrixTag;
};
template< class G, class M, class V>
struct TensorTraits< Helmholtz<G, M, V> >
{
    using value_type  = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
template< class G, class M, class V>
struct TensorTraits< Helmholtz2<G, M, V> >
{
    using value_type  = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
///@endcond


} //namespace dg

