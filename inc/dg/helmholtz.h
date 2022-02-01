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
 * @brief Matrix class that represents a Helmholtz-type operator \f$ (\chi+\alpha\Delta) \f$
 *
 * @ingroup matrixoperators
 *
 * Discretization of \f[ (\chi+\alpha\Delta) \f]
 * where \f$ \chi\f$ is a function and \f$\alpha\f$ a scalar.
 * @attention The Laplacian in this formula is positive as opposed to the negative sign in the \c Elliptic operator
 *
 * Can be used by the CG class. The following example shows how the class can be used to act as a \c Helmholtz2 operator:
 @snippet helmholtzg2_b.cu doxygen
 * @copydoc hide_geometry_matrix_container
 */
template<class Geometry, class Matrix, class Container>
struct Helmholtz
{
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    Helmholtz() {}

    /**
     * @brief Construct from given elliptic object
     * @param alpha Scalar in the above formula
     * @param elliptic an existing elliptic operator
     */
    Helmholtz( value_type alpha, Elliptic<Geometry, Matrix, Container> elliptic):
        m_alpha(alpha), m_laplaceM(elliptic),
        m_chi( m_laplaceM.weights())
    {
        dg::blas1::copy( 1., m_chi);
    }
    /**
     * @brief Construct
     *
     * @param g The grid to use (boundary conditions are taken from there)
     * @param alpha Scalar in the above formula
     * @param dir Direction of the Laplace operator
     * @param jfactor The jfactor used in the Laplace operator (probably 1 is always the best factor but one never knows...)
     * @note The default value of \f$\chi\f$ is one.
     */
    Helmholtz( const Geometry& g, value_type alpha = 1., direction dir = dg::forward, value_type jfactor=1.):
        Helmholtz( g, g.bcx(), g.bcy(), alpha, dir, jfactor)
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
     * @note The default value of \f$\chi\f$ is one.
     */
    Helmholtz( const Geometry& g, bc bcx, bc bcy, value_type alpha = 1., direction dir = dg::forward, value_type jfactor=1.):
        m_laplaceM( g, bcx, bcy, dir, jfactor),
        m_chi( dg::construct<Container>( dg::evaluate( dg::one, g))),
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
        *this = Helmholtz( std::forward<Params>( ps)...);
    }
    /**
     * @brief apply operator
     *
     * Computes
     * \f[ y = W( \chi + \alpha\Delta) x \f] to make the matrix symmetric
     * @param x lhs (is constant up to changes in ghost cells)
     * @param y rhs contains solution
     * @tparam ContainerTypes must be usable with \c Container_type in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y)
    {
        if( m_alpha != 0)
            blas2::symv( m_laplaceM, x, y);
        dg::blas1::pointwiseDot( 1., m_chi, x, -m_alpha, y);

    }
    ///@copydoc Elliptic::weights()const
    const Container& weights()const {return m_laplaceM.weights();}
    /**
     * @brief Preconditioner to use in conjugate gradient solvers
     *
     * @return 1 (by default)
     */
    const Container& precond()const {return m_laplaceM.precond();}
    ///Write access to the elliptic class
    Elliptic<Geometry,Matrix,Container>& elliptic(){
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
     * @param chi new Container
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0>
    void set_chi( const ContainerType0& chi) {
        dg::blas1::copy( chi, m_chi);
    }
    /**
     * @brief Access chi
     * @return chi
     */
    const Container& chi() const{return m_chi;}
  private:
    Elliptic<Geometry, Matrix, Container> m_laplaceM;
    Container m_chi;
    value_type m_alpha;
};

/**
 * @brief Matrix class that represents a 3d Helmholtz-type operator \f$ (\chi+\alpha\Delta) \f$
 *
 * @ingroup matrixoperators
 *
 * This is the three-dimensional version of \c Helmholtz
 * Discretization of \f[ (\chi+\alpha\Delta) \f]
 * where \f$ \chi\f$ is a function and \f$\alpha\f$ a scalar.
 * Can be used by the CG class.
 * @attention The Laplacian in this formula is positive as opposed to the negative sign in the \c Elliptic operator
 * @copydoc hide_geometry_matrix_container
 */
template<class Geometry, class Matrix, class Container>
struct Helmholtz3d
{
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    Helmholtz3d() {}
    ///@copydoc Helmholtz::Helmholtz(value_type,Elliptic<Geometry,Matrix,Container>)
    Helmholtz3d( value_type alpha, Elliptic3d<Geometry, Matrix, Container> elliptic):
        m_alpha(alpha), m_laplaceM(elliptic),
        m_chi( elliptic.copyable())
    {
        dg::blas1::copy( 1., m_chi);
    }
    ///@copydoc Helmholtz::Helmholtz(const Geometry&,value_type,direction,value_type)
    Helmholtz3d( const Geometry& g, value_type alpha = 1., direction dir = dg::forward, value_type jfactor=1.):
        Helmholtz3d( g, g.bcx(), g.bcy(), g.bcz(), alpha, dir, jfactor)
    {
    }
    /**
     * @brief Construct
     *
     * @param g The grid to use
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param bcz boundary contition in z
     * @param alpha Scalar in the above formula
     * @param dir Direction of the Laplace operator
     * @param jfactor The jfactor used in the Laplace operator (probably 1 is always the best factor but one never knows...)
     * @note The default value of \f$\chi\f$ is one.
     */
    Helmholtz3d( const Geometry& g, bc bcx, bc bcy, bc bcz, value_type alpha = 1., direction dir = dg::forward, value_type jfactor=1.):
        m_laplaceM( g, bcx, bcy, bcz, dir, jfactor),
        m_chi( dg::construct<Container>( dg::evaluate( dg::one, g))),
        m_alpha(alpha)
    {
    }
    ///@copydoc Helmholtz::construct()
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = Helmholtz3d( std::forward<Params>( ps)...);
    }
    ///@copydoc Helmholtz::symv()
    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y)
    {
        if( m_alpha != 0)
            blas2::symv( m_laplaceM, x, y);
        dg::blas1::pointwiseDot( 1., m_chi, x, -m_alpha, y);

    }
    ///@copydoc Elliptic::weights()const
    const Container& weights()const {return m_laplaceM.weights();}
    ///@copydoc Helmholtz::precond()const
    const Container& precond()const {return m_laplaceM.precond();}
    ///Write access to the elliptic class
    Elliptic3d<Geometry, Matrix, Container>& elliptic(){
        return m_laplaceM;
    }
    ///@copydoc Helmholtz::alpha()
    value_type& alpha( ){  return m_alpha;}
    ///@copydoc Helmholtz::alpha()const
    value_type alpha( ) const  {return m_alpha;}
    ///@copydoc Helmholtz::set_chi()
    template<class ContainerType0>
    void set_chi( const ContainerType0& chi) {
        dg::blas1::copy( chi, m_chi);
    }
    ///@copydoc Helmholtz::chi()
    const Container& chi() const{return m_chi;}
  private:
    Elliptic3d<Geometry, Matrix, Container> m_laplaceM;
    Container m_chi;
    value_type m_alpha;
};


///@copydoc Helmholtz
///@ingroup matrixoperators
template<class Geometry, class Matrix, class Container>
using Helmholtz2d = Helmholtz<Geometry, Matrix, Container>;

/**
 * @brief DEPRECATED, Matrix class that represents a more general Helmholtz-type operator
 *
 * @ingroup matrixoperators
 *
 * Discretization of
 * \f[ \left[ \chi +2 \alpha\Delta +  \alpha^2\Delta \left(\chi^{-1}\Delta \right)\right] \f]
 * where \f$ \chi\f$ is a function and \f$\alpha\f$ a scalar.
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
        m_laplaceM.construct( g, bcx, bcy, dir, jfactor);
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
            blas2::symv( m_laplaceM, x, temp1_); // temp1_ = -nabla_perp^2 x
            blas1::pointwiseDivide(temp1_, chi_, y); //temp2_ = (chi^-1)*W*nabla_perp^2 x
            blas2::symv( m_laplaceM, y, temp2_);//temp2_ = nabla_perp^2 *(chi^-1)*nabla_perp^2 x
        }
        blas1::pointwiseDot( chi_, x, y); //y = chi*x
        blas1::axpby( 1., y, -2.*alpha_, temp1_, y);
        blas1::axpby( alpha_*alpha_, temp2_, 1., y, y);
    }
    ///@copydoc Elliptic::weights()const
    const Container& weights()const {return m_laplaceM.weights();}
    /**
     * @brief Preconditioner to use in conjugate gradient solvers
     *
     * @return 1 by default
     */
    const Container& precond()const {return m_laplaceM.precond();}
    /**
     * @brief Change alpha
     * @return reference to alpha
     */
    value_type& alpha( ){  return alpha_;}
    /**
     * @brief Access alpha
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
    Elliptic<Geometry, Matrix, Container> m_laplaceM;
    Container temp1_, temp2_;
    Container chi_;
    value_type alpha_;
};
///@cond


template< class G, class M, class V>
struct TensorTraits< Helmholtz3d<G, M, V> >
{
    using value_type  = get_value_type<V>;
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

