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
 * @brief A general Helmholtz-type operator \f$ (\chi-\alpha F) \f$
 *
 * @ingroup matrixoperators
 *
 * where \f$ \chi\f$ is a vector and \f$\alpha\f$ a scalar and \f$F\f$ is an operator.
 * @attention If \f$ F\f$ is the \c Elliptic operator then the Laplacian in
 * this formula becomes positive as opposed to the negative sign in the \c Elliptic
 * operator
 *
 * Can be used by the \c dg::PCG class. The following example shows how the class can be used to act as a \c Helmholtz2 operator:
 @snippet helmholtzg2_b.cpp doxygen
 * @copydoc hide_geometry_matrix_container
 * @note The intention is for Matrix to be one of the Elliptic classes with
 * the \c weights() and \c precond() methods defined. If Matrix is to be an
 * arbitrary functor then it is more convenient to simply directly use a lambda
 * function to achieve the computation of \f$ y = (\chi - \alpha F)x\f$
 */
template<class Matrix, class Container>
struct GeneralHelmholtz
{
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;

    ///@brief empty object ( no memory allocation)
    GeneralHelmholtz() = default;

    /**
     * @brief Construct from given Matrix object
     * @param alpha Scalar in the above formula
     * @param matrix an existing elliptic operator
     */
    GeneralHelmholtz( value_type alpha, Matrix matrix):
        m_alpha(alpha), m_matrix(matrix), m_chi( m_matrix.weights())
    {
        dg::blas1::copy( 1., m_chi);
    }

    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = GeneralHelmholtz( std::forward<Params>( ps)...);
    }

    ///Call weights() of Matrix class
    const Container& weights()const {return m_matrix.weights();}
    ///Call precond() of Matrix class
    const Container& precond()const {return m_matrix.precond();}

    /**
     * @brief Compute \f[ y = ( \chi - \alpha M) x \f]
     *
     * @param x lhs
     * @param y rhs contains solution
     * @tparam ContainerTypes must be usable with \c Container_type in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y)
    {
        if( m_alpha != 0)
            blas2::symv( m_matrix, x, y);
        dg::blas1::pointwiseDot( 1., m_chi, x, -m_alpha, y);

    }

    ///Write access to Matrix object
    Matrix& matrix(){
        return m_matrix;
    }
    ///Read access to Matrix object
    const Matrix& matrix()const{
        return m_matrix;
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
    value_type m_alpha;
    Matrix m_matrix;
    Container m_chi;
};

///@brief a 2d Helmholtz opereator \f$ (\chi - \alpha F)\f$ with \f$ F = -\Delta\f$
///@copydetails GeneralHelmholtz
///@ingroup matrixoperators
template<class Geometry, class Matrix, class Container>
using Helmholtz = GeneralHelmholtz<dg::Elliptic2d<Geometry,Matrix,Container>, Container>;
///@brief a 1d Helmholtz opereator \f$ (\chi - \alpha F)\f$ with \f$ F = -\partial_x^2\f$
///@copydetails GeneralHelmholtz
///@ingroup matrixoperators
template<class Geometry, class Matrix, class Container>
using Helmholtz1d = GeneralHelmholtz<dg::Elliptic1d<Geometry,Matrix,Container>, Container>;
///@brief a 2d Helmholtz opereator \f$ (\chi - \alpha F)\f$ with \f$ F = -\Delta\f$
///@copydetails GeneralHelmholtz
///@ingroup matrixoperators
template<class Geometry, class Matrix, class Container>
using Helmholtz2d = GeneralHelmholtz<dg::Elliptic2d<Geometry,Matrix,Container>, Container>;
///@brief a 3d Helmholtz opereator \f$ (\chi - \alpha F)\f$ with \f$ F = -\Delta\f$
///@copydetails GeneralHelmholtz
///@ingroup matrixoperators
template<class Geometry, class Matrix, class Container>
using Helmholtz3d = GeneralHelmholtz<dg::Elliptic3d<Geometry,Matrix,Container>, Container>;

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
 @snippet helmholtzg2_b.cpp doxygen
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
     * \f[ y = \left[ \chi +2 \alpha\Delta +  \alpha^2\Delta \left(\chi^{-1}\Delta \right)\right] x \f] to make the matrix symmetric
     * @param x lhs (is constant up to changes in ghost cells)
     * @param y rhs contains solution
     * @note Takes care of sign in laplaceM and thus multiplies by -alpha
     */
    void symv(const Container& x, Container& y)
    {
        if( alpha_ != 0)
        {
            blas2::symv( m_laplaceM, x, temp1_); // temp1_ = -nabla_perp^2 x
            blas1::pointwiseDivide(temp1_, chi_, y); //temp2_ = (chi^-1)*nabla_perp^2 x
            blas2::symv( m_laplaceM, y, temp2_);//temp2_ = nabla_perp^2 *(chi^-1)*nabla_perp^2 x
        }
        blas1::pointwiseDot( chi_, x, y); //y = chi*x
        blas1::axpby( 1., y, -2.*alpha_, temp1_, y);
        blas1::axpby( alpha_*alpha_, temp2_, 1., y, y);
    }
    ///@copydoc Elliptic2d::weights()const
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
    Elliptic2d<Geometry, Matrix, Container> m_laplaceM;
    Container temp1_, temp2_;
    Container chi_;
    value_type alpha_;
};
///@cond
template< class M, class V>
struct TensorTraits< GeneralHelmholtz<M, V> >
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

