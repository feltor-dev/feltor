#ifndef _DG_POISSON_CUH
#define _DG_POISSON_CUH

#include "blas.h"
#include "topology/geometry.h"
#include "enums.h"
#include "topology/evaluation.h"
#include "topology/derivativesA.h"
#ifdef MPI_VERSION
#include "topology/mpi_evaluation.h"
#endif

/*! @file

  object for computation of Poisson bracket
  */
namespace dg
{

/**
 * @brief Direct discretization of %Poisson bracket \f$ \{ f,g\} \f$
 *
 * Computes \f[ \{f,g\} := \chi/\sqrt{g_{2d}}\left(\partial_x f\partial_y g - \partial_y f\partial_x g\right) \f]
 * where \f$ g_{2d} = g/g_{zz}\f$ is the two-dimensional volume element of the plane in 2x1 product space and \f$ \chi\f$ is an optional factor.
 * Has the possitility to use mixed boundary conditions
 * @snippet poisson_t.cpp doxygen
 * @sa A discussion of this and other advection schemes can also be found here https://mwiesenberger.github.io/advection
 * @ingroup arakawa
 * @copydoc hide_geometry_matrix_container
 */
template< class Geometry, class Matrix, class Container >
struct Poisson
{
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    Poisson() = default;
    /**
     * @brief Create Poisson on a grid
     * @param g The grid
     * @note chi defaults to 1
     */
    Poisson( const Geometry& g);
    /**
     * @brief Create Poisson on a grid using different boundary conditions
     * @param g The grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     * @note chi defaults to 1
     */
    Poisson( const Geometry& g, bc bcx, bc bcy);
    /**
     * @brief Create Poisson on a grid using different boundary conditions
     * @param g The grid
     * @param bcxlhs The lhs boundary condition in x
     * @param bcxrhs The rhs boundary condition in x
     * @param bcylhs The lhs boundary condition in y
     * @param bcyrhs The rhs boundary condition in y
     * @note chi defaults to 1
     */
    Poisson( const Geometry& g, bc bcxlhs, bc bcylhs, bc bcxrhs, bc bcyrhs );
    /**
     * @brief Compute poisson's bracket
     *
     * Computes \f[ [f,g] := 1/\sqrt{g_{2d}}\left(\partial_x f\partial_y g - \partial_y f\partial_x g\right) \f]
     * where \f$ g_{2d} = g/g_{zz}\f$ is the two-dimensional volume element of the plane in 2x1 product space.
     * @param lhs left hand side in x-space
     * @param rhs rights hand side in x-space
     * @param result Poisson's bracket in x-space
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void operator()( const ContainerType0& lhs, const ContainerType1& rhs, ContainerType2& result);
    /**
     * @brief Change Chi
     *
     * @param new_chi The new chi
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0>
    void set_chi( const ContainerType0& new_chi) {
        dg::blas1::pointwiseDivide( new_chi, m_perp_vol, m_chi);
    }

    /**
     * @brief Return internally used x - derivative
     *
     * The same as a call to \c dg::create::dx( g, bcxlhs)
     * @return derivative
     */
    const Matrix& dxlhs() const {
        return m_dxlhs;
    }
    /**
     * @brief Return internally used y - derivative
     *
     * The same as a call to \c dg::create::dy( g, bcylhs)
     * @return derivative
     */
    const Matrix& dylhs() const {
        return m_dylhs;
    }
    /**
     * @brief Return internally used x - derivative
     *
     * The same as a call to \c dg::create::dx( g, bcxrhs)
     * @return derivative
     */
    const Matrix& dxrhs() const {
        return m_dxrhs;}
    /**
     * @brief Return internally used y - derivative
     *
     * The same as a call to \c dg::create::dy( g, bcyrhs)
     * @return derivative
     */
    const Matrix& dyrhs() {
        return m_dyrhs;
    }

  private:
    Container m_dxlhslhs, m_dxrhsrhs, m_dylhslhs, m_dyrhsrhs, m_helper;
    Matrix m_dxlhs, m_dylhs, m_dxrhs, m_dyrhs;
    Container m_chi, m_perp_vol;
};

///@cond
//idea: backward transform lhs and rhs and then use bdxf and bdyf , then forward transform
//needs less memory!! and is faster
template< class Geometry, class Matrix, class Container>
Poisson<Geometry, Matrix, Container>::Poisson( const Geometry& g ):
    Poisson( g, g.bcx(), g.bcy(), g.bcx(), g.bcy()){}

template< class Geometry, class Matrix, class Container>
Poisson<Geometry, Matrix, Container>::Poisson( const Geometry& g, bc bcx, bc bcy):
    Poisson( g, bcx, bcy, bcx, bcy){}

template< class Geometry, class Matrix, class Container>
Poisson<Geometry, Matrix, Container>::Poisson(  const Geometry& g, bc bcxlhs, bc bcylhs, bc bcxrhs, bc bcyrhs):
    m_dxlhslhs( dg::evaluate( one, g) ), m_dxrhsrhs(m_dxlhslhs), m_dylhslhs(m_dxlhslhs), m_dyrhsrhs( m_dxlhslhs), m_helper( m_dxlhslhs),
    m_dxlhs(dg::create::dx( g, bcxlhs, dg::centered)),
    m_dylhs(dg::create::dy( g, bcylhs, dg::centered)),
    m_dxrhs(dg::create::dx( g, bcxrhs, dg::centered)),
    m_dyrhs(dg::create::dy( g, bcyrhs, dg::centered))
{
    m_chi = m_perp_vol = dg::tensor::volume2d(g.metric());
    dg::blas1::pointwiseDivide( 1., m_perp_vol, m_chi);
}

template< class Geometry, class Matrix, class Container>
template<class ContainerType0, class ContainerType1, class ContainerType2>
void Poisson< Geometry, Matrix, Container>::operator()( const ContainerType0& lhs, const ContainerType1& rhs, ContainerType2& result)
{
    blas2::symv(  m_dxlhs, lhs,  m_dxlhslhs); //dx_lhs lhs
    blas2::symv(  m_dylhs, lhs,  m_dylhslhs); //dy_lhs lhs
    blas2::symv(  m_dxrhs, rhs,  m_dxrhsrhs); //dx_rhs rhs
    blas2::symv(  m_dyrhs, rhs,  m_dyrhsrhs); //dy_rhs rhs

    blas1::pointwiseDot( 1., m_dxlhslhs, m_dyrhsrhs, -1., m_dylhslhs, m_dxrhsrhs, 0., result);
    blas1::pointwiseDot( m_chi, result, result);
}
///@endcond

}//namespace dg

#endif //_DG_POISSON_CUH
