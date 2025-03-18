#ifndef _DG_ARAKAWA_CUH
#define _DG_ARAKAWA_CUH

#include "blas.h"
#include "topology/geometry.h"
#include "enums.h"
#include "topology/evaluation.h"
#include "topology/derivativesA.h"
#ifdef MPI_VERSION
#include "topology/mpi_evaluation.h"
#endif

/*! @file

  @brief object for computation of Poisson bracket
  */
namespace dg
{
/**
 * @brief Arakawa's scheme for %Poisson bracket \f$ \{ f,g\} \f$
 *
 * Computes \f[ \{f,g\} := \chi/\sqrt{g_{2d}}\left(\partial_x f\partial_y g - \partial_y f\partial_x g\right) \f]
 * where \f$ g_{2d} = g/g_{zz}\f$ is the two-dimensional volume element of the plane in 2x1 product space and \f$ \chi\f$ is an optional factor.
 * If \f$ \chi=1\f$, then the discretization conserves, mass, energy and enstrophy.
 * @snippet arakawa_t.cpp function
 * @snippet arakawa_t.cpp doxygen
 * @note This is the algorithm published in
 * <a href="https://doi.org/10.1016/j.cpc.2014.07.007">L. Einkemmer, M. Wiesenberger A conservative discontinuous Galerkin scheme for the 2D incompressible Navier-Stokes equations Computer Physics Communications 185, 2865-2873 (2014)</a>
 * @sa A discussion of this and other advection schemes can also be found here https://mwiesenberger.github.io/advection
 * @copydoc hide_geometry_matrix_container
 * @ingroup arakawa
 */
template< class Geometry, class Matrix, class Container >
struct ArakawaX
{
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ArakawaX() = default;
    /**
     * @brief Create Arakawa on a grid
     * @param g The grid
     * @note chi defaults to 1
     */
    ArakawaX( const Geometry& g);
    /**
     * @brief Create Arakawa on a grid using different boundary conditions
     * @param g The grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     * @note chi defaults to 1
     */
    ArakawaX( const Geometry& g, bc bcx, bc bcy);

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
        *this = ArakawaX( std::forward<Params>( ps)...);
    }

    /**
     * @brief Compute Poisson bracket
     *
     * Computes \f[ [f,g] := 1/\sqrt{g_{2d}}\left(\partial_x f\partial_y g - \partial_y f\partial_x g\right) \f]
     * where \f$ g_{2d} = g/g_{zz}\f$ is the two-dimensional volume element of the plane in 2x1 product space.
     * @param lhs left hand side in x-space
     * @param rhs rights hand side in x-space
     * @param result Poisson's bracket in x-space
     * @note memops: 30
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void operator()( const ContainerType0& lhs, const ContainerType1& rhs, ContainerType2& result){
        return this->operator()( 1., lhs, rhs, 0., result);
    }
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void operator()( value_type alpha, const ContainerType0& lhs, const ContainerType1& rhs, value_type beta, ContainerType2& result);
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
     * The same as a call to dg::create::dx( g, bcx)
     * @return derivative
     */
    const Matrix& dx() const {
        return m_bdxf;
    }
    /**
     * @brief Return internally used y - derivative
     *
     * The same as a call to dg::create::dy( g, bcy)
     * @return derivative
     */
    const Matrix& dy() const {
        return m_bdyf;
    }

  private:
    Container m_dxlhs, m_dxrhs, m_dylhs, m_dyrhs, m_helper;
    Matrix m_bdxf, m_bdyf;
    Container m_chi, m_perp_vol;
};
///@cond
template<class Geometry, class Matrix, class Container>
ArakawaX<Geometry, Matrix, Container>::ArakawaX( const Geometry& g ):
    ArakawaX( g, g.bcx(), g.bcy()) { }

template<class Geometry, class Matrix, class Container>
ArakawaX<Geometry, Matrix, Container>::ArakawaX( const Geometry& g, bc bcx, bc bcy):
    m_dxlhs( dg::construct<Container>(dg::evaluate( one, g)) ), m_dxrhs(m_dxlhs), m_dylhs(m_dxlhs), m_dyrhs( m_dxlhs), m_helper( m_dxlhs),
    m_bdxf(dg::create::dx( g, bcx, dg::centered)),
    m_bdyf(dg::create::dy( g, bcy, dg::centered))
{
    m_chi = m_perp_vol = dg::tensor::volume2d(g.metric());
    dg::blas1::pointwiseDivide( 1., m_perp_vol, m_chi);
}

template<class T>
struct ArakawaFunctor
{
    DG_DEVICE
    void operator()(T lhs, T rhs, T dxlhs, T& dylhs, T& dxrhs, T& dyrhs) const
    {
        T result = T(0);
        result = DG_FMA(  (1./3.)*dxlhs, dyrhs, result);
        result = DG_FMA( -(1./3.)*dylhs, dxrhs, result);
        T temp = T(0);
        temp = DG_FMA(  (1./3.)*lhs, dyrhs, temp);
        dyrhs = result;
        temp = DG_FMA( -(1./3.)*dylhs, rhs, temp);
        dylhs = temp;
        temp = T(0);
        temp = DG_FMA(  (1./3.)*dxlhs, rhs, temp);
        temp = DG_FMA( -(1./3.)*lhs, dxrhs, temp);
        dxrhs = temp;
    }
};

template< class Geometry, class Matrix, class Container>
template<class ContainerType0, class ContainerType1, class ContainerType2>
void ArakawaX< Geometry, Matrix, Container>::operator()( value_type alpha, const ContainerType0& lhs, const ContainerType1& rhs, value_type beta, ContainerType2& result)
{
    //compute derivatives in x-space
    blas2::symv( m_bdxf, lhs, m_dxlhs);
    blas2::symv( m_bdyf, lhs, m_dylhs);
    blas2::symv( m_bdxf, rhs, m_dxrhs);
    blas2::symv( m_bdyf, rhs, m_dyrhs);
    blas1::subroutine( ArakawaFunctor<get_value_type<Container>>(), lhs, rhs, m_dxlhs, m_dylhs, m_dxrhs, m_dyrhs);

    blas2::symv( 1., m_bdxf, m_dylhs, 1., m_dyrhs);
    blas2::symv( 1., m_bdyf, m_dxrhs, 1., m_dyrhs);
    blas1::pointwiseDot( alpha, m_chi, m_dyrhs, beta, result);
}
///@endcond

}//namespace dg

#endif //_DG_ARAKAWA_CUH
