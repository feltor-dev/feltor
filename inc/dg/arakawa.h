#ifndef _DG_ARAKAWA_CUH
#define _DG_ARAKAWA_CUH

#include "blas.h"
#include "geometry/geometry.h"
#include "enums.h"
#include "geometry/evaluation.h"
#include "geometry/derivatives.h"
#ifdef MPI_VERSION
#include "geometry/mpi_derivatives.h"
#include "geometry/mpi_evaluation.h"
#endif

/*! @file

  @brief object for computation of Poisson bracket
  */
namespace dg
{
//citation missing in documentation
/**
 * @brief X-space generalized version of Arakawa's scheme
 *
 * Computes \f[ [f,g] := \chi/\sqrt{g_{2d}}\left(\partial_x f\partial_y g - \partial_y f\partial_x g\right) \f]
 * where \f$ g_{2d} = g/g_{zz}\f$ is the two-dimensional volume element of the plane in 2x1 product space and \f$ \chi\f$ is an optional factor.
 * If \f$ \chi=1\f$, then the discretization conserves, mass, energy and enstrophy.
 * @snippet arakawa_t.cu function
 * @snippet arakawa_t.cu doxygen
 * @copydoc hide_geometry_matrix_container
 * @ingroup arakawa
 */
template< class Geometry, class Matrix, class container >
struct ArakawaX
{
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
     * @brief Compute Poisson bracket
     *
     * Computes \f[ [f,g] := 1/\sqrt{g_{2d}}\left(\partial_x f\partial_y g - \partial_y f\partial_x g\right) \f]
     * where \f$ g_{2d} = g/g_{zz}\f$ is the two-dimensional volume element of the plane in 2x1 product space.
     * @param lhs left hand side in x-space
     * @param rhs rights hand side in x-space
     * @param result Poisson's bracket in x-space
     * @note memops: 30
     * @tparam ContainerTypes must be usable with \c container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void operator()( const ContainerType0& lhs, const ContainerType1& rhs, ContainerType2& result);
    /**
     * @brief Change Chi
     *
     * @param new_chi The new chi
     * @tparam ContainerTypes must be usable with \c container in \ref dispatch
     */
    template<class ContainerType0>
    void set_chi( const ContainerType0& new_chi) {
        dg::blas1::pointwiseDot( new_chi, m_inv_perp_vol, m_chi);
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

    /**
     * @brief Compute the total variation integrand
     *
     * Computes \f[ (\nabla\phi)^2 = \partial_i \phi g^{ij}\partial_j \phi \f]
     * in the plane of a 2x1 product space
     * @param phi function
     * @param varphi may equal phi, contains result on output
     * @tparam ContainerTypes must be usable with \c container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void variation( const ContainerType0& phi, ContainerType1& varphi)
    {
        blas2::symv( m_bdxf, phi, m_dxrhs);
        blas2::symv( m_bdyf, phi, m_dyrhs);
        tensor::multiply2d( m_metric, m_dxrhs, m_dyrhs, varphi, m_helper);
        blas1::pointwiseDot( 1., varphi, m_dxrhs, 1., m_helper, m_dyrhs, 0., varphi);
    }

  private:
    container m_dxlhs, m_dxrhs, m_dylhs, m_dyrhs, m_helper;
    Matrix m_bdxf, m_bdyf;
    SparseTensor<container> m_metric;
    container m_chi, m_inv_perp_vol;
};
///@cond
template<class Geometry, class Matrix, class container>
ArakawaX<Geometry, Matrix, container>::ArakawaX( const Geometry& g ):
    ArakawaX( g, g.bcx(), g.bcy()) { }

template<class Geometry, class Matrix, class container>
ArakawaX<Geometry, Matrix, container>::ArakawaX( const Geometry& g, bc bcx, bc bcy):
    m_dxlhs( dg::transfer<container>(dg::evaluate( one, g)) ), m_dxrhs(m_dxlhs), m_dylhs(m_dxlhs), m_dyrhs( m_dxlhs), m_helper( m_dxlhs),
    m_bdxf(dg::create::dx( g, bcx, dg::centered)),
    m_bdyf(dg::create::dy( g, bcy, dg::centered))
{
    m_metric=g.metric();
    m_chi = dg::tensor::determinant2d(m_metric);
    dg::blas1::transform(m_chi, m_chi, dg::SQRT<get_value_type<container>>());
    m_inv_perp_vol = m_chi;
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

template< class Geometry, class Matrix, class container>
template<class ContainerType0, class ContainerType1, class ContainerType2>
void ArakawaX< Geometry, Matrix, container>::operator()( const ContainerType0& lhs, const ContainerType1& rhs, ContainerType2& result)
{
    //compute derivatives in x-space
    blas2::symv( m_bdxf, lhs, m_dxlhs);
    blas2::symv( m_bdyf, lhs, m_dylhs);
    blas2::symv( m_bdxf, rhs, m_dxrhs);
    blas2::symv( m_bdyf, rhs, result);
    blas1::evaluate( ArakawaFunctor<get_value_type<container>>(), lhs, rhs, m_dxlhs, m_dylhs, m_dxrhs, result);

    blas2::symv( 1., m_bdxf, m_dylhs, 1., result);
    blas2::symv( 1., m_bdyf, m_dxrhs, 1., result);
    blas1::pointwiseDot( m_chi, result, result);
}
///@endcond

}//namespace dg

#endif //_DG_ARAKAWA_CUH
