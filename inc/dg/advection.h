#ifndef _DG_ADVECTION_H
#define _DG_ADVECTION_H
#include "blas.h"
#include "topology/geometry.h"
#include "enums.h"
#include "topology/evaluation.h"
#include "topology/derivativesA.h"
#ifdef MPI_VERSION
#include "topology/mpi_evaluation.h"
#endif
/*! @file

  @brief Computation of advection, gradients and divergences
  */
namespace dg
{

    //MW this scheme cannot be formulated as a weak form

/**
 * @brief %Upwind discretization of advection operator \f$ \vec v\cdot\nabla f\f$
 *
 * This is the upwind scheme where a backward derivative is used if v is
 * positive and a forward derivative else
 * For example
 * @code
// v_x  = -dy phi
dg::blas2::symv( -1., dy, phi, 0., vx);
// v_y = dx phi
dg::blas2::symv( 1., dx, phi, 0., vy);
// compute on Cartesian grid in 2d on device
dg::Advection < dg::CartesianGrid2d, dg::DMatrix, dg::DVec> advection(grid);
// df = - v Grad f
advection.upwind( -1., vx, vy, f, 0., df);
@endcode
 * @note This scheme brings its own numerical diffusion and thus does not need any other artificial viscosity mechanisms. The only places where the scheme might run into oscillations is if there is a stagnation point with v==0 at a fixed position
 * @sa A discussion of this and other advection schemes can be found here https://mwiesenberger.github.io/advection
 * @copydoc hide_geometry_matrix_container
 * @ingroup arakawa
 */
template< class Geometry, class Matrix, class Container >
struct Advection
{
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    Advection() = default;
    /**
     * @brief Create Arakawa on a grid
     * @param g The grid
     */
    Advection( const Geometry& g);
    /**
     * @brief Create Advection on a grid using different boundary conditions
     * @param g The grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    Advection( const Geometry& g, bc bcx, bc bcy);

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
        *this = Advection( std::forward<Params>( ps)...);
    }

    /**
     * @brief Compute Advection term \f$ y =  \alpha \vec v\cdot\nabla f + \beta y \f$
     *
     * This uses the upwind advection mechanism, i.e. a backward derivative for positive v and a forward derivative else
     * @param alpha constant
     * @param vx Velocity in x direction
     * @param vy Velocity in y direction
     * @param f function
     * @param beta constant
     * @param result Result
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3>
    void upwind( value_type alpha, const ContainerType0& vx, const ContainerType1& vy, const ContainerType2& f, value_type beta, ContainerType3& result);

  private:
    Container m_temp0, m_temp1;
    Matrix m_dxf, m_dyf, m_dxb, m_dyb;
};

///@cond
template<class Geometry, class Matrix, class Container>
Advection<Geometry, Matrix, Container>::Advection( const Geometry& g ):
    Advection( g, g.bcx(), g.bcy()) { }

template<class Geometry, class Matrix, class Container>
Advection<Geometry, Matrix, Container>::Advection( const Geometry& g, bc bcx, bc bcy):
    m_temp0( dg::construct<Container>(dg::evaluate( one, g)) ), m_temp1(m_temp0),
    m_dxf(dg::create::dx( g, bcx, dg::forward)),
    m_dyf(dg::create::dy( g, bcy, dg::forward)),
    m_dxb(dg::create::dx( g, bcx, dg::backward)),
    m_dyb(dg::create::dy( g, bcy, dg::backward))
{
}

template< class Geometry, class Matrix, class Container>
template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3>
void Advection<Geometry, Matrix, Container>::upwind( value_type alpha, const ContainerType0& vx, const ContainerType1& vy, const ContainerType2& f, value_type beta, ContainerType3& result)
{
    blas2::symv( m_dxb, f, m_temp0);
    blas2::symv( m_dxf, f, m_temp1);
    blas1::evaluate( result, dg::Axpby( alpha, beta), dg::UpwindProduct(), vx, m_temp0, m_temp1);
    blas2::symv( m_dyb, f, m_temp0);
    blas2::symv( m_dyf, f, m_temp1);
    blas1::evaluate( result, dg::Axpby( alpha, value_type(1)), dg::UpwindProduct(), vy, m_temp0, m_temp1);
}
///@endcond

}//namespace dg


#endif //_DG_ADVECTION_H
