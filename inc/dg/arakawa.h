#ifndef _DG_ARAKAWA_CUH
#define _DG_ARAKAWA_CUH

#include "blas.h"
#include "geometry/geometry.h"
#include "enums.h"
#include "backend/evaluation.cuh"
#include "backend/derivatives.h"
#ifdef MPI_VERSION
#include "backend/mpi_derivatives.h"
#include "backend/mpi_evaluation.h"
#endif

/*! @file 
  
  object for computation of Poisson bracket
  */
namespace dg
{

/**
 * @brief X-space generalized version of Arakawa's scheme
 *
 * @copydoc hide_matrix_container
 * @copydoc hide_geometry
 * @ingroup arakawa
 */
template< class Geometry, class Matrix, class container >
struct ArakawaX
{
    /**
     * @brief Create Arakawa on a grid
     * @param g The grid
     */
    ArakawaX( const Geometry& g);
    /**
     * @brief Create Arakawa on a grid using different boundary conditions
     * @param g The grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    ArakawaX( const Geometry& g, bc bcx, bc bcy);

    /**
     * @brief Compute poisson's bracket
     *
     * Computes \f[ [f,g] := 1/\sqrt{g_{2d}}\left(\partial_x f\partial_y g - \partial_y f\partial_x g\right) \f]
     * where \f$ g_{2d} = g/g_{zz}\f$ is the two-dimensional volume element of the plane in 2x1 product space. 
     * @param lhs left hand side in x-space
     * @param rhs rights hand side in x-space
     * @param result Poisson's bracket in x-space
     */
    void operator()( const container& lhs, const container& rhs, container& result);

    /**
     * @brief Return internally used x - derivative 
     *
     * The same as a call to dg::create::dx( g, bcx)
     * @return derivative
     */
    const Matrix& dx() {return bdxf;}
    /**
     * @brief Return internally used y - derivative
     *
     * The same as a call to dg::create::dy( g, bcy)
     * @return derivative
     */
    const Matrix& dy() {return bdyf;}

    /**
     * @brief Compute the total variation integrand 
     *
     * Computes \f[ (\nabla\phi)^2 = \partial_i \phi g^{ij}\partial_j \phi \f]
     * in the plane of a 2x1 product space
     * @param phi function 
     * @param varphi may equal phi, contains result on output
     */
    void variation( const container& phi, container& varphi)
    {
        blas2::symv( bdxf, phi, dxrhs);
        blas2::symv( bdyf, phi, dyrhs);
        tensor::multiply2d( metric_, dxrhs, dyrhs, varphi, helper_);
        blas1::pointwiseDot( varphi, dxrhs, varphi);
        blas1::pointwiseDot( 1., helper_, dyrhs,1., varphi );
    }

  private:
    container dxlhs, dxrhs, dylhs, dyrhs, helper_;
    Matrix bdxf, bdyf;
    SparseElement<container> perp_vol_inv_;
    SparseTensor<container> metric_;
};

template<class Geometry, class Matrix, class container>
ArakawaX<Geometry, Matrix, container>::ArakawaX( const Geometry& g ): 
    dxlhs( dg::evaluate( one, g) ), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper_( dxlhs), 
    bdxf( dg::create::dx( g, g.bcx())),
    bdyf( dg::create::dy( g, g.bcy()))
{
    metric_=g.metric().perp();
    perp_vol_inv_ = dg::tensor::determinant(metric_);
    dg::tensor::sqrt(perp_vol_inv_);
}
template<class Geometry, class Matrix, class container>
ArakawaX<Geometry, Matrix, container>::ArakawaX( const Geometry& g, bc bcx, bc bcy): 
    dxlhs( dg::evaluate( one, g) ), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper_( dxlhs),
    bdxf(dg::create::dx( g, bcx)),
    bdyf(dg::create::dy( g, bcy))
{ 
    metric_=g.metric().perp();
    perp_vol_inv_ = dg::tensor::determinant(metric_);
    dg::tensor::sqrt(perp_vol_inv_);
}

template< class Geometry, class Matrix, class container>
void ArakawaX< Geometry, Matrix, container>::operator()( const container& lhs, const container& rhs, container& result)
{
    //compute derivatives in x-space
    blas2::symv( bdxf, lhs, dxlhs);
    blas2::symv( bdyf, lhs, dylhs);
    blas2::symv( bdxf, rhs, dxrhs);
    blas2::symv( bdyf, rhs, dyrhs);

    // order is important now
    // +x (1) -> result und (2) -> blhs
    blas1::pointwiseDot( lhs, dyrhs, result);
    blas1::pointwiseDot( lhs, dxrhs, helper_);

    // ++ (1) -> dyrhs and (2) -> dxrhs
    blas1::pointwiseDot( dxlhs, dyrhs, dyrhs);
    blas1::pointwiseDot( dylhs, dxrhs, dxrhs);

    // x+ (1) -> dxlhs and (2) -> dylhs
    blas1::pointwiseDot( dxlhs, rhs, dxlhs);
    blas1::pointwiseDot( dylhs, rhs, dylhs);

    blas1::axpby( 1./3., dyrhs, -1./3., dxrhs);  //dxl*dyr - dyl*dxr -> dxrhs
    //everything which needs a dx 
    blas1::axpby( 1./3., dxlhs, -1./3., helper_);   //dxl*r - l*dxr     -> helper 
    //everything which needs a dy
    blas1::axpby( 1./3., result, -1./3., dylhs); //l*dyr - dyl*r     -> dylhs

    //blas1::axpby( 0., dyrhs,  -0., dxrhs); //++
    ////for testing purposes (note that you need to set criss-cross)
    //blas1::axpby( 1., dxlhs,  -0., helper); //x+ - +x
    //blas1::axpby( 0., result, -1., dylhs);  //+x - x+

    blas2::symv( bdyf, helper_, result);      //dy*(dxl*r - l*dxr) -> result
    blas2::symv( bdxf, dylhs, dxlhs);      //dx*(l*dyr - dyl*r) -> dxlhs
    //now sum everything up
    blas1::axpby( 1., dxlhs, 1., result); //result + dxlhs -> result
    blas1::axpby( 1., dxrhs, 1., result); //result + dyrhs -> result
    tensor::pointwiseDot( perp_vol_inv_, result, result);
}

}//namespace dg

#endif //_DG_ARAKAWA_CUH
