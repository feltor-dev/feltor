#ifndef _DG_ARAKAWA_CUH
#define _DG_ARAKAWA_CUH

#include "blas.h"
#include "enums.h"
#include "backend/derivatives.cuh"
#ifdef MPI_VERSION
#include "backend/mpi_matrix.h"
#include "backend/mpi_derivatives.h"
#endif

/*! @file 
  
  objects for computation of Poisson bracket
  */
namespace dg
{

/**
 * @brief X-space generalized version of Arakawa's scheme
 *
 * @ingroup arakawa
 * @tparam container The vector class on which to operate on
 */
template< class Matrix, class container >
struct ArakawaX
{
    /**
     * @brief Create Arakawa on a grid
     *
     * @param g The 2D grid
     */
    template< class Grid>
    ArakawaX( const Grid& g);
    /**
     * @brief Create Arakawa on a grid using different boundary conditions
     *
     * @param g The 2D grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    template< class Grid>
    ArakawaX( const Grid& g, bc bcx, bc bcy);

    /**
     * @brief Compute poisson's bracket
     *
     * @param lhs left hand side in x-space
     * @param rhs rights hand side in x-space
     * @param result Poisson's bracket in x-space
     */
    void operator()( container& lhs, container& rhs, container& result);

    /**
     * @brief Return internally used 2d - x - derivative in ell format in XSPACE
     *
     * The same as a call to 
     * dg::create::dx( g, bcx, XSPACE)
     * but the format is the fast ell_matrix format
     * @return derivative
     */
    const Matrix& dx() {return bdxf;}
    /**
     * @brief Return internally used 2d - y - derivative in ell format in XSPACE
     *
     * The same as a call to 
     * dg::create::dy( g, bcy, XSPACE)
     * but the format is the fast ell_matrix format
     * @return derivative
     */
    const Matrix& dy() {return bdyf;}

    /**
     * @brief Compute the total variation integrand in 2D
     *
     * Computes \f[ (\nabla\phi)^2 \f]
     * @param phi function 
     * @param varphi may equal phi, contains result on output
     */
    void variation( container& phi, container& varphi)
    {
        blas2::symv( bdxf, phi, dxlhs);
        blas2::symv( bdyf, phi, dylhs);
        blas1::pointwiseDot( dxlhs, dxlhs, dxlhs);
        blas1::pointwiseDot( dylhs, dylhs, dylhs);
        blas1::axpby( 1., dxlhs, 1., dylhs, varphi);
        //typedef typename VectorTraits<container>::value_type value_type; 
        //blas1::transform( varphi, varphi, dg::SQRT<value_type>() );
    }
    /**
     * @brief Compute the "symmetric bracket"
     *
     * Computes \f[ [f,g] := \partial_x f\partial_x g + \partial_y f\partial_y g \f]

     * @param lhs The left hand side
     * @param rhs The right hand side (may equal lhs)
     * @param result The result (write only, may equal lhs or rhs)
     */
    void bracketS( container& lhs, container& rhs, container& result)
    {
        blas2::symv( bdxf, lhs, dxlhs);
        blas2::symv( bdyf, lhs, dylhs);
        if( &lhs != &rhs)
        {
            blas2::symv( bdxf, rhs, dxrhs);
            blas2::symv( bdyf, rhs, dyrhs);
            blas1::pointwiseDot( dxlhs, dxrhs, dxlhs);
            blas1::pointwiseDot( dylhs, dyrhs, dylhs);
        }
        else
        {
            blas1::pointwiseDot( dxlhs, dxlhs, dxlhs);
            blas1::pointwiseDot( dylhs, dylhs, dylhs);
        }
        blas1::axpby( 1., dxlhs, 1., dylhs, result);
    }

  private:
    container dxlhs, dxrhs, dylhs, dyrhs, helper;
    Matrix bdxf, bdyf;
};

//idea: backward transform lhs and rhs and then use bdxf and bdyf , then forward transform
//needs less memory!! and is faster
template< class Matrix, class container>
template< class Grid>
ArakawaX<Matrix, container>::ArakawaX( const Grid& g ): 
    dxlhs( dg::evaluate( one, g) ), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper( dxlhs), 
    bdxf( dg::create::dx( g, g.bcx())),
    bdyf( dg::create::dy( g, g.bcy()))
{ }
template< class Matrix, class container>
template< class Grid>
ArakawaX<Matrix, container>::ArakawaX( const Grid& g, bc bcx, bc bcy): 
    dxlhs( dg::evaluate( one, g) ), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper( dxlhs),
    bdxf(dg::create::dx( g, bcx)),
    bdyf(dg::create::dy( g, bcy))
{
}

template< class Matrix, class container>
void ArakawaX< Matrix, container>::operator()( container& lhs, container& rhs, container& result)
{
    //compute derivatives in x-space
    blas2::symv( bdxf, lhs, dxlhs);
    blas2::symv( bdyf, lhs, dylhs);
    blas2::symv( bdxf, rhs, dxrhs);
    blas2::symv( bdyf, rhs, dyrhs);

    // order is important now
    // +x (1) -> result und (2) -> blhs
    blas1::pointwiseDot( lhs, dyrhs, result);
    blas1::pointwiseDot( lhs, dxrhs, helper);

    // ++ (1) -> dyrhs and (2) -> dxrhs
    blas1::pointwiseDot( dxlhs, dyrhs, dyrhs);
    blas1::pointwiseDot( dylhs, dxrhs, dxrhs);

    // x+ (1) -> dxlhs and (2) -> dylhs
    blas1::pointwiseDot( dxlhs, rhs, dxlhs);
    blas1::pointwiseDot( dylhs, rhs, dylhs);

    blas1::axpby( 1./3., dyrhs, -1./3., dxrhs);  //dxl*dyr - dyl*dxr -> dxrhs
    //everything which needs a dx 
    blas1::axpby( 1./3., dxlhs, -1./3., helper);   //dxl*r - l*dxr     -> helper 
    //everything which needs a dy
    blas1::axpby( 1./3., result, -1./3., dylhs); //l*dyr - dyl*r     -> dylhs

    //blas1::axpby( 0., dyrhs,  -0., dxrhs); //++
    ////for testing purposes (note that you need to set criss-cross)
    //blas1::axpby( 1., dxlhs,  -0., helper); //x+ - +x
    //blas1::axpby( 0., result, -1., dylhs);  //+x - x+

    blas2::symv( bdyf, helper, result);      //dy*(dxl*r - l*dxr) -> result
    blas2::symv( bdxf, dylhs, dxlhs);      //dx*(l*dyr - dyl*r) -> dxlhs
    //now sum everything up
    blas1::axpby( 1., dxlhs, 1., result); //result + dxlhs -> result
    blas1::axpby( 1., dxrhs, 1., result); //result + dyrhs -> result
}

}//namespace dg

#endif //_DG_ARAKAWA_CUH
