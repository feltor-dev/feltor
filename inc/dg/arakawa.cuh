#ifndef _DG_ARAKAWA_CUH
#define _DG_ARAKAWA_CUH

#include <cusp/ell_matrix.h>

#include "blas.h"
//#include "dlt.h"
#include "vector_traits.h"
#include "typedefs.cuh"

#include "derivatives.cuh"

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
template< class container=thrust::device_vector<double> >
struct ArakawaX
{
    typedef typename container::value_type value_type; //!< value type of container
    //typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    typedef dg::DMatrix Matrix; //!< always use device matrix
    /**
     * @brief Create Arakawa on a grid
     *
     * @param g The 2D grid
     */
    ArakawaX( const Grid2d<value_type>& g);
    /**
     * @brief Create Arakawa on a grid using different boundary conditions
     *
     * @param g The 2D grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    ArakawaX( const Grid2d<value_type>& g, bc bcx, bc bcy);
    /**
     * @brief Create Arakawa on a grid
     *
     * @param g The 3D grid
     */
    ArakawaX( const Grid3d<value_type>& g);
    /**
     * @brief Create Arakawa on a grid using different boundary conditions
     *
     * @param g The 3D grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    ArakawaX( const Grid3d<value_type>& g, bc bcx, bc bcy);
    //ArakawaX( unsigned Nx, unsigned Ny, double hx, double hy, int bcx, int bcy); //deprecated

    /**
     * @brief Compute poisson's bracket
     *
     * @param lhs left hand side in x-space
     * @param rhs rights hand side in x-space
     * @param result Poisson's bracket in x-space
     */
    void operator()( const container& lhs, const container& rhs, container& result);

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

  private:
    Matrix bdxf, bdyf;
    container dxlhs, dxrhs, dylhs, dyrhs, helper;
};

//idea: backward transform lhs and rhs and then use bdxf and bdyf , then forward transform
//needs less memory!! and is faster
template< class container>
ArakawaX<container>::ArakawaX( const Grid2d<value_type>& g): dxlhs( g.size()), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper( dxlhs)
{
    bdxf = dg::create::dx( g, g.bcx(), XSPACE);
    bdyf = dg::create::dy( g, g.bcy(), XSPACE);
}
template< class container>
ArakawaX<container>::ArakawaX( const Grid2d<value_type>& g, bc bcx, bc bcy): dxlhs( g.size()), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper( dxlhs)
{
    bdxf = dg::create::dx( g, bcx, XSPACE);
    bdyf = dg::create::dy( g, bcy, XSPACE);
}
template< class container>
ArakawaX<container>::ArakawaX( const Grid3d<value_type>& g): dxlhs( g.size()), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper( dxlhs)
{
    bdxf = dg::create::dx( g, g.bcx(), XSPACE);
    bdyf = dg::create::dy( g, g.bcy(), XSPACE);
}
template< class container>
ArakawaX<container>::ArakawaX( const Grid3d<value_type>& g, bc bcx, bc bcy): dxlhs( g.size()), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper( dxlhs)
{
    bdxf = dg::create::dx( g, bcx, XSPACE);
    bdyf = dg::create::dy( g, bcy, XSPACE);
}

template< class container>
void ArakawaX< container>::operator()( const container& lhs, const container& rhs, container& result)
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
