#ifndef _DG_POISSON_CUH
#define _DG_POISSON_CUH

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
 * @brief Poisson bracket scheme
 *
 * Equal to the Arakawa class except for the possitility to use mixed boundary conditions
 * @ingroup arakawa
 * @copydoc hide_geometry_matrix_container
 */
template< class Geometry, class Matrix, class container >
struct Poisson
{
    /**
     * @brief Create Poisson on a grid
     * @param g The grid
     */
    Poisson( const Geometry& g);
    /**
     * @brief Create Poisson on a grid using different boundary conditions
     * @param g The grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    Poisson( const Geometry& g, bc bcx, bc bcy);
    /**
     * @brief Create Poisson on a grid using different boundary conditions
     * @param g The grid
     * @param bcxlhs The lhs boundary condition in x
     * @param bcxrhs The rhs boundary condition in x
     * @param bcylhs The lhs boundary condition in y
     * @param bcyrhs The rhs boundary condition in y
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
     */
    void operator()( const container& lhs, const container& rhs, container& result);

    /**
     * @brief Return internally used x - derivative 
     *
     * The same as a call to dg::create::dx( g, bcx)
     * @return derivative
     */
    const Matrix& dxlhs() {return dxlhs_;}
    /**
     * @brief Return internally used y - derivative 
     *
     * The same as a call to dg::create::dy( g, bcy)
     * @return derivative
     */
    const Matrix& dylhs() {return dylhs_;}
    /**
     * @brief Return internally used x - derivative 
     *
     * The same as a call to dg::create::dx( g, bcx)
     * @return derivative
     */
    const Matrix& dxrhs() {return dxrhs_;}
    /**
     * @brief Return internally used y - derivative
     *
     * The same as a call to dg::create::dy( g, bcy)
     * @return derivative
     */
    const Matrix& dyrhs() {return dyrhs_;}
    /**
     * @brief Compute the total variation integrand, uses bc of rhs of poisson bracket
     *
     * Computes \f[ (\nabla\phi)^2 = \partial_i \phi g^{ij}\partial_j \phi \f]
     * in the plane of a 2x1 product space
     * @param phi function 
     * @param varphi may equal phi, contains result on output
     */
    void variationRHS( const container& phi, container& varphi)
    {
        blas2::symv( dxrhs_, phi, dxrhsrhs_);
        blas2::symv( dyrhs_, phi, dyrhsrhs_);
        tensor::multiply2d( metric_, dxrhsrhs_, dyrhsrhs_, varphi, helper_);
        blas1::pointwiseDot( varphi, dxrhsrhs_, varphi);
        blas1::pointwiseDot( 1., helper_, dyrhsrhs_,1., varphi );
    }

  private:
    container dxlhslhs_, dxrhsrhs_, dylhslhs_, dyrhsrhs_, helper_;
    Matrix dxlhs_, dylhs_, dxrhs_, dyrhs_;
    SparseElement<container> perp_vol_inv_;
    SparseTensor<container> metric_;
};

///@cond
//idea: backward transform lhs and rhs and then use bdxf and bdyf , then forward transform
//needs less memory!! and is faster
template< class Geometry, class Matrix, class container>
Poisson<Geometry, Matrix, container>::Poisson( const Geometry& g ): 
    dxlhslhs_( dg::evaluate( one, g) ), dxrhsrhs_(dxlhslhs_), dylhslhs_(dxlhslhs_), dyrhsrhs_( dxlhslhs_), helper_( dxlhslhs_),
    dxlhs_(dg::create::dx( g, g.bcx(),dg::centered)),
    dylhs_(dg::create::dy( g, g.bcy(),dg::centered)),
    dxrhs_(dg::create::dx( g, g.bcx(),dg::centered)),
    dyrhs_(dg::create::dy( g, g.bcy(),dg::centered))
{
    metric_=g.metric().perp();
    perp_vol_inv_ = dg::tensor::determinant(metric_);
    dg::tensor::sqrt(perp_vol_inv_);
}

template< class Geometry, class Matrix, class container>
Poisson<Geometry, Matrix, container>::Poisson( const Geometry& g, bc bcx, bc bcy): 
    dxlhslhs_( dg::evaluate( one, g) ), dxrhsrhs_(dxlhslhs_), dylhslhs_(dxlhslhs_), dyrhsrhs_( dxlhslhs_), helper_( dxlhslhs_),
    dxlhs_(dg::create::dx( g, bcx,dg::centered)),
    dylhs_(dg::create::dy( g, bcy,dg::centered)),
    dxrhs_(dg::create::dx( g, bcx,dg::centered)),
    dyrhs_(dg::create::dy( g, bcy,dg::centered))
{ 
    metric_=g.metric().perp();
    perp_vol_inv_ = dg::tensor::determinant(metric_);
    dg::tensor::sqrt(perp_vol_inv_);
}

template< class Geometry, class Matrix, class container>
Poisson<Geometry, Matrix, container>::Poisson(  const Geometry& g, bc bcxlhs, bc bcylhs, bc bcxrhs, bc bcyrhs): 
    dxlhslhs_( dg::evaluate( one, g) ), dxrhsrhs_(dxlhslhs_), dylhslhs_(dxlhslhs_), dyrhsrhs_( dxlhslhs_), helper_( dxlhslhs_),
    dxlhs_(dg::create::dx( g, bcxlhs,dg::centered)),
    dylhs_(dg::create::dy( g, bcylhs,dg::centered)),
    dxrhs_(dg::create::dx( g, bcxrhs,dg::centered)),
    dyrhs_(dg::create::dy( g, bcyrhs,dg::centered))
{ 
    metric_=g.metric().perp();
    perp_vol_inv_ = dg::tensor::determinant(metric_);
    dg::tensor::sqrt(perp_vol_inv_);
}

template< class Geometry, class Matrix, class container>
void Poisson< Geometry, Matrix, container>::operator()( const container& lhs, const container& rhs, container& result)
{
    blas2::symv(  dxlhs_, lhs,  dxlhslhs_); //dx_lhs lhs
    blas2::symv(  dylhs_, lhs,  dylhslhs_); //dy_lhs lhs
    blas2::symv(  dxrhs_, rhs,  dxrhsrhs_); //dx_rhs rhs
    blas2::symv(  dyrhs_, rhs,  dyrhsrhs_); //dy_rhs rhs
    
    blas1::pointwiseDot( 1., dxlhslhs_, dyrhsrhs_, -1., dylhslhs_, dxrhsrhs_, 0., result);   
    tensor::pointwiseDot( perp_vol_inv_, result, result);
}
///@endcond

}//namespace dg

#endif //_DG_POISSON_CUH
