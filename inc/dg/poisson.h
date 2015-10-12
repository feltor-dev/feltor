#ifndef _DG_POISSON_CUH
#define _DG_POISSON_CUH

#include "blas.h"
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
 * @tparam Matrix The Matrix class to use
 * @tparam container The vector class on which to operate on. The blas2 function symv( m, x, y) must be callable and may not change x. 
 */
template< class Matrix, class container >
struct Poisson
{
    /**
     * @brief Create Poisson on a grid
     *
     * @tparam Grid The Grid class. The functions dg::create::dx( g, bcx) and
     * dg::create::dy( g, bcy) must be callable and return an instance of the Matrix class. Furthermore dg::evaluate( one, g) must return an instance of the container class.
     * @param g The grid
     */
    template< class Grid>
    Poisson( const Grid& g);
    /**
     * @brief Create Poisson on a grid using different boundary conditions
     *
     * @tparam Grid The Grid class. The functions dg::create::dx( g, bcx) and
     * dg::create::dy( g, bcy) must be callable and return an instance of the Matrix class. Furthermore dg::evaluate( one, g) must return an instance of the container class.
     * @param g The grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    template< class Grid>
    Poisson( const Grid& g, bc bcx, bc bcy);
    /**
     * @brief Create Poisson on a grid using different boundary conditions
     *
     * @tparam Grid The Grid class. The functions dg::create::dx( g, bcx) and
     * dg::create::dy( g, bcy) must be callable and return an instance of the Matrix class. Furthermore dg::evaluate( one, g) must return an instance of the container class.
     * @param g The grid
     * @param bcxlhs The lhs boundary condition in x
     * @param bcxrhs The rhs boundary condition in x
     * @param bcylhs The lhs boundary condition in y
     * @param bcyrhs The rhs boundary condition in y
     */
    template< class Grid>
    Poisson( const Grid& g, bc bcxlhs, bc bcylhs, bc bcxrhs, bc bcyrhs );
    /**
     * @brief Compute poisson's bracket
     *
     * Computes \f[ [f,g] := \partial_x f\partial_y g - \partial_y f\partial_x g \f]
     * @param lhs left hand side in x-space
     * @param rhs rights hand side in x-space
     * @param result Poisson's bracket in x-space
     */
    void operator()( container& lhs, container& rhs, container& result);

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
     * Computes \f[ (\nabla\phi)^2 \f]
     * @param phi function 
     * @param varphi may equal phi, contains result on output
     * @note same as a call to bracketS( phi, phi, varphi)
     */
    void variationRHS( container& phi, container& varphi)
    {
        blas2::symv( dxrhs_, phi, dxrhsrhs_);
        blas2::symv( dyrhs_, phi, dyrhsrhs_);
        blas1::pointwiseDot(dxrhsrhs_, dxrhsrhs_, helper_);
        blas1::pointwiseDot(dyrhsrhs_, dyrhsrhs_, varphi);
        blas1::axpby( 1.,helper_, 1., varphi, varphi);
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
        blas2::symv(  dxlhs_, lhs,  dxlhslhs_); //dx_lhs lhs
        blas2::symv(  dylhs_, lhs,  dylhslhs_); //dy_lhs lhs
        blas2::symv(  dxrhs_, rhs,  dxrhsrhs_); //dx_rhs rhs
        blas2::symv(  dyrhs_, rhs,  dyrhsrhs_); //dy_rhs rhs
        
        blas1::pointwiseDot( dxlhslhs_, dyrhsrhs_, helper_);   //dx_lhs lhs * dy_rhs rhs
        blas1::pointwiseDot( dylhslhs_, dyrhsrhs_, result);    //dy_lhs lhs * dx_rhs rhs
        
        blas1::axpby( 1., helper_, 1., result,result);        //dx_lhs lhs * dy_rhs rhs + dy_lhs lhs * dx_rhs rhs
    }

  private:
    container dxlhslhs_,dxrhsrhs_,dylhslhs_,dyrhsrhs_,helper_;
    Matrix dxlhs_, dylhs_,dxrhs_,dyrhs_;
};

//idea: backward transform lhs and rhs and then use bdxf and bdyf , then forward transform
//needs less memory!! and is faster
template< class Matrix, class container>
template< class Grid>
Poisson<Matrix, container>::Poisson( const Grid& g ): 
    dxlhslhs_( dg::evaluate( one, g) ), dxrhsrhs_(dxlhslhs_), dylhslhs_(dxlhslhs_), dyrhsrhs_( dxlhslhs_), helper_( dxlhslhs_),
    dxlhs_(dg::create::dx( g, g.bcx(),dg::centered)),
    dylhs_(dg::create::dy( g, g.bcy(),dg::centered)),
    dxrhs_(dg::create::dx( g, g.bcx(),dg::centered)),
    dyrhs_(dg::create::dy( g, g.bcy(),dg::centered))
{ }
template< class Matrix, class container>
template< class Grid>
Poisson<Matrix, container>::Poisson( const Grid& g, bc bcx, bc bcy): 
    dxlhslhs_( dg::evaluate( one, g) ), dxrhsrhs_(dxlhslhs_), dylhslhs_(dxlhslhs_), dyrhsrhs_( dxlhslhs_), helper_( dxlhslhs_),
    dxlhs_(dg::create::dx( g, bcx,dg::centered)),
    dylhs_(dg::create::dy( g, bcy,dg::centered)),
    dxrhs_(dg::create::dx( g, bcx,dg::centered)),
    dyrhs_(dg::create::dy( g, bcy,dg::centered))
{
}
template< class Matrix, class container>
template< class Grid>
Poisson<Matrix, container>::Poisson(  const Grid& g, bc bcxlhs, bc bcylhs, bc bcxrhs, bc bcyrhs): 
    dxlhslhs_( dg::evaluate( one, g) ), dxrhsrhs_(dxlhslhs_), dylhslhs_(dxlhslhs_), dyrhsrhs_( dxlhslhs_), helper_( dxlhslhs_),
    dxlhs_(dg::create::dx( g, bcxlhs,dg::centered)),
    dylhs_(dg::create::dy( g, bcylhs,dg::centered)),
    dxrhs_(dg::create::dx( g, bcxrhs,dg::centered)),
    dyrhs_(dg::create::dy( g, bcyrhs,dg::centered))
{
}
template< class Matrix, class container>
void Poisson< Matrix, container>::operator()( container& lhs, container& rhs, container& result)
{
    blas2::symv(  dxlhs_, lhs,  dxlhslhs_); //dx_lhs lhs
    blas2::symv(  dylhs_, lhs,  dylhslhs_); //dy_lhs lhs
    blas2::symv(  dxrhs_, rhs,  dxrhsrhs_); //dx_rhs rhs
    blas2::symv(  dyrhs_, rhs,  dyrhsrhs_); //dy_rhs rhs
    
    blas1::pointwiseDot( dxlhslhs_, dyrhsrhs_, helper_);   //dx_lhs lhs * dy_rhs rhs
    blas1::pointwiseDot( dylhslhs_, dxrhsrhs_, result);    //dy_lhs lhs * dx_rhs rhs
    
    blas1::axpby( 1., helper_, -1., result,result);        //dx_lhs lhs * dy_rhs rhs - dy_lhs lhs * dx_rhs rhs
}

}//namespace dg

#endif //_DG_POISSON_CUH
