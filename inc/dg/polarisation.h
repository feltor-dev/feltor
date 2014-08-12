#pragma once

#include "blas.h"
#include "enums.h"
#include "backend/evaluation.cuh"
#include "backend/derivatives.cuh"
#ifdef MPI_VERSION
#include "backend/mpi_derivatives.h"
#include "backend/mpi_evaluation.h"
#endif


//#include "cusp_eigen.h"
//CAN'T BE TRANSVERSE SYMMETRIC?

/*! @file 

  Contains object for the polarisation matrix creation
  */
namespace dg
{

/**
 * @brief X-space version of polarisation term
 *
 * @ingroup highlevel
 *
 * The term discretized is \f[ -\nabla ( \chi \nabla ) \f]
 * @tparam Matrix The Matrix class to use
 * @tparam Vector The Vector class to use
 * @tparam Preconditioner The Preconditioner class to use
 * This class has the SelfMadeMatrixTag so it can be used in blas2::symv functions 
 * and thus in a conjugate gradient solver. 
 *
 */
template <class Matrix, class Vector, class Preconditioner>
class Polarisation
{
    public:
    /**
     * @brief Construct from Grid
     *
     * @tparam Grid The Grid class. A call to dg::evaluate( one, g) must return an instance of the Vector class, a call to dg::create::weights(g) and dg::create::precond(g)
     * must return instances of the Preconditioner class and calls to dg::create::dx( g, not_normed, backward) and jump2d( g, bcx, bcy) are made.
     * @param g The Grid, boundary conditions are taken from here
     */
    template< class Grid>
    Polarisation( const Grid& g): 
        xchi( dg::evaluate( one, g) ), xx(xchi), temp( xx),
        weights_(dg::create::weights(g, dg::cylindrical)), precond_(dg::create::precond(g, dg::cylindrical)), 
        rightx( dg::create::dx( g, g.bcx(), normed, forward)),
        righty( dg::create::dy( g, g.bcy(), normed, forward)),
        leftx ( dg::create::dx( g, inverse( g.bcx()), normed, backward)),
        lefty ( dg::create::dy( g, inverse( g.bcy()), normed, backward)),
        jump  ( dg::create::jump2d( g, g.bcx(), g.bcy()) ) 
    { }
    /**
     * @brief Construct from grid and boundary conditions
     *
     * @tparam Grid The Grid class. A call to dg::evaluate( one, g) must return an instance of the Vector class, a call to dg::create::weights(g) and dg::create::precond(g)
     * must return instances of the Preconditioner class and calls to dg::create::dx( g, not_normed, backward) and jump2d( g, bcx, bcy) are made.
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     */
    template< class Grid>
    Polarisation( const Grid& g, bc bcx, bc bcy): 
        xchi( dg::evaluate(one, g)), xx(xchi), temp( xx),
        weights_(dg::create::weights(g)), precond_(dg::create::precond(g)),
        rightx(dg::create::dx( g,bcx, normed, forward)),
        righty(dg::create::dy( g,bcy, normed, forward)),
        leftx (dg::create::dx( g, inverse(bcx), normed, backward)),
        lefty (dg::create::dy( g, inverse(bcy), normed, backward)),
        jump  (dg::create::jump2d( g, bcx, bcy))
    { }

    /**
     * @brief Change Chi
     *
     * @param chi The new chi
     */
    void set_chi( const Vector& chi)
    {
        xchi = chi;
        //dg::blas1::pointwiseDot( weights_, chi, xchi);
    }
    /**
     * @brief Returns the weights to use in conjugate gradient
     *
     * @return weights
     */
    const Preconditioner& weights()const {return weights_;}
    /**
     * @brief Returns the preconditioner to use in conjugate gradient
     *
     * @return inverse weights
     */
    const Preconditioner& precond()const {return precond_;}

    /**
     * @brief Computes the polarisation term
     *
     * @param x left-hand-side
     * @param y result
     */
    void symv( Vector& x, Vector& y) 
    {
        dg::blas2::gemv( rightx, x, temp); //R_x*x 
        dg::blas1::pointwiseDot( xchi, temp, temp); //Chi*R_x*x 
        dg::blas2::gemv( leftx, temp, xx); //L_x*Chi*R_x*x
        dg::blas2::symv( weights_, xx, xx);

        dg::blas2::gemv( righty, x, temp);
        dg::blas1::pointwiseDot( xchi, temp, temp);
        dg::blas2::gemv( lefty, temp, y);
        dg::blas2::symv( weights_, y, y);
        
        dg::blas2::symv( jump, x, temp);
        dg::blas1::axpby( -1., xx, -1., y, xx); //-D_xx - D_yy + J
        dg::blas1::axpby( +1., temp, 1., xx, y); 
    }
    private:
    bc inverse( bc bound)
    {
        if( bound == DIR) return NEU;
        if( bound == NEU) return DIR;
        if( bound == DIR_NEU) return NEU_DIR;
        if( bound == NEU_DIR) return DIR_NEU;
        return PER;
    }
    Matrix leftx, lefty, rightx, righty, jump;
    Preconditioner weights_, precond_; //contain coeffs for chi multiplication
    Vector xchi, xx, temp;
};

///@cond
template< class M, class V, class P>
struct MatrixTraits< Polarisation<M, V, P> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

} //namespace dg

