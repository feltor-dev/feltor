#pragma once

#include "blas.h"
#include "enums.h"
#include "backend/evaluation.cuh"
#include "backend/derivatives.cuh"
#ifdef MPI_VERSION
#include "backend/mpi_derivatives.h"
#include "backend/mpi_evaluation.h"
#endif


/*! @file 

  Contains the general negative elliptic operator
  */
namespace dg
{

/**
 * @brief Operator that acts as a 2d negative elliptic differential operator
 *
 * @ingroup operators
 *
 * The term discretized is \f[ -\nabla \cdot ( \chi \nabla_\perp ) \f]
 * where \f$ \nabla_\perp \f$ is the perpendicular gradient. In cartesian 
 * coordinates that means \f[ -\partial_x(\chi\partial_x) - \partial_y(\chi\partial_y)\f]
 * is discretized while in cylindrical coordinates
 * \f[ - \frac{1}{R}\partial_R( R\chi\partial_R) - \partial_Z(\chi\partial_Z)\f]
 * is discretized.
 * @tparam Matrix The Matrix class to use
 * @tparam Vector The Vector class to use
 * @tparam Preconditioner The Preconditioner class to use
 * This class has the SelfMadeMatrixTag so it can be used in blas2::symv functions 
 * and thus in a conjugate gradient solver. 
 * @note The constructors initialize \f$ \chi=1\f$ so that a negative laplacian operator
 * results
 * @attention Pay attention to the negative sign 
 */
template <class Matrix, class Vector, class Preconditioner>
class Elliptic
{
    public:
    /**
     * @brief Construct from Grid
     *
     * @tparam Grid The Grid class. A call to dg::evaluate( one, g) must return an instance of the Vector class, 
     * a call to dg::create::weights(g) and dg::create::inv_weights(g)
     * must return instances of the Preconditioner class and 
     * calls to dg::create::dx( g, no, backward) and jump2d( g, bcx, bcy, no) are made.
     * @param g The Grid, boundary conditions are taken from here
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative
     */
    template< class Grid>
    Elliptic( const Grid& g, norm no = not_normed, direction dir = forward): 
        leftx ( dg::create::dx( g, inverse( g.bcx()), no, inverse(dir))),
        lefty ( dg::create::dy( g, inverse( g.bcy()), no, inverse(dir))),
        rightx( dg::create::dx( g, g.bcx(), normed, dir)),
        righty( dg::create::dy( g, g.bcy(), normed, dir)),
        jump  ( dg::create::jump2d( g, g.bcx(), g.bcy(), no )),
        weights_(dg::create::weights(g)), precond_(dg::create::inv_weights(g)), 
        xchi( dg::evaluate( one, g) ), xx(xchi), temp( xx), R(xchi),
        no_(no)
    { 
        if( g.system() == cylindrical)
        {
            R = dg::evaluate( dg::coo1, g);
            dg::blas1::pointwiseDot( R, xchi, xchi); 
        }
    }
    /**
     * @brief Construct from grid and boundary conditions
     *
     * @tparam Grid The Grid class. A call to dg::evaluate( one, g) must return an instance of the Vector class, 
     * a call to dg::create::weights(g) and dg::create::inv_weights(g)
     * must return instances of the Preconditioner class and 
     * calls to dg::create::dx( g, no, backward) and jump2d( g, bcx, bcy, no) are made.
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative (i.e. forward, backward or centered)
     */
    template< class Grid>
    Elliptic( const Grid& g, bc bcx, bc bcy, norm no = not_normed, direction dir = forward): 
        leftx (dg::create::dx( g, inverse(bcx), no, inverse(dir))),
        lefty (dg::create::dy( g, inverse(bcy), no, inverse(dir))),
        rightx(dg::create::dx( g,bcx, normed, dir)),
        righty(dg::create::dy( g,bcy, normed, dir)),
        jump  (dg::create::jump2d( g, bcx, bcy, no)),
        weights_(dg::create::weights(g)), precond_(dg::create::inv_weights(g)),
        xchi( dg::evaluate(one, g)), xx(xchi), temp( xx), R(xchi),
        no_(no)
    { 
        if( g.system() == cylindrical)
        {
            R = dg::evaluate( dg::coo1, g);
            dg::blas1::pointwiseDot( R, xchi, xchi); 
        }
    }

    /**
     * @brief Change Chi 
     *
     * @param chi The new chi
     */
    void set_chi( const Vector& chi)
    {
        xchi = chi;
        dg::blas1::pointwiseDot( R, xchi, xchi); 
    }
    /**
     * @brief Returns the weights used to make the matrix symmetric 
     *
     * @return weights
     */
    const Preconditioner& weights()const {return weights_;}
    /**
     * @brief Returns the preconditioner to use in conjugate gradient
     *
     * In this case inverse weights are the best choice
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

        dg::blas2::gemv( righty, x, temp);
        dg::blas1::pointwiseDot( xchi, temp, temp);
        dg::blas2::gemv( lefty, temp, y);
        
        dg::blas2::symv( jump, x, temp);
        dg::blas1::axpby( -1., xx, -1., y, xx); //-D_xx - D_yy + J
        if(no_==normed) //if cartesian then R = 1
            dg::blas1::pointwiseDivide( xx, R, xx);
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
    direction inverse( direction dir)
    {
        if( dir == forward) return backward;
        if( dir == backward) return forward;
        return centered;
    }
    Matrix leftx, lefty, rightx, righty, jump;
    Preconditioner weights_, precond_; //contain coeffs for chi multiplication
    Vector xchi, xx, temp, R;
    norm no_;
};

///@cond
template< class M, class V, class P>
struct MatrixTraits< Elliptic<M, V, P> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

} //namespace dg

