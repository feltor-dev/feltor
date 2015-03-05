#pragma once


#include "blas.h"
#include "enums.h"
#include "backend/evaluation.cuh"
#include "backend/derivatives.cuh"



namespace dg{


template<class Matrix, class Vector, class Preconditioner> 
struct GeneralElliptic
{
    template< class Grid>
    GeneralElliptic( const Grid& g, norm no = not_normed, direction dir = forward): 
        leftx ( dg::create::dx( g, inverse( g.bcx()), no, inverse(dir))),
        lefty ( dg::create::dy( g, inverse( g.bcy()), no, inverse(dir))),
        leftz ( dg::create::dz( g, inverse( g.bcz()), no, inverse(dir))),
        rightx( dg::create::dx( g, g.bcx(), normed, dir)),
        righty( dg::create::dy( g, g.bcy(), normed, dir)),
        rightz( dg::create::dz( g, g.bcz(), normed, dir)),
        jump  ( dg::create::jump2d( g, g.bcx(), g.bcy(), no )),
        weights_(dg::create::weights(g)), precond_(dg::create::inv_weights(g)), 
        xchi( dg::evaluate( one, g) ), ychi( xchi), zchi( xchi), 
        xx(xchi), yy(xx), zz(xx), temp0( xx), temp1(temp0), R(xchi),
        no_(no)
    { 
        if( g.system() == cylindrical)
        {
            R = dg::evaluate( dg::coo1, g);
        }
    }

    void set_x( const Vector& chi)
    {
        xchi = chi;
    }
    void set_y( const Vector& chi)
    {
        ychi = chi;
    }
    void set_z( const Vector& chi)
    {
        zchi = chi;
    }
    void set( const std::vector<Vector>& chi)
    {
        xchi = chi[0];
        ychi = chi[1];
        zchi = chi[2];
        dg::blas1::pointwiseDot( R, xchi, xchi); 
        dg::blas1::pointwiseDot( R, ychi, ychi); 
        dg::blas1::pointwiseDot( R, zchi, zchi); 
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
        dg::blas2::gemv( rightx, x, temp0); //R_x*x 
        dg::blas1::pointwiseDot( xchi, temp0, xx); //Chi_x*R_x*x 

        dg::blas2::gemv( righty, x, temp0);//R_y*x
        dg::blas1::pointwiseDot( ychi, temp0, yy);//Chi_y*R_y*x

        dg::blas2::gemv( rightz, x, temp0); // R_z*x
        dg::blas1::pointwiseDot( zchi, temp0, zz); //Chi_z*R_z*x

        dg::blas1::axpby( 1., xx, 1., yy, temp0);
        dg::blas1::axpby( 1., zz, 1., temp0, temp0); //gradpar x 
        dg::blas1::pointwiseDot( R, temp0, temp0);

        dg::blas1::pointwiseDot( xchi, temp0, temp1); 
        dg::blas2::gemv( leftx, temp1, xx); 

        dg::blas1::pointwiseDot( ychi, temp0, temp1);
        dg::blas2::gemv( lefty, temp1, yy);

        dg::blas1::pointwiseDot( zchi, temp0, temp1); 
        dg::blas2::gemv( leftz, temp1, zz); 

        dg::blas1::axpby( -1., xx, -1., yy, y);
        dg::blas1::axpby( -1., zz, +1., y, y); 
        
        dg::blas2::symv( jump, x, temp0);
        if(no_==normed) //if cartesian then R = 1
            dg::blas1::pointwiseDivide( y, R, y);
        dg::blas1::axpby( +1., temp0, 1., y, y); 
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
    Matrix leftx, lefty, leftz, rightx, righty, rightz, jump;
    Preconditioner weights_, precond_; //contain coeffs for chi multiplication
    Vector xchi, ychi, zchi, xx, yy, zz, temp0, temp1,  R;
    norm no_;


};
///@cond
template< class M, class V, class P>
struct MatrixTraits< GeneralElliptic<M, V, P> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

}//namespace dg
