#pragma once

#include "blas.h"
#include "creation.h"


//#include "cusp_eigen.h"
//CAN'T BE TRANSVERSE SYMMETRIC?

/*! @file 

  Contains object for the polarisation matrix creation
  */
namespace dg
{

template <class Matrix, class Vector, class Preconditioner>
class Polarisation
{
    public:
    template< class Grid>
    Polarisation( const Grid& g, const Vector& copyable): 
        xchi( copyable), xx(copyable), temp( copyable),
        weights_(dg::create::weights(g)), precond_(dg::create::precond(g))
    {
        rightx=dg::create::dx( g, g.bcx(), normed, forward);
        righty=dg::create::dy( g, g.bcy(), normed, forward);
        leftx =dg::create::dx( g, inverse( g.bcx()), not_normed, backward);
        lefty =dg::create::dy( g, inverse( g.bcy()), not_normed, backward);
    //cusp::transpose( rightx, leftx); 
    //cusp::transpose( righty, lefty); 
        jump  =dg::create::jump2d( g, g.bcx(), g.bcy());
    }
    template< class Grid>
    Polarisation( const Grid& g, bc bcx, bc bcy, const Vector& copyable): 
        xchi( copyable), xx(copyable), temp( copyable),
        weights_(dg::create::weights(g)), precond_(dg::create::precond(g))
    {
        rightx=dg::create::dx( g,bcx, normed, forward);
        righty=dg::create::dy( g,bcy, normed, forward);
        
        leftx =dg::create::dx( g, inverse(bcx), not_normed, backward);
        lefty =dg::create::dy( g, inverse(bcy), not_normed, backward);
        jump  =dg::create::jump( g, bcx, bcy);
    }

    void set_chi( const Vector& chi)
    {
        xchi = chi;
        //dg::blas1::pointwiseDot( weights_, chi, xchi);
    }
    const Preconditioner& weights()const {return weights_;}
    const Preconditioner& precond()const {return precond_;}

    void symv( const Vector& x, Vector& y) 
    {
        dg::blas2::gemv( rightx, x, temp); //R_x*x 
        dg::blas1::pointwiseDot( xchi, temp, temp); //Chi*R_x*x 
        dg::blas2::gemv( leftx, temp, xx); //L_x*Chi*R_x*x

        dg::blas2::gemv( righty, x, temp);
        dg::blas1::pointwiseDot( xchi, temp, temp);
        dg::blas2::gemv( lefty, temp, y);
        
        dg::blas2::symv( jump, x, temp);
        dg::blas1::axpby( -1., xx, -1., y, xx); //D_xx + D_yy
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

