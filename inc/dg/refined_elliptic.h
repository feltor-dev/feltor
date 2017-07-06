#pragma once

#include "elliptic.h"
#include "geometry/refined_grid.h"

/*! @file 

  Contains an elliptic method on a refined grid
  */
namespace dg
{

template < class Geometry,class IMatrix, class Matrix, class Vector>
class RefinedElliptic
{
    public:
    /**
     * @brief Construct from Grid
     *
     * @tparam Geometry The Grid class. A call to dg::evaluate( one, g) must return an instance of the Vector class, 
     * a call to dg::create::weights(g) and dg::create::inv_weights(g)
     * must return instances of the Vector class and 
     * calls to dg::create::dx( g, no, backward) and jump2d( g, bcx, bcy, no) are made.
     * @param g The Grid, boundary conditions are taken from here
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative
     */
    RefinedElliptic( Geometry g, norm no = not_normed, direction dir = forward): 
        no_(no), elliptic_( g, no, dir)
    { 
        construct( g, g.bcx(), g.bcy(), dir);
    }

    /**
     * @brief Construct from grid and boundary conditions
     *
     * @tparam Geometry The Grid class. A call to dg::evaluate( one, g) must return an instance of the Vector class, 
     * a call to dg::create::weights(g) and dg::create::inv_weights(g)
     * must return instances of the Vector class and 
     * calls to dg::create::dx( g, no, backward) and jump2d( g, bcx, bcy, no) are made.
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative (i.e. forward, backward or centered)
     */
    RefinedElliptic( Geometry g, bc bcx, bc bcy, norm no = not_normed, direction dir = forward): 
        no_(no), elliptic_( g, bcx, bcy, no, dir)
    { 
        construct( g, bcx, bcy, dir);
    }

    /**
     * @brief Change Chi 
     *
     * @param chi The new chi
     */
    void set_chi( const Vector& chi)
    {
        //dg::blas2::gemv( Q_, chi, temp1_);
        //elliptic_.set_chi( temp1_);
        elliptic_.set_chi( chi);
    }

    /**
     * @brief Returns the weights used to make the matrix symmetric 
     *
     * @return weights
     */
    const Vector& weights()const {return weights_;}
    /**
     * @brief Returns the preconditioner to use in conjugate gradient
     *
     * In this case inverse weights are the best choice
     * @return inverse weights
     */
    const Vector& precond()const {return inv_weights_;}

    /**
     * @brief Computes the polarisation term
     *
     * @param x left-hand-side
     * @param y result
     */
    void symv( const Vector& x, Vector& y) 
    {
        dg::blas2::gemv( Q_, x, temp1_); 
        elliptic_.symv( temp1_, temp2_);
        if( no_ == normed) 
        {
            dg::blas2::gemv( P_, temp2_, y); 
            return;
        }
        else 
        {
            dg::blas2::gemv( QT_, temp2_, y);
            return;
        }
    }

    /**
     * @brief Compute the Right hand side
     *
     * \f[P\sqrt{g} Q \rho\f]
     * @param rhs the original right hand side
     * @param rhs_mod the modified right hand side of the same size (may equal rhs)
     */
    void compute_rhs( const Vector& rhs, Vector& rhs_mod )
    {
        //dg::blas2::gemv( Q_, rhs, temp1_);
        //dg::blas1::pointwiseDot( vol_, temp1_, temp1_);
        dg::blas1::pointwiseDot( vol_, rhs, temp1_);
        dg::blas2::gemv( QT_, temp1_, rhs_mod);
        dg::blas2::symv( inv_weights_, rhs_mod, rhs_mod);
    }
    private:
    void construct( Geometry g, bc bcx, bc bcy, direction dir)
    {
        dg::blas2::transfer( dg::create::interpolation( g), Q_);
        dg::blas2::transfer( dg::create::interpolationT( g), QT_);
        dg::blas2::transfer( dg::create::projection( g), P_);

        dg::blas1::transfer( dg::evaluate( dg::one, g), temp1_);
        dg::blas1::transfer( dg::evaluate( dg::one, g), temp2_);
        dg::blas1::transfer( dg::create::weights( g.associated()), weights_);
        dg::blas1::transfer( dg::create::inv_weights( g.associated()), inv_weights_);
        dg::blas1::transfer( dg::create::volume( g), vol_);
        dg::blas1::transfer( dg::create::inv_volume( g), inv_vol_);
    }
    norm no_;
    IMatrix P_, Q_, QT_;
    Elliptic<Geometry, Matrix, Vector> elliptic_;
    Vector temp1_, temp2_;
    Vector weights_, inv_weights_;
    Vector vol_, inv_vol_;
};


///@cond
template< class G, class IM, class M, class V>
struct MatrixTraits< RefinedElliptic<G, IM, M, V> >
{
    typedef typename VectorTraits<V>::value_type  value_type;
    typedef SelfMadeMatrixTag matrix_category;
};

///@endcond

} //namespace dg

