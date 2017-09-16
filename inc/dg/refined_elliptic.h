#pragma once

#include "backend/interpolation.cuh"
#include "backend/projection.cuh"
#include "elliptic.h"
#include "geometry/refined_grid.h"

/*! @file 

  @brief contains an elliptic method on a refined grid
  */
namespace dg
{

template < class Geometry,class IMatrix, class Matrix, class container>
class RefinedElliptic
{
    public:
    /**
     * @brief Construct from Grid
     *
     * @param g_coarse The coarse Grid
     * @param g_fine The fine Grid, boundary conditions are taken from here
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative
     */
    RefinedElliptic( const Geometry& g_coarse, const Geometry& g_fine, norm no = not_normed, direction dir = forward): 
        no_(no), elliptic_( g_fine, no, dir)
    { 
        construct( g_coarse, g_fine, g_fine.bcx(), g_fine.bcy(), dir);
    }

    /**
     * @brief Construct from grid and boundary conditions
     *
     * @param g_coarse The coarse Grid
     * @param g_fine The fine Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative (i.e. forward, backward or centered)
     */
    RefinedElliptic( const Geometry& g_coarse, const Geometry& g_fine, bc bcx, bc bcy, norm no = not_normed, direction dir = forward): 
        no_(no), elliptic_( g_fine, bcx, bcy, no, dir)
    { 
        construct( g_coarse, g_fine, bcx, bcy, dir);
    }

    /**
     * @brief Change Chi 
     *
     * @param chi The new chi
     */
    void set_chi( const container& chi)
    {
        //dg::blas2::gemv( Q_, chi, temp1_);
        //elliptic_.set_chi( temp1_);
        elliptic_.set_chi( chi);
    }

    /**
     * @brief Returns the inverse weights used to make the matrix normed
     * @return inverse weights
     */
    const container& inv_weights()const {return inv_weights_;}
    /**
     * @brief Returns the preconditioner to use in conjugate gradient
     *
     * In this case inverse weights are the best choice
     * @return inverse weights
     */
    const container& precond()const {return inv_weights_;}

    /**
     * @brief Computes the polarisation term
     *
     * @param x left-hand-side
     * @param y result
     */
    void symv( const container& x, container& y) 
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
    void compute_rhs( const container& rhs, container& rhs_mod )
    {
        //dg::blas2::gemv( Q_, rhs, temp1_);
        //dg::blas1::pointwiseDot( vol_, temp1_, temp1_);
        dg::tensor::pointwiseDot( vol_, rhs, temp1_);
        dg::blas2::gemv( QT_, temp1_, rhs_mod);
        dg::blas2::symv( inv_weights_, rhs_mod, rhs_mod);
    }
    private:
    void construct( const Geometry& g_coarse, const Geometry& g_fine, bc bcx, bc bcy, direction dir)
    {
        dg::blas2::transfer( dg::create::interpolation( g_fine, g_coarse), Q_);
        dg::blas2::transfer( dg::create::interpolationT( g_coarse, g_fine), QT_);
        dg::blas2::transfer( dg::create::projection( g_coarse, g_fine), P_);

        dg::blas1::transfer( dg::evaluate( dg::one, g_fine), temp1_);
        dg::blas1::transfer( dg::evaluate( dg::one, g_fine), temp2_);
        dg::blas1::transfer( dg::create::weights( g_coarse), weights_);
        dg::blas1::transfer( dg::create::inv_weights( g_coarse), inv_weights_);
        inv_vol_ = vol_ = dg::tensor::volume( g_fine.metric());
        dg::tensor::invert( inv_vol_);

    }
    norm no_;
    IMatrix P_, Q_, QT_;
    Elliptic<Geometry, Matrix, container> elliptic_;
    container temp1_, temp2_;
    container weights_, inv_weights_;
    dg::SparseElement<container> vol_, inv_vol_;
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

