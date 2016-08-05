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
        dg::blas2::gemv( Q_, chi, temp1_);
        elliptic_.set_chi( temp1_);
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
     * P\sqrt{g} Q \rho
     * @param rhs the original right hand side
     * @param rhs_mod the modified right hand side of the same size (may equal rhs)
     */
    void compute_rhs( const Vector& rhs, Vector& rhs_mod )
    {
        dg::blas2::gemv( Q_, rhs, temp1_);
        //reminder: vol_ contains the weight function of the refined grid 
        dg::blas1::pointwiseDot( vol_, temp1_, temp1_);
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

/**
 * @brief Operator that acts as a 2d negative elliptic differential operator
 *
 * @ingroup matrixoperators
 *
 * The term discretized is \f[ -\nabla \cdot ( \chi \nabla_\perp ) \f]
 * where \f$ \nabla_\perp \f$ is the perpendicular gradient. In general 
 * coordinates that means 
 * \f[ -\frac{1}{\sqrt{g}}\left( 
 * \partial_x\left(\sqrt{g}\chi \left(g^{xx}\partial_x + g^{xy}\partial_y \right)\right) 
 + \partial_y\left(\sqrt{g}\chi \left(g^{yx}\partial_x + g^{yy}\partial_y \right)\right) \right)\f]
 is discretized
 * @tparam Geometry The geometry sets the metric of the grid
 * @tparam Matrix The Matrix class to use
 * @tparam Vector The Vector class to use
 * @tparam Vector The Vector class to use
 * This class has the SelfMadeMatrixTag so it can be used in blas2::symv functions 
 * and thus in a conjugate gradient solver. 
 * @note The constructors initialize \f$ \chi=1\f$ so that a negative laplacian operator
 * results
 * @attention Pay attention to the negative sign 
 */
template < class Geometry,class IMatrix, class Matrix, class Vector>
class AltRefinedElliptic
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
    AltRefinedElliptic( Geometry g, norm no = not_normed, direction dir = forward): 
        no_(no), g_(g)
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
    AltRefinedElliptic( Geometry g, bc bcx, bc bcy, norm no = not_normed, direction dir = forward): 
        no_(no), g_(g)
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
        xchiC = chi;
    }
    /**
     * @brief Compute the Right hand side
     *
     * P\sqrt{g} Q \rho
     * @param rhs the original right hand side
     * @param rhs_mod the modified right hand side of the same size (may equal rhs)
     */
    void compute_rhs( const Vector& rhs, Vector& rhs_mod )
    {
        dg::blas2::gemv( Q_, rhs, tempxF);
        dg::geo::multiplyVolume( tempxF, g_);
        dg::blas1::pointwiseDot( tempxF, weightsF_, tempxF);
        dg::blas2::gemv( QT_, tempxF, rhs_mod);
        dg::blas2::symv( inv_weightsC_, rhs_mod, rhs_mod);
    }

    /**
     * @brief Returns the weights used to make the matrix symmetric 
     *
     * i.e. the volume form 
     * @return weights
     */
    const Vector& weights()const {return weightsC_;}
    /**
     * @brief Returns the preconditioner to use in conjugate gradient
     *
     * In this case inverse weights are the best choice
     * @return inverse weights
     */
    const Vector& precond()const {return inv_weightsC_;}

    /**
     * @brief Computes the polarisation term
     *
     * @param x left-hand-side
     * @param y result
     */
    void symv( const Vector& x, Vector& y) 
    {
        //compute gradient
        dg::blas2::gemv( rightxC, x, tempxC); //R_x*f 
        dg::blas2::gemv( rightyC, x, tempyC); //R_y*f
        //multiply with chi 
        dg::blas1::pointwiseDot( xchiC, tempxC, tempxC); //Chi*R_x*x 
        dg::blas1::pointwiseDot( xchiC, tempyC, tempyC); //Chi*R_x*x 
        //now interpolate to fine grid
        dg::blas2::gemv( Q_, tempxC, tempxF);
        dg::blas2::gemv( Q_, tempyC, tempyF);

        //raise index and multiply volume form
        dg::geo::raisePerpIndex( tempxF, tempyF, gradXF, gradYF, g_);
        dg::geo::multiplyVolume( gradXF, g_); //gradXF is the d^x 
        dg::geo::multiplyVolume( gradYF, g_); //gradYF is the d^y

        //now project back to coarse grid
        dg::blas1::pointwiseDot( weightsF_, gradXF, gradXF);
        dg::blas1::pointwiseDot( weightsF_, gradYF, gradYF);
        dg::blas2::gemv( QT_, gradXF, tempxC);
        dg::blas2::gemv( QT_, gradYF, tempyC);
        dg::blas1::pointwiseDot( inv_weightsC_, tempxC, tempxC);
        dg::blas1::pointwiseDot( inv_weightsC_, tempyC, tempyC);
        //now take divergence
        dg::blas2::gemv( leftxC, tempxC, gradxC);  
        dg::blas2::gemv( leftyC, tempyC,  y);  
        dg::blas1::axpby( -1., gradxC, -1., y, y); //-D_xx - D_yy 
        if( no_ == normed)
            dg::geo::divideVolume( y, g_);

        //add jump terms
        dg::blas2::symv( jumpXC, x, tempxC);
        dg::blas1::axpby( +1., tempxC, 1., y, y); 
        dg::blas2::symv( jumpYC, x, tempyC);
        dg::blas1::axpby( +1., tempyC, 1., y, y); 
        if( no_ == not_normed)//multiply weights without volume
            dg::blas2::symv( weights_wo_volC, y, y);

    }
    private:
    void construct( Geometry g, bc bcx, bc bcy, direction dir)
    {
        dg::blas2::transfer( dg::create::dx( g.associated(), inverse( bcx), inverse(dir)), leftxC);
        dg::blas2::transfer( dg::create::dy( g.associated(), inverse( bcy), inverse(dir)), leftyC);
        dg::blas2::transfer( dg::create::dx( g.associated(), bcx, dir), rightxC);
        dg::blas2::transfer( dg::create::dy( g.associated(), bcy, dir), rightyC);
        dg::blas2::transfer( dg::create::jumpX( g.associated(), bcx),   jumpXC);
        dg::blas2::transfer( dg::create::jumpY( g.associated(), bcy),   jumpYC);
        dg::blas1::transfer( dg::create::weights(g.associated()),        weightsC_);
        dg::blas1::transfer( dg::create::inv_weights(g.associated()),    inv_weightsC_);
        dg::blas1::transfer( dg::create::weights(g),        weightsF_);
        dg::blas1::transfer( dg::create::inv_weights(g),    inv_weightsF_);
        dg::blas1::transfer( dg::evaluate( dg::one, g.associated()),    xchiC);
        tempxC = tempyC = gradxC = xchiC;
        dg::blas1::transfer( dg::evaluate( dg::one, g),    gradYF);
        tempxF = tempyF = gradXF = gradYF;

        dg::blas1::transfer( dg::create::volume(g.associated()),        weights_wo_volC);
        dg::geo::divideVolume( weights_wo_volC, g_);

        dg::blas2::transfer( dg::create::interpolation( g), Q_);
        dg::blas2::transfer( dg::create::interpolationT( g), QT_);
        dg::blas2::transfer( dg::create::projection( g), P_);
    }
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
    IMatrix P_, Q_, QT_;
    Matrix leftxC, leftyC, rightxC, rightyC, jumpXC, jumpYC;
    Vector weightsC_, inv_weightsC_, weights_wo_volC;  //contain
    Vector weightsF_, inv_weightsF_;
    Vector xchiC, tempxC, tempyC, gradxC;
    Vector gradYF, tempxF, tempyF, gradXF;
    norm no_;
    Geometry g_;
};

///@cond
template< class G, class IM, class M, class V>
struct MatrixTraits< AltRefinedElliptic<G, IM, M, V> >
{
    typedef typename VectorTraits<V>::value_type  value_type;
    typedef SelfMadeMatrixTag matrix_category;
};

///@endcond

} //namespace dg

