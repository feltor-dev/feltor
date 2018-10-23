#pragma once

#include "topology/interpolation.h"
#include "topology/projection.h"
#include "elliptic.h"
#include "topology/refined_grid.h"
#ifdef MPI_VERSION
#include "topology/mpi_projection.h"
#endif

/*! @file

  @brief contains an elliptic method on a refined grid
  */
namespace dg
{

 /*!@brief The refined version of \c Elliptic

 * Holds an \c Elliptic object on the fine grid and brackets every call to symv with %interpolation and %projection matrices
 * @copydoc hide_geometry_matrix_container
 * @ingroup matrixoperators
 * @attention This class is still under construction!
 */
template < class Geometry,class IMatrix, class Matrix, class container>
class RefinedElliptic
{
    public:
    /**
     * @brief Construct from a coarse and a fine grid
     *
     * @param g_coarse The coarse Grid
     * @param g_fine The fine Grid, boundary conditions are taken from here
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative
     */
    RefinedElliptic( const Geometry& g_coarse, const Geometry& g_fine, norm no = not_normed, direction dir = forward): RefinedElliptic( g_coarse, g_fine, g_fine.bcx(), g_fine.bcy(), no, dir){}

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
     * @tparam ContainerTypes must be usable with \c container in \ref dispatch
     */
    template<class ContainerType0>
    void set_chi( const ContainerType0& chi)
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
    const container& weights()const {return weights_;}
    /**
     * @brief Returns the preconditioner to use in conjugate gradient
     * @return inverse weights
     */
    const container& precond()const {return inv_weights_;}

    /**
     * @brief Computes the polarisation term
     *
     * @param x left-hand-side
     * @param y result
     * @tparam ContainerTypes must be usable with \c container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y)
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
     * @tparam ContainerTypes must be usable with \c container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void compute_rhs( const ContainerType0& rhs, ContainerType1& rhs_mod )
    {
        //dg::blas2::gemv( Q_, rhs, temp1_);
        //dg::blas1::pointwiseDot( vol_, temp1_, temp1_);
        dg::blas1::pointwiseDot( vol_, rhs, temp1_);
        dg::blas2::gemv( QT_, temp1_, rhs_mod);
        dg::blas2::symv( inv_weights_, rhs_mod, rhs_mod);
    }
    private:
    void construct( const Geometry& g_coarse, const Geometry& g_fine, bc bcx, bc bcy, direction dir)
    {
        dg::blas2::transfer( dg::create::interpolation( g_fine, g_coarse), Q_);
        dg::blas2::transfer( dg::create::interpolationT( g_coarse, g_fine), QT_);
        dg::blas2::transfer( dg::create::projection( g_coarse, g_fine), P_);

        dg::assign( dg::evaluate( dg::one, g_fine), temp1_);
        dg::assign( dg::evaluate( dg::one, g_fine), temp2_);
        dg::assign( dg::create::weights( g_coarse), weights_);
        dg::assign( dg::create::inv_weights( g_coarse), inv_weights_);
        vol_ = dg::tensor::volume( g_fine.metric());

    }
    norm no_;
    IMatrix P_, Q_, QT_;
    Elliptic<Geometry, Matrix, container> elliptic_;
    container temp1_, temp2_;
    container weights_, inv_weights_;
    container vol_;
};


///@cond
template< class G, class IM, class M, class V>
struct TensorTraits< RefinedElliptic<G, IM, M, V> >
{
    using value_type  = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};

///@endcond

} //namespace dg

