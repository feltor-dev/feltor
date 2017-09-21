#pragma once

#include "blas.h"
#include "enums.h"
#include "backend/memory.h"
#include "backend/evaluation.cuh"
#include "backend/derivatives.h"
#ifdef MPI_VERSION
#include "backend/mpi_derivatives.h"
#include "backend/mpi_evaluation.h"
#endif
#include "geometry/geometry.h"

/*! @file 

  @brief General negative elliptic operators
  */
namespace dg
{

/**
 * @brief %Operator that acts as a 2d negative elliptic differential operator
 *
 * @ingroup matrixoperators
 *
 * The term discretized is \f[ -\nabla \cdot ( \chi \nabla_\perp ) \f]
 * where \f$ \nabla_\perp \f$ is the perpendicular gradient. In general 
 * coordinates that means 
 * \f[ -\frac{1}{\sqrt{g}}\left( 
 * \partial_x\left(\sqrt{g}\chi \left(g^{xx}\partial_x + g^{xy}\partial_y \right)\right) 
 + \partial_y\left(\sqrt{g}\chi \left(g^{yx}\partial_x + g^{yy}\partial_y \right)\right) \right)\f]
 is discretized. Note that the discontinuous Galerkin discretization adds so-called
 jump terms via a jump matrix 
 \f[ D^\dagger_x \chi D_x + \alpha J \f]
 where \f$\alpha\f$  is a scale factor ( = jfactor). Usually the default \f$ \alpha=1 \f$ is a good choice.
 However, in some cases, e.g. when \f$ \chi \f$ exhibits very large variations
 \f$ \alpha=0.1\f$ or \f$ \alpha=0.01\f$ might be better values. 
 In a time dependent problem the value of \f$\alpha\f$ determines the 
 numerical diffusion, i.e. for low values numerical oscillations may appear. 
 Also note that a forward discretization has more diffusion than a centered discretization.

 * @copydoc hide_geometry_matrix_container
 * This class has the SelfMadeMatrixTag so it can be used in blas2::symv functions 
 * and thus in a conjugate gradient solver. 
 * @note The constructors initialize \f$ \chi=1\f$ so that a negative laplacian operator
 * results
 * @note The inverse dG weights make a good general purpose preconditioner, but 
 * the inverse of \f$ \chi\f$ should also seriously be considered
 * @note the jump term \f$ \alpha J\f$  adds artificial numerical diffusion
 * @attention Pay attention to the negative sign 
 */
template <class Geometry, class Matrix, class container>
class Elliptic
{
    public:
    ///@brief empty object ( no memory allocation)
    Elliptic(){}
    /**
     * @brief Construct from Grid
     *
     * @param g The Grid, boundary conditions are taken from here
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative

     * @param jfactor (\f$ = \alpha \f$ ) scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     * @note chi is assumed 1 per default
     */
    Elliptic( const Geometry& g, norm no = not_normed, direction dir = forward, double jfactor=1.)
    { 
        construct( g, g.bcx(), g.bcy(), no, dir, jfactor);
    }

    ///@copydoc Elliptic::construct()
    Elliptic( const Geometry& g, bc bcx, bc bcy, norm no = not_normed, direction dir = forward, double jfactor=1.)
    { 
        construct( g, bcx, bcy, no, dir, jfactor);
    }

    /**
     * @brief Construct from grid and boundary conditions
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative (i.e. forward, backward or centered)
     * @param jfactor scale jump terms (1 is a good value but in some cases 0.1 or 0.01 might be better)
     */
    void construct( const Geometry& g, bc bcx, bc bcy, norm no = not_normed, direction dir = forward, double jfactor = 1.)
    {
        no_=no, jfactor_=jfactor;
        dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), leftx);
        dg::blas2::transfer( dg::create::dy( g, inverse( bcy), inverse(dir)), lefty);
        dg::blas2::transfer( dg::create::dx( g, bcx, dir), rightx);
        dg::blas2::transfer( dg::create::dy( g, bcy, dir), righty);
        dg::blas2::transfer( dg::create::jumpX( g, bcx),   jumpX);
        dg::blas2::transfer( dg::create::jumpY( g, bcy),   jumpY);

        dg::blas1::transfer( dg::create::inv_volume(g),    inv_weights_);
        dg::blas1::transfer( dg::create::volume(g),        weights_);
        dg::blas1::transfer( dg::create::inv_weights(g),   precond_); 
        tempx = tempy = gradx = inv_weights_;
        chi_=g.metric();
        vol_=dg::tensor::volume(chi_);
        dg::tensor::scal( chi_, vol_);
        dg::blas1::transfer( dg::create::weights(g), weights_wo_vol);
    }

    ///@copydoc  Elliptic::Elliptic(const Geometry&,norm,direction,double)
    void construct( const Geometry& g, norm no = not_normed, direction dir = forward, double jfactor = 1.){
        construct( g, g.bcx(), g.bcy(), no, dir, jfactor);
    }

    /**
     * @brief Change Chi 
     *
     * @param chi The new chi (all elements must be >0)
     * @note There is no get_chi because chi is multiplied with volume elements
     */
    void set_chi( const container& chi)
    {
        if( !chi_old_.isSet()) 
        {
            dg::tensor::scal( chi_, chi);
            dg::blas1::pointwiseDivide( precond_, chi, precond_);
            chi_old_.value() = chi;
            return;
        }
        dg::blas1::pointwiseDivide( chi, chi_old_.value(), tempx);
        dg::blas1::pointwiseDivide( precond_, tempx, precond_);
        dg::tensor::scal( chi_, tempx);
        chi_old_.value()=chi;
    }

    /**
     * @brief Return the vector missing in the un-normed symmetric matrix 
     *
     * i.e. the inverse of the weights() function
     * @return inverse volume form including inverse weights 
     */
    const container& inv_weights()const {return inv_weights_;}
    /**
     * @brief Return the vector making the matrix symmetric
     *
     * i.e. the volume form 
     * @return volume form including weights 
     */
    const container& weights()const {return weights_;}
    /**
     * @brief Return the default preconditioner to use in conjugate gradient
     *
     * Currently returns the inverse of the weights without volume elment multiplied by the inverse of \f$ \chi\f$. 
     * This is especially good when \f$ \chi\f$ exhibits large amplitudes or variations
     * @return the inverse of \f$\chi\f$.       
     */
    const container& precond()const {return precond_;}
    /**
     * @brief Set the currently used jfactor
     *
     * @param new_jfactor The new scale factor for jump terms
     */
    void set_jfactor( double new_jfactor) {jfactor_ = new_jfactor;}
    /**
     * @brief Get the currently used jfactor
     *
     * @return  The current scale factor for jump terms
     */
    double get_jfactor() const {return jfactor_;}

    /**
     * @brief Computes the polarisation term
     *
     * @param x left-hand-side
     * @param y result
     */
    void symv( const container& x, container& y) 
    {
        //compute gradient
        dg::blas2::gemv( rightx, x, tempx); //R_x*f 
        dg::blas2::gemv( righty, x, tempy); //R_y*f

        //multiply with tensor (note the alias)
        dg::tensor::multiply2d(chi_, tempx, tempy, gradx, tempy);

        //now take divergence
        dg::blas2::symv( lefty, tempy, y);  
        dg::blas2::symv( -1., leftx, gradx, -1., y);  
        if( no_ == normed)
            dg::tensor::pointwiseDivide( y, vol_, y);

        //add jump terms
        dg::blas2::symv( jfactor_, jumpX, x, 1., y);
        dg::blas2::symv( jfactor_, jumpY, x, 1., y);
        if( no_ == not_normed)//multiply weights without volume
            dg::blas2::symv( weights_wo_vol, y, y);

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
    Matrix leftx, lefty, rightx, righty, jumpX, jumpY;
    container weights_, inv_weights_, precond_, weights_wo_vol; 
    container tempx, tempy, gradx;
    norm no_;
    SparseTensor<container> chi_;
    SparseElement<container> chi_old_, vol_;
    double jfactor_;
};


/**
 * @brief %Operator that acts as a 3d negative elliptic differential operator
 *
 * @ingroup matrixoperators
 *
 * The term discretized is 
 * \f[ -\nabla \cdot ( \mathbf b  \mathbf b \cdot \nabla ) \f]
  In general that means 
 * \f[ 
 * \begin{align}
 * v = b^x \partial_x f + b^y\partial_y f + b^z \partial_z f \\
 * -\frac{1}{\sqrt{g}} \left(\partial_x(\sqrt{g} b^x v ) + \partial_y(\sqrt{g}b^y v) + \partial_z(\sqrt{g} b^z v)\right)
 *  \end{align}
 *  \f] 
 * is discretized, with \f$ b^i\f$ being the contravariant components of \f$\mathbf b\f$ . 
 * @copydoc hide_geometry_matrix_container
 * This class has the SelfMadeMatrixTag so it can be used in blas2::symv functions 
 * and thus in a conjugate gradient solver. 
 * @note The constructors initialize \f$ b^x = b^y = b^z=1\f$ 
 * @attention Pay attention to the negative sign 
 */
template< class Geometry, class Matrix, class container> 
struct GeneralElliptic
{
    /**
     * @brief Construct from Grid
     *
     * @param g The Grid, boundary conditions are taken from here
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative
     */
    GeneralElliptic( const Geometry& g, norm no = not_normed, direction dir = forward): 
        leftx ( dg::create::dx( g, inverse( g.bcx()), inverse(dir))),
        lefty ( dg::create::dy( g, inverse( g.bcy()), inverse(dir))),
        leftz ( dg::create::dz( g, inverse( g.bcz()), inverse(dir))),
        rightx( dg::create::dx( g, g.bcx(), dir)),
        righty( dg::create::dy( g, g.bcy(), dir)),
        rightz( dg::create::dz( g, g.bcz(), dir)),
        jumpX ( dg::create::jumpX( g, g.bcx())),
        jumpY ( dg::create::jumpY( g, g.bcy())),
        weights_(dg::create::volume(g)), inv_weights_(dg::create::inv_volume(g)), precond_(dg::create::inv_weights(g)), 
        xchi( dg::evaluate( one, g) ), ychi( xchi), zchi( xchi), 
        xx(xchi), yy(xx), zz(xx), temp0( xx), temp1(temp0),
        no_(no)
    { 
        vol_=dg::tensor::determinant(g.metric());
        dg::tensor::invert(vol_);
        dg::tensor::sqrt(vol_); //now we have volume element
    }
    /**
     * @brief Construct from Grid and bc 
     *
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param bcz boundary contition in z
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative
     */
    GeneralElliptic( const Geometry& g, bc bcx, bc bcy, bc bcz, norm no = not_normed, direction dir = forward): 
        leftx ( dg::create::dx( g, inverse( bcx), inverse(dir))),
        lefty ( dg::create::dy( g, inverse( bcy), inverse(dir))),
        leftz ( dg::create::dz( g, inverse( bcz), inverse(dir))),
        rightx( dg::create::dx( g, bcx, dir)),
        righty( dg::create::dy( g, bcy, dir)),
        rightz( dg::create::dz( g, bcz, dir)),
        jumpX ( dg::create::jumpX( g, bcx)),
        jumpY ( dg::create::jumpY( g, bcy)),
        weights_(dg::create::volume(g)), inv_weights_(dg::create::inv_volume(g)), precond_(dg::create::inv_weights(g)), 
        xchi( dg::evaluate( one, g) ), ychi( xchi), zchi( xchi), 
        xx(xchi), yy(xx), zz(xx), temp0( xx), temp1(temp0),
        no_(no)
    { 
        vol_=dg::tensor::determinant(g.metric());
        dg::tensor::invert(vol_);
        dg::tensor::sqrt(vol_); //now we have volume element
    }
    /**
     * @brief Set x-component of \f$ \chi\f$
     *
     * @param chi new x-component
     */
    void set_x( const container& chi)
    {
        xchi = chi;
    }
    /**
     * @brief Set y-component of \f$ \chi\f$
     *
     * @param chi new y-component
     */
    void set_y( const container& chi)
    {
        ychi = chi;
    }
    /**
     * @brief Set z-component of \f$ \chi\f$
     *
     * @param chi new z-component
     */
    void set_z( const container& chi)
    {
        zchi = chi;
    }

    /**
     * @brief Set new components for \f$ \chi\f$
     *
     * @param chi chi[0] is new x-component, chi[1] the new y-component, chi[2] z-component
     */
    void set( const std::vector<container>& chi)
    {
        xchi = chi[0];
        ychi = chi[1];
        zchi = chi[2];
    }

    ///@copydoc Elliptic::inv_weights()
    const container& inv_weights()const {return inv_weights_;}
    /**
     * @brief Returns the preconditioner to use in conjugate gradient
     *
     * In this case inverse weights (without volume element) are returned
     * @return inverse weights
     */
    const container& precond()const {return precond_;}

    ///@copydoc Elliptic::symv()
    void symv( const container& x, container& y) 
    {
        dg::blas2::gemv( rightx, x, temp0); //R_x*x 
        dg::blas1::pointwiseDot( 1., xchi, temp0, 0., xx);//Chi_x*R_x*x

        dg::blas2::gemv( righty, x, temp0);//R_y*x
        dg::blas1::pointwiseDot( 1., ychi, temp0, 1., xx);//Chi_y*R_y*x

        dg::blas2::gemv( rightz, x, temp0); // R_z*x
        dg::blas1::pointwiseDot( 1., zchi, temp0, 1., xx);//Chi_z*R_z*x

        dg::tensor::pointwiseDot( vol_, xx, temp0);

        dg::blas1::pointwiseDot( xchi, temp0, temp1); 
        dg::blas2::gemv( -1., leftx, temp1, 0., y); 

        dg::blas1::pointwiseDot( ychi, temp0, temp1);
        dg::blas2::gemv( -1., lefty, temp1, 1., y);

        dg::blas1::pointwiseDot( zchi, temp0, temp1); 
        dg::blas2::gemv( -1., leftz, temp1, 1., y); 

        if( no_==normed) 
            dg::tensor::pointwiseDivide( temp0, vol_, temp0);
        
        dg::blas2::symv( +1., jumpX, x, 1., y);
        dg::blas2::symv( +1., jumpY, x, 1., y);
        if( no_==not_normed)//multiply weights w/o volume
        {
            dg::tensor::pointwiseDivide( y, vol_, y);
            dg::blas1::pointwiseDivide( y, inv_weights_, y);
        }
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
    Matrix leftx, lefty, leftz, rightx, righty, rightz, jumpX, jumpY;
    container weights_, inv_weights_, precond_; //contain coeffs for chi multiplication
    container xchi, ychi, zchi, xx, yy, zz, temp0, temp1;
    norm no_;
    SparseElement<container> vol_;
};

/**
 * @brief %Operator that acts as a 3d negative elliptic differential operator. Is the symmetric of the GeneralElliptic with 
 * 0.5(D_+ + D_-) or vice versa
 *
 * @ingroup matrixoperators
 *
 * The term discretized is 
 * \f[ -\nabla \cdot ( \mathbf b  \mathbf b \cdot \nabla ) \f]
  In general that means 
 * \f[ 
 * \begin{align}
 * v = b^x \partial_x f + b^y\partial_y f + b^z \partial_z f \\
 * -\frac{1}{\sqrt{g}} \left(\partial_x(\sqrt{g} b^x v ) + \partial_y(\sqrt{g}b^y v) + \partial_z(\sqrt{g} b^z v)\right)
 *  \end{align}
 *  \f] 
 * is discretized, with \f$ b^i\f$ being the contravariant components of \f$\mathbf b\f$ . 
 * @copydoc hide_geometry_matrix_container
 * This class has the SelfMadeMatrixTag so it can be used in blas2::symv functions 
 * and thus in a conjugate gradient solver. 
 * @note The constructors initialize \f$ \chi_x = \chi_y = \chi_z=1\f$ 
 * @attention Pay attention to the negative sign 
 */
template<class Geometry, class Matrix, class container> 
struct GeneralEllipticSym
{
    /**
     * @brief Construct from Grid
     *
     * @param g The Grid, boundary conditions are taken from here
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative
     */
    GeneralEllipticSym( const Geometry& g, norm no = not_normed, direction dir = forward): 
        ellipticForward_( g, no, dir), ellipticBackward_(g,no,inverse(dir)),
        temp_( dg::evaluate( one, g) )
    { }

        /**
     * @brief Construct from Grid and bc
     *
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param bcz boundary contition in z
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative
     */
    GeneralEllipticSym( const Geometry& g, bc bcx, bc bcy,bc bcz, norm no = not_normed, direction dir = forward): 
        ellipticForward_( g, bcx, bcy, no, dir), ellipticBackward_(g,bcx,bcy,no,inverse(dir)),
        temp_( dg::evaluate( one, g) )
    { 
    }
    /**
     * @brief Set x-component of \f$ chi\f$
     *
     * @param chi new x-component
     */
    void set_x( const container& chi)
    {
        ellipticForward_.set_x( chi);
        ellipticBackward_.set_x( chi);
    }
    /**
     * @brief Set y-component of \f$ chi\f$
     *
     * @param chi new y-component
     */
    void set_y( const container& chi)
    {
        ellipticForward_.set_y( chi);
        ellipticBackward_.set_y( chi);
    }
    /**
     * @brief Set z-component of \f$ chi\f$
     *
     * @param chi new z-component
     */
    void set_z( const container& chi)
    {
        ellipticForward_.set_z( chi);
        ellipticBackward_.set_z( chi);
    }

    /**
     * @brief Set new components for \f$ chi\f$
     *
     * @param chi chi[0] is new x-component, chi[1] the new y-component, chi[2] z-component
     */
    void set( const std::vector<container>& chi)
    {
        ellipticForward_.set( chi);
        ellipticBackward_.set( chi);
    }

    ///@copydoc Elliptic::weights()
    const container& weights()const {return ellipticForward_.weights();}
    ///@copydoc Elliptic::inv_weights()
    const container& inv_weights()const {return ellipticForward_.inv_weights();}
    ///@copydoc GeneralElliptic::precond()
    const container& precond()const {return ellipticForward_.precond();}

    ///@copydoc Elliptic::symv()
    void symv( const container& x, container& y) 
    {
        ellipticForward_.symv( x,y);
        ellipticBackward_.symv( x,temp_);
        dg::blas1::axpby( 0.5, temp_, 0.5, y);
    }
    private:
    direction inverse( direction dir)
    {
        if( dir == forward) return backward;
        if( dir == backward) return forward;
        return centered;
    }
    dg::GeneralElliptic<Geometry, Matrix, container> ellipticForward_, ellipticBackward_;
    container temp_;
};

/**
 * @brief %Operator that acts as a 2d negative elliptic differential operator
 *
 * @ingroup matrixoperators
 *
 * The term discretized is 
 * \f[ -\nabla \cdot ( \chi \cdot \nabla_\perp ) \f]
 * where \f$\chi\f$ is a symmetric tensor
  In general that means 
 * \f[ 
 * \begin{align}
 * v^x = \chi^{xx} \partial_x f + \chi^{xy}\partial_y f \\
 * v^y = \chi^{yx} \partial_x f + \chi^{yy}\partial_y f \\
 * -\frac{1}{\sqrt{g}} \left(\partial_x(\sqrt{g} v^x ) + \partial_y(\sqrt{g} v^y) \right)
 *  \end{align}
 *  \f] 
 * @copydoc hide_geometry_matrix_container
 * This class has the SelfMadeMatrixTag so it can be used in blas2::symv functions 
 * and thus in a conjugate gradient solver. 
 * @note The constructors initialize \f$ \chi = I\f$ 
 * @attention Pay attention to the negative sign 
 */
template< class Geometry, class Matrix, class container> 
struct TensorElliptic
{
    /**
     * @brief Construct from Grid
     * @param g The Grid, boundary conditions are taken from here
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative
     */
    TensorElliptic( const Geometry& g, norm no = not_normed, direction dir = forward): 
        no_(no), g_(g)
    { 
        construct( g, g.bcx(), g.bcy(), dir);
    }
    /**
     * @brief Construct from Grid and bc 
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param no Not normed for elliptic equations, normed else
     * @param dir Direction of the right first derivative
     */
    TensorElliptic( const Geometry& g, bc bcx, bc bcy, norm no = not_normed, direction dir = forward): 
        no_(no), g_(g)
    { 
        construct( g, bcx, bcy, dir);
    }

    /**
     * @brief Set new components for \f$ chi\f$
     *
     * @param chiXX The new xx component
     * @param chiXY The new xy component
     * @param chiYY The new yy component
     * @note Components need to be already transformed into the current coordinate system
     */
    void set( const container& chiXX, const container& chiXY, const container& chiYY)
    {
        dg::tensor::pointwiseDot( vol_, chiXX, chixx_);
        dg::tensor::pointwiseDot( vol_, chiXY, chixy_);
        dg::tensor::pointwiseDot( vol_, chiYY, chiyy_);
    }

    /**
     * @brief Transform components to the current coordinate system
     */
    template<class ChiRR, class ChiRZ, class ChiZZ>
    void transform_and_set( const ChiRR& chiRR, const ChiRZ& chiRZ, const ChiZZ& chiZZ)
    {
        typename GeometryTraits<Geometry>::host_vector chiXX, chiXY, chiYY;
        dg::pushForwardPerp( chiRR, chiRZ, chiZZ, chiXX, chiXY, chiYY, g_.get());
        dg::blas1::transfer( chiXX, chixx_);
        dg::blas1::transfer( chiXY, chixy_);
        dg::blas1::transfer( chiYY, chiyy_);
        set( chixx_, chixy_, chiyy_);
    }

    ///@copydoc Elliptic::inv_weights()
    const container& inv_weights()const {return inv_weights_;}
    ///@copydoc GeneralElliptic::precond()
    const container& precond()const {return precond_;}

    ///@copydoc Elliptic::symv()
    void symv( const container& x, container& y) 
    {
        //compute gradient
        dg::blas2::gemv( rightx, x, tempx_); //R_x*f 
        dg::blas2::gemv( righty, x, tempy_); //R_y*f

        //multiply with chi 
        dg::blas1::pointwiseDot( 1., chixx_, tempx_, 1., chixy_, tempy_, 0., gradx_);//gxy*v_y
        dg::blas1::pointwiseDot( 1., chixy_, tempx_, 1., chiyy_, tempy_, 1., tempy_); //gyy*v_y

        //now take divergence
        dg::blas2::gemv( -1., leftx, gradx_, 0., y);  
        dg::blas2::gemv( -1., lefty, tempy_, 1., y);  
        if( no_ == normed)
            dg::tensor::pointwiseDivide( y, vol_,y);

        //add jump terms
        dg::blas2::symv( +1., jumpX, x, 1., y);
        dg::blas2::symv( +1., jumpY, x, 1., y);
        if( no_ == not_normed)//multiply weights without volume
            dg::blas2::symv( weights_wo_vol, y, y);
    }
    private:
    void construct( const Geometry& g, bc bcx, bc bcy, direction dir)
    {
        dg::blas2::transfer( dg::create::dx( g, inverse( bcx), inverse(dir)), leftx);
        dg::blas2::transfer( dg::create::dy( g, inverse( bcy), inverse(dir)), lefty);
        dg::blas2::transfer( dg::create::dx( g, bcx, dir), rightx);
        dg::blas2::transfer( dg::create::dy( g, bcy, dir), righty);
        dg::blas2::transfer( dg::create::jumpX( g, bcx),   jumpX);
        dg::blas2::transfer( dg::create::jumpY( g, bcy),   jumpY);
        dg::blas1::transfer( dg::create::volume(g),        weights_);
        dg::blas1::transfer( dg::create::inv_volume(g),    inv_weights_);
        dg::blas1::transfer( dg::create::inv_weights(g),   precond_); //weights are better preconditioners than volume
        dg::blas1::transfer( dg::evaluate( dg::one, g),    chixx_);
        dg::blas1::transfer( dg::evaluate( dg::zero,g),    chixy_);
        dg::blas1::transfer( dg::evaluate( dg::one, g),    chiyy_);
        tempx_ = tempy_ = gradx_ = chixx_;
        dg::blas1::transfer( dg::create::weights(g), weights_wo_vol);

        vol_=dg::tensor::volume(g.metric());
        dg::tensor::pointwiseDot( vol_, chixx_, chixx_); 
        dg::tensor::pointwiseDot( vol_, chixy_, chixy_); 
        dg::tensor::pointwiseDot( vol_, chiyy_, chiyy_); 
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
    Matrix leftx, lefty, rightx, righty, jumpX, jumpY;
    container weights_, inv_weights_, weights_wo_vol, precond_; //contain coeffs for chi multiplication
    container chixx_, chixy_, chiyy_, tempx_, tempy_, gradx_;
    SparseElement<container> vol_;
    norm no_;
    Handle<Geometry> g_;
};

///@cond
template< class G, class M, class V>
struct MatrixTraits< Elliptic<G, M, V> >
{
    typedef typename VectorTraits<V>::value_type  value_type;
    typedef SelfMadeMatrixTag matrix_category;
};

template< class G, class M, class V>
struct MatrixTraits< GeneralElliptic<G, M, V> >
{
    typedef typename VectorTraits<V>::value_type  value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template< class G, class M, class V>
struct MatrixTraits< GeneralEllipticSym<G, M, V> >
{
    typedef typename VectorTraits<V>::value_type  value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
template< class G, class M, class V>
struct MatrixTraits< TensorElliptic<G, M, V> >
{
    typedef typename VectorTraits<V>::value_type  value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

} //namespace dg

