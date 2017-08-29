#pragma once

#include "dg/blas.h"
#include "dg/geometry/geometry.h"
#include "dg/backend/derivatives.h"
#include "fieldaligned.h"
#ifdef MPI_VERSION
#include "backend/mpi_derivatives.h"
#include "mpi_fieldaligned.h"
#endif //MPI_VERSION
#include "magnetic_field.h"

/*!@file 
 *
 * This file includes the appropriate headers for parallel derivatives
 */

//TODO: use buffers to make symv const
namespace dg{

/**
* @brief Class for the evaluation of a parallel derivative
*
* This class discretizes the operators \f$ \nabla_\parallel = 
\mathbf{b}\cdot \nabla = b_R\partial_R + b_Z\partial_Z + b_\phi\partial_\phi \f$, \f$\nabla_\parallel^\dagger\f$ and \f$\Delta_\parallel=\nabla_\parallel^\dagger\cdot\nabla_\parallel\f$ in
arbitrary coordinates
* @ingroup fieldaligned
* @tparam IMatrix The type of the interpolation matrix
* @tparam Matrix The matrix class of the jump matrix
* @tparam container The container-class on which the interpolation matrix operates on (does not need to be dg::HVec)
*/
template< class Geometry, class IMatrix, class Matrix, class container >
struct DS
{
    /**
    * @brief Construct from a field and a grid
    *
    * @param mag Take the magnetic field as vector field
    * @param g  the boundary conditions are also taken from here
    * @param no norm or not_normed affects the behaviour of the symv function
    * @param dir the direction affects both the operator() and the symv function
    * @param dependsOnX performance indicator for the fine 2 coarse operations (elements of vec depend on first coordinate yes or no) also determines if a jump matrix is added in the x-direction
    * @param dependsOnY performance indicator for the fine 2 coarse operations (elements of vec depend on second coordinate yes or no)
    * @param mx Multiplication factor in x
    * @param my Multiplication factor in y
    */
    DS(const dg::geo::TokamakMagneticField& mag, const Geometry& g, dg::norm no=dg::normed, dg::direction dir = dg::centered, bool dependsOnX = true, bool dependsOnY=true, unsigned mx=1, unsigned my=1);

    /**
    * @brief Apply the forward derivative on a 3d vector
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void forward( const container& f, container& dsf);
    /**
    * @brief Apply the backward derivative on a 3d vector
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void backward( const container& f, container& dsf);
    /**
    * @brief Apply the centered derivative on a 3d vector
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void centered( const container& f, container& dsf);

    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void forwardAdj( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void backwardAdj( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void centeredAdjDir( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void forwardAdjDir( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void backwardAdjDir( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void centeredAdj( const container& f, container& dsf);

    /**
    * @brief compute parallel derivative
    *
    * dependent on dir redirects to either forward(), backward() or centered()
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void operator()( const container& f, container& dsf);


    /**
     * @brief Discretizes the parallel Laplacian as a symmetric matrix
     *
     * if direction is centered then centered followed by centeredAdj and adding jump terms
     * @param f The vector to derive
     * @param dsTdsf contains result on output (write only)
     * @note if apply_jumpX is false then no jumpy terms will be added in the x-direction
     */
    void symv( const container& f, container& dsTdsf);

    /**
    * @brief Set boundary conditions in the limiter region
    *
    * if Dirichlet boundaries are used the left value is the left function
    value, if Neumann boundaries are used the left value is the left derivative value
    * @param bcz boundary condition
    * @param left left boundary value
    * @param right right boundary value
    */
    void set_boundaries( dg::bc bcz, double left, double right)
    {
        f_.set_boundaries( bcz, left, right);
    }
    /**
     * @brief Set boundary conditions in the limiter region
     *
     * if Dirichlet boundaries are used the left value is the left function
     value, if Neumann boundaries are used the left value is the left derivative value
     * @param bcz boundary condition
     * @param left left boundary value
     * @param right right boundary value
    */
    void set_boundaries( dg::bc bcz, const container& left, const container& right)
    {
        f_.set_boundaries( bcz, left, right);
    }
    /**
     * @brief Set boundary conditions in the limiter region
     *
     * if Dirichlet boundaries are used the left value is the left function
     value, if Neumann boundaries are used the left value is the left derivative value
     * @param bcz boundary condition
     * @param global 3D vector containing boundary values
     * @param scal_left left scaling factor
     * @param scal_right right scaling factor
     */
    void set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right)
    {
        f_.set_boundaries( bcz, global, scal_left, scal_right);
    }

    /**
     * @brief Returns the weights used to make the matrix symmetric 
     *
     * needed by invert class
     * @return weights
     */
    const container& weights()const {return vol3d;}
    /**
     * @brief Returns the preconditioner to use in conjugate gradient
     *
     * needed by invert class
     * In this case inverse weights are the best choice
     * @return inverse weights
     */
    const container& precond()const {return inv3d;}

    /**
    * @brief access the underlying Fielaligned object for evaluate
    *
    * @return acces to fieldaligned object
    */
    const FieldAligned<Geometry, IMatrix, container>& fieldaligned() const{return f_;}
    private:
    FieldAligned<Geometry,IMatrix,container> f_;
    Matrix jumpX, jumpY;
    container tempP, temp0, tempM;
    container f, dsf;
    container vol3d, inv3d;
    container invB;
    //container R_;
    dg::norm no_;
    dg::direction dir_;
    bool apply_jumpX_;
    dg::SparseElement<container> volume_;
};

///@cond
////////////////////////////////////DEFINITIONS////////////////////////////////////////

template<class Geometry, class I, class M, class container>
DS<Geometry, I, M,container>::DS(const dg::geo::TokamakMagneticField& mag, const Geometry& grid, dg::norm no, dg::direction dir, bool jumpX, bool jumpY, unsigned mx, unsigned my):
        f_( dg::geo::BinaryVectorLvl0( dg::geo::BHatR(mag), dg::geo::BHatZ(mag), dg::geo::BHatP(mag)), grid, mx, my, 1e-5, FullLimiter(), grid.bcx(), grid.bcy()),
        jumpX( dg::create::jumpX( grid)),
        jumpY( dg::create::jumpY( grid)),
        tempP( dg::evaluate( dg::zero, grid)), temp0( tempP), tempM( tempP), 
        f(tempP), dsf(tempP),
        vol3d( dg::create::volume( grid)), inv3d( dg::create::inv_volume( grid)),
        invB(dg::pullback(dg::geo::InvB(mag), grid)), 
        no_(no), dir_(dir), apply_jumpX_(jumpX)
{
    volume_ = dg::tensor::volume(grid.metric());
}

template<class G, class I, class M, class container>
inline void DS<G,I,M,container>::operator()( const container& f, container& dsf) { 
    if( dir_ == dg::centered)
        return centered( f, dsf);
    else if( dir_ == dg::forward)
        return forward( f, dsf);
    else
        return backward( f, dsf);
}


template<class G, class I, class M, class container>
void DS<G, I,M,container>::centered( const container& f, container& dsf)
{
    //direct discretisation
    assert( &f != &dsf);
    f_(einsPlus, f, tempP);
    f_(einsMinus, f, tempM);
    dg::blas1::axpby( 1., tempP, -1., tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hz(), dsf);
}

template<class G, class I, class M, class container>
void DS<G, I,M,container>::centeredAdj( const container& f, container& dsf)
{               
    //adjoint discretisation
    assert( &f != &dsf);    
    dg::blas1::pointwiseDot( vol3d, f, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hz(), dsf);
    f_(einsPlusT, dsf, tempP);
    f_(einsMinusT, dsf, tempM);
    dg::blas1::axpby( 1., tempM, -1., tempP, dsf);        
    dg::blas1::pointwiseDot( inv3d, dsf, dsf); 
}

template<class G, class I, class M, class container>
void DS<G,I,M,container>::centeredAdjDir( const container& f, container& dsf)
{       
//     Direct discretisation
    assert( &f != &dsf);    
    dg::blas1::pointwiseDot( f, invB, dsf);
    f_(einsPlus, dsf, tempP);
    f_(einsMinus, dsf, tempM);
    dg::blas1::axpby( 1., tempP, -1., tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hz(), dsf);        
    dg::blas1::pointwiseDivide( dsf, invB, dsf);
}

template<class G, class I, class M, class container>
void DS<G,I,M,container>::forward( const container& f, container& dsf)
{
    //direct
    assert( &f != &dsf);
    f_(einsPlus, f, tempP);
    dg::blas1::axpby( 1., tempP, -1., f, tempP);
    dg::blas1::pointwiseDivide( tempP, f_.hp(), dsf);
}

template<class G, class I, class M, class container>
void DS<G,I,M,container>::forwardAdj( const container& f, container& dsf)
{    
    //adjoint discretisation
    assert( &f != &dsf);
    dg::blas1::pointwiseDot( vol3d, f, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hp(), dsf);
    f_(einsPlusT, dsf, tempP);
    dg::blas1::axpby( -1., tempP, 1., dsf, dsf);
    dg::blas1::pointwiseDot( inv3d, dsf, dsf); 
}

template<class G, class I, class M, class container>
void DS<G,I,M,container>::forwardAdjDir( const container& f, container& dsf)
{
    //direct discretisation
    assert( &f != &dsf);
    dg::blas1::pointwiseDot( f, invB, dsf);
    f_(einsMinus, dsf, tempP);
    dg::blas1::axpby( -1., tempP, 1., dsf, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hm(), dsf);        
    dg::blas1::pointwiseDivide( dsf, invB, dsf);
}

template<class G,class I, class M, class container>
void DS<G,I,M,container>::backward( const container& f, container& dsf)
{
    //direct
    assert( &f != &dsf);
    f_(einsMinus, f, tempM);
    dg::blas1::axpby( 1., tempM, -1., f, tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hm(), dsf);
}

template<class G,class I, class M, class container>
void DS<G,I,M,container>::backwardAdj( const container& f, container& dsf)
{    
    //adjoint discretisation
    assert( &f != &dsf);
    dg::blas1::pointwiseDot( vol3d, f, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hm(), dsf);
    f_(einsMinusT, dsf, tempM);
    dg::blas1::axpby( -1., tempM, 1., dsf, dsf);
    dg::blas1::pointwiseDot( inv3d, dsf, dsf); 
}

template<class G,class I, class M, class container>
void DS<G,I,M,container>::backwardAdjDir( const container& f, container& dsf)
{
    //direct
    assert( &f != &dsf);
    dg::blas1::pointwiseDot( f, invB, dsf);
    f_(einsPlus, dsf, tempM);
    dg::blas1::axpby( -1., tempM, 1., dsf, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hp(), dsf);        
    dg::blas1::pointwiseDivide( dsf, invB, dsf);
}

template<class G,class I, class M, class container>
void DS<G,I,M,container>::symv( const container& f, container& dsTdsf)
{
    if(dir_ == dg::centered)
    {
        centered( f, tempP);
        centeredAdj( tempP, dsTdsf);
    }
    else 
    {
        forward( f, tempP);
        forwardAdj( tempP, dsTdsf);
        backward( f, tempM);
        backwardAdj( tempM, temp0);
        dg::blas1::axpby(0.5,temp0,0.5,dsTdsf,dsTdsf);
    }
//     add jump term 

    if(apply_jumpX_)
    {
        dg::blas2::symv( jumpX, f, temp0);
        dg::tensor::pointwiseDivide( temp0, volume_, temp0);
        dg::blas1::axpby( -1., temp0, 1., dsTdsf, dsTdsf);
    }
    dg::blas2::symv( jumpY, f, temp0);
    dg::tensor::pointwiseDivide( temp0, volume_, temp0);
    dg::blas1::axpby( -1., temp0, 1., dsTdsf, dsTdsf);
    if( no_ == not_normed)
    {
        dg::blas1::pointwiseDot( vol3d, dsTdsf, dsTdsf); //make it symmetric
    }
}

//enables the use of the dg::blas2::symv function 
template< class G, class I, class M, class V>
struct MatrixTraits< DS<G,I,M, V> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};

///@endcond

}//namespace dg

