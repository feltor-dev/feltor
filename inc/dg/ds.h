#pragma once

#include "blas.h"
#include "geometry.h"
#include "backend/derivatives.h"
#include "geometry/fieldaligned.h"
#ifdef MPI_VERSION
#include "backend/mpi_derivatives.h"
#include "geometry/mpi_fieldaligned.h"
#endif //MPI_VERSION

/*!@file 
 *
 * This file includes the appropriate headers for parallel derivatives
 */

namespace dg{

/**
* @brief Class for the evaluation of a parallel derivative
*
* This class discretizes the operators \f$ \nabla_\parallel = 
\mathbf{b}\cdot \nabla = b_R\partial_R + b_Z\partial_Z + b_\phi\partial_\phi \f$, \f$\nabla_\parallel^\dagger\f$ and \f$\Delta_\parallel=\nabla_\parallel^\dagger\cdot\nabla_\parallel\f$ in
cylindrical coordinates
* @ingroup fieldaligned
* @tparam FA Engine class for interpolation, provides the necessary interpolation operations
* @tparam Matrix The matrix class of the jump matrix
* @tparam container The container-class on which the interpolation matrix operates on (does not need to be dg::HVec)
*/
template< class FA, class Matrix, class container >
struct DS
{
    typedef FA FieldAligned;//!< typedef for easier construction of corresponding fieldaligned object

    /**
    * @brief Construct from a field and a grid
    *
    * @tparam InvB The inverse magnitude of the magnetic field \f$ \frac{1}{B}\f$
    * @param field The fieldaligned object containing interpolation matrices
    * @param invB The inverse magentic field strength
    * @param no norm or not_normed affects the behaviour of the symv function
    * @param dir the direction affects both the operator() and the symv function
    @param jumpX determines if a jump matrix is added in X-direction
    */
    template<class InvB>
    DS(const FA& field, InvB invB, dg::norm no=dg::normed, dg::direction dir = dg::centered, bool jumpX = true);

    /**
    * @brief Apply the derivative on a 3d vector
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void forward( const container& f, container& dsf);
    /**
    * @brief Apply the derivative on a 3d vector
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void backward( const container& f, container& dsf);
    /**
    * @brief Apply the derivative on a 3d vector
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void centered( const container& f, container& dsf);

    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void forwardT( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void backwardT( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void centeredTD( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void forwardTD( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void backwardTD( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void centeredT( const container& f, container& dsf);

    /**
    * @brief is the negative transposed of centered
    *
    * redirects to centered
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void operator()( const container& f, container& dsf);


    /**
     * @brief Compute the second derivative using finite differences
     *
     * discretizes \f$ \nabla_\parallel\cdot \nabla_\parallel\f$
     * @param f input function
     * @param dssf output (write-only)
     */
    void dss( const container& f, container& dssf);

    /**
     * @brief Discretizes the parallel Laplacian as a symmetric matrix
     *
     * forward followed by forwardT and adding jump terms
     * @param f The vector to derive
     * @param dsTdsf contains result on output (write only)
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
    const FA& fieldaligned() const{return f_;}
    private:
    FA f_;
    Matrix jumpX, jumpY;
    container tempP, temp0, tempM;
    container vol3d, inv3d;
    container invB;
    //container R_;
    dg::norm no_;
    dg::direction dir_;
    bool apply_jumpX_;
};

///@cond
////////////////////////////////////DEFINITIONS////////////////////////////////////////

template<class FA, class M, class container>
template <class Field>
DS<FA, M,container>::DS(const FA& field, Field inverseB, dg::norm no, dg::direction dir, bool jumpX):
        f_(field),
        jumpX( dg::create::jumpX( field.grid())),
        jumpY( dg::create::jumpY( field.grid())),
        tempP( dg::evaluate( dg::zero, field.grid())), temp0( tempP), tempM( tempP), 
        vol3d( dg::create::volume( field.grid())), inv3d( dg::create::inv_volume( field.grid())),
        invB(dg::pullback(inverseB,field.grid())), //R_(dg::evaluate(dg::coo1,grid)), 
        no_(no), dir_(dir), apply_jumpX_(jumpX)
{ }

template<class F, class M, class container>
inline void DS<F,M,container>::operator()( const container& f, container& dsf) { 
    if( dir_ == dg::centered)
        return centered( f, dsf);
    else if( dir_ == dg::forward)
        return forward( f, dsf);
    else
        return backward( f, dsf);
}


template<class F, class M, class container>
void DS<F,M,container>::centered( const container& f, container& dsf)
{
    //direct discretisation
    assert( &f != &dsf);
    f_.einsPlus( f, tempP);
    f_.einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempP, -1., tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hz(), dsf);
    
    ////adjoint discretisation
/*    assert( &f != &dsf);    
    dg::blas1::pointwiseDot( vol3d, f, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hz(), dsf);
    dg::blas1::pointwiseDivide( dsf, invB, dsf);

    einsPlusT( dsf, tempP);
    einsMinusT( dsf, tempM);
    dg::blas1::axpby( 1., tempM, -1., tempP);
    dg::blas1::pointwiseDot( inv3d, tempP, dsf);
    dg::blas1::pointwiseDot( dsf, invB, dsf);  */  

}

template<class F, class M, class container>
void DS<F,M,container>::centeredT( const container& f, container& dsf)
{               
//     //adjoint discretisation
    assert( &f != &dsf);    
    dg::blas1::pointwiseDot( vol3d, f, dsf);

    dg::blas1::pointwiseDivide( dsf, f_.hz(), dsf);
    f_.einsPlusT( dsf, tempP);
    f_.einsMinusT( dsf, tempM);
    dg::blas1::axpby( 1., tempM, -1., tempP);        
    dg::blas1::pointwiseDot( inv3d, tempP, dsf); 

//       dg::blas1::pointwiseDot( inv3d, tempP,tempP); //make it symmetric
        //stegmeir weights
//         dg::blas1::pointwiseDot( f_.hz()h, f, dsf);
//         dg::blas1::pointwiseDot( invB, dsf, dsf);
//         dg::blas1::pointwiseDot( w2d, dsf, dsf);
//         dg::blas1::pointwiseDivide( dsf, f_.hz(), dsf);
//         einsPlusT( dsf, tempP);
//         einsMinusT( dsf, tempM);
//         dg::blas1::axpby( 1., tempM, -1., tempP);        
//         dg::blas1::pointwiseDot( inv3d, tempP, dsf);
//         dg::blas1::scal(dsf,0.5);
//         dg::blas1::pointwiseDivide( tempP,f_.hz()h,  dsf);
//         dg::blas1::pointwiseDivide(  dsf,invB, dsf);
//         dg::blas1::pointwiseDivide( dsf,w2d,  dsf);  
}

template<class F, class M, class container>
void DS<F,M,container>::centeredTD( const container& f, container& dsf)
{       
//     Direct discretisation
    assert( &f != &dsf);    
    dg::blas1::pointwiseDot( f, invB, dsf);
    f_.einsPlus( dsf, tempP);
    f_.einsMinus( dsf, tempM);
    dg::blas1::axpby( 1., tempP, -1., tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hz(), dsf);        
    dg::blas1::pointwiseDivide( dsf, invB, dsf);
}

template<class F, class M, class container>
void DS<F,M,container>::forward( const container& f, container& dsf)
{
    //direct
    assert( &f != &dsf);
    f_.einsPlus( f, tempP);
    dg::blas1::axpby( 1., tempP, -1., f, tempP);
    dg::blas1::pointwiseDivide( tempP, f_.hp(), dsf);
    //adjoint discretisation
//     assert( &f != &dsf);    
//     dg::blas1::pointwiseDot( vol3d, f, dsf);
//     dg::blas1::pointwiseDivide( dsf, f_.hm(), dsf);
//     dg::blas1::pointwiseDivide( dsf, invB, dsf);
//     einsMinusT( dsf, tempP);
//     dg::blas1::axpby( 1., tempP,-1.,dsf,dsf);
//     dg::blas1::pointwiseDot( inv3d, dsf, dsf);
//     dg::blas1::pointwiseDot( dsf, invB, dsf);
}

template<class F, class M, class container>
void DS<F,M,container>::forwardT( const container& f, container& dsf)
{    
    //adjoint discretisation
    assert( &f != &dsf);
    dg::blas1::pointwiseDot( vol3d, f, dsf);   
    dg::blas1::pointwiseDivide( dsf, f_.hp(), dsf);
    f_.einsPlusT( dsf, tempP);
    dg::blas1::axpby( -1., tempP, 1., dsf, dsf);
    dg::blas1::pointwiseDot( inv3d, dsf, dsf);
}

template<class F, class M, class container>
void DS<F,M,container>::forwardTD( const container& f, container& dsf)
{
    //direct discretisation
    assert( &f != &dsf);    
    dg::blas1::pointwiseDot( f, invB, dsf);
    f_.einsMinus( dsf, tempP);
    dg::blas1::axpby( -1., tempP, 1., dsf, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hm(), dsf);        
    dg::blas1::pointwiseDivide( dsf, invB, dsf);


}
template<class F, class M, class container>
void DS<F,M,container>::backward( const container& f, container& dsf)
{
    //direct
    assert( &f != &dsf);
    f_.einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempM, -1., f, tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hm(), dsf);
    
    //adjoint discretisation
//     assert( &f != &dsf);    
//     dg::blas1::pointwiseDot( vol3d, f, dsf);
//     dg::blas1::pointwiseDivide( dsf, f_.hp(), dsf);
//     dg::blas1::pointwiseDivide( dsf, invB, dsf);
//     einsPlusT( dsf, tempM);
//     dg::blas1::axpby( 1., tempM, -1.,dsf,dsf);
//     dg::blas1::pointwiseDot( inv3d,dsf, dsf);
//     dg::blas1::pointwiseDot( dsf, invB, dsf);
}
template<class F, class M, class container>
void DS<F,M,container>::backwardT( const container& f, container& dsf)
{    
    //adjoint discretisation
    assert( &f != &dsf);
    dg::blas1::pointwiseDot( vol3d, f, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hm(), dsf);
    f_.einsMinusT( dsf, tempM);
    dg::blas1::axpby( -1., tempM, 1., dsf, dsf);
    dg::blas1::pointwiseDot( inv3d, dsf, dsf);   
}

template<class F, class M, class container>
void DS<F,M,container>::backwardTD( const container& f, container& dsf)
{
    //direct
    assert( &f != &dsf);    
    dg::blas1::pointwiseDot( f, invB, dsf);
    f_.einsPlus( dsf, tempM);
    dg::blas1::axpby( -1., tempM, 1., dsf, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hp(), dsf);        
    dg::blas1::pointwiseDivide( dsf, invB, dsf);
}

template< class F, class M, class container >
void DS<F,M,container>::symv( const container& f, container& dsTdsf)
{
    if(dir_ == dg::centered)
    {
        centered( f, tempP);
        centeredT( tempP, dsTdsf);
    }
    else 
    {
        forward( f, tempP);
        forwardT( tempP, dsTdsf);
        backward( f, tempM);
        backwardT( tempM, temp0);
        dg::blas1::axpby(0.5,temp0,0.5,dsTdsf,dsTdsf);
    }
//     add jump term 

    if(apply_jumpX_)
    {
        dg::blas2::symv( jumpX, f, temp0);
        dg::geo::divideVolume( temp0, f_.grid());
        dg::blas1::axpby( -1., temp0, 1., dsTdsf, dsTdsf);
    }
    dg::blas2::symv( jumpY, f, temp0);
    dg::geo::divideVolume( temp0, f_.grid());
    //dg::blas1::pointwiseDivide( temp0, R_, temp0);
    dg::blas1::axpby( -1., temp0, 1., dsTdsf, dsTdsf);
    if( no_ == not_normed)
    {
        dg::blas1::pointwiseDot( vol3d, dsTdsf, dsTdsf); //make it symmetric
    }
}

template< class F, class M, class container >
void DS<F,M,container>::dss( const container& f, container& dssf)
{
    assert( &f != &dssf);
    f_.einsPlus(  f, tempP);
    f_.einsMinus( f, tempM);
    dg::blas1::pointwiseDivide( tempP, f_.hp(), tempP);
    dg::blas1::pointwiseDivide( tempP, f_.hz(), tempP);
    dg::blas1::pointwiseDivide( f, f_.hp(), temp0);
    dg::blas1::pointwiseDivide( temp0, f_.hm(), temp0);
    dg::blas1::pointwiseDivide( tempM, f_.hm(), tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hz(), tempM);
    dg::blas1::axpby(  2., tempP, +2., tempM); //fp+fm
    dg::blas1::axpby( -2., temp0, +1., tempM, dssf); 
}


//enables the use of the dg::blas2::symv function 
template< class F, class M, class V>
struct MatrixTraits< DS<F,M, V> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};

///@endcond

///@addtogroup typedefs
///@{
typedef dg::DS<dg::FieldAligned<dg::CylindricalGrid3d<dg::DVec>, dg::IDMatrix, dg::DVec>, dg::DMatrix, dg::DVec> DDS;//!< device DS type
typedef dg::DS<dg::FieldAligned<dg::CylindricalGrid3d<dg::HVec>, dg::IHMatrix, dg::HVec>, dg::HMatrix, dg::HVec> HDS; //!< host DS type
#ifdef MPI_VERSION
typedef dg::DS< dg::MPI_FieldAligned<dg::CylindricalMPIGrid3d<dg::MDVec>, dg::IDMatrix, dg::BijectiveComm< dg::iDVec, dg::DVec >, dg::DVec>, dg::MDMatrix, dg::MDVec > MDDS; //!< MPI device DS type
typedef dg::DS< dg::MPI_FieldAligned<dg::CylindricalMPIGrid3d<dg::MHVec>, dg::IHMatrix, dg::BijectiveComm< dg::iHVec, dg::HVec >, dg::HVec>, dg::MHMatrix, dg::MHVec > MHDS; //!< MPI host DS type
#endif //MPI_VERSION
///@}


}//namespace dg

