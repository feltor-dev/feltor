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
    * forward derivative \f$ g_i = \alpha \frac{1}{h_z^+}(f_{i+1} - f_{i}) + \beta g_i\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void forward( double alpha, const container& f, double beta, container& dsf){
        do_forward( alpha, f, beta, dsf);
    }
    /**
    * @brief Apply the backward derivative on a 3d vector
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void backward( double alpha, const container& f, double beta, container& dsf){
        do_backward( alpha, f, beta, dsf);
    }
    /**
    * @brief Apply the centered derivative on a 3d vector
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void centered( double alpha, const container& f, double beta, container& dsf){
        do_centered( alpha, f, beta, dsf);
    }

    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void forwardAdj( double alpha, const container& f, double beta, container& dsf){
        do_forwardAdj( alpha, f, beta, dsf, dg::normed);
    }
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void backwardAdj( double alpha, const container& f, double beta, container& dsf){
        do_backwardAdj( alpha, f, beta, dsf, dg::normed);
    }
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void centeredAdj(double alpha, const container& f, double beta, container& dsf){
        do_centeredAdj( alpha, f, beta, dsf, dg::normed);
    }
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void forwardAdjDir( double alpha, const container& f, double beta, container& dsf)
    {
        dg::blas1::pointwiseDivide( f, m_B, m_temp);
        forward( m_temp, m_temp);
        dg::blas1::pointwiseDot( alpha, m_temp, m_B, beta, dsf);
    }
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void backwardAdjDir( double alpha, const container& f, double beta, container& dsf)
    {
        dg::blas1::pointwiseDivide( f, m_B, m_temp);
        backward( 1., m_temp, 0., m_temp);
        dg::blas1::pointwiseDot( alpha, m_temp, m_B, beta, dsf);
    }
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void centeredAdjDir( double alpha, const container& f, double beta, container& dsf)
    {
        dg::blas1::pointwiseDivide( f, m_B, m_temp);
        backward( 1., m_temp, m_temp);
        dg::blas1::pointwiseDot( alpha, m_temp, m_B, beta, dsf);
    }

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
    void symv( const container& f, container& dsTdsf){ do_symv( f, dsTdsf);}

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

    const container& weights()const {return vol3d;}
    const container& inv_weights()const {return inv3d;}
    const container& precond()const {return inv3d;}

    /**
    * @brief access the underlying Fielaligned object for evaluate
    *
    * @return acces to fieldaligned object
    */
    const FieldAligned<Geometry, IMatrix, container>& fieldaligned() const{return f_;}
    private:
    void do_forward(double alpha, const container& f, double beta, container& dsf);
    void do_backward(double alpha, const container& f, double beta, container& dsf);
    void do_centered(double alpha, const container& f, double beta, container& dsf);
    void do_forwardAdj(double alpha, const container& f, double beta, container& dsf, dg::norm no);
    void do_backwardAdj(double alpha, const container& f, double beta, container& dsf, dg::norm no);
    void do_centeredAdj(double alpha, const container& f, double beta, container& dsf, dg::norm no);
    void do_symv(const container& f, container& dsf);

    FieldAligned<Geometry,IMatrix,container> m_fa;
    Matrix m_jumpX, m_jumpY;
    container m_temp;
    container m_tempP, m_temp0, m_tempM;
    container m_vol3d, m_inv3d, m_weights_wo_vol;
    container m_B;
    //container R_;
    dg::norm m_no;
    dg::direction m_dir;
    bool m_apply_jumpX, m_apply_jumpY;
};

///@cond
////////////////////////////////////DEFINITIONS////////////////////////////////////////

template<class Geometry, class I, class M, class container>
DS<Geometry, I, M,container>::DS(const dg::geo::TokamakMagneticField& mag, const Geometry& grid, dg::norm no, dg::direction dir, bool jumpX, bool jumpY, unsigned mx, unsigned my):
        m_fa( dg::geo::BinaryVectorLvl0( dg::geo::BHatR(mag), dg::geo::BHatZ(mag), dg::geo::BHatP(mag)), grid, mx, my, 1e-5, FullLimiter(), grid.bcx(), grid.bcy()),
        m_no(no), m_dir(dir), m_apply_jumpX(jumpX), m_apply_jumpY(jumpY)
{
    dg::blas1::transfer( dg::pullback( dg::geo::Bmodule(mag), grid), m_B);
    m_temp = m_B, m_tempP = m_B, m_temp0 = m_b, m_tempM = m_b;
    dg::blas1::transfer( dg::create::volume(     grid), m_vol3d); 
    dg::blas1::transfer( dg::create::weights(    grid), m_weights_wo_vol); 
    dg::blas1::transfer( dg::create::inv_volume( grid), m_inv3d); 
    dg::blas2::transfer( dg::create::jumpX( grid), jumpX);
    dg::blas2::transfer( dg::create::jumpY( grid), jumpY);
}

template<class G, class I, class M, class container>
inline void DS<G,I,M,container>::operator()( const container& f, container& dsf) { 
    if( dir_ == dg::centered)
        return centered( 1., f, 0., dsf);
    else if( dir_ == dg::forward)
        return forward( 1., f, 0., dsf);
    else
        return backward( 1., f, 0., dsf);
}

template<class G, class I, class M, class container>
void DS<G,I,M,container>::do_forward( double alpha, const container& f, double beta, container& dsf)
{
    //direct
    m_fa(einsPlus, f, m_tempP);
    dg::blas1::axpby( 1., m_tempP, -1., f, m_tempP);
    dg::blas1::pointwiseDot( alpha, m_tempP, m_fa.hp_inv(), beta, dsf);
}
template<class G,class I, class M, class container>
void DS<G,I,M,container>::do_backward( double alpha, const container& f, double beta, container& dsf)
{
    //direct
    m_fa(einsMinus, f, m_tempM);
    dg::blas1::axpby( 1., m_tempM, -1., f, m_tempM);
    dg::blas1::pointwiseDot( alpha, m_tempM, m_fa.hp_inv(), beta, dsf);
}
template<class G, class I, class M, class container>
void DS<G, I,M,container>::do_centered( double alpha, const container& f, double beta, container& dsf)
{
    //direct discretisation
    m_fa(einsPlus, f, m_tempP);
    m_fa(einsMinus, f, m_tempM);
    dg::blas1::axpby( 1., m_tempP, -1., m_tempM);
    dg::blas1::pointwiseDot( alpha, m_tempM, m_fa.hz_inv(), beta, dsf);
}
template<class G, class I, class M, class container>
void DS<G,I,M,container>::do_forwardAdj( double alpha, const container& f, double beta, container& dsf, dg::norm no)
{    
    //adjoint discretisation
    dg::blas1::pointwiseDot( m_vol3d, f, m_temp0);
    dg::blas1::pointwiseDot( m_temp0, m_fa.hp_inv(), m_temp0);
    m_fa(einsPlusT, m_temp0, m_tempP);
    dg::blas1::axpby( -1., m_tempP, 1., m_temp0, m_temp0);
    if(no == dg::normed) 
        dg::blas1::pointwiseDot( alpha, m_inv3d, m_temp0, beta, dsf); 
}
template<class G,class I, class M, class container>
void DS<G,I,M,container>::do_backwardAdj( double alpha, const container& f, double beta, container& dsf, dg::norm no)
{    
    //adjoint discretisation
    dg::blas1::pointwiseDot( m_vol3d, m_temp0, m_temp0);
    dg::blas1::pointwiseDot( m_temp0, m_fa.hm_inv(), m_temp0);
    m_fa(einsMinusT, m_temp0, m_tempM);
    dg::blas1::axpby( -1., m_tempM, 1., m_temp0, m_temp0);
    if(no == dg::normed) 
        dg::blas1::pointwiseDot( alpha, m_inv3d, m_temp0, beta, dsf); 
}
template<class G, class I, class M, class container>
void DS<G, I,M,container>::do_centeredAdj( double alpha, container& f, double beta, container& dsf, dg::norm no)
{               
    //adjoint discretisation
    dg::blas1::pointwiseDot( m_vol3d, f, m_temp0);
    dg::blas1::pointwiseDot( m_temp0, m_fa.hz_inv(), m_temp0);
    m_fa(einsPlusT,  m_temp0, m_tempP);
    m_fa(einsMinusT, m_temp0, m_tempM);
    dg::blas1::axpby( 1., m_tempP, -1., m_tempM);
    if(no == dg::normed) 
        dg::blas1::pointwiseDot( alpha, m_inv3d, m_tempM, beta, dsf); 
}

template<class G,class I, class M, class container>
void DS<G,I,M,container>::do_symv( const container& f, container& dsTdsf)
{
    if(m_dir == dg::centered)
    {
        do_centered( f, m_tempP);
        do_centeredAdj( m_tempP, dsTdsf, dg::not_normed);
    }
    else 
    {
        do_forward( f, m_tempP);
        do_forwardAdj( m_tempP, m_temp0, dg::not_normed);
        do_backward( f, m_tempM);
        do_backwardAdj( m_tempM, dsTdsf, dg::not_normed);
        dg::blas1::axpby(0.5,m_temp0,0.5,dsTdsf);
    }
    dg::blas1::pointwiseDivide( dsTdsf, m_weights_wo_vol, dsTdsf);
    //     add jump term 
    if(m_apply_jumpX)
        dg::blas2::symv( -1., jumpX, f, 1., dsTdsf);
    if(m_apply_jumpY)
        dg::blas2::symv( -1., jumpY, f, 1., dsTdsf);
    dg::blas1::pointwiseDot( m_weights_wo_vol, dsTdsf, dsTdsf); //make it symmetric
    if( m_no == dg::normed)
        dg::blas1::pointwiseDot( m_inv3d, dsTdsf, dsTdsf); //make it symmetric
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

