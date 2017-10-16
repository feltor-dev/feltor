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
namespace geo{

/**
* @brief Class for the evaluation of a parallel derivative
*
* This class discretizes the operators 
\f$ \nabla_\parallel = \mathbf{b}\cdot \nabla = b_R\partial_R + b_Z\partial_Z + b_\phi\partial_\phi \f$, 
\f$\nabla_\parallel^\dagger\f$ and 
\f$\Delta_\parallel=\nabla_\parallel^\dagger\cdot\nabla_\parallel\f$
in arbitrary coordinates
* @ingroup fieldaligned
* @tparam ProductGeometry must be either aProductGeometry3d or aProductMPIGeometry3d or any derivative 
* @tparam IMatrix The type of the interpolation matrix 
    -dg::IHMatrix, or dg::IDMatrix, dg::MIHMatrix, or dg::MIDMatrix
* @tparam Matrix The matrix class of the jump matrix   
    -dg::HMatrix, or dg::DMatrix, dg::MHMatrix, or dg::MDMatrix
* @tparam container The container-class on which the interpolation matrix operates on
    -dg::HVec, or dg::DVec, dg::MHVec, or dg::MDVec
*/
template< class ProductGeometry, class IMatrix, class Matrix, class container >
struct DS
{
    DS(){}
    DS(const dg::geo::TokamakMagneticField& mag, const ProductGeometry& g, unsigned mx=1, unsigned my=1, double eps = 1e-5, dg::norm no=dg::normed, dg::direction dir = dg::centered, bool dependsOnX = true, bool dependsOnY=true)
    {
        dg::geo::BinaryVectorLvl0 vec( dg::geo::BHatR(mag), dg::geo::BHatZ(mag), dg::geo::BHatP(mag));
        m_fa.construct( vec, grid, mx, my, eps, FullLimiter(), grid.bcx(), grid.bcy());
        construct( m_fa, g, no, dir, dependsOnX, dependsOnY);
    }
    DS(const dg::geo::BinaryVectorLvl0& vec, const ProductGeometry& g, unsigned mx=1, unsigned my=1, double eps = 1e-5, dg::norm no=dg::normed, dg::direction dir = dg::centered, bool dependsOnX = true, bool dependsOnY=true)
    {
        m_fa.construct( vec, grid, mx, my, eps, FullLimiter(), grid.bcx(), grid.bcy());
        construct( m_fa, g, no, dir, dependsOnX, dependsOnY);
    }
    DS(const dg::geo::Fieldaligned<ProductGeometry, I, M, container>& fa, const ProductGeometry& g, dg::norm no=dg::normed, dg::direction dir = dg::centered, bool dependsOnX = true, bool dependsOnY=true)
    {
        construct( fa, g, no, dir, dependsOnX, dependsOnY);
    }
    void construct(const dg::geo::Fieldaligned<ProductGeometry, I, M, container>& fa, const ProductGeometry& g, dg::norm no=dg::normed, dg::direction dir = dg::centered, bool dependsOnX = true, bool dependsOnY=true);

    /**
    * @brief Apply the forward derivative on a 3d vector
    *
    * forward derivative \f$ g_i = \alpha \frac{1}{h_z^+}(f_{i+1} - f_{i}) + \beta g_i\f$
    * @param alpha Scalar
    * @param f The vector to derive
    * @param beta Scalar
    * @param g contains result on output (write only)
    */
    void forward( double alpha, const container& f, double beta, container& g){
        do_forward( alpha, f, beta, g);
    }
    /**
    * @brief Apply the backward derivative on a 3d vector
    *
    * backward derivative \f$ g_i = \alpha \frac{1}{2h_z^-}(f_{i} - f_{i-1}) + \beta g_i \f$
    * @param alpha Scalar
    * @param f The vector to derive
    * @param beta Scalar
    * @param g contains result on output (write only)
    */
    void backward( double alpha, const container& f, double beta, container& g){
        do_backward( alpha, f, beta, g);
    }
    /**
    * @brief Apply the centered derivative on a 3d vector
    *
    * centered derivative \f$ g_i = \alpha \frac{1}{2h_z}(f_{i+1} - f_{i-1}) + \beta g_i\f$
    * @param alpha Scalar
    * @param f The vector to derive
    * @param beta Scalar
    * @param g contains result on output (write only)
    */
    void centered( double alpha, const container& f, double beta, container& g){
        do_centered( alpha, f, beta, g);
    }

    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * @param alpha Scalar
    * @param f The vector to derive
    * @param beta Scalar
    * @param g contains result on output (write only)
    */
    void forwardAdj( double alpha, const container& f, double beta, container& g){
        do_forwardAdj( alpha, f, beta, g, dg::normed);
    }
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * @param alpha Scalar
    * @param f The vector to derive
    * @param beta Scalar
    * @param g contains result on output (write only)
    */
    void backwardAdj( double alpha, const container& f, double beta, container& g){
        do_backwardAdj( alpha, f, beta, g, dg::normed);
    }
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * @param alpha Scalar
    * @param f The vector to derive
    * @param beta Scalar
    * @param g contains result on output (write only)
    */
    void centeredAdj(double alpha, const container& f, double beta, container& g){
        do_centeredAdj( alpha, f, beta, g, dg::normed);
    }

    /**
    * @brief compute parallel derivative
    *
    * dependent on dir redirects to either forward(), backward() or centered()
    * @param f The vector to derive
    * @param g contains result on output (write only)
    */
    void operator()( const container& f, container& g);


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
    const dg::geo::FieldAligned<ProductGeometry, IMatrix, container>& fieldaligned() const{return f_;}
    private:
    void do_forward(double alpha, const container& f, double beta, container& dsf);
    void do_backward(double alpha, const container& f, double beta, container& dsf);
    void do_centered(double alpha, const container& f, double beta, container& dsf);
    void do_forwardAdj(double alpha, const container& f, double beta, container& dsf, dg::norm no);
    void do_backwardAdj(double alpha, const container& f, double beta, container& dsf, dg::norm no);
    void do_centeredAdj(double alpha, const container& f, double beta, container& dsf, dg::norm no);
    void do_symv(const container& f, container& dsf);

    FieldAligned<ProductGeometry,IMatrix,container> m_fa;
    Matrix m_jumpX, m_jumpY;
    container m_temp;
    container m_tempP, m_temp0, m_tempM;
    container m_vol3d, m_inv3d, m_weights_wo_vol;
    dg::norm m_no;
    dg::direction m_dir;
    bool m_apply_jumpX, m_apply_jumpY;
};

///@cond
////////////////////////////////////DEFINITIONS////////////////////////////////////////

template<class Geometry, class I, class M, class container>
void DS<Geometry, I, M,container>::construct(const Fieldaligned<Geometry, I, M, container>& fa, const Geometry& grid, dg::norm no, dg::direction dir, bool jumpX, bool jumpY)
{
    m_fa=fa;
    m_no=no, m_dir=dir, m_apply_jumpX=jumpX, m_apply_jumpY=jumpY;

    dg::blas1::transfer( dg::create::volume(     grid), m_vol3d); 
    dg::blas1::transfer( dg::create::weights(    grid), m_weights_wo_vol); 
    dg::blas1::transfer( dg::create::inv_volume( grid), m_inv3d); 
    dg::blas2::transfer( dg::create::jumpX( grid), jumpX);
    dg::blas2::transfer( dg::create::jumpY( grid), jumpY);
    m_temp = m_vol3d, m_tempP = m_temp, m_temp0 = m_temp, m_tempM = m_temp;
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

}//namespace geo
}//namespace dg

