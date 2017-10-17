#pragma once

#include "dg/blas.h"
#include "dg/geometry/geometry.h"
#include "dg/backend/derivatives.h"
#include "fieldaligned.h"
#ifdef MPI_VERSION
#include "dg/backend/mpi_derivatives.h"
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
* @brief Class for the evaluation of parallel derivatives
*
* This class discretizes the operators 
\f$ \nabla_\parallel = \mathbf{v}\cdot \nabla = v^\zeta\partial_\zeta + v^\eta\partial_\eta + v^\varphi\partial_\varphi \f$, 
\f$\nabla_\parallel^\dagger\f$ and 
\f$\Delta_\parallel=\nabla_\parallel^\dagger\cdot\nabla_\parallel\f$
in arbitrary coordinates
* @ingroup fieldaligned
* @tparam ProductGeometry must be either dg::aProductGeometry3d or dg::aProductMPIGeometry3d or any derivative 
* @tparam IMatrix The type of the interpolation matrix 
    - dg::IHMatrix, or dg::IDMatrix, dg::MIHMatrix, or dg::MIDMatrix
* @tparam Matrix The matrix class of the jump matrix   
    - dg::HMatrix, or dg::DMatrix, dg::MHMatrix, or dg::MDMatrix
* @tparam container The container-class on which the interpolation matrix operates on
    - dg::HVec, or dg::DVec, dg::MHVec, or dg::MDVec
* @sa The pdf <a href="./parallel.pdf" target="_blank">parallel derivative</a> writeup 
*/
template< class ProductGeometry, class IMatrix, class Matrix, class container >
struct DS
{
    ///@brief No memory allocation; all member calls except construct are invalid
    DS(){}
    
    /**
     * @brief Create the magnetic unit vector field and construct
     * @copydetails DS(const dg::geo::BinaryVectorLvl0&,const ProductGeometry&,unsigned,unsigned,bool,bool,double,dg::norm,dg::direction)
     */
    DS(const dg::geo::TokamakMagneticField& vec, const ProductGeometry& grid, unsigned multiplyX=1, unsigned multiplyY=1, bool dependsOnX = true, bool dependsOnY=true, double eps = 1e-5, dg::norm no=dg::normed, dg::direction dir = dg::centered)
    {
        dg::geo::BinaryVectorLvl0 bhat( (dg::geo::BHatR)(vec), (dg::geo::BHatZ)(vec), (dg::geo::BHatP)(vec));
        m_fa.construct( bhat, grid, multiplyX, multiplyY, dependsOnX, dependsOnY, eps, grid.bcx(), grid.bcy(), FullLimiter());
        construct( m_fa, no, dir);
    }
    /**
     * @brief Create a Fieldaligned object and construct
     *
     * @param vec The vector field to integrate
     * @param grid The grid on which to operate defines the parallel boundary condition in case there is a limiter.
     * @param multiplyX defines the resolution in X of the fine grid relative to grid
     * @param multiplyY defines the resolution in Y of the fine grid relative to grid
     * @param dependsOnX indicates, whether the given vector field vec depends on the first coordinate
     * @param dependsOnY indicates, whether the given vector field vec depends on the second coordinate
     * @param eps Desired accuracy of the fieldline integrator
     * @param no indicate if the symv function should be symmetric (not_normed) or not
     * @param dir indicate the direction in the bracket operator and in symv
     *@note globalbcx and globalbcy  as well as bcz are taken from grid with full limter 
     * @sa Fieldaligned
     */
    DS(const dg::geo::BinaryVectorLvl0& vec, const ProductGeometry& grid, unsigned multiplyX=1, unsigned multiplyY=1, bool dependsOnX = true, bool dependsOnY=true, double eps = 1e-5, dg::norm no=dg::normed, dg::direction dir = dg::centered)
    {
        m_fa.construct( vec, grid, multiplyX, multiplyY, dependsOnX, dependsOnY, eps, grid.bcx(), grid.bcy(), FullLimiter());
        construct( m_fa, no, dir);
    }
    ///@copydoc construct
    DS(const dg::geo::Fieldaligned<ProductGeometry, IMatrix, container>& fa, dg::norm no=dg::normed, dg::direction dir = dg::centered)
    {
        construct( fa, no, dir);
    }
    /**
     * @brief Re-construct from a given Fieldaligned object
     *
     * @param fa this object will be used in all further member calls
     * @param no indicate if the symv function should be symmetric (not_normed) or not
     * @param dir indicate the direction in the bracket operator and in symv
     */
    void construct(const dg::geo::Fieldaligned<ProductGeometry, IMatrix, container>& fa, dg::norm no=dg::normed, dg::direction dir = dg::centered);

    /**
    * @brief forward derivative \f$ g_i = \alpha \frac{1}{h_z^+}(f_{i+1} - f_{i}) + \beta g_i\f$
    *
    * @param alpha Scalar
    * @param f The vector to derive
    * @param beta Scalar
    * @param g contains result on output (write only)
    * @note the vector sizes need to equal the grid size in the constructor
    */
    void forward( double alpha, const container& f, double beta, container& g){
        do_forward( alpha, f, beta, g);
    }
    ///@brief backward derivative \f$ g_i = \alpha \frac{1}{2h_z^-}(f_{i} - f_{i-1}) + \beta g_i \f$
    ///@copydetails forward
    void backward( double alpha, const container& f, double beta, container& g){
        do_backward( alpha, f, beta, g);
    }
    ///@brief centered derivative \f$ g_i = \alpha \frac{1}{2h_z}(f_{i+1} - f_{i-1}) + \beta g_i\f$
    ///@copydetails forward
    void centered( double alpha, const container& f, double beta, container& g){
        do_centered( alpha, f, beta, g);
    }

    ///@brief Apply the negative forward adjoint derivative on a 3d vector
    ///@copydetails forward
    void forwardAdj( double alpha, const container& f, double beta, container& g){
        do_forwardAdj( alpha, f, beta, g, dg::normed);
    }
    ///@brief Apply the negative backward adjoint derivative on a 3d vector
    ///@copydetails forward
    void backwardAdj( double alpha, const container& f, double beta, container& g){
        do_backwardAdj( alpha, f, beta, g, dg::normed);
    }
    ///@brief Apply the negative centered adjoint derivative on a 3d vector
    ///@copydetails forward
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
    void operator()( const container& f, container& g){operator()(1., f, 0., g);}
    void operator()(double alpha, const container& f, double beta, container& g);


    /**
     * @brief Discretizes \f$ \nabla\cdot ( \vec v \vec v \cdot \nabla . )\f$ as a symmetric matrix
     *
     * if direction is centered then centered followed by centeredAdj and adding jump terms
     * @param f The vector to derive
     * @param dsTdsf contains result on output (write only)
     * @note if dependsOnX is false then no jump terms will be added in the x-direction and similar in y
     */
    void symv( const container& f, container& dsTdsf){ do_symv( f, dsTdsf);}

    ///@copydoc FieldAligned::set_boundaries(dg::bc,double,double)
    void set_boundaries( dg::bc bcz, double left, double right)
    {
        m_fa.set_boundaries( bcz, left, right);
    }
    ///@copydoc FieldAligned::set_boundaries(dg::bc,const container&,const container&)
    void set_boundaries( dg::bc bcz, const container& left, const container& right)
    {
        m_fa.set_boundaries( bcz, left, right);
    }
    ///@copydoc FieldAligned::set_boundaries(dg::bc,const container&,double,double)
    void set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right)
    {
        m_fa.set_boundaries( bcz, global, scal_left, scal_right);
    }

    const container& weights()const {return m_vol3d;}
    const container& inv_weights()const {return m_inv3d;}
    const container& precond()const {return m_inv3d;}

    /**
    * @brief access the underlying Fielaligned object for evaluate
    *
    * @return acces to fieldaligned object
    */
    const Fieldaligned<ProductGeometry, IMatrix, container>& fieldaligned() const{return m_fa;}
    private:
    void do_forward(double alpha, const container& f, double beta, container& dsf);
    void do_backward(double alpha, const container& f, double beta, container& dsf);
    void do_centered(double alpha, const container& f, double beta, container& dsf);
    void do_forwardAdj(double alpha, const container& f, double beta, container& dsf, dg::norm no);
    void do_backwardAdj(double alpha, const container& f, double beta, container& dsf, dg::norm no);
    void do_centeredAdj(double alpha, const container& f, double beta, container& dsf, dg::norm no);
    void do_symv(const container& f, container& dsf);

    Fieldaligned<ProductGeometry, IMatrix, container> m_fa;
    container m_temp;
    container m_tempP, m_temp0, m_tempM;
    container m_vol3d, m_inv3d, m_weights_wo_vol;
    dg::norm m_no;
    dg::direction m_dir;
    Matrix m_jumpX, m_jumpY;
};

///@cond
////////////////////////////////////DEFINITIONS////////////////////////////////////////

template<class Geometry, class I, class M, class container>
void DS<Geometry, I, M,container>::construct(const Fieldaligned<Geometry, I, container>& fa, dg::norm no, dg::direction dir)
{
    m_fa=fa;
    m_no=no, m_dir=dir;

    dg::blas1::transfer( dg::create::volume(     fa.grid()), m_vol3d); 
    dg::blas1::transfer( dg::create::weights(    fa.grid()), m_weights_wo_vol); 
    dg::blas1::transfer( dg::create::inv_volume( fa.grid()), m_inv3d); 
    dg::blas2::transfer( dg::create::jumpX( fa.grid()), m_jumpX);
    dg::blas2::transfer( dg::create::jumpY( fa.grid()), m_jumpY);
    m_temp = m_vol3d, m_tempP = m_temp, m_temp0 = m_temp, m_tempM = m_temp;
}

template<class G, class I, class M, class container>
inline void DS<G,I,M,container>::operator()( double alpha, const container& f, double beta, container& dsf) { 
    if( m_dir == dg::centered)
        return centered( alpha, f, beta, dsf);
    else if( m_dir == dg::forward)
        return forward( alpha, f, beta, dsf);
    else
        return backward( alpha, f, beta, dsf);
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
void DS<G, I,M,container>::do_centeredAdj( double alpha, const container& f, double beta, container& dsf, dg::norm no)
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
        do_centered( 1., f, 0., m_tempP);
        do_centeredAdj( 1., m_tempP, 0., dsTdsf, dg::not_normed);
    }
    else 
    {
        do_forward( 1., f, 0., m_tempP);
        do_forwardAdj( 1., m_tempP, 0., m_temp0, dg::not_normed);
        do_backward( 1., f, 0., m_tempM);
        do_backwardAdj( 1., m_tempM, 0., dsTdsf, dg::not_normed);
        dg::blas1::axpby(0.5,m_temp0,0.5,dsTdsf);
    }
    dg::blas1::pointwiseDivide( dsTdsf, m_weights_wo_vol, dsTdsf);
    //     add jump term 
    if(m_fa.dependsOnX())
        dg::blas2::symv( -1., m_jumpX, f, 1., dsTdsf);
    if(m_fa.dependsOnY())
        dg::blas2::symv( -1., m_jumpY, f, 1., dsTdsf);
    dg::blas1::pointwiseDot( m_weights_wo_vol, dsTdsf, dsTdsf); //make it symmetric
    if( m_no == dg::normed)
        dg::blas1::pointwiseDot( m_inv3d, dsTdsf, dsTdsf); //make it symmetric
}
///@endcond


}//namespace geo

///@cond
template< class G, class I, class M, class V>
struct MatrixTraits< geo::DS<G,I,M, V> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond
}//namespace dg
