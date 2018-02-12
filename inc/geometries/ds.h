#pragma once

#include "dg/blas.h"
#include "dg/geometry/geometry.h"
#include "dg/geometry/derivatives.h"
#include "fieldaligned.h"
#ifdef MPI_VERSION
#include "dg/geometry/mpi_derivatives.h"
#include "mpi_fieldaligned.h"
#endif //MPI_VERSION
#include "magnetic_field.h"

/*!@file 
 *
 * This file includes the appropriate headers for parallel derivatives
 */

namespace dg{
namespace geo{

    /*!@class hide_ds_parameters2
    * @param f The vector to derive
    * @param g contains result on output (write only)
    * @note the vector sizes need to equal the grid size in the constructor
    */
    /*!@class hide_ds_parameters4
    * @param alpha Scalar
    * @param f The vector to derive
    * @param beta Scalar
    * @param g contains result on output (write only)
    * @note the vector sizes need to equal the grid size in the constructor
    */

/**
* @brief Class for the evaluation of parallel derivatives
*
* This class discretizes the operators 
\f$ \nabla_\parallel = \vec{v}\cdot \nabla = v^\zeta\partial_\zeta + v^\eta\partial_\eta + v^\varphi\partial_\varphi \f$, 
\f$\nabla_\parallel^\dagger = -\nabla\cdot(\vec v .)\f$ and 
\f$\Delta_\parallel=-\nabla_\parallel^\dagger\cdot\nabla_\parallel\f$
in arbitrary coordinates
@snippet ds_t.cu doxygen
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
    typedef dg::geo::Fieldaligned<ProductGeometry, IMatrix, container> FA; //!< conveniently abbreviates underlying \c Fieldaligned type
    ///@brief No memory allocation; all member calls except construct are invalid
    DS(){}
    
    /**
     * @brief Create the magnetic unit vector field and construct

     * @copydoc hide_fieldaligned_physics_parameters
     * @param no indicate if the symv function should be symmetric (not_normed) or not
     * @param dir indicate the direction in the bracket operator and in symv
     * @copydoc hide_fieldaligned_numerics_parameters
     * @sa \c Fieldaligned
     */
    template <class Limiter>
    DS(const dg::geo::TokamakMagneticField& vec, const ProductGeometry& grid, 
        dg::bc bcx = dg::NEU, 
        dg::bc bcy = dg::NEU, 
        Limiter limit = FullLimiter(), 
        dg::norm no=dg::normed, dg::direction dir = dg::centered, 
        double eps = 1e-5, unsigned multiplyX=10, unsigned multiplyY=10, bool dependsOnX = true, bool dependsOnY=true, bool integrateAll=true, double deltaPhi=-1)
    {
        dg::geo::BinaryVectorLvl0 bhat( (dg::geo::BHatR)(vec), (dg::geo::BHatZ)(vec), (dg::geo::BHatP)(vec));
        m_fa.construct( bhat, grid, bcx, bcy, limit, eps, multiplyX, multiplyY, dependsOnX, dependsOnY,integrateAll,deltaPhi);
        construct( m_fa, no, dir);
    }
    /**
     * @brief Use the given vector field to construct
     *
     * @copydoc hide_fieldaligned_physics_parameters
     * @param no indicate if the symv function should be symmetric (not_normed) or not
     * @param dir indicate the direction in the bracket operator and in symv
     * @copydoc hide_fieldaligned_numerics_parameters
     * @sa \c Fieldaligned
     */
    template<class Limiter>
    DS(const dg::geo::BinaryVectorLvl0& vec, const ProductGeometry& grid, 
        dg::bc bcx = dg::NEU, 
        dg::bc bcy = dg::NEU, 
        Limiter limit = FullLimiter(), 
        dg::norm no=dg::normed, dg::direction dir = dg::centered, 
        double eps = 1e-5, unsigned multiplyX=10, unsigned multiplyY=10, bool dependsOnX = true, bool dependsOnY=true, bool integrateAll=true, double deltaPhi=-1)
    {
        m_fa.construct( vec, grid, bcx, bcy, limit, eps, multiplyX, multiplyY, dependsOnX, dependsOnY, integrateAll,deltaPhi);
        construct( m_fa, no, dir);
    }
    ///@copydoc construct
    DS(const FA& fieldaligned, dg::norm no=dg::normed, dg::direction dir = dg::centered)
    {
        construct( fieldaligned, no, dir);
    }
    /**
     * @brief Re-construct from a given \c Fieldaligned object
     *
     * @param fieldaligned this object will be used in all further member calls
     * @param no indicate if the symv function should be symmetric (not_normed) or not
     * @param dir indicate the direction in the bracket operator and in symv
     */
    void construct(const FA& fieldaligned, dg::norm no=dg::normed, dg::direction dir = dg::centered);

    ///@copydoc Fieldaligned::set_boundaries(dg::bc,double,double)
    void set_boundaries( dg::bc bcz, double left, double right){
        m_fa.set_boundaries( bcz, left, right);
    }
    ///@copydoc Fieldaligned::set_boundaries(dg::bc,const container&,const container&)
    void set_boundaries( dg::bc bcz, const container& left, const container& right){
        m_fa.set_boundaries( bcz, left, right);
    }
    ///@copydoc Fieldaligned::set_boundaries(dg::bc,const container&,double,double)
    void set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right){
        m_fa.set_boundaries( bcz, global, scal_left, scal_right);
    }

    /**
    * @brief forward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
    *
    * forward derivative \f$ g_i = \alpha \frac{1}{h_z^+}(f_{i+1} - f_{i}) + \beta g_i\f$
    * @copydoc hide_ds_parameters4
    */
    void forward( double alpha, const container& f, double beta, container& g){
        do_forward( alpha, f, beta, g);
    }
    /**
    * @brief backward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
    *
    * backward derivative \f$ g_i = \alpha \frac{1}{2h_z^-}(f_{i} - f_{i-1}) + \beta g_i \f$
    * @copydoc hide_ds_parameters4
    */
    void backward( double alpha, const container& f, double beta, container& g){
        do_backward( alpha, f, beta, g);
    }
    /**
    * @brief centered derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
    *
    * centered derivative \f$ g_i = \alpha \frac{1}{2h_z^0}(f_{i+1} - f_{i-1}) + \beta g_i\f$
    * @copydoc hide_ds_parameters4
    */
    void centered( double alpha, const container& f, double beta, container& g){
        do_centered( alpha, f, beta, g);
    }
    /**
    * @brief backward derivative \f$ g = \vec v \cdot \nabla f \f$
    *
    * backward derivative \f$ g_i = \frac{1}{2h_z^-}(f_{i} - f_{i-1}) \f$
    * @copydoc hide_ds_parameters2
    */
    void backward( const container& f, container& g){
        do_backward(1.,f,0.,g);
    }
    /**
    * @brief forward derivative \f$ g = \vec v \cdot \nabla f \f$
    *
    * forward derivative \f$ g_i = \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @copydoc hide_ds_parameters2
    */
    void forward( const container& f, container& g){
        do_forward(1.,f,0.,g);
    }
    /**
    * @brief centered derivative \f$ g = \vec v \cdot \nabla f \f$
    *
    * centered derivative \f$ g_i = \frac{1}{2h_z^0}(f_{i+1} - f_{i-1})\f$
    * @copydoc hide_ds_parameters2
    */
    void centered( const container& f, container& g){
        do_centered(1.,f,0.,g);
    }

    ///@brief forward divergence \f$ g = \alpha \nabla\cdot(\vec v f) + \beta g\f$
    ///@copydoc hide_ds_parameters4
    ///@note forwardDiv is the negative adjoint of backward
    void forwardDiv( double alpha, const container& f, double beta, container& g){
        do_forwardDiv( alpha, f, beta, g, dg::normed);
    }
    ///@brief backward divergence \f$ g = \alpha \nabla\cdot(\vec v f) + \beta g\f$
    ///@copydoc hide_ds_parameters4
    ///@note backwardDiv is the negative adjoint of forward
    void backwardDiv( double alpha, const container& f, double beta, container& g){
        do_backwardDiv( alpha, f, beta, g, dg::normed);
    }
    ///@brief centered divergence \f$ g = \alpha \nabla\cdot(\vec v f) + \beta g\f$
    ///@copydoc hide_ds_parameters4
    ///@note centeredDiv is the negative adjoint of centered
    void centeredDiv(double alpha, const container& f, double beta, container& g){
        do_centeredDiv( alpha, f, beta, g, dg::normed);
    }
    ///@brief forward divergence \f$ g = \nabla\cdot(\vec v f)\f$
    ///@copydoc hide_ds_parameters2
    ///@note forwardDiv is the negative adjoint of backward
    void forwardDiv(const container& f, container& g){
        do_forwardDiv( 1.,f,0.,g, dg::normed);
    }
    ///@brief backward divergence \f$ g = \nabla\cdot(\vec v f)\f$
    ///@copydoc hide_ds_parameters2
    ///@note backwardDiv is the negative adjoint of forward
    void backwardDiv(const container& f, container& g){
        do_backwardDiv( 1.,f,0.,g, dg::normed);
    }
    ///@brief centered divergence \f$ g = \nabla\cdot(\vec v f) g\f$
    ///@copydoc hide_ds_parameters2
    ///@note centeredDiv is the negative adjoint of centered
    void centeredDiv(const container& f, container& g){
        do_centeredDiv( 1.,f,0.,g, dg::normed);
    }

    /**
    * @brief Discretizes \f$ g = \vec v\cdot \nabla f \f$
    *
    * dependent on dir given in constructor redirects to either \c forward(), \c backward() or \c centered()
    * @copydoc hide_ds_parameters2
    */
    void operator()( const container& f, container& g){operator()(1., f, 0., g);}
    /**
    * @brief Discretizes \f$ g = \alpha \vec v\cdot \nabla f + \beta g \f$
    *
    * dependent on dir given in constructor redirects to either \c forward(), \c backward() or \c centered()
    * @copydoc hide_ds_parameters4
    */
    void operator()(double alpha, const container& f, double beta, container& g);


    /**
     * @brief Discretizes \f$ g = \nabla\cdot ( \vec v \vec v \cdot \nabla f )\f$ as a symmetric matrix
     *
     * if direction given in constructor is centered then centered followed by centeredDiv and adding jump terms is called, else a symmetric forward/backward discretization is chosen.
     * @copydoc hide_ds_parameters2
     * @note if dependsOnX is false then no jump terms will be added in the x-direction; analogous in y
     */
    void symv( const container& f, container& g){ do_symv( 1., f, 0., g);}
    /**
     * @brief Discretizes \f$ g = \alpha \nabla\cdot ( \vec v \vec v \cdot \nabla f ) + \beta g\f$ as a symmetric matrix
     *
     * if direction given in constructor is centered then centered followed by centeredDiv and adding jump terms is called, else a symmetric forward/backward discretization is chosen.
     * @copydoc hide_ds_parameters4
     * @note if dependsOnX is false then no jump terms will be added in the x-direction; analogous in y
     */
    void symv( double alpha, const container& f, double beta, container& g){ do_symv( alpha, f, beta, g);}
    /**
     * @brief Discretizes \f$ g = (\vec v\cdot \nabla)^2 f \f$ 
     *
     * The formula used is \f[ \nabla_\parallel^2 f = 2\left(\frac{f^+}{h_z^+h_z^0} - \frac{f^0}{h_z^+h_z^-} + \frac{f^-}{h_z^-h_z^0}\right) \f]
     * @copydoc hide_ds_parameters2
     */
    void dss( const container& f, container& g){ do_dss( 1., f, 0., g);}
    /**
     * @brief Discretizes \f$ g = \alpha (\vec v\cdot \nabla)^2 f + \beta g \f$ 
     *
     * The formula used is \f[ \nabla_\parallel^2 f = 2\left(\frac{f^+}{h_z^+h_z^0} - \frac{f^0}{h_z^+h_z^-} + \frac{f^-}{h_z^-h_z^0}\right) \f]
     * @copydoc hide_ds_parameters4
     */
    void dss( double alpha, const container& f, double beta, container& g){ do_symv( alpha, f, beta, g);}

    const container& weights()const {return m_vol3d;}
    const container& inv_weights()const {return m_inv3d;}
    const container& precond()const {return m_inv3d;}

    /**
    * @brief access the underlying Fieldaligned object for evaluate
    *
    * @return acces to fieldaligned object
    */
    const FA& fieldaligned() const{return m_fa;}
    private:
    void do_forward(double alpha, const container& f, double beta, container& dsf);
    void do_backward(double alpha, const container& f, double beta, container& dsf);
    void do_centered(double alpha, const container& f, double beta, container& dsf);
    void do_forwardDiv(double alpha, const container& f, double beta, container& dsf, dg::norm no);
    void do_backwardDiv(double alpha, const container& f, double beta, container& dsf, dg::norm no);
    void do_centeredDiv(double alpha, const container& f, double beta, container& dsf, dg::norm no);
    void do_symv(double alpha, const container& f, double beta, container& dsf);
    void do_dss(double alpha, const container& f, double beta, container& dsf);

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
    dg::blas1::axpby( 1., f, -1., m_tempM, m_tempM);
    dg::blas1::pointwiseDot( alpha, m_tempM, m_fa.hm_inv(), beta, dsf);
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
void DS<G,I,M,container>::do_backwardDiv( double alpha, const container& f, double beta, container& dsf, dg::norm no)
{    
    //adjoint discretisation
    dg::blas1::pointwiseDot( 1., m_vol3d, f, m_fa.hp_inv(), 0., m_temp0);
    m_fa(einsPlusT, m_temp0, m_tempP);
    if(no == dg::normed) 
    {
        dg::blas1::axpby( 1., m_temp0, -1., m_tempP, m_temp0);
        dg::blas1::pointwiseDot( alpha, m_inv3d, m_temp0, beta, dsf); 
    }
    else
        dg::blas1::axpbypgz( alpha, m_temp0, -alpha, m_tempP, beta, dsf);
}
template<class G,class I, class M, class container>
void DS<G,I,M,container>::do_forwardDiv( double alpha, const container& f, double beta, container& dsf, dg::norm no)
{    
    //adjoint discretisation
    dg::blas1::pointwiseDot( 1., m_vol3d, f, m_fa.hm_inv(),0., m_temp0);
    m_fa(einsMinusT, m_temp0, m_tempM);
    if(no == dg::normed) 
    {
        dg::blas1::axpby( 1., m_tempM, -1., m_temp0, m_temp0);
        dg::blas1::pointwiseDot( alpha, m_inv3d, m_temp0, beta, dsf); 
    }
    else
        dg::blas1::axpbypgz( alpha, m_tempM, -alpha, m_temp0, beta, dsf);
}
template<class G, class I, class M, class container>
void DS<G, I,M,container>::do_centeredDiv( double alpha, const container& f, double beta, container& dsf, dg::norm no)
{               
    //adjoint discretisation
    dg::blas1::pointwiseDot( 1., m_vol3d, f, m_fa.hz_inv(), 0.,m_temp0);
    m_fa(einsPlusT,  m_temp0, m_tempP);
    m_fa(einsMinusT, m_temp0, m_tempM);
    if(no == dg::normed) 
    {
        dg::blas1::axpby( 1., m_tempM, -1., m_tempP);
        dg::blas1::pointwiseDot( alpha, m_inv3d, m_tempP, beta, dsf); 
    }
    else
        dg::blas1::axpbypgz( alpha, m_tempM, -alpha, m_tempP, beta, dsf);

}

template<class G,class I, class M, class container>
void DS<G,I,M,container>::do_symv( double alpha, const container& f, double beta, container& dsTdsf)
{
    if(m_dir == dg::centered)
    {
        do_centered( 1., f, 0., m_tempP);
        do_centeredDiv( 1., m_tempP, 0., m_temp, dg::not_normed);
    }
    else 
    {
        do_forward( 1., f, 0., m_tempP);
        do_backwardDiv( 1., m_tempP, 0., m_temp, dg::not_normed);
        do_backward( 1., f, 0., m_tempM);
        do_forwardDiv( 0.5, m_tempM, 0.5, m_temp, dg::not_normed);
    }
    dg::blas1::pointwiseDivide( m_temp, m_weights_wo_vol, m_temp);
    //     add jump term 
    if(m_fa.dependsOnX())
        dg::blas2::symv( -1., m_jumpX, f, 1., m_temp);
    if(m_fa.dependsOnY())
        dg::blas2::symv( -1., m_jumpY, f, 1., m_temp);

    if( m_no == dg::normed)
        dg::blas1::pointwiseDot( alpha, m_inv3d, m_weights_wo_vol, m_temp, beta, dsTdsf);
    else
        dg::blas1::pointwiseDot( alpha, m_weights_wo_vol, m_temp, beta, dsTdsf);
}

template<class G,class I, class M, class container>
void DS<G,I,M,container>::do_dss( double alpha, const container& f, double beta, container& dssf)
{
    m_fa(einsPlus,  f, m_tempP);
    m_fa(einsMinus, f, m_tempM);
    dg::blas1::pointwiseDot( 1., m_tempP, m_fa.hp_inv(), 1., m_tempM, m_fa.hm_inv(), 0., m_tempM);
    dg::blas1::pointwiseDot( -2.*alpha, f,  m_fa.hp_inv(), m_fa.hm_inv(), beta, dssf);
    dg::blas1::pointwiseDot( 2.*alpha, m_fa.hz_inv(), m_tempM, 1., dssf);

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
