#pragma once

#include "dg/blas.h"
#include "dg/topology/geometry.h"
#include "dg/topology/derivatives.h"
#include "fieldaligned.h"
#ifdef MPI_VERSION
#include "dg/topology/mpi_derivatives.h"
#include "mpi_fieldaligned.h"
#endif //MPI_VERSION
#include "magnetic_field.h"

/*!@file
 *
 * This file includes the appropriate headers for parallel derivatives
 */

namespace dg{
namespace geo{
///@cond
namespace detail{

struct ComputeSymv{
    DG_DEVICE
    void operator()( double& fp, double fm, double hp, double hm,  double vol3d) const{
        fp = ( fp-fm)/(hp+hm);
        fp = vol3d*fp/(hp+hm);
    }
    DG_DEVICE
    void operator()( double& fp, double& fm, double f0, double hp, double hm, double vol3d) const{
        fp = ( fp-f0)/hp;
        fp = 0.5*vol3d*fp/hp;
        fm = ( f0-fm)/hm;
        fm = 0.5*vol3d*fm/hm;
    }
};
struct ComputeSymvEnd{
    DG_DEVICE
    void operator()( double& fm, double fp, double weights) const{
        fm = ( fp-fm)/weights;
    }
    DG_DEVICE
    void operator()( double& efm, double fm, double fp, double efp, double weights) const{
        efm = ( efm- fm + fp -efp)/weights;
    }
};

struct ComputeDSForward{
    ComputeDSForward( double alpha, double beta):m_alpha(alpha), m_beta(beta){}
    DG_DEVICE
    void operator()( double& dsf, double fo, double fp,
            double hp)
    {
        dsf = m_alpha*( fp - fo)/hp
            + m_beta*dsf;
    }
    DG_DEVICE
    void operator()( double& dsf, double fo, double fp, double fpp,
            double hp)
    {
        dsf = m_alpha*( -3.*fo + 4.*fp - fpp)/2./hp
            + m_beta*dsf;
    }
    private:
    double m_alpha, m_beta;
};
struct ComputeDSBackward{
    ComputeDSBackward( double alpha, double beta):m_alpha(alpha), m_beta(beta){}
    DG_DEVICE
    void operator()( double& dsf, double fo, double fm,
            double hm)
    {
        dsf = m_alpha*( fo - fm)/hm
            + m_beta*dsf;
    }
    DG_DEVICE
    void operator()( double& dsf, double fo, double fm, double fmm,
            double hm)
    {
        dsf = m_alpha*( 3.*fo - 4.*fm + fmm)/2./hm
            + m_beta*dsf;
    }
    private:
    double m_alpha, m_beta;
};

struct ComputeDSCentered{

    ComputeDSCentered( double alpha, double beta):m_alpha(alpha), m_beta(beta){}
    DG_DEVICE
    void operator()( double& dsf, double fm, double fo, double fp,
            double hm, double hp)
    {
        dsf = m_alpha*( fm*( 1./(hp+hm) - 1/hm)
            + fo*( 1./hm - 1./hp)
            + fp*( 1./hp - 1/(hp+hm))
            ) + m_beta*dsf;
    }
    private:
    double m_alpha, m_beta;
};
struct ComputeDSCenteredNEU{
    ComputeDSCenteredNEU( double alpha, double beta, std::array<double,2> b_value):m_alpha(alpha), m_beta(beta), m_bm(b_value[0]), m_bp( b_value[1]),  m_ds(1.,0.){}
    DG_DEVICE
    void operator()( double& dsf, double fm, double fo, double fp,
            double hm, double hp, double hbm, double hbp,
            double bpm, double bpo, double bpp)
    {
        double inner=0, plus=0, minus=0, both=0;
        m_ds( inner, fm, fo, fp, hm, hp);
        plus  = ( 1./hm - 1./( 2.*hbp + hm))*(fo-fm) + m_bp * hm /(2.*hbp + hm);
        minus = ( 1./hp - 1./( 2.*hbm + hp))*(fp-fm) + m_bm * hp /(2.*hbm + hp);
        both = (m_bp*hbm+hbp*m_bm)/(hbp+hbm);
        dsf = m_alpha*((1.-bpm-bpo-bpp)*inner + bpp*plus + bpm*minus + bpo*both)
            + m_beta*dsf;
    }
    private:
    double m_alpha, m_beta;
    double m_bm, m_bp;
    ComputeDSCentered m_ds;
};
struct ComputeDSCenteredDIR{
    ComputeDSCenteredDIR( double alpha, double beta, std::array<double,2> b_value):m_alpha(alpha), m_beta(beta), m_bm(b_value[0]), m_bp( b_value[1]),  m_ds(1.,0.){}
    DG_DEVICE
    void operator()( double& dsf, double fm, double fo, double fp,
            double hm, double hp, double hbm, double hbp,
            double bpm, double bpo, double bpp)
    {
        double inner=0, plus=0, minus=0, both=0;
        m_ds( inner, fm,   fo, fp,   hm,  hp);
        m_ds( plus,  fm,   fo, m_bp, hm,  hbp);
        m_ds( minus, m_bm, fo, fp,   hbm, hp);
        m_ds( both,  m_bm, fo, m_bp, hbm, hbp);
        dsf = m_alpha*((1.-bpm-bpo-bpp)*inner + bpp*plus + bpm*minus + bpo*both)
            + m_beta*dsf;
    }
    private:
    double m_alpha, m_beta;
    double m_bm, m_bp;
    ComputeDSCentered m_ds;
};
struct ComputeDSS{
    ComputeDSS( double alpha, double beta):m_alpha(alpha), m_beta(beta){}
    DG_DEVICE
    void operator()( double& dssf, double fm, double fo, double fp,
            double hm, double hp)
    {
        dssf = m_alpha*(
               2.*fm/(hp+hm)/hm
             - 2.*fo/hp/hm
             + 2.*fp/(hp+hm)/hp) + m_beta*dssf;
    }
    private:
    double m_alpha, m_beta;
};
struct ComputeDSSNEU{
    ComputeDSSNEU( double alpha, double beta, std::array<double,2> b_value):m_alpha(alpha), m_beta(beta), m_bm(b_value[0]), m_bp( b_value[1]),  m_dss(1.,0.){}
    DG_DEVICE
    void operator()( double& dssf, double fm, double fo, double fp,
            double hm, double hp, double hbm, double hbp,
            double bpm, double bpo, double bpp)
    {
        double inner=0, plus=0, minus=0, both=0;
        m_dss( inner, fm, fo, fp, hm, hp);
        plus  =  2./( 2.*hbp + hm)*( m_bp - (fo-fm)/hm );
        minus =  2./( 2.*hbm + hp)*( (fp-fo)/hp - m_bm );
        both = (m_bp-m_bm)/(hbp+hbm);
        dssf = m_alpha*((1.-bpm-bpo-bpp)*inner + bpp*plus + bpm*minus + bpo*both)
            + m_beta*dssf;
    }
    private:
    double m_alpha, m_beta;
    double m_bm, m_bp;
    ComputeDSS m_dss;
};

struct ComputeDSSDIR{
    ComputeDSSDIR( double alpha, double beta, std::array<double,2> b_value):m_alpha(alpha), m_beta(beta), m_bm(b_value[0]), m_bp( b_value[1]),  m_dss(1.,0.){}
    DG_DEVICE
    void operator()( double& dssf, double fm, double fo, double fp,
            double hm, double hp, double hbm, double hbp,
            double bpm, double bpo, double bpp)
    {
        double inner=0, plus=0, minus=0, both=0;
        m_dss( inner, fm, fo, fp, hm, hp);
        m_dss( plus, fm, fo, m_bp, hm, hbp);
        m_dss( minus, m_bm, fo, fp, hbm, hp);
        m_dss( both, m_bm, fo, m_bp, hbm, hbp);
        dssf = m_alpha*((1.-bpm-bpo-bpp)*inner + bpp*plus + bpm*minus + bpo*both)
            +m_beta*dssf;
    }
    private:
    double m_alpha, m_beta;
    double m_bm, m_bp;
    ComputeDSS m_dss;
};

}//namespace detail
///@endcond

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
/*!@class hide_ds_fm
 * @param fm fieldaligned(einsMinus, f, fm)
 */
/*!@class hide_ds_fp
 * @param fp fieldaligned(einsPlus, f, fp)
 */
/*!@class hide_ds_dir
 * @param dir indicate the direction in the bracket operator and in symv
 */
/*!@class hide_ds_freestanding
 * @note This function computes the same as the corresponding member function of DS
 * and you have to compute the einsPlus and einsMinus interpolations from dg::Fieldaligned yourself. The reasoning for this function is that you can re-use the latter interpolations if you compute for example both first and second derivative of a function.
 */


/*!@class hide_ds_attention
@attention The \c div and \c symv member functions reliably converge only if fieldlines
do not(!) intersect the boundary and then only if the \c mx and \c my
    parameters are sufficiently high
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
 * @note The parallel Laplacian cannot be inverted as long as there are
 * closed fieldlines somewhere in the domain (which is virtually always true). There is therefore no norm parameter in the class.
@attention The \c div and \c symv member functions reliably work only if fieldlines
do not(!) intersect the boundary and then only if the \c mx and \c my
    parameters are sufficiently high
* @ingroup fieldaligned
* @tparam ProductGeometry must be either \c dg::aProductGeometry3d or \c dg::aProductMPIGeometry3d or any derivative
* @tparam IMatrix The type of the interpolation matrix
    - \c dg::IHMatrix, or \c dg::IDMatrix, \c dg::MIHMatrix, or \c dg::MIDMatrix
* @tparam Matrix The matrix class of the jump matrix
    - \c dg::HMatrix, or \c dg::DMatrix, \c dg::MHMatrix, or \c dg::MDMatrix
* @tparam container The container-class on which the interpolation matrix operates on
    - \c dg::HVec, or \c dg::DVec, \c dg::MHVec, or \c dg::MDVec
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
     * @copydoc hide_ds_dir
     * @copydoc hide_fieldaligned_numerics_parameters
     * @sa \c Fieldaligned
     */
    template <class Limiter>
    DS(const dg::geo::TokamakMagneticField& vec, const ProductGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        dg::direction dir = dg::centered,
        double eps = 1e-5,
        unsigned mx=10, unsigned my=10,
        double deltaPhi=-1):
        DS( FA( vec, grid, bcx, bcy, limit, eps, mx, my, deltaPhi), dir )
    {
    }
    /**
     * @brief Use the given vector field to construct
     *
     * @copydoc hide_fieldaligned_physics_parameters
     * @copydoc hide_ds_dir
     * @copydoc hide_fieldaligned_numerics_parameters
     * @sa \c Fieldaligned
     */
    template<class Limiter>
    DS(const dg::geo::CylindricalVectorLvl0& vec, const ProductGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        dg::direction dir = dg::centered,
        double eps = 1e-5,
        unsigned mx=10, unsigned my=10,
        double deltaPhi=-1):
        DS( FA( vec, grid, bcx, bcy, limit, eps, mx, my, deltaPhi), dir)
    {
    }
    /**
     * @brief Re-construct from a given \c Fieldaligned object
     *
     * @param fieldaligned this object will be used in all further member calls
     * @copydoc hide_ds_dir
     */
    DS( FA fieldaligned, dg::direction dir = dg::centered);
    /**
    * @brief Perfect forward parameters to one of the constructors
    * @tparam Params deduced by the compiler
    * @param ps parameters forwarded to constructors
    */
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = DS( std::forward<Params>( ps)...);
    }

    /*!@brief Set the direction used in the operator() and symv member
     * @param dir new direction
     */
    void set_direction( dg::direction dir){
        m_dir = dir;
    }

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
        m_fa(einsPlus, f, m_tempP);
        ds_forward( m_fa, alpha, f, m_tempP, beta, g);
    }
    /**
    * @brief backward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
    *
    * backward derivative \f$ g_i = \alpha \frac{1}{h_z^-}(f_{i} - f_{i-1}) + \beta g_i \f$
    * @copydoc hide_ds_parameters4
    */
    void backward( double alpha, const container& f, double beta, container& g){
        m_fa(einsMinus, f, m_tempM);
        ds_backward( m_fa, alpha, m_tempM, f, beta, g);
    }
    /**
    * @brief centered derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
    *
    * The centered derivative is constructed by fitting a polynomial through the plus point the minus point and the center point and evaluating its derivative at the center point. For the exact resulting formula consult the <a href="./parallel.pdf" target="_blank">parallel derivative</a> writeup
    * @copydoc hide_ds_parameters4
    */
    void centered( double alpha, const container& f, double beta, container& g){
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        ds_centered( m_fa, alpha, m_tempM, f, m_tempP, beta, g);
    }
    /**
    * @brief backward derivative \f$ g = \vec v \cdot \nabla f \f$
    *
    * backward derivative \f$ g_i = \frac{1}{h_z^-}(f_{i} - f_{i-1}) \f$
    * @copydoc hide_ds_parameters2
    */
    void backward( const container& f, container& g){
        backward(1., f,0.,g);
    }
    /**
    * @brief forward derivative \f$ g = \vec v \cdot \nabla f \f$
    *
    * forward derivative \f$ g_i = \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @copydoc hide_ds_parameters2
    */
    void forward( const container& f, container& g){
        forward(1.,f, 0.,g);
    }
    /**
    * @brief centered derivative \f$ g = \vec v \cdot \nabla f \f$
    *
    * The centered derivative is constructed by fitting a polynomial through the plus point the minus point and the center point and evaluating its derivative at the center point. For the exact resulting formula consult the <a href="./parallel.pdf" target="_blank">parallel derivative</a> writeup
    * @copydoc hide_ds_parameters2
    */
    void centered( const container& f, container& g){
        centered(1.,f,0.,g);
    }

    ///@brief forward divergence \f$ g = \alpha \nabla\cdot(\vec v f) + \beta g\f$
    ///@copydoc hide_ds_parameters4
    ///@copydoc hide_ds_attention
    ///@note divForward is the negative adjoint of backward
    void divForward( double alpha, const container& f, double beta, container& g){
        do_divForward( alpha, f, beta, g);
    }
    ///@brief backward divergence \f$ g = \alpha \nabla\cdot(\vec v f) + \beta g\f$
    ///@copydoc hide_ds_parameters4
    ///@copydoc hide_ds_attention
    ///@note divBackward is the negative adjoint of forward
    void divBackward( double alpha, const container& f, double beta, container& g){
        do_divBackward( alpha, f, beta, g);
    }
    ///@brief centered divergence \f$ g = \alpha \nabla\cdot(\vec v f) + \beta g\f$
    ///@copydoc hide_ds_parameters4
    ///@copydoc hide_ds_attention
    ///@note divCentered is the negative adjoint of centered
    void divCentered(double alpha, const container& f, double beta, container& g){
        do_divCentered( alpha, f, beta, g);
    }
    ///@brief forward divergence \f$ g = \nabla\cdot(\vec v f)\f$
    ///@copydoc hide_ds_parameters2
    ///@copydoc hide_ds_attention
    ///@note divForward is the negative adjoint of backward
    void divForward(const container& f, container& g){
        do_divForward( 1.,f,0.,g);
    }
    ///@brief backward divergence \f$ g = \nabla\cdot(\vec v f)\f$
    ///@copydoc hide_ds_parameters2
    ///@copydoc hide_ds_attention
    ///@note divBackward is the negative adjoint of forward
    void divBackward(const container& f, container& g){
        do_divBackward( 1.,f,0.,g);
    }
    ///@brief centered divergence \f$ g = \nabla\cdot(\vec v f)\f$
    ///@copydoc hide_ds_parameters2
    ///@copydoc hide_ds_attention
    ///@note divCentered is the negative adjoint of centered
    void divCentered(const container& f, container& g){
        do_divCentered( 1.,f,0.,g);
    }

    /**
    * @brief Discretizes \f$ g = \vec v\cdot \nabla f \f$
    *
    * dependent on dir given in constructor redirects to either \c forward(), \c backward() or \c centered()
    * @copydoc hide_ds_parameters2
    */
    void operator()( const container& f, container& g){
        ds( m_dir, 1., f, 0., g);
    }
    /**
    * @brief Discretizes \f$ g = \alpha \vec v\cdot \nabla f + \beta g \f$
    *
    * dependent on dir given in constructor redirects to either \c forward(), \c backward() or \c centered()
    * @copydoc hide_ds_parameters4
    */
    void operator()(double alpha, const container& f, double beta, container& g){
        ds(m_dir, alpha, f, beta, g);
    }


    /**
    * @brief Discretizes \f$ g = \vec v\cdot \nabla f \f$
    *
    * @param dir redirects to either \c forward(), \c backward() or \c centered()
    * @copydoc hide_ds_parameters2
    */
    void ds(dg::direction dir,  const container& f, container& g){
        ds(dir, 1., f, 0., g);
    }
    /**
    * @brief Discretizes \f$ g = \alpha \vec v\cdot \nabla f + \beta g \f$
    *
    * @param dir redirects to either \c forward(), \c backward() or \c centered()
    * @copydoc hide_ds_parameters4
    */
    void ds(dg::direction dir, double alpha, const container& f, double beta, container& g);
    /**
    * @brief Discretizes \f$ g = \nabla \cdot ( \vec v f) \f$
    *
    * @param dir redirects to either \c divForward(), \c divBackward() or \c divCentered()
    * @copydoc hide_ds_parameters2
     * @copydoc hide_ds_attention
    */
    void div(dg::direction dir,  const container& f, container& g){
        div(dir, 1., f, 0., g);
    }
    /**
    * @brief Discretizes \f$ g = \alpha \nabla \cdot( \vec v f) + \beta g \f$
    *
    * @param dir redirects to either \c divForward(), \c divBackward() or \c divCentered()
    * @copydoc hide_ds_parameters4
     * @copydoc hide_ds_attention
    */
    void div(dg::direction dir, double alpha, const container& f, double beta, container& g);

    /**
     * @brief Discretizes \f$ g = \nabla\cdot ( \vec v \vec v \cdot \nabla f )\f$
     *
     * If direction given in constructor is centered then centered followed by divCentered and adding jump terms is called, else a symmetric forward/backward discretization is chosen.
     * @copydoc hide_ds_parameters2
     * @copydoc hide_ds_attention
     */
    void symv( const container& f, container& g){ do_symv( 1., f, 0., g);}
    /**
     * @brief Discretizes \f$ g = \alpha \nabla\cdot ( \vec v \vec v \cdot \nabla f ) + \beta g\f$ as a symmetric matrix
     *
     * if direction given in constructor is centered then centered followed by divCentered and adding jump terms is called, else a symmetric forward/backward discretization is chosen.
     * @copydoc hide_ds_parameters4
     * @copydoc hide_ds_attention
     */
    void symv( double alpha, const container& f, double beta, container& g){ do_symv( alpha, f, beta, g);}
    /**
     * @brief Discretizes \f$ g = (\vec v\cdot \nabla)^2 f \f$
     *
     * The formula used is \f[ \nabla_\parallel^2 f = 2\left(\frac{f^+}{h_z^+ h_z^0} - \frac{f^0}{h_z^- h_z^+} + \frac{f^-}{h_z^-h_z^0}\right) \f]
     * which is the second derivative of a 2nd order polynomial fitted through the plus, minus and centre points
     * @copydoc hide_ds_parameters2
     */
    void dss( const container& f, container& g){
        dss( 1., f, 0., g);}
    /**
     * @brief Discretizes \f$ g = \alpha (\vec v\cdot \nabla)^2 f + \beta g \f$
     *
     * The formula used is \f[ \nabla_\parallel^2 f = 2\left(\frac{f^+}{h_z^+ h_z^0} - \frac{f^0}{h_z^- h_z^+} + \frac{f^-}{h_z^-h_z^0}\right) \f]
     * which is the second derivative of a 2nd order polynomial fitted through the plus, minus and centre points
     * @copydoc hide_ds_parameters4
     */
    void dss( double alpha, const container& f, double beta, container& g){
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        dss_centered( m_fa, alpha, m_tempM, f, m_tempP, beta, g);
    }

    const container& weights()const {
        return m_vol3d;
    }
    const container& inv_weights()const {
        return m_inv3d;
    }
    const container& precond()const {
        return m_inv3d;
    }

    /**
    * @brief access the underlying Fieldaligned object
    *
    * @return acces to Fieldaligned object
    */
    const FA& fieldaligned() const{return m_fa;}
    private:
    void do_divForward(double alpha, const container& f, double beta, container& dsf);
    void do_divBackward(double alpha, const container& f, double beta, container& dsf);
    void do_divCentered(double alpha, const container& f, double beta, container& dsf);
    void do_symv(double alpha, const container& f, double beta, container& dsf);

    Fieldaligned<ProductGeometry, IMatrix, container> m_fa;
    container m_temp;
    container m_tempP, m_temp0, m_tempM;
    container m_vol3d, m_inv3d, m_weights_wo_vol;
    dg::direction m_dir;
    Matrix m_jumpX, m_jumpY;
};

///@cond
////////////////////////////////////DEFINITIONS////////////////////////////////////////

template<class Geometry, class I, class M, class container>
DS<Geometry, I, M,container>::DS( Fieldaligned<Geometry, I, container> fa, dg::direction dir): m_fa(fa), m_dir(dir)
{
    dg::assign( dg::create::volume(     fa.grid()), m_vol3d);
    dg::assign( dg::create::weights(    fa.grid()), m_weights_wo_vol);
    dg::assign( dg::create::inv_volume( fa.grid()), m_inv3d);
    dg::blas2::transfer( dg::create::jumpX( fa.grid(), fa.bcx()), m_jumpX);
    dg::blas2::transfer( dg::create::jumpY( fa.grid(), fa.bcy()), m_jumpY);
    m_temp = m_vol3d, m_tempP = m_temp, m_temp0 = m_temp, m_tempM = m_temp;
}

template<class G, class I, class M, class container>
inline void DS<G,I,M,container>::ds( dg::direction dir, double alpha, const container& f, double beta, container& dsf) {
    switch( dir){
        case dg::centered:
        return centered( alpha, f, beta, dsf);
        case dg::forward:
        return forward( alpha, f, beta, dsf);
        case dg::backward:
        return backward( alpha, f, beta, dsf);
    }
}
template<class G, class I, class M, class container>
inline void DS<G,I,M,container>::div( dg::direction dir, double alpha, const container& f, double beta, container& dsf) {
    switch( dir){
        case dg::centered:
        return divCentered( alpha, f, beta, dsf);
        case dg::forward:
        return divForward( alpha, f, beta, dsf);
        case dg::backward:
        return divBackward( alpha, f, beta, dsf);
    }
}


template<class G, class I, class M, class container>
void DS<G,I,M,container>::do_divBackward( double alpha, const container& f, double beta, container& dsf)
{
    //adjoint discretisation
    dg::blas1::pointwiseDot(  m_vol3d, f, m_temp0);
    dg::blas1::pointwiseDivide( m_temp0, m_fa.hp(), m_temp0);
    m_fa(einsPlusT, m_temp0, m_tempP);
    dg::blas1::pointwiseDot( alpha, m_temp0, m_inv3d, -alpha, m_tempP, m_inv3d, beta, dsf);
}
template<class G,class I, class M, class container>
void DS<G,I,M,container>::do_divForward( double alpha, const container& f, double beta, container& dsf)
{
    //adjoint discretisation
    dg::blas1::pointwiseDot(  m_vol3d, f, m_temp0);
    dg::blas1::pointwiseDivide( m_temp0, m_fa.hm(), m_temp0);
    m_fa(einsMinusT, m_temp0, m_tempM);
    dg::blas1::pointwiseDot( alpha, m_tempM, m_inv3d, -alpha, m_temp0, m_inv3d, beta, dsf);
}
template<class G, class I, class M, class container>
void DS<G, I,M,container>::do_divCentered( double alpha, const container& f, double beta, container& dsf)
{
    //adjoint discretisation
    dg::blas1::pointwiseDot(  m_vol3d, f, m_temp0);
    dg::blas1::axpby( 1., m_fa.hp(), 1., m_fa.hm(), m_tempP);
    dg::blas1::pointwiseDivide( m_temp0, m_tempP, m_temp0);
    m_fa(einsPlusT,  m_temp0, m_tempP);
    m_fa(einsMinusT, m_temp0, m_tempM);
    dg::blas1::pointwiseDot( alpha, m_tempM, m_inv3d, -alpha, m_tempP, m_inv3d, beta, dsf);

}

template<class G,class I, class M, class container>
void DS<G,I,M,container>::do_symv( double alpha, const container& f, double beta, container& dsTdsf)
{

    if(m_dir == dg::centered) //does not converge with BC!!
    {
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        dg::blas1::subroutine( detail::ComputeSymv(), m_tempP, m_tempM,
                m_fa.hp(), m_fa.hm(), m_vol3d);
        m_fa(einsPlusT,  m_tempP, m_temp);
        m_fa(einsMinusT, m_tempP, m_tempM);
        dg::blas1::subroutine( detail::ComputeSymvEnd(), m_temp,
            m_tempM, m_weights_wo_vol);
    }
    else
    {
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        dg::blas1::subroutine( detail::ComputeSymv(), m_tempP, m_tempM, f,
            m_fa.hp(), m_fa.hm(), m_vol3d);
        m_fa(einsPlusT, m_tempP, m_temp0);
        m_fa(einsMinusT, m_tempM, m_temp);
        dg::blas1::subroutine( detail::ComputeSymvEnd(), m_temp,
            m_tempM, m_tempP, m_temp0, m_weights_wo_vol);
    }

    //     add jump terms
    dg::blas2::symv( -1., m_jumpX, f, 1., m_temp);
    dg::blas2::symv( -1., m_jumpY, f, 1., m_temp);
    //now volume without weights is missing

    dg::blas1::pointwiseDot( alpha, m_inv3d, m_weights_wo_vol, m_temp, beta, dsTdsf);
}
///@endcond

/**
* @brief forward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
*
* forward derivative \f$ g_i = \alpha \frac{1}{h_z^+}(f_{i+1} - f_{i}) + \beta g_i\f$
* @param fa this object will be used to get grid distances
* @copydoc hide_ds_parameters4
* @copydoc hide_ds_fp
* @ingroup fieldaligned
* @copydoc hide_ds_freestanding
*/
template<class FieldAligned, class container>
void ds_forward(const FieldAligned& fa, double alpha, const container& f, const container& fp, double beta, container& g)
{
    //direct
    dg::blas1::subroutine( detail::ComputeDSForward( alpha, beta),
            g, f, fp, fa.hp());
}
/**
* @brief backward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
*
* backward derivative \f$ g_i = \alpha \frac{1}{h_z^-}(f_{i} - f_{i-1}) + \beta g_i \f$
* @param fa this object will be used to get grid distances
* @copydoc hide_ds_parameters4
* @copydoc hide_ds_fm
* @ingroup fieldaligned
* @copydoc hide_ds_freestanding
*/
template<class FieldAligned, class container>
void ds_backward( const FieldAligned& fa, double alpha, const container& fm, const container& f, double beta, container& g)
{
    //direct
    dg::blas1::subroutine( detail::ComputeDSBackward( alpha, beta),
            g, f, fm, fa.hm());
}
/**
* @brief centered derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
*
* The centered derivative is constructed by fitting a polynomial through the plus point the minus point and the center point and evaluating its derivative at the center point. For the exact resulting formula consult the <a href="./parallel.pdf" target="_blank">parallel derivative</a> writeup
* @param fa this object will be used to get grid distances
* @copydoc hide_ds_parameters4
* @copydoc hide_ds_fm
* @copydoc hide_ds_fp
* @ingroup fieldaligned
* @copydoc hide_ds_freestanding
*/
template<class FieldAligned, class container>
void ds_centered( const FieldAligned& fa, double alpha, const container& fm,
        const container& f, const container& fp, double beta, container& g)
{
    //direct discretisation
    dg::blas1::subroutine( detail::ComputeDSCentered( alpha, beta),
            g, fm, f, fp, fa.hm(), fa.hp());
}
/**
 * @brief Centered derivative \f$ g = \alpha (\vec v\cdot \nabla)^2 f + \beta g \f$
 *
 * The formula used is \f[ \nabla_\parallel^2 f = 2\left(\frac{f^+}{h_z^+ h_z^0} - \frac{f^0}{h_z^- h_z^+} + \frac{f^-}{h_z^-h_z^0}\right) \f]
 * which is the second derivative of a 2nd order polynomial fitted through the plus, minus and centre points
 * the boundary conditions are implemented by
 * mirroring points perpendicular to the boundary, which has some drawbacks as
 * to the numerical stability and toroidal resolution.
 * @param fa this object will be used to get grid distances
 * @copydoc hide_ds_parameters4
 * @copydoc hide_ds_fm
 * @copydoc hide_ds_fp
 * @ingroup fieldaligned
 * @copydoc hide_ds_freestanding
 */
template<class FieldAligned, class container>
void dss_centered( const FieldAligned& fa, double alpha, const container&
        fm,const container& f, const container& fp, double beta, container& g)
{
    dg::blas1::subroutine( detail::ComputeDSS( alpha, beta),
            g, fm, f, fp, fa.hm(), fa.hp());
}

/**
 * @brief centered derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
 *
 * The centered derivative is constructed by fitting a polynomial through the plus point the minus point and the center point and evaluating its derivative at the center point. For the exact resulting formula consult the <a href="./parallel.pdf" target="_blank">parallel derivative</a> writeup.
 * the boundary condition is implemented
 * along the field-line, that is the boundary condition is used as part of the polynomial interpolation.
 * @param fa this object will be used to get grid distances
 * @copydoc hide_ds_parameters4
 * @copydoc hide_ds_fm
 * @copydoc hide_ds_fp
 * @param bound either dg::NEU or dg::DIR (rest not implemented yet)
 * @param boundary_value first value is for incoming fieldlines, second one for outgoing
 * @ingroup fieldaligned
 */
template<class FieldAligned, class container>
void ds_centered_bc_along_field( const FieldAligned& fa, double alpha, const container& fm,
        const container& f, const container& fp, double beta, container& g,
        dg::bc bound, std::array<double,2> boundary_value = {0,0})
{
    //direct discretisation
    if( bound == dg::NEU)
    {
        dg::blas1::subroutine( detail::ComputeDSCenteredNEU( alpha, beta, boundary_value),
                g, fm, f, fp, fa.hm(), fa.hp(), fa.hbm(),
                fa.hbp(), fa.bbm(), fa.bbo(), fa.bbp());
    }
    else// if( bound == dg::DIR)
    {
        dg::blas1::subroutine( detail::ComputeDSCenteredDIR( alpha, beta, boundary_value),
                g, fm, f, fp, fa.hm(), fa.hp(), fa.hbm(),
                fa.hbp(), fa.bbm(), fa.bbo(), fa.bbp());
    }
}
/**
 * @brief Centered derivative \f$ g = \alpha (\vec v\cdot \nabla)^2 f + \beta g \f$
 *
 * The formula used is \f[ \nabla_\parallel^2 f = 2\left(\frac{f^+}{h_z^+ h_z^0} - \frac{f^0}{h_z^- h_z^+} + \frac{f^-}{h_z^-h_z^0}\right) \f]
 * which is the second derivative of a 2nd order polynomial fitted through the plus, minus and centre points
 * the boundary condition is implemented
 * along the field-line, that is the boundary condition is used as part of the polynomial interpolation.
 * @param fa this object will be used to get grid distances
 * @copydoc hide_ds_parameters4
* @copydoc hide_ds_fm
* @copydoc hide_ds_fp
* @param bound either dg::NEU or dg::DIR (rest not implemented yet)
* @param boundary_value first value is for incoming fieldlines, second one for outgoing
* @ingroup fieldaligned
 */
template<class FieldAligned, class container>
void dss_centered_bc_along_field( const FieldAligned& fa, double alpha, const
        container& fm,const container& f, const container& fp, double beta,
        container& g, dg::bc bound, std::array<double,2> boundary_value =
        {0,0})
{
    if( bound == dg::NEU)
    {
        dg::blas1::subroutine( detail::ComputeDSSNEU( alpha, beta, boundary_value),
                g, fm, f, fp, fa.hm(), fa.hp(), fa.hbm(),
                fa.hbp(), fa.bbm(), fa.bbo(), fa.bbp());
    }
    else// if( bound == dg:DIR)
    {
        dg::blas1::subroutine( detail::ComputeDSSDIR( alpha, beta, boundary_value),
                g, fm, f, fp, fa.hm(), fa.hp(), fa.hbm(),
                fa.hbp(), fa.bbm(), fa.bbo(), fa.bbp());
    }
}

}//namespace geo

///@cond
template< class G, class I, class M, class V>
struct TensorTraits< geo::DS<G,I,M, V> >
{
    using value_type = double;
    using tensor_category = SelfMadeMatrixTag;
};
///@endcond
}//namespace dg
