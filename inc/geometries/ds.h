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

/*!@class hide_ds_parameters2
* @param f The vector to derive
* @param g contains result on output (may alias input vectors)
* @note the vector sizes need to equal the grid size in the constructor
*/
/*!@class hide_ds_parameters4
* @param alpha Scalar
* @param f The vector to derive
* @param beta Scalar
* @param g contains result on output (may alias input vectors)
* @note the vector sizes need to equal the grid size in the constructor
*/
/*!@class hide_ds_fm
 * @param fm fieldaligned(einsMinus, f, fm)
 */
/*!@class hide_ds_fp
 * @param fp fieldaligned(einsPlus, f, fp)
 */
/*!@class hide_ds_fmm
 * @param fm fieldaligned(einsMinus, f, fm)
 * @param fmm twice apply fieldaligned(einsMinus, f, fm)
 */
/*!@class hide_ds_fpp
 * @param fp fieldaligned(einsPlus, f, fp)
 * @param fpp twice apply fieldaligned(einsPlus, f, fp)
 */
/*!@class hide_ds_dir
 * @param dir indicate the direction in the bracket operator and in symv
 */
/*!@class hide_ds_freestanding
 * @note This function computes the same as the corresponding member function
 * of \c dg::geo::DS but you have to compute the einsPlus and einsMinus
 * interpolations from \c dg::geo::Fieldaligned yourself. The reasoning for
 * this function is that you can re-use the latter interpolations if you
 * compute for example both first and second derivative of a function
 * or even if you compute it for difference boundary conditions using
 * dg::geo::assign_bc_along_field_2nd, dg::geo::assign_bc_along_field_1st or
 * dg::geo::swap_bc_perp
 */
/*!@class hide_ds_along_field
 * @attention The boundary condition given as a parameter to this function
 * overrules the boundary conditions used in the given \c dg::geo::Fieldaligned
 * object and the ones implicit in the einsPlus and einsMinus interpolations
 * (Independently of whether the magnetic field was periodified or not).
 * In this way it is possible to apply the same Fieldaligned object to
 * several quantities with different boundary conditions and save quite a bit
 * of memory consumption.
 */


/**
 * @brief Assign boundary conditions along magnetic field lines
 * interpolating a 2nd order polynomial
 *
 * **Call this function before one of the freestanding ds functions to
 * replace the default boundary conditions with along field boundary conditions**
 *
 * This function replaces the values of the plus and minus fields that
 * are outside of the domain with ghost values. These values are
 * constructed by fitting a polynomial through the
 * boundary point, the plus or minus point and the center point.
 * For the exact resulting formula consult the
 * <a href="./parallel.pdf" target="_blank">parallel derivative</a> writeup.
 * This is achieved using masks that mark the points where fieldlines intersect
 * the domain boundary and replace the interpolated boundary values.
 * @tparam FieldAligned
 * @tparam container
 * @param fa this object will be used to get grid distances and masking regions
 * @copydoc hide_ds_fm
 * @param f
 * @copydoc hide_ds_fp
 * @param fmg resulting eMinus field (can alias fm)
 * @param fpg resulting ePlus field (can alias fp)
 * @param bound either dg::NEU or dg::DIR (rest not implemented yet)
 * @param boundary_value first value is for incoming fieldlines, second one for outgoing
 * @ingroup fieldaligned
 * @copydoc hide_ds_along_field
 */
template<class FieldAligned, class container>
void assign_bc_along_field_2nd( const FieldAligned& fa, const container& fm,
        const container& f, const container& fp, container& fmg, container& fpg,
        dg::bc bound, std::array<double,2> boundary_value = {0,0})
{
    if( bound == dg::NEU)
    {
        double dbm = boundary_value[0], dbp = boundary_value[1];
        dg::blas1::subroutine( [dbm, dbp]DG_DEVICE( double fm, double fo,
                    double fp, double& fmg, double& fpg, double hm, double hp,
                    double hbm, double hbp, double bbm, double bbo, double bbp
                    ){
            double plus=0, minus=0, bothP=0, bothM = 0;
            plus = dbp*hp*(hm+hp)/(2.*hbp+hm) +
                fo*(2.*hbp+hm-hp)*(hm+hp)/hm/(2.*hbp+hm) + fm*hp*(-2.*hbp +
                    hp)/hm/(2.*hbp + hm);
            minus = fp*hm*(-2.*hbm+hm)/hp/(2.*hbm+hp) -
                dbm*hm*(hm+hp)/(2.*hbm+hp) +
                fo*(2.*hbm-hm+hp)*(hm+hp)/hp/(2.*hbm+hp);
            bothM = fo + dbp*hm*(-2.*hbm + hm)/2./(hbm+hbp) -
                dbm*hm*(2.*hbp+hm)/2./(hbm+hbp);
            bothP = fo + dbp*hp*(2.*hbm + hp)/2./(hbm+hbp) +
                dbm*hp*(2.*hbp-hp)/2./(hbm+hbp);
            fmg = (1.-bbo-bbm)*fm + bbm*minus + bbo*bothM;
            fpg = (1.-bbo-bbp)*fp + bbp*plus  + bbo*bothP;
        }, fm, f, fp, fmg, fpg, fa.hm(), fa.hp(), fa.hbm(), fa.hbp(), fa.bbm(),
                fa.bbo(), fa.bbp() );
    }
    else// if( bound == dg:DIR)
    {
        double fbm = boundary_value[0], fbp = boundary_value[1];
        dg::blas1::subroutine( [fbm, fbp]DG_DEVICE( double fm, double fo,
                    double fp, double& fmg, double& fpg, double hm, double hp,
                    double hbm, double hbp, double bbm, double bbo, double bbp
                    ){
            double plus=0, minus=0, bothP=0, bothM = 0;
            plus  = fm*hp*(-hbp + hp)/hm/(hbp+hm) + fo*(hbp-hp)*(hm+hp)/hbp/hm
                +fbp*hp*(hm+hp)/hbp/(hbp+hm);
            minus = +fo*(hbm-hm)*(hm+hp)/hbm/hp + fbm*hm*(hm+hp)/hbm/(hbm+hp)
                + fp*hm*(-hbm+hm)/hp/(hbm+hp);
            bothM = fbp*hm*(-hbm+hm)/hbp/(hbm+hbp) +
                fo*(hbm-hm)*(hbp+hm)/hbm/hbp + fbm*hm*(hbp+hm)/hbm/(hbm+hbp);
            bothP = fo*(hbp-hp)*(hbm+hp)/hbm/hbp +
                fbp*hp*(hbm+hp)/hbp/(hbm+hbp) + fbm*hp*(-hbp+hp)/hbm/(hbm+hbp);
            fmg = (1.-bbo-bbm)*fm + bbm*minus + bbo*bothM;
            fpg = (1.-bbo-bbp)*fp + bbp*plus  + bbo*bothP;
        }, fm, f, fp, fmg, fpg, fa.hm(), fa.hp(), fa.hbm(), fa.hbp(), fa.bbm(),
                fa.bbo(), fa.bbp());
    }

}
/**
 * @brief Assign boundary conditions along magnetic field lines
 * interpolating a 1st order polynomial (a line)
 *
 * @tparam FieldAligned
 * @tparam container
 * @param fa
 * @param fm
 * @param fp
 * @param fmg result (can alias fm)
 * @param fpg result (can alias fp)
 * @param bound either dg::NEU or dg::DIR (rest not implemented yet)
 * @param boundary_value first value is for incoming fieldlines, second one for outgoing
 * @ingroup fieldaligned
 * @copydoc hide_ds_along_field
 */
template<class FieldAligned, class container>
void assign_bc_along_field_1st( const FieldAligned& fa, const container& fm,
        const container& fp, container& fmg, container& fpg,
        dg::bc bound, std::array<double,2> boundary_value = {0,0})
{
    if( bound == dg::NEU)
    {
        double dbm = boundary_value[0], dbp = boundary_value[1];
        dg::blas1::subroutine( [dbm, dbp]DG_DEVICE( double fm, double fp,
                    double& fmg, double& fpg, double hm, double hp,
                    double bbm, double bbp
                    ){
            double plus=0, minus=0;
            plus = fm + dbp*(hp+hm);
            minus = fp - dbm*(hp+hm);
            fmg = (1.-bbm)*fm + bbm*minus;
            fpg = (1.-bbp)*fp + bbp*plus;
        }, fm, fp, fmg, fpg, fa.hm(), fa.hp(), fa.bbm(), fa.bbp() );
    }
    else// if( bound == dg:DIR)
    {
        double fbm = boundary_value[0], fbp = boundary_value[1];
        dg::blas1::subroutine( [fbm, fbp]DG_DEVICE( double fm, double fp,
                    double& fmg, double& fpg, double hm, double hp, double hbm,
                    double hbp, double bbm, double bbo, double bbp
                    ){
            double plus=0, minus=0, bothP=0, bothM = 0;
            plus  = fm + (fbp-fm)/(hbp+hm)*(hp+hm) ;
            minus = fp - (hp+hm)*(fp-fbm)/(hp+hbm);
            bothM = fbp + (fbp-fbm)/(hbp+hbm)*(hp+hbm);
            bothP = fbp - (fbp-fbm)/(hbp+hbm)*(hbp+hm);
            fmg = (1.-bbo-bbm)*fm + bbm*minus + bbo*bothM;
            fpg = (1.-bbo-bbp)*fp + bbp*plus  + bbo*bothP;
        }, fm, fp, fmg, fpg, fa.hm(), fa.hp(), fa.hbm(), fa.hbp(), fa.bbm(),
                fa.bbo(), fa.bbp());
    }
}

/**
 * @brief Swap the perp boundary condition
 *
 * This function multiplies (-1) to every value that lies outside the box.
 * This effectively swaps the boundary conditions in the Fourier boundary
 * mode, i.e. if NEU was used in Fieldaligned, then now they are DIR and
 * vice versa.
 *
 * @tparam FieldAligned
 * @tparam container
 * @param fa this object will be used to get masking regions
 * @copydoc hide_ds_fm
 * @copydoc hide_ds_fp
 * @param fmg resulting eMinus field (can alias fm)
 * @param fpg resulting ePlus field (can alias fp)
 * @note The motivation for this function is to avoid reconstructing a
 * fieldaligned object since its constructor is quite expensive in runtime and
 * memory. The idea is to construct Fieldaligned with NEU boundary conditions
 * and then use this function to swap them to DIR whenever needed.
 * @note The swapping of bc is not exactly (but almost) the same as
 * re-constructing the fieldaligned object in case the support operator is used
 * (mx, my > 1), this is because the swap is performed only on the coarse grid.
 * In all tests the error is negligible so far.
 * @ingroup fieldaligned
 */
template<class FieldAligned, class container>
void swap_bc_perp( const FieldAligned& fa, const container& fm,
        const container& fp, container& fmg, container& fpg)
{
    dg::blas1::subroutine( []DG_DEVICE( double fm, double fp,
                double& fmg, double& fpg,
                double bbm, double bbo, double bbp
                ){
        fmg = (1.-bbo-bbm)*fm + (bbm+bbo)*(-fm);
        fpg = (1.-bbo-bbp)*fp + (bbp+bbo)*(-fp);
    }, fm, fp, fmg, fpg, fa.bbm(), fa.bbo(), fa.bbp() );

}

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
        double deltaPhi=-1, std::string interpolation_method = "dg"):
        DS( FA( vec, grid, bcx, bcy, limit, eps, mx, my, deltaPhi,
                    interpolation_method), dir )
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
        double deltaPhi=-1, std::string interpolation_method = "dg"):
        DS( FA( vec, grid, bcx, bcy, limit, eps, mx, my, deltaPhi,
                    interpolation_method), dir)
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
    * @brief 2nd order forward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
    *
    * forward derivative \f$ g_i = \alpha \frac{1}{2h_z^+}(-f_{i+2} + 4f_{i+1} - 3f_{i}) + \beta g_i\f$
    * @copydoc hide_ds_parameters4
    */
    void forward2( double alpha, const container& f, double beta, container& g){
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsPlus, m_tempP, m_tempM);
        ds_forward2( m_fa, alpha, f, m_tempP, m_tempM, beta, g);
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
    * @brief 2nd order backward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
    *
    * backward derivative \f$ g_i = \alpha \frac{1}{2h_z^-}(3f_{i} - 4f_{i-1} + f_{i-2}) + \beta g_i \f$
    * @copydoc hide_ds_parameters4
    */
    void backward2( double alpha, const container& f, double beta, container& g){
        m_fa(einsMinus, f, m_tempM);
        m_fa(einsMinus, m_tempM, m_tempP);
        ds_backward2( m_fa, alpha, m_tempP, m_tempM, f, beta, g);
    }
    /**
    * @brief centered derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
    *
    * The centered derivative is constructed by fitting a polynomial through
    * the plus point the minus point and the center point and evaluating its
    * derivative at the center point. For the exact resulting formula consult
    * the <a href="./parallel.pdf" target="_blank">parallel derivative</a>
    * writeup
    * @copydoc hide_ds_parameters4
    */
    void centered( double alpha, const container& f, double beta, container& g){
        m_fa(einsPlus,  f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        ds_centered( m_fa, alpha, m_tempM, f, m_tempP, beta, g);
    }
    /// Same as \c dg::geo::ds_centered after \c dg::geo::ds_assign_bc_along_field_2nd
    void centered_bc_along_field(
        double alpha, const container& f, double beta, container& g, dg::bc bound,
        std::array<double,2> boundary_value = {0,0}){
        m_fa(einsPlus,  f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        assign_bc_along_field_2nd( m_fa, m_tempM, f, m_tempP, m_tempM, m_tempP,
                bound, boundary_value);
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
    /// Same as \c dg::geo::divDirectCentered
    void divDirect( double alpha, const container& divv, const
            container& f, double beta, container& g){
        m_fa(einsPlus,  f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        divDirectCentered( divv, m_fa, alpha, m_tempM, f, m_tempP, beta, g);
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
    void symv( const container& f, container& g){ symv( 1., f, 0., g);}
    /**
     * @brief Discretizes \f$ g = \alpha \nabla\cdot ( \vec v \vec v \cdot \nabla f ) + \beta g\f$ as a symmetric matrix
     *
     * if direction given in constructor is centered then centered followed by divCentered and adding jump terms is called, else a symmetric forward/backward discretization is chosen.
     * @copydoc hide_ds_parameters4
     * @copydoc hide_ds_attention
     */
    void symv( double alpha, const container& f, double beta, container& g);
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
    /// Same as \c dg::geo::dss_centered after \c dg::geo::ds_assign_bc_along_field_2nd
    void dss_bc_along_field(
        double alpha, const container& f, double beta, container& g, dg::bc bound,
        std::array<double,2> boundary_value = {0,0}){
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        assign_bc_along_field_2nd( m_fa, m_tempM, f, m_tempP, m_tempM, m_tempP,
                bound, boundary_value);
        dss_centered( m_fa, alpha, m_tempM, f, m_tempP, beta, g);
    }
    /// Same as \c dg::geo::dssd_centered
    void dssd( double alpha, const container& divv, const container& f, double
            beta, container& g){
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        dssd_centered( divv, m_fa, alpha, m_tempM, f, m_tempP, beta, g);
    }
    /// Same as \c dg::geo::dssd_centered after \c dg::geo::ds_assign_bc_along_field_2nd
    void dssd_bc_along_field( double alpha, const container& divv, const
            container& f, double beta, container& g, dg::bc bound,
            std::array<double,2> boundary_value = {0,0}){
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        assign_bc_along_field_2nd( m_fa, m_tempM, f, m_tempP, m_tempM, m_tempP,
                bound, boundary_value);
        dssd_centered( divv, m_fa, alpha, m_tempM, f, m_tempP, beta, g);
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
void DS<G,I,M,container>::symv( double alpha, const container& f, double beta, container& dsTdsf)
{
    //CUDA LAMBDAS CANNOT BE CONTAINED IN A PRIVATE OR PROTECTED METHOD
    if(m_dir == dg::centered) //does not converge with BC!!
    {
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        dg::blas1::subroutine( []DG_DEVICE(
            double& fp, double fm, double hp, double hm,  double vol3d){
                fp = ( fp-fm)/(hp+hm);
                fp = vol3d*fp/(hp+hm);
            }, m_tempP, m_tempM, m_fa.hp(), m_fa.hm(), m_vol3d);
        m_fa(einsPlusT,  m_tempP, m_temp);
        m_fa(einsMinusT, m_tempP, m_tempM);
        dg::blas1::subroutine([]DG_DEVICE (
            double& fm, double fp, double weights) { fm = ( fp-fm)/weights; },
                m_temp, m_tempM, m_weights_wo_vol);
    }
    else
    {
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        dg::blas1::subroutine( []DG_DEVICE(
            double& fp, double& fm, double f0, double hp, double hm, double vol3d) {
                fp = ( fp-f0)/hp;
                fp = 0.5*vol3d*fp/hp;
                fm = ( f0-fm)/hm;
                fm = 0.5*vol3d*fm/hm;
            }, m_tempP, m_tempM, f, m_fa.hp(), m_fa.hm(), m_vol3d);
        m_fa(einsPlusT, m_tempP, m_temp0);
        m_fa(einsMinusT, m_tempM, m_temp);
        dg::blas1::subroutine( []DG_DEVICE (
            double& efm, double fm, double fp, double efp, double weights){
                efm = ( efm- fm + fp -efp)/weights;
            }, m_temp, m_tempM, m_tempP, m_temp0, m_weights_wo_vol);
    }

    //     add jump terms
    dg::blas2::symv( -1., m_jumpX, f, 1., m_temp);
    dg::blas2::symv( -1., m_jumpY, f, 1., m_temp);
    //now volume without weights is missing

    dg::blas1::pointwiseDot( alpha, m_inv3d, m_weights_wo_vol, m_temp, beta, dsTdsf);
};
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
    dg::blas1::subroutine( [ alpha, beta]DG_DEVICE(
            double& dsf, double fo, double fp, double hp){
                dsf = alpha*( fp - fo)/hp + beta*dsf;
            },
            g, f, fp, fa.hp());
}
/**
* @brief 2nd order forward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
*
* forward derivative \f$ g_i = \alpha \frac{1}{2h_z^+}(-f_{i+2} + 4f_{i+1} - 3f_{i}) + \beta g_i\f$
* @param fa this object will be used to get grid distances
* @copydoc hide_ds_parameters4
* @copydoc hide_ds_fpp
* @ingroup fieldaligned
* @copydoc hide_ds_freestanding
*/
template<class FieldAligned, class container>
void ds_forward2(const FieldAligned& fa, double alpha, const container& f, const container& fp, const container& fpp, double beta, container& g)
{
    //direct
    dg::blas1::subroutine( [ alpha, beta]DG_DEVICE(
            double& dsf, double fo, double fp, double fpp, double hp){
                dsf = alpha*( -3.*fo + 4.*fp - fpp)/2./hp
                     + beta*dsf;
            },
            g, f, fp, fpp, fa.hp());
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
    dg::blas1::subroutine( [ alpha, beta] DG_DEVICE(
            double& dsf, double fo, double fm, double hm){
                dsf = alpha*( fo - fm)/hm + beta*dsf;
            },
            g, f, fm, fa.hm());

}
/**
* @brief 2nd order backward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
*
* backward derivative \f$ g_i = \alpha \frac{1}{2h_z^-}(3f_{i} - 4f_{i-1} + f_{i-2}) + \beta g_i \f$
* @param fa this object will be used to get grid distances
* @copydoc hide_ds_parameters4
* @copydoc hide_ds_fmm
* @ingroup fieldaligned
* @copydoc hide_ds_freestanding
*/
template<class FieldAligned, class container>
void ds_backward2( const FieldAligned& fa, double alpha, const container& fmm, const container& fm, const container& f, double beta, container& g)
{
    //direct
    dg::blas1::subroutine( [ alpha, beta] DG_DEVICE(
            double& dsf, double fo, double fm,  double fmm, double hm){
                dsf = alpha*( 3.*fo - 4.*fm + fmm)/2./hm
                    + beta*dsf;
            },
            g, f, fm, fmm, fa.hm());

}

///@cond
namespace detail
{
struct DSCentered
{
    DSCentered( double alpha, double beta) : m_alpha(alpha), m_beta(beta){}
    DG_DEVICE
    void operator()( double& dsf, double fm, double fo, double fp, double hm,
            double hp)
    {
        dsf = m_alpha*(
            fm*( 1./(hp+hm) - 1./hm) +
            fo*( 1./hm - 1./hp) +
            fp*( 1./hp - 1./(hp+hm))
            ) + m_beta*dsf;
    };

    private:
    double m_alpha, m_beta;
};
struct DSSCentered
{
    DSSCentered( double alpha, double beta) : m_alpha(alpha), m_beta(beta){}
    DG_DEVICE
    void operator()( double& dssf, double fm, double fo, double fp, double hm,
            double hp)
    {
        dssf = m_alpha*(
                    2.*fm/(hp+hm)/hm - 2.*fo/hp/hm + 2.*fp/(hp+hm)/hp
               ) + m_beta*dssf;
    };

    private:
    double m_alpha, m_beta;
};
struct DSSDCentered
{
    DSSDCentered( double alpha, double beta) : m_ds(1.,0.), m_dss(1., 0.),
        m_alpha( alpha), m_beta(beta){}
    DG_DEVICE
    void operator()( double& dssdf, double fm, double fo, double fp, double hm,
            double hp, double divv)
    {
        // Delta_par = divv * ds + dss
        double ds = 0., dss = 0.;
        m_ds(   ds, fm, fo, fp, hm, hp);
        m_dss( dss, fm, fo, fp, hm, hp);
        dssdf = m_alpha*( ds*divv + dss) + m_beta*dssdf;
    };

    private:
    DSCentered m_ds;
    DSSCentered m_dss;
    double m_alpha, m_beta;
};

}//namespace detail
///@endcond

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
    dg::blas1::subroutine( detail::DSCentered(alpha, beta),
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
void dss_centered( const FieldAligned& fa, double alpha, const container& fm,
        const container& f, const container& fp, double beta, container& g)
{
    dg::blas1::subroutine( detail::DSSCentered( alpha, beta),
        g, fm, f, fp, fa.hm(), fa.hp());
}
/**
 * @brief Centered derivative \f$ g = \alpha \nabla\cdot(\vec v \vec v\cdot \nabla) f + \beta g \f$
 *
 * The formula used is \f[ \Delta_\parallel f = \nabla\cdot \vec v \nabla_\parallel f + \nabla_\parallel^2 f \f]
 * @param divv The divergence of the vector field
 * @param fa this object will be used to get grid distances
 * @copydoc hide_ds_parameters4
 * @copydoc hide_ds_fm
 * @copydoc hide_ds_fp
 * @ingroup fieldaligned
 * @copydoc hide_ds_freestanding
 */
template<class FieldAligned, class container>
void dssd_centered( const container& divv, const FieldAligned& fa, double
        alpha, const container& fm, const container& f, const container& fp,
        double beta, container& g)
{
    dg::blas1::subroutine( detail::DSSDCentered( alpha, beta),
        g, fm, f, fp, fa.hm(), fa.hp(), divv);
}

/**
 * @brief centered derivative \f$ g = \alpha \nabla \cdot \vec v f + \beta g\f$
 *
 * This is the direct derivative (i.e. product rule) \f$ \nabla \cdot \vec v
 * f = \vec v \cdot \nabla f + f \nabla\cdot \vec v\f$ where for the first
 * derivative
 * we use a centered derivative and for the second f we use the average
 * along the fieldline. ( This has advantageous conservative properties)
 * @param divv The divergence of the vector field
 * @param fa this object will be used to get grid distances
 * @copydoc hide_ds_parameters4
 * @copydoc hide_ds_fm
 * @copydoc hide_ds_fp
 * @ingroup fieldaligned
 * @copydoc hide_ds_freestanding
*/
template<class FieldAligned, class container>
void divDirectCentered( const container& divv, const FieldAligned& fa, double
        alpha, const container& fm, const container& f, const container& fp,
        double beta, container& g){
    ds_centered( fa, alpha, fm, f, fp, beta, g);
    dg::blas1::subroutine( [alpha]DG_DEVICE( double& g, double minus,
                double plus, double hp, double hm, double divv){
            g += alpha*(hm*plus+hp*minus)/(hp+hm)*divv;
            }, g, fm, fp, fa.hp(), fa.hm(), divv);
}

/**
 * @brief Compute average along a fieldline \f$ g = \alpha \frac{h^- f_{k+1} + h^+ f_{k-1}}{h^- + h^+} + \beta g\f$
 *
 * @note The idea is to use this function on staggered grid implementations. There
 * it can be used to convert a function from one grid to the staggered grid and
 * back
 * @param fa this object will be used to get grid distances
 * @param alpha Scalar
 * @copydoc hide_ds_fm
 * @copydoc hide_ds_fp
 * @param beta Scalar
 * @param g contains result on output (may alias input vectors)
 * @ingroup fieldaligned
 * @copydoc hide_ds_freestanding
 */
template<class FieldAligned, class container>
void ds_average( const FieldAligned& fa, double alpha,
        const container& fm, const container& fp, double beta, container& g)
{
    dg::blas1::subroutine( [alpha,beta]DG_DEVICE( double& g, double fm, double fp,
                double hp, double hm){
            g = alpha*(hm*fp+hp*fm)/(hp+hm) + beta*g;
            }, g, fm, fp, fa.hp(), fa.hm());
}
/**
 * @brief Compute simple slope along a fieldline \f$ g = \alpha \frac{f_{k+1} - f_{k-1}}{h^- + h^+} + \beta g\f$
 *
 * @note The idea is to use this function on staggered grid implementations. There
 * it can be used to compute the gradient from one grid onto the staggered grid
 * and vice versa
 * @param fa this object will be used to get grid distances
 * @param alpha Scalar
 * @copydoc hide_ds_fm
 * @copydoc hide_ds_fp
 * @param beta Scalar
 * @param g contains result on output (may alias input vectors)
 * @ingroup fieldaligned
 * @copydoc hide_ds_freestanding
 */
template<class FieldAligned, class container>
void ds_slope( const FieldAligned& fa, double alpha,
        const container& fm, const container& fp, double beta, container& g)
{
    dg::blas1::subroutine( [alpha,beta]DG_DEVICE( double& g, double fm, double fp,
                double hp, double hm){
            g = alpha*(fp-fm)/(hp+hm) + beta*g;
            }, g, fm, fp, fa.hp(), fa.hm());
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
