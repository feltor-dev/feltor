#pragma once

#include "dg/algorithm.h"
#include "fieldaligned.h"
#ifdef MPI_VERSION
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
    DSSCentered( double alpha, double beta, double delta) : m_alpha(alpha), m_beta(beta), m_delta(delta){}
    DG_DEVICE
    void operator()( double& dssf, double fm, double fo, double fp, double hm,
            double hp)
    {
        dssf = m_alpha*(
                    2.*fm/(hp+hm)/hm - 2.*fo/hp/hm + 2.*fp/(hp+hm)/hp
               ) + m_beta*dssf;
    };

    DG_DEVICE
    void operator()( double& dssf, double fm, double fo, double fp,
        double bPm, double bP0, double bPp)
    {
        double bP2 = (bPp+bP0)/2.;
        double bM2 = (bPm+bP0)/2.;
        double fm2 = (fo-fm)/m_delta;
        double fp2 = (fp-fo)/m_delta;

        dssf = m_alpha*bP0*( bP2*fp2 - bM2*fm2)/m_delta + m_beta*dssf;
    }

    private:
    double m_alpha, m_beta, m_delta;
};
struct DSSDCentered
{
    DSSDCentered( double alpha, double beta, double delta) :
        m_alpha( alpha), m_beta(beta), m_delta(delta){}
    DG_DEVICE
    void operator()( double& dssdf, double fm, double fo, double fp, double Gm,
        double Go, double Gp, double bPm, double bP0, double bPp)
    {
        // various combinations of bP do not seem to matter
        double bP2 = (bPp+bP0)/2.;
        double bM2 = (bPm+bP0)/2.;
        double fm2 = (fo-fm)/m_delta;
        double fp2 = (fp-fo)/m_delta;
        double gp2 = (Gp + Go)/Go/2.;
        double gm2 = (Gm + Go)/Go/2.;

        dssdf = m_alpha*( gp2*fp2*bP2*bP2 - bM2*bM2*gm2*fm2)/m_delta + m_beta*dssdf;

        // does not seem to conserve nicely
        //dssdf = m_alpha*( fp*gp2*bP2*bP2 - 2*bP0*bP0*fo + bM2*bM2*gm2*fm)/(hp)/(hm) + m_beta*dssdf;
        //dssdf = m_alpha*( fp*Gp/Go*bPp*bPp - 2*bP0*bP0*fo + bPm*bPm*Gm/Go*fm)/(hp)/(hm) + m_beta*dssdf;
    };

    private:
    double m_alpha, m_beta, m_delta;
};

}//namespace detail
///@endcond

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
 * <a href="https://www.overleaf.com/read/jjvstccqzcjv" target="_blank">parallel derivative</a> writeup.
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
    double delta = fa.deltaPhi();
    if( bound == dg::NEU)
    {
        double dbm = boundary_value[0], dbp = boundary_value[1];
        dg::blas1::subroutine( [dbm, dbp, delta]DG_DEVICE( double fm, double fo,
                    double fp, double& fmg, double& fpg,
                    double hbm, double hbp, double bbm, double bbo, double bbp
                    ){
            //formula derived for non-equidistant grid
            double hm = delta, hp = delta;
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
        }, fm, f, fp, fmg, fpg, fa.hbm(), fa.hbp(), fa.bbm(),
                fa.bbo(), fa.bbp() );
    }
    else// if( bound == dg:DIR)
    {
        double fbm = boundary_value[0], fbp = boundary_value[1];
        dg::blas1::subroutine( [fbm, fbp, delta]DG_DEVICE( double fm, double fo,
                    double fp, double& fmg, double& fpg,
                    double hbm, double hbp, double bbm, double bbo, double bbp
                    ){
            //formula derived for non-equidistant grid
            double hm = delta, hp = delta;
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
        }, fm, f, fp, fmg, fpg, fa.hbm(), fa.hbp(), fa.bbm(),
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
 * @copydoc hide_ds_fm
 * @copydoc hide_ds_fp
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
    double delta = fa.deltaPhi();
    if( bound == dg::NEU)
    {
        double dbm = boundary_value[0], dbp = boundary_value[1];
        dg::blas1::subroutine( [dbm, dbp, delta]DG_DEVICE( double fm, double fp,
                    double& fmg, double& fpg, double bbm, double bbp
                    ){
            double hm = delta, hp = delta;
            double plus=0, minus=0;
            plus = fm + dbp*(hp+hm);
            minus = fp - dbm*(hp+hm);
            fmg = (1.-bbm)*fm + bbm*minus;
            fpg = (1.-bbp)*fp + bbp*plus;
        }, fm, fp, fmg, fpg, fa.bbm(), fa.bbp() );
    }
    else// if( bound == dg:DIR)
    {
        double fbm = boundary_value[0], fbp = boundary_value[1];
        dg::blas1::subroutine( [fbm, fbp, delta]DG_DEVICE( double fm, double fp,
                    double& fmg, double& fpg, double hbm,
                    double hbp, double bbm, double bbo, double bbp
                    ){
            double hm = delta, hp = delta;
            double plus=0, minus=0, bothP=0, bothM = 0;
            plus  = fm + (fbp-fm)/(hbp+hm)*(hp+hm) ;
            minus = fp - (hp+hm)*(fp-fbm)/(hp+hbm);
            bothM = fbp + (fbp-fbm)/(hbp+hbm)*(hp+hbm);
            bothP = fbp - (fbp-fbm)/(hbp+hbm)*(hbp+hm);
            fmg = (1.-bbo-bbm)*fm + bbm*minus + bbo*bothM;
            fpg = (1.-bbo-bbp)*fp + bbp*plus  + bbo*bothP;
        }, fm, fp, fmg, fpg, fa.hbm(), fa.hbp(), fa.bbm(),
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
@note The \c div and \c symv member functions are close to but not exactly volume conserving
(at least if not the linear interpolation is chosen in Fieldaligned)*/

/**
* @brief Class for the evaluation of parallel derivatives
*
* This class discretizes the operators
\f$ \nabla_\parallel = \vec{v}\cdot \nabla = v^\zeta\partial_\zeta + v^\eta\partial_\eta + v^\varphi\partial_\varphi \f$,
\f$\nabla_\parallel^\dagger = -\nabla\cdot(\vec v .)\f$ and
\f$\Delta_\parallel=-\nabla_\parallel^\dagger\cdot\nabla_\parallel\f$
in arbitrary coordinates
@snippet ds_t.cpp doxygen
 * @note The parallel Laplacian cannot be inverted as long as there are
 * closed fieldlines somewhere in the domain (which is virtually always true).
@note The \c div and \c symv member functions are close to but not exactly volume conserving
(at least if not the linear interpolation is chosen in Fieldaligned)
* @ingroup fieldaligned
* @tparam ProductGeometry must be either \c dg::aProductGeometry3d or \c dg::aProductMPIGeometry3d or any derivative
* @tparam IMatrix The type of the interpolation matrix
    - \c dg::IHMatrix, or \c dg::IDMatrix, \c dg::MIHMatrix, or \c dg::MIDMatrix
* @tparam container The container-class on which the interpolation matrix operates on
    - \c dg::HVec, or \c dg::DVec, \c dg::MHVec, or \c dg::MDVec
* @note The \c dg::TensorTraits are defined for this class and thus \c DS
* is usable as a Matrix class (calling \c symv )
* @sa The pdf <a href="https://www.overleaf.com/read/jjvstccqzcjv" target="_blank">parallel derivative</a> writeup
*/
template< class ProductGeometry, class IMatrix, class container >
struct DS
{
    typedef dg::geo::Fieldaligned<ProductGeometry, IMatrix, container> FA; //!< conveniently abbreviates underlying \c Fieldaligned type
    ///@brief No memory allocation; all member calls except construct are invalid
    DS(){}

    /**
     * @brief Create the magnetic unit vector field and construct

     * @copydoc hide_fieldaligned_physics_parameters
     * @copydoc hide_fieldaligned_numerics_parameters
     * @sa \c Fieldaligned
     */
    template <class Limiter>
    DS(const dg::geo::TokamakMagneticField& vec, const ProductGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned mx=10, unsigned my=10,
        double deltaPhi=-1, std::string interpolation_method = "dg",
        bool benchmark=true):
        DS( FA( vec, grid, bcx, bcy, limit, eps, mx, my, deltaPhi,
                    interpolation_method, benchmark) )
    {
    }
    /**
     * @brief Use the given vector field to construct
     *
     * @copydoc hide_fieldaligned_physics_parameters
     * @copydoc hide_fieldaligned_numerics_parameters
     * @sa \c Fieldaligned
     */
    template<class Limiter>
    DS(const dg::geo::CylindricalVectorLvl1& vec, const ProductGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned mx=10, unsigned my=10,
        double deltaPhi=-1, std::string interpolation_method = "dg",
        bool benchmark=true):
        DS( FA( vec, grid, bcx, bcy, limit, eps, mx, my, deltaPhi,
                    interpolation_method, benchmark))
    {
    }
    /**
     * @brief Re-construct from a given \c Fieldaligned object
     *
     * @param fieldaligned this object will be used in all further member calls
     */
    DS( FA fieldaligned);
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
        m_fa(zeroForw, f, m_tempO);
        m_fa(einsPlus, f, m_tempP);
        ds_forward( m_fa, alpha, m_tempO, m_tempP, beta, g);
    }
    /**
    * @brief 2nd order forward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
    *
    * forward derivative \f$ g_i = \alpha \frac{1}{2h_z^+}(-f_{i+2} + 4f_{i+1} - 3f_{i}) + \beta g_i\f$
    * @copydoc hide_ds_parameters4
    */
    void forward2( double alpha, const container& f, double beta, container& g){
        m_fa(zeroForw, f, m_tempO);
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsPlus, m_tempP, m_tempM);
        ds_forward2( m_fa, alpha, m_tempO, m_tempP, m_tempM, beta, g);
    }
    /**
    * @brief backward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
    *
    * backward derivative \f$ g_i = \alpha \frac{1}{h_z^-}(f_{i} - f_{i-1}) + \beta g_i \f$
    * @copydoc hide_ds_parameters4
    */
    void backward( double alpha, const container& f, double beta, container& g){
        m_fa(einsMinus, f, m_tempM);
        m_fa(zeroForw, f, m_tempO);
        ds_backward( m_fa, alpha, m_tempM, m_tempO, beta, g);
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
        m_fa(zeroForw, f, m_tempO);
        ds_backward2( m_fa, alpha, m_tempP, m_tempM, m_tempO, beta, g);
    }
    /**
    * @brief centered derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
    *
    * The centered derivative is constructed by fitting a polynomial through
    * the plus point the minus point and the center point and evaluating its
    * derivative at the center point. For the exact resulting formula consult
    * the <a href="https://www.overleaf.com/read/jjvstccqzcjv" target="_blank">parallel derivative</a>
    * writeup
    * @copydoc hide_ds_parameters4
    */
    void centered( double alpha, const container& f, double beta, container& g){
        m_fa(einsPlus,  f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        ds_centered( m_fa, alpha, m_tempM, m_tempP, beta, g);
    }
    /// Same as \c dg::geo::ds_centered after \c dg::geo::ds_assign_bc_along_field_2nd
    void centered_bc_along_field(
        double alpha, const container& f, double beta, container& g, dg::bc bound,
        std::array<double,2> boundary_value = {0,0}){
        m_fa(einsPlus,  f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        assign_bc_along_field_2nd( m_fa, m_tempM, f, m_tempP, m_tempM, m_tempP,
                bound, boundary_value);
        ds_centered( m_fa, alpha, m_tempM, m_tempP, beta, g);

    }
    /**
    * @brief backward derivative \f$ g = \vec v \cdot \nabla f \f$
    *
    * backward derivative \f$ g_i = \frac{v^\varphi}{\Delta\varphi}(f_{i} - f_{i-1}) \f$
    * @copydoc hide_ds_parameters2
    */
    void backward( const container& f, container& g){
        backward(1., f,0.,g);
    }
    /**
    * @brief forward derivative \f$ g = \vec v \cdot \nabla f \f$
    *
    * forward derivative \f$ g_i = \frac{v^\varphi}{\Delta\varphi}(f_{i+1} - f_{i})\f$
    * @copydoc hide_ds_parameters2
    */
    void forward( const container& f, container& g){
        forward(1.,f, 0.,g);
    }
    /**
    * @brief centered derivative \f$ g = \vec v \cdot \nabla f \f$
    *
    * centered derivative \f$ g_i = \frac{v^\varphi}{2\Delta\varphi}(f_{i+1} - f_{i-1})\f$
    * @copydoc hide_ds_parameters2
    */
    void centered( const container& f, container& g){
        centered(1.,f,0.,g);
    }

    ///@brief forward divergence \f$ g = \alpha \nabla\cdot(\vec v f) + \beta g\f$
    ///@copydoc hide_ds_parameters4
    ///@copydoc hide_ds_attention
    void divForward( double alpha, const container& f, double beta, container& g){
        m_fa(einsPlus,  f, m_tempP);
        m_fa(zeroForw,  f, m_tempO);
        ds_divForward( m_fa, alpha, m_tempO, m_tempP, beta, g);
    }
    ///@brief backward divergence \f$ g = \alpha \nabla\cdot(\vec v f) + \beta g\f$
    ///@copydoc hide_ds_parameters4
    ///@copydoc hide_ds_attention
    void divBackward( double alpha, const container& f, double beta, container& g){
        m_fa(einsMinus,  f, m_tempM);
        m_fa(zeroForw,  f, m_tempO);
        ds_divBackward( m_fa, alpha, m_tempM, m_tempO, beta, g);
    }
    ///@brief centered divergence \f$ g = \alpha \nabla\cdot(\vec v f) + \beta g\f$
    ///@copydoc hide_ds_parameters4
    ///@copydoc hide_ds_attention
    void divCentered(double alpha, const container& f, double beta, container& g){
        m_fa(einsPlus,  f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        ds_divCentered( m_fa, alpha, m_tempM, m_tempP, beta, g);
    }
    ///@brief forward divergence \f$ g = \nabla\cdot(\vec v f)\f$
    ///@copydoc hide_ds_parameters2
    ///@copydoc hide_ds_attention
    void divForward(const container& f, container& g){
        divForward( 1.,f,0.,g);
    }
    ///@brief backward divergence \f$ g = \nabla\cdot(\vec v f)\f$
    ///@copydoc hide_ds_parameters2
    ///@copydoc hide_ds_attention
    void divBackward(const container& f, container& g){
        divBackward( 1.,f,0.,g);
    }
    ///@brief centered divergence \f$ g = \nabla\cdot(\vec v f)\f$
    ///@copydoc hide_ds_parameters2
    ///@copydoc hide_ds_attention
    void divCentered(const container& f, container& g){
        divCentered( 1.,f,0.,g);
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
     * using the \c dg::geo::dssd_centered method.
     * @copydoc hide_ds_parameters2
     * @copydoc hide_ds_attention
     * @note The \c dg::TensorTraits are defined for this class and thus \c DS
     * is usable as a Matrix class (calling \c symv )
     */
    void symv( const container& f, container& g){ symv( 1., f, 0., g);}
    /**
     * @brief Discretizes \f$ g = \alpha \nabla\cdot ( \vec v \vec v \cdot \nabla f ) + \beta g\f$ as a symmetric matrix
     *
     * using the \c dg::geo::dssd_centered method.
     * @copydoc hide_ds_parameters4
     * @copydoc hide_ds_attention
     * @note The \c dg::TensorTraits are defined for this class and thus \c DS
     * is usable as a Matrix class (calling \c symv )
     */
    void symv( double alpha, const container& f, double beta, container& g);
    /**
     * @brief Discretizes \f$ g = (\vec v\cdot \nabla)^2 f \f$
     *
     * Same as \c forward followed by \c backward
     * @copydoc hide_ds_parameters2
     */
    void dss( const container& f, container& g){
        dss( 1., f, 0., g);}
    /**
     * @brief Discretizes \f$ g = \alpha (\vec v\cdot \nabla)^2 f + \beta g \f$
     *
     * Same as \c forward followed by \c backward
     * @copydoc hide_ds_parameters4
     */
    void dss( double alpha, const container& f, double beta, container& g){
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        m_fa(zeroForw,  f, m_tempO);
        dss_centered( m_fa, alpha, m_tempM, m_tempO, m_tempP, beta, g);
    }
    /// Same as \c dg::geo::dss_centered after \c dg::geo::ds_assign_bc_along_field_2nd
    void dss_bc_along_field(
        double alpha, const container& f, double beta, container& g, dg::bc bound,
        std::array<double,2> boundary_value = {0,0}){
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        m_fa(zeroForw,  f, m_tempO);
        assign_bc_along_field_2nd( m_fa, m_tempM, m_tempO, m_tempP, m_tempM, m_tempP,
                bound, boundary_value);
        dss_centered( m_fa, alpha, m_tempM, m_tempO, m_tempP, beta, g);
    }
    /// Same as \c dg::geo::dssd_centered
    void dssd( double alpha, const container& f, double
            beta, container& g){
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        m_fa(zeroForw,  f, m_tempO);
        dssd_centered( m_fa, alpha, m_tempM, m_tempO, m_tempP, beta, g);
    }
    /// Same as \c dg::geo::dssd_centered after \c dg::geo::ds_assign_bc_along_field_2nd
    void dssd_bc_along_field( double alpha, const
            container& f, double beta, container& g, dg::bc bound,
            std::array<double,2> boundary_value = {0,0}){
        m_fa(einsPlus, f, m_tempP);
        m_fa(einsMinus, f, m_tempM);
        m_fa(zeroForw,  f, m_tempO);
        assign_bc_along_field_2nd( m_fa, m_tempM, m_tempO, m_tempP, m_tempM, m_tempP,
                bound, boundary_value);
        dssd_centered( m_fa, alpha, m_tempM, f, m_tempP, beta, g);
    }

    /// The volume form with dG weights
    const container& weights()const {
        return m_fa.sqrtG();
    }

    /**
    * @brief access the underlying Fieldaligned object
    *
    * @return acces to Fieldaligned object
    */
    FA& fieldaligned(){return m_fa;}
    const FA& fieldaligned()const{return m_fa;}
    private:
    Fieldaligned<ProductGeometry, IMatrix, container> m_fa;
    container m_tempP, m_tempO, m_tempM;
};

///@cond
////////////////////////////////////DEFINITIONS////////////////////////////////////////

template<class Geometry, class I, class container>
DS<Geometry, I, container>::DS( Fieldaligned<Geometry, I, container> fa): m_fa(fa)
{
    m_tempP = fa.sqrtG(), m_tempM = m_tempO = m_tempP;
}

template<class G, class I, class container>
inline void DS<G,I,container>::ds( dg::direction dir, double alpha,
    const container& f, double beta, container& dsf) {
    switch( dir){
        case dg::centered:
        return centered( alpha, f, beta, dsf);
        case dg::forward:
        return forward( alpha, f, beta, dsf);
        case dg::backward:
        return backward( alpha, f, beta, dsf);
    }
}
template<class G, class I, class container>
inline void DS<G,I,container>::div( dg::direction dir, double alpha,
    const container& f, double beta, container& dsf) {
    switch( dir){
        case dg::centered:
        return divCentered( alpha, f, beta, dsf);
        case dg::forward:
        return divForward( alpha, f, beta, dsf);
        case dg::backward:
        return divBackward( alpha, f, beta, dsf);
    }
}


template<class G,class I, class container>
void DS<G,I,container>::symv( double alpha, const container& f, double beta, container& dsTdsf)
{
    dssd( alpha, f, beta, dsTdsf);
};
///@endcond

/**
* @brief forward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
*
* forward derivative \f$ g_i = \alpha \frac{v^\varphi}{\Delta\varphi}(f_{i+1} - f_{i}) + \beta g_i\f$
* @param fa this object will be used to get grid distances
* @copydoc hide_ds_parameters4
* @copydoc hide_ds_fp
* @ingroup fieldaligned
* @copydoc hide_ds_freestanding
*/
template<class FieldAligned, class container>
void ds_forward(const FieldAligned& fa, double alpha, const container& f,
    const container& fp, double beta, container& g)
{
    //direct
    double delta = fa.deltaPhi();
    dg::blas1::subroutine( [ alpha, beta, delta]DG_DEVICE(
            double& dsf, double fo, double fp, double bphi){
                dsf = alpha*bphi*( fp - fo)/delta + beta*dsf;
            },
            g, f, fp, fa.bphi());
}
/**
* @brief 2nd order forward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
*
* forward derivative \f$ g_i = \alpha \frac{v^\varphi}{2\Delta\varphi}(-f_{i+2} + 4f_{i+1} - 3f_{i}) + \beta g_i\f$
* @param fa this object will be used to get grid distances
* @copydoc hide_ds_parameters4
* @copydoc hide_ds_fpp
* @ingroup fieldaligned
* @copydoc hide_ds_freestanding
*/
template<class FieldAligned, class container>
void ds_forward2(const FieldAligned& fa, double alpha, const container& f,
    const container& fp, const container& fpp, double beta, container& g)
{
    //direct
    double delta = fa.deltaPhi();
    dg::blas1::subroutine( [ alpha, beta, delta]DG_DEVICE(
            double& dsf, double fo, double fp, double fpp, double bphi){
                dsf = alpha*bphi*( -3.*fo + 4.*fp - fpp)/2./delta
                     + beta*dsf;
            },
            g, f, fp, fpp, fa.bphi());
}

/**
* @brief backward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
*
* backward derivative \f$ g_i = \alpha \frac{v^\varphi}{\Delta\varphi}(f_{i} - f_{i-1}) + \beta g_i \f$
* @param fa this object will be used to get grid distances
* @copydoc hide_ds_parameters4
* @copydoc hide_ds_fm
* @ingroup fieldaligned
* @copydoc hide_ds_freestanding
*/
template<class FieldAligned, class container>
void ds_backward( const FieldAligned& fa, double alpha, const container& fm,
    const container& f, double beta, container& g)
{
    //direct
    double delta = fa.deltaPhi();
    dg::blas1::subroutine( [ alpha, beta, delta] DG_DEVICE(
            double& dsf, double fo, double fm, double bphi){
                dsf = alpha*bphi*( fo - fm)/delta + beta*dsf;
            },
            g, f, fm, fa.bphi());

}
/**
* @brief 2nd order backward derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
*
* backward derivative \f$ g_i = \alpha \frac{v^\varphi}{2\Delta\varphi}(3f_{i} - 4f_{i-1} + f_{i-2}) + \beta g_i \f$
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
    double delta = fa.deltaPhi();
    dg::blas1::subroutine( [ alpha, beta, delta] DG_DEVICE(
            double& dsf, double fo, double fm,  double fmm, double bphi){
                dsf = alpha*bphi*( 3.*fo - 4.*fm + fmm)/2./delta
                    + beta*dsf;
            },
            g, f, fm, fmm, fa.bphi());

}


/**
* @brief centered derivative \f$ g = \alpha \vec v \cdot \nabla f + \beta g\f$
*
* The formula used is
* \f$ g_i = \alpha \frac{v^\varphi}{2\Delta\varphi}(f_{i+1} - f_{i-1}) + \beta g_i \f$
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
void ds_centered( const FieldAligned& fa, double alpha, const container& fm,
        const container& fp, double beta, container& g)
{
    //direct discretisation
    double delta=fa.deltaPhi();
    dg::blas1::subroutine( [alpha,beta,delta]DG_DEVICE( double& g, double fm,
        double fp, double bphi){
            g = alpha*bphi*(fp-fm)/2./delta + beta*g;
            }, g, fm, fp, fa.bphi());
}
/**
 * @brief Centered derivative \f$ g = \alpha (\vec v\cdot \nabla)^2 f + \beta g \f$
 *
 * The formula used is the one that is obtained if one thinks of \f$ \nabla_\parallel f\f$ on a staggered grid and then deriving the result i.e. \c dg::geo::ds_forward followed by \c dg::geo::ds_backward
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
    dg::blas1::subroutine( detail::DSSCentered( alpha, beta, fa.deltaPhi()),
        g, fm, f, fp, fa.bphiM(), fa.bphi(), fa.bphiP());
}
/**
 * @brief Centered derivative \f$ g = \alpha \nabla\cdot(\vec v \vec v\cdot \nabla) f + \beta g \f$
 *
 * The formula used is \f[ \Delta_\parallel f = \nabla\cdot (\vec v \nabla_\parallel f )\f]
 * @param fa this object will be used to get grid distances
 * @copydoc hide_ds_parameters4
 * @copydoc hide_ds_fm
 * @copydoc hide_ds_fp
 * @ingroup fieldaligned
 * @copydoc hide_ds_freestanding
 */
template<class FieldAligned, class container>
void dssd_centered( const FieldAligned& fa, double alpha, const container& fm,
    const container& f, const container& fp, double beta, container& g)
{
    dg::blas1::subroutine( detail::DSSDCentered( alpha, beta, fa.deltaPhi()),
        g, fm, f, fp, fa.sqrtGm(), fa.sqrtG(), fa.sqrtGp(),
        fa.bphiM(), fa.bphi(), fa.bphiP());
}

/**
 * @brief backward derivative \f$ g = \alpha \nabla \cdot \vec v f + \beta g\f$
 *
 * backward derivative \f$ g_i = \alpha \frac{1}{\Delta\varphi\sqrt{G_i}}(\sqrt{G_{i}}v^\varphi_{i}f_{i} - \sqrt{G_{i-1}}v^\varphi_{i-1}f_{i-1}) + \beta g_i\f$
 * @param fa this object will be used to get grid distances
 * @param alpha Scalar
 * @copydoc hide_ds_fm
 * @param f The vector to derive
 * @param beta Scalar
 * @param g contains result on output (may alias input vectors)
 * @ingroup fieldaligned
 * @copydoc hide_ds_freestanding
*/
template<class FieldAligned, class container>
void ds_divBackward( const FieldAligned& fa, double alpha, const container& fm,
    const container& f, double beta, container& g)
{
    double delta = fa.deltaPhi();
    dg::blas1::subroutine( [alpha,beta,delta] DG_DEVICE( double& dsf, double f0,
            double f1, double Gm, double G0, double bPm, double bP0){
                dsf = alpha*(bP0*G0*f0 - bPm*Gm*f1)/G0/delta + beta*dsf; },
            g, f, fm, fa.sqrtGm(), fa.sqrtG(), fa.bphiM(), fa.bphi());
}

/**
 * @brief forward derivative \f$ g = \alpha \nabla \cdot \vec v f + \beta g\f$
 *
 * forward derivative \f$ g_i = \alpha \frac{1}{\Delta\varphi\sqrt{G_i}}(\sqrt{G_{i+1}}v^\varphi_{i+1}f_{i+1} - \sqrt{G_{i}}v^\varphi_{i}f_{i}) + \beta g_i\f$
 * @param fa this object will be used to get grid distances
 * @param alpha Scalar
 * @param f The vector to derive
 * @copydoc hide_ds_fp
 * @param beta Scalar
 * @param g contains result on output (may alias input vectors)
 * @ingroup fieldaligned
 * @copydoc hide_ds_freestanding
*/
template<class FieldAligned, class container>
void ds_divForward( const FieldAligned& fa, double alpha, const container& f,
    const container& fp, double beta, container& g)
{
    double delta = fa.deltaPhi();
    dg::blas1::subroutine( [alpha,beta,delta] DG_DEVICE( double& dsf, double f0,
            double f1, double Gp, double G0, double bPp, double bP0){
                dsf = alpha*(bPp*Gp*f1 - bP0*G0*f0)/G0/delta + beta*dsf; },
            g, f, fp, fa.sqrtGp(), fa.sqrtG(), fa.bphiP(), fa.bphi());
}
/**
 * @brief centered derivative \f$ g = \alpha \nabla \cdot \vec v f + \beta g\f$
 *
 * centered derivative \f$ g_i = \alpha \frac{1}{2\Delta\varphi\sqrt{G_i}}(\sqrt{G_{i+1}}v^\varphi_{i+1}f_{i+1} - \sqrt{G_{i-1}}v^\varphi_{i-1}f_{i-1}) + \beta g_i\f$
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
void ds_divCentered( const FieldAligned& fa, double alpha, const container& fm, const container& fp,
    double beta, container& g)
{
    double delta = fa.deltaPhi();
    dg::blas1::subroutine( [alpha,beta,delta]DG_DEVICE( double& dsf, double fm,
        double fp, double Gm, double Gp, double G0,
        double bPm, double /*bP0*/, double bPp)
        {
            dsf = alpha*( fp*Gp*bPp - fm*Gm*bPm )/G0/2./delta + beta*dsf;
        }, g, fm, fp, fa.sqrtGm(),
        fa.sqrtGp(), fa.sqrtG(), fa.bphiM(), fa.bphi(), fa.bphiP());

}

/**
 * @brief Compute average along a fieldline \f$ g = \alpha \frac{f_{k+1} + f_{k-1}}{2} + \beta g\f$
 *
 * @note The idea is to use this function on staggered grid implementations. There
 * it can be used to convert a function from one grid to the staggered grid and
 * back
 *
 * [un-named FieldAligned] this object is ignored in this function but it is
 * kept to keep the interface consistent with the other ds-type functions
 * @param alpha Scalar
 * @copydoc hide_ds_fm
 * @copydoc hide_ds_fp
 * @param beta Scalar
 * @param g contains result on output (may alias input vectors)
 * @ingroup fieldaligned
 * @copydoc hide_ds_freestanding
 */
template<class FieldAligned, class container>
void ds_average( const FieldAligned&, double alpha,
        const container& fm, const container& fp, double beta, container& g)
{
    dg::blas1::subroutine( [alpha,beta]DG_DEVICE( double& g, double fm, double fp
                ){
            g = alpha*(fp+fm)/2. + beta*g;
            }, g, fm, fp);
}
/**
 * @brief Compute simple slope along a fieldline \f$ g = \alpha v^\varphi\frac{f_{k+1} - f_{k-1}}{2\Delta\varphi} + \beta g\f$
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
 * @sa Does exactly the same as \c ds_centered
 */
template<class FieldAligned, class container>
void ds_slope( const FieldAligned& fa, double alpha,
        const container& fm, const container& fp, double beta, container& g)
{
    ds_centered( fa, alpha, fm, fp, beta, g);
}


}//namespace geo

///@cond
template< class G, class I, class V>
struct TensorTraits< geo::DS<G,I, V> >
{
    using value_type = double;
    using tensor_category = SelfMadeMatrixTag;
};
///@endcond
}//namespace dg
