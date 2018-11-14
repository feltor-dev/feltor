#pragma once
//#include "solovev.h"
#include "fluxfunctions.h"
#include "solovev_parameters.h"

/*!@file
 *
 * Initialize and Damping objects
 */
namespace dg
{
namespace geo
{

/**
* @brief Composition \f[ f\circ\psi = f(\psi(R,Z)) \f]
*
* @tparam UnaryFunctor A unary Functor with interface <tt>double (double x)</tt>
* @ingroup profiles
*/
template<class UnaryFunctor>
struct Compose : public aCloneableCylindricalFunctor<Compose<UnaryFunctor>>
{
    /**
    * @brief Construct from 2d functor and forward any parameters to \c UnaryFunctor
    *
    * @param psi A binary functor
    * @param ps Parameters that are forwarded to the constructor of \c UnaryFunctor
    * @tparam FunctorParams Determined by Compiler
    */
    template<class ...FunctorParams>
    Compose ( CylindricalFunctor psi, FunctorParams&& ... ps): m_psip(psi),
        m_f(std::forward<FunctorParams>(ps)...){}
    private:
    double do_compute( double R, double Z) const
    {
        return m_f(m_psip(R,Z));
    }
    CylindricalFunctor m_psip;
    UnaryFunctor m_f;
};
///@addtogroup profiles
///@{

/**
 * @brief Returns zero outside psipmax and inside psipmin, otherwise 1
     \f[ \begin{cases}
        1  \text{ if } \psi_{p,min} < \psi_p(R,Z) < \psi_{p,max}\\
        0  \text{ else}
     \end{cases}\f]
 */
using Iris = Compose<dg::Iris>;
/**
 * @brief Returns zero outside psipmax, otherwise 1
     \f[ \begin{cases}
        0  \text{ if } \psi_p(R,Z) > \psi_{p,max} \\
        1  \text{ else}
     \end{cases}\f]
 */
using Pupil = Compose<dg::Pupil>;
/**
 * @brief Returns psi inside psipmax and psipmax outside psipmax
     \f[ \begin{cases}
        \psi_{p,max}  \text{ if } \psi_p(R,Z) > \psi_{p,max} \\
        \psi_p(R,Z) \text{ else}
     \end{cases}\f]
 */
using PsiPupil = Compose<dg::PsiPupil>;

/**
 * @brief Sets values to one outside psipmax, zero else
     \f[ \begin{cases}
        1  \text{ if } \psi_p(R,Z) > \psi_{p,\max} \\
        0  \text{ else}
     \end{cases}\f]
 *
 */
using PsiLimiter = dg::geo::Compose<dg::Heaviside>;

/**
 * @brief Damps the outer boundary in a zone
 * from psipmax to psipmax+ 4*alpha with a normal distribution
 * Returns 1 inside, zero outside and a gaussian within
     \f[ \begin{cases}
 1 \text{ if } \psi_p(R,Z) < \psi_{p,max}\\
 0 \text{ if } \psi_p(R,Z) > (\psi_{p,max} + 4\alpha) \\
 \exp\left( - \frac{(\psi_p - \psi_{p,max})^2}{2\alpha^2}\right), \text{ else}
 \end{cases}
   \f]
 *
 */
using GaussianDamping = dg::geo::Compose<dg::GaussianDamping>;

/**
 * @brief
 * A tanh profile
 \f[ 0.5\left( 1 + \tanh\left( \frac{\psi_p(R,Z) - \psi_{p,max} }{\alpha}\right)\right) \f]

 Similar to GaussianProfDamping is zero outside given \c psipmax and
 one inside of it.
 */
using TanhDamping = dg::geo::Compose<dg::TanhProfX>;

/**
 * @brief Density profile with variable peak and background amplitude
 *\f[ N(R,Z)= B + A\psi_p(R,Z)\f]
 */
using Nprofile = dg::geo::Compose<dg::LinearX>;

/**
 * @brief Zonal flow field
     \f[ N(R,Z)= B + A \sin(k_\psi \psi_p(R,Z) ) \f]
 */
using ZonalFlow = dg::geo::Compose<dg::SinX>;

/**
 * @brief Cut everything below X-point
 \f[ \begin{cases}
 1 \text{ if } Z > Z_X \\
 0 \text{ else }
 \end{cases}
 \f]
 */
struct ZCutter : public aCloneableCylindricalFunctor<ZCutter>
{
    ZCutter(double ZX): Z_X(ZX){}
    private:
    double do_compute(double R, double Z) const {
        if( Z > Z_X)
            return 1;
        return 0;
    }
    double Z_X;
};


///@}
}//namespace functors
}//namespace dg

