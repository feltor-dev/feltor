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

///@addtogroup profiles
///@{

/**
 * @brief <tt>(CylindricalFunctor psip, double psip_min, double psip_max)</tt>
     \f[ f(R,Z) = \begin{cases}
        1  \text{ if } \psi_{p,min} < \psi_p(R,Z) < \psi_{p,max}\\
        0  \text{ else}
     \end{cases}\f]
 */
using Iris = Compose<dg::Iris>;
/**
 * @brief <tt> (CylindricalFunctor psip, double psipmax) </tt>
     \f[ f(R,Z) = \begin{cases}
        0  \text{ if } \psi_p(R,Z) > \psi_{p,max} \\
        1  \text{ else}
     \end{cases}\f]
 */
using Pupil = Compose<dg::Pupil>;

/**
 * @brief <tt> (CylindricalFunctor psip, double psipmax)</tt>
     \f[ f(R,Z) = \begin{cases}
        \psi_{p,max}  \text{ if } \psi_p(R,Z) > \psi_{p,max} \\
        \psi_p(R,Z) \text{ else}
     \end{cases}\f]
 */
using PsiPupil = Compose<dg::PsiPupil>;

/**
 * @brief <tt> (CylindricalFunctor psip, double psipmax) </tt>
     \f[ f(R,Z) = \begin{cases}
        1  \text{ if } \psi_p(R,Z) > \psi_{p,\max} \\
        0  \text{ else}
     \end{cases}\f]
 */
using PsiLimiter = dg::geo::Compose<dg::Heaviside>;

/**
 * @brief <tt>(CylindricalFunctor psip, double psipmax, double alpha)</tt>
 * \f[ f(R,Z) = \begin{cases}
 1 \text{ if } \psi_p(R,Z) < \psi_{p,max}\\
 0 \text{ if } \psi_p(R,Z) > (\psi_{p,max} + 4\alpha) \\
 \exp\left( - \frac{(\psi_p(R,Z) - \psi_{p,max})^2}{2\alpha^2}\right), \text{ else}
 \end{cases}
   \f]
 */
using GaussianDamping = dg::geo::Compose<dg::GaussianDamping>;

/**
 * @brief <tt>(CylindricalFunctor psip, double psipmax, double alpha, int sign =1,double B = 0., double A = 1.) </tt>
 * \f[ f(R,Z) = B + 0.5 A(1+ \text{sign} \tanh((\psi_p(R,Z)-\psi_{p,max})/\alpha ) ) \f]

 Similar to GaussianProfDamping is zero outside given \c psipmax and
 one inside of it.
 */
using TanhDamping = dg::geo::Compose<dg::TanhProfX>;

/**
 * @brief <tt> (CylindricalFunctor psip, double A, double B)</tt>
 *\f[ f(R,Z)= B + A\psi_p(R,Z)\f]
 */
using Nprofile = dg::geo::Compose<dg::LinearX>;

/**
 * @brief <tt>(CylindricalFunctor psip, double A, double B, double k_psi)</tt>
     \f[ f(R,Z)= B + A \sin(k_\psi \psi_p(R,Z) ) \f]
 */
using ZonalFlow = dg::geo::Compose<dg::SinX>;

/**
 * @brief <tt> (double Z_X)</tt>
 \f[ f(R,Z)= \begin{cases}
 1 \text{ if } Z > Z_X \\
 0 \text{ else }
 \end{cases}
 \f]
 */
struct ZCutter : public aCylindricalFunctor<ZCutter>
{
    ZCutter(double ZX): Z_X(ZX){}
    double do_compute(double R, double Z) const {
        if( Z > Z_X)
            return 1;
        return 0;
    }
    private:
    double Z_X;
};


///@}
}//namespace functors
}//namespace dg

