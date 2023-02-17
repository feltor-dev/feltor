#include "dg/algorithm.h"
#include "matrixsqrt.h"
#include "lanczos.h"
#include "tableau.h"

namespace dg {
namespace mat {

/**
* @brief Exponential Runge-Kutta fixed-step time-integration for \f$ \dot y = A y + \hat E(t,y)\f$
*
* We follow  <a href="https://www.cambridge.org/core/journals/acta-numerica/article/exponential-integrators/8ED12FD70C2491C4F3FB7A0ACF922FCD" target="_blank">Hochbruck and Ostermann, Exponential Integrators, Acta Numerica (2010)</a>
* \f[
 \begin{align}
    u^{n+1} = \exp(-\Delta t A) u^n + \Delta t \sum_{i=1}^s b_i(-\Delta t A) \hat E(t^n + c_i \Delta t, U_{ni} \\
    U_{ni}  = \exp(-c_i\Delta t A) u^n + \Delta t \sum_{j=1}^{i-1} a_{ij}(-\Delta t A) \hat E(t^n + c_j \Delta t, U_{nj}
 \end{align}
\f]

The method is defined by the coefficient matrix functions \f$a_{ij}(z),\ b_i(z),\ c_i = \sum_{j=1}^{i-1} a_{ij}(0)\f$
 and \c s is the number
of stages. For \f$ A=0\f$ the underlying Runge-Kutta method with Butcher Tableau
\f$ a_{ij} = a_{ij}(0),\ b_i=b_i(0)\f$ is recovered.
We introduce the functions
\begin{align}
\varphi_{k+1}(z) := \frac{\varphi_k(z) - \varphi_k(0)}{z} \\
\varphi_0(z) = \exp(z)
\end{align}
and the shorthand notation \f$ \varphi_{j,k} := \varphi_j(-c_k\Delta t A),\ \varphi_j := \varphi_j(-\Delta tA)\f$.

You can provide your own coefficients or use one of our predefined methods:
@copydoc hide_func_explicit_butcher_tableaus
*
* @note Uses only \c dg::blas1 routines to integrate one step.
* @copydoc hide_ContainerType
* @ingroup exp_int
*/
template<class ContainerType>
struct ExpRungeKutta
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ExpRungeKutta() = default;
    const ContainerType& copyable()const{ return m_k;}

    template< class ExplicitRHS, class MatrixFunction>
    void step( const std::tuple<ExplicitRHS, MatrixFunction>& ode, value_type t0, const ContainerType& u0, value_type& t1, ContainerType& u1, value_type dt)
    {

        std::get<0>(ode)(t0, u0, m_k);
        std::get<1>(ode)( [dt](value_type x){ return (exp( -dt*x-1))/(-dt*x);}, m_k, m_tmp);
        std::get<1>(ode)( [dt](value_type x){ return exp( -dt*x);}, u0, u1);
        dg::blas1::axpby( 1., m_tmp, 1., u1);
    }
    private:
    ContainerType m_k, m_tmp;
    FunctionalButcherTableau<value_type> m_b;

};

} //namespace matrix
} //namespace dg
