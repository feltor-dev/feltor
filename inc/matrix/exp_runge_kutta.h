#include "dg/algorithm.h"
#include "matrixsqrt.h"
#include "lanczos.h"
#include "tableau.h"

namespace dg {
namespace mat {

/**
* @brief Exponential one step time-integration for \f$ \dot y = A y \f$
*
* \f[
 \begin{align}
    y^{n+1} = \exp(-\Delta t A) y^n
 \end{align}
\f]

* @copydoc hide_ContainerType
* @ingroup exp_int
*/
template<class ContainerType>
struct ExponentialStep
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ExponentialStep() = default;
    ExponentialStep( const ContainerType& copyable): m_u(copyable)
    { }
    template< class MatrixFunction>
    void step( const MatrixFunction& ode, value_type t0,
            const ContainerType& u0, value_type& t1, ContainerType& u1,
            value_type dt)
    {
        ode( [dt](value_type x){
                        return (exp( dt*x));}, u0, m_u);
        dg::blas1::copy( m_u, u1);
        t1 = t0 + dt;
    }
    private:
    ContainerType m_u;
};

/**
* @brief Exponential Runge-Kutta fixed-step time-integration for \f$ \dot y = A y + \hat E(t,y)\f$
*
* We follow <a href="https://doi.org/10.1017/S0962492910000048" target="_blank">Hochbruck and Ostermann, Exponential Integrators, Acta Numerica (2010)</a>
* \f[
 \begin{align}
    u^{n+1} = \exp(\Delta t A) u^n + \Delta t \sum_{i=1}^s b_i(\Delta t A) \hat E(t^n + c_i \Delta t, U_{ni}) \\
    U_{ni}  = \exp(c_i\Delta t A) u^n + \Delta t \sum_{j=1}^{i-1} a_{ij}(\Delta t A) \hat E(t^n + c_j \Delta t, U_{nj})
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
@note In exponential Rosenbrock type schemes it is assumed that \f$ A\f$ is the
Jacobian of the system. If it is not, then the order conditions are different
and the order and embedded orders are not what is indicated in our names.
*
* @note Uses only \c dg::blas1 routines to integrate one step.
* @copydoc hide_ContainerType
* @ingroup exp_int
*/
template<class ContainerType>
struct ExponentialERKStep
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ExponentialERKStep() = default;
    ExponentialERKStep( ConvertsToFunctionalButcherTableau<value_type> tableau, const
            ContainerType& copyable): m_rk(tableau), m_k(m_rk.num_stages(),
                copyable), m_tmp(copyable), m_u(copyable)
    { }
    const ContainerType& copyable()const{ return m_tmp;}

    template< class ExplicitRHS, class MatrixFunction>
    void step( const std::tuple<ExplicitRHS,MatrixFunction>& ode, value_type t0, const
            ContainerType& u0, value_type& t1, ContainerType& u1, value_type
            dt, ContainerType& delta)
    {
        step ( ode, t0, u0, t1, u1, dt, delta, true);
    }

    template< class ExplicitRHS, class MatrixFunction>
    void step( const std::tuple<ExplicitRHS, MatrixFunction>& ode, value_type t0,
            const ContainerType& u0, value_type& t1, ContainerType& u1,
            value_type dt)
    {
        step ( ode, t0, u0, t1, u1, dt, m_tmp, false);
    }
    ///global order of the method given by the current Butcher Tableau
    unsigned order() const {
        return m_rk.order();
    }
    ///global order of the embedding given by the current Butcher Tableau
    unsigned embedded_order() const {
        return m_rk.embedded_order();
    }
    ///number of stages of the method given by the current Butcher Tableau
    unsigned num_stages() const{
        return m_rk.num_stages();
    }

    private:
    template< class ExplicitRHS, class MatrixFunction>
    void step( const std::tuple<ExplicitRHS, MatrixFunction>& ode, value_type t0, const ContainerType& u0, value_type& t1, ContainerType& u1, value_type dt, ContainerType& delta, bool compute_delta)
    {
        // There may be a formulation that saves on a few matrix function applications
        unsigned s = m_rk.num_stages();
        value_type tu = t0;
        if( t0 != m_t1 )
        {
            std::get<0>(ode)(t0, u0, m_k[0]); //freshly compute k_0
            //std::cout << "t0 "<<t0<<" u0 "<<u0<<" k0 "<<m_k[0]<<"\n";
        }
        for ( unsigned i=1; i<s; i++)
        {
            std::get<1>(ode)( [&](value_type x){
                        return (exp( m_rk.c(i)*dt*x));}, u0, m_u);
            for( unsigned j=0; j<i; j++)
            {
                //std::cout << "a(i j) "<<i<<" "<<j<<"\n";
                if ( m_rk.a(i,j)(0) != 0)
                {
                    std::get<1>(ode)([&](value_type x){ return
                            m_rk.a(i,j)(dt*x);}, m_k[j], m_tmp);
                    dg::blas1::axpby( dt, m_tmp, 1., m_u);
                }
            }
            tu = DG_FMA( dt,m_rk.c(i),t0); //l=0
            std::get<0>(ode)( tu, m_u, m_k[i]);
        }
        //Now add everything up to get solution and error estimate
        // can't use u1 cause u0 can alias u1
        std::get<1>(ode)( [dt](value_type x){
                    return (exp( dt*x));}, u0, m_u);
        //std::cout <<" exp ( dt A) u0 = "<< m_u<<"\n";
        if( compute_delta)
            dg::blas1::copy( u1, delta);
        for( unsigned i=0; i<s; i++)
        {
            //std::cout << "b(i) "<<i<<"\n";
            if( m_rk.b(i)(0) != 0)
            {
                std::get<1>(ode)([&](value_type x){ return
                            m_rk.b(i)(dt*x);}, m_k[i], m_tmp);
                //std::cout << "b_i ( h A) (k) = "<<m_tmp<<"\n";
                dg::blas1::axpby( dt, m_tmp, 1., m_u);
            }
            if( compute_delta && m_rk.bt(i)(0) != 0)
            {
                std::get<1>(ode)([&](value_type x){ return
                            m_rk.bt(i)(dt*x);}, m_k[i], m_tmp);
                dg::blas1::axpby( dt, m_tmp, 1., delta);
            }
        }
        dg::blas1::copy( m_u, u1);
        if( compute_delta)
            dg::blas1::axpby( 1., u1, -1., delta);
        //make sure (t1,u1) is the last call to f
        m_t1 = t1 = t0 + dt;
        std::get<0>(ode)( t1, u1, m_k[0]);
        //std::cout << "t1 "<<t1<<" u1 "<<u0<<" k0 "<<m_k[0]<<"\n";
    }
    FunctionalButcherTableau<value_type> m_rk;
    std::vector<ContainerType> m_k;
    value_type m_t1 = 1e300;//remember the last timestep at which ERK is called
    ContainerType m_tmp, m_u;

};

} //namespace matrix
} //namespace dg
