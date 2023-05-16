#include "dg/algorithm.h"
#include "matrixsqrt.h"
#include "lanczos.h"
#include "tableau.h"

namespace dg {
namespace mat {

/** @class hide_explicit_rhs
 * @tparam ExplicitRHS The explicit (part of the) right hand side
 * is a functor type with no return value (subroutine)
 * of signature <tt> void operator()(value_type, const ContainerType&, ContainerType&)</tt>
 * The first argument is the time, the second is the input vector, which the
 * functor may \b not override, and the third is the output, i.e. y' = E(t, y)
 * translates to E(t, y, y').
 * The two ContainerType arguments never alias each other in calls to the functor.
 * The functor can throw to indicate failure. Exceptions should derive from
 * \c std::exception.
 */
/** @class hide_matrix_function
 * @tparam MatrixFunction The exponential of the (part of the) right hand side
 * is a functor type with no return value (subroutine)
 * of signature <tt> void operator()(UnaryOp, const ContainerType&, ContainerType&)</tt>
 * The first argument is the matrix function to compute, the second is the input vector, which the
 * functor may \b not override, and the third is the output. The timestepper will call
 * <tt> MatrixFunction(f, y, yp) </tt> where f typically is the exponential
 * or \c dg::mat::phi1, \c dg::mat::phi2, etc.
 * The FunctionalButcherTableau determines which matrix functions are used.
 * The timestepper expects that <tt> MatrixFunction(f, y, yp) </tt>  computes \f$ y' = f( A) y\f$.
 * The two ContainerType arguments never alias each other in calls to the functor.
 * The functor can throw to indicate failure. Exceptions should derive from
 * \c std::exception.
 */

/**
* @brief Exponential one step time-integration for \f$ \dot y = A y \f$
*
* This integrator computes the exact solution
* \f[
 \begin{align}
    y^{n+1} = \exp(-\Delta t A) y^n
 \end{align}
\f]

* This class is usable in the \c dg::SinglestepTimeloop
* @copydoc hide_ContainerType
* @ingroup exp_int
*/
template<class ContainerType>
struct ExponentialStep
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    /// no memory allocation
    ExponentialStep() = default;
    /**
    * @brief Reserve internal workspace for the integration
    *
    * @param copyable vector of the size that is later used in \c step (
     it does not matter what values \c copyable contains, but its size is important;
     the \c step method can only be called with vectors of the same size)
    */
    ExponentialStep( const ContainerType& copyable): m_u(copyable)
    { }
    const ContainerType& copyable()const{ return m_u;}
    /**
    * @brief Advance one step via \f$ u_1 = \exp( A \Delta t) u_0\f$, \f$ t_1 = t_0 + \Delta_t\f$
    *
    * @copydoc hide_matrix_function
    * @param ode object that computes matrix functions (for this class always the exponential)
    * of the right hand side
    * @param t0 start time
    * @param u0 value at \c t0
    * @param t1 (write only) end time ( equals \c t0+dt on return, may alias \c t0)
    * @param u1 (write only) contains result on return (may alias u0)
    * @param dt timestep
    */
    template< class MatrixFunction>
    void step( MatrixFunction& ode, value_type t0,
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
* @brief Exponential Runge-Kutta fixed-step time-integration for \f$ \dot y = A y + g(t,y)\f$
*
* We follow <a href="https://doi.org/10.1017/S0962492910000048" target="_blank">Hochbruck and Ostermann, Exponential Integrators, Acta Numerica (2010)</a>
* \f[
 \begin{align}
    u^{n+1} = \exp(\Delta t A) u^n + \Delta t \sum_{i=1}^s b_i(\Delta t A) g(t^n + c_i \Delta t, U_{ni}) \\
    U_{ni}  = \exp(c_i\Delta t A) u^n + \Delta t \sum_{j=1}^{i-1} a_{ij}(\Delta t A) g(t^n + c_j \Delta t, U_{nj})
 \end{align}
\f]

The method is defined by the coefficient matrix functions \f$a_{ij}(z),\ b_i(z),\ c_i = \sum_{j=1}^{i-1} a_{ij}(0)\f$
 and \c s is the number
of stages. For \f$ A=0\f$ the underlying Runge-Kutta method with Butcher Tableau
\f$ a_{ij} = a_{ij}(0),\ b_i=b_i(0)\f$ is recovered.
We introduce the functions
\f[
\begin{align}
\varphi_{k+1}(z) := \frac{\varphi_k(z) - \varphi_k(0)}{z} \\
\varphi_0(z) = \exp(z)
\end{align}
\f]
and the shorthand notation \f$ \varphi_{j,k} := \varphi_j(-c_k\Delta t A),\ \varphi_j := \varphi_j(-\Delta tA)\f$.

You can provide your own coefficients or use one of our predefined methods:
@copydoc hide_func_explicit_butcher_tableaus
*
* This class is usable in the \c dg::SinglestepTimeloop and \c dg::AdaptiveTimeloop
* @copydoc hide_ContainerType
* @ingroup exp_int
*/
template<class ContainerType>
struct ExponentialERKStep
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ExponentialERKStep() = default;
    /**
    * @brief Reserve internal workspace for the integration
    *
    * @param tableau Tableau, name or identifier that \c ConvertsToFunctionalButcherTableau
    * @param copyable vector of the size that is later used in \c step (
     it does not matter what values \c copyable contains, but its size is important;
     the \c step method can only be called with vectors of the same size)
    */
    ExponentialERKStep( ConvertsToFunctionalButcherTableau<value_type> tableau, const
            ContainerType& copyable): m_rk(tableau), m_k(m_rk.num_stages(),
                copyable), m_tmp(copyable), m_u(copyable)
    { }
    const ContainerType& copyable()const{ return m_tmp;}

    /// @brief Advance one step with error estimate
    ///@copydetails step(const std::tuple<ExplicitRHS,MatrixFunction>&,value_type,const ContainerType&,value_type&,ContainerType&,value_type)
    ///@param delta Contains error estimate (u1 - tilde u1) on return (must have equal size as \c u0)
    template< class ExplicitRHS, class MatrixFunction>
    void step( const std::tuple<ExplicitRHS,MatrixFunction>& ode, value_type t0, const
            ContainerType& u0, value_type& t1, ContainerType& u1, value_type
            dt, ContainerType& delta)
    {
        step ( ode, t0, u0, t1, u1, dt, delta, true);
    }

    /**
    * @brief Advance one step ignoring error estimate and embedded method
    *
    * @copydoc hide_explicit_rhs
    * @copydoc hide_matrix_function
    * @param ode the <explicit rhs, matrix_function> functor.
    * Typically \c std::tie(explicit_rhs, matrix_function)
    * @param t0 start time
    * @param u0 value at \c t0
    * @param t1 (write only) end time ( equals \c t0+dt on return, may alias \c t0)
    * @param u1 (write only) contains result on return (may alias u0)
    * @param dt timestep
    * @note on return \c explicit_rhs(t1, u1) will be the last call to \c explicit_rhs (this is
    * useful if \c ExplicitRHS holds state, which is thus updated to the current
    * timestep)
    */
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
