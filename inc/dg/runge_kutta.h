#ifndef _DG_RK_
#define _DG_RK_

#include <cassert>
#include <array>
#include <tuple>

#include "ode.h"
#include "backend/exceptions.h"
#include "tableau.h"
#include "blas1.h"
#include "implicit.h"

/*! @file
 * @brief Runge-Kutta explicit ODE-integrators
 */

namespace dg{
///@cond
namespace detail{
// this is a dense matrix-matrix multiplication
// a generalization of the corresponding blas2::gemv DenseMatrix algorithm
template<class ContainerType>
void gemm(
        const std::array<double,2>& alpha,
        const DenseMatrix<ContainerType>& m,
        const std::array<const std::vector<double>*,2> x,
        const std::array<double,2>& beta,
        std::array<ContainerType*,2>&& y)
{
    const unsigned size = m.num_cols();
    unsigned i=0;
    if( size >= 8)
        for( i=0; i<size/8; i++)
            dg::blas1::subroutine( dg::EmbeddedPairSum(),
                    *y[0], *y[1], i==0 ? beta[0] : 1., i==0 ? beta[1] : 1.,
                    alpha[0]*(*x[0])[i*8+0], alpha[1]*(*x[1])[i*8+0], *m.get()[i*8+0],
                    alpha[0]*(*x[0])[i*8+1], alpha[1]*(*x[1])[i*8+1], *m.get()[i*8+1],
                    alpha[0]*(*x[0])[i*8+2], alpha[1]*(*x[1])[i*8+2], *m.get()[i*8+2],
                    alpha[0]*(*x[0])[i*8+3], alpha[1]*(*x[1])[i*8+3], *m.get()[i*8+3],
                    alpha[0]*(*x[0])[i*8+4], alpha[1]*(*x[1])[i*8+4], *m.get()[i*8+4],
                    alpha[0]*(*x[0])[i*8+5], alpha[1]*(*x[1])[i*8+5], *m.get()[i*8+5],
                    alpha[0]*(*x[0])[i*8+6], alpha[1]*(*x[1])[i*8+6], *m.get()[i*8+6],
                    alpha[0]*(*x[0])[i*8+7], alpha[1]*(*x[1])[i*8+7], *m.get()[i*8+7]);
    unsigned l=0;
    if( size%8 >= 4)
        for( l=0; l<(size%8)/4; l++)
            dg::blas1::subroutine( dg::EmbeddedPairSum(),
                    *y[0], *y[1], size < 8 ?  beta[0]  : 1., size < 8 ? beta[1] : 1.,
                    alpha[0]*(*x[0])[i*8+l*4+0], alpha[1]*(*x[1])[i*8+l*4+0], *m.get()[i*8+l*4+0],
                    alpha[0]*(*x[0])[i*8+l*4+1], alpha[1]*(*x[1])[i*8+l*4+1], *m.get()[i*8+l*4+1],
                    alpha[0]*(*x[0])[i*8+l*4+2], alpha[1]*(*x[1])[i*8+l*4+2], *m.get()[i*8+l*4+2],
                    alpha[0]*(*x[0])[i*8+l*4+3], alpha[1]*(*x[1])[i*8+l*4+3], *m.get()[i*8+l*4+3]);
    unsigned k=0;
    if( (size%8)%4 >= 2)
        for( k=0; k<((size%8)%4)/2; k++)
            dg::blas1::subroutine( dg::EmbeddedPairSum(),
                    *y[0], *y[1], size < 4 ?  beta[0]  : 1., size < 4 ? beta[1] : 1.,
                    alpha[0]*(*x[0])[i*8+l*4+k*2+0], alpha[1]*(*x[1])[i*8+l*4+k*2+0], *m.get()[i*8+l*4+k*2+0],
                    alpha[0]*(*x[0])[i*8+l*4+k*2+1], alpha[1]*(*x[1])[i*8+l*4+k*2+1], *m.get()[i*8+l*4+k*2+1]);
    if( ((size%8)%4)%2 == 1)
        dg::blas1::subroutine( dg::EmbeddedPairSum(),
                    *y[0], *y[1], size < 2 ?  beta[0]  : 1., size < 2 ? beta[1] : 1.,
                    alpha[0]*(*x[0])[i*8+l*4+k*2], alpha[1]*(*x[1])[i*8+l*4+k*2], *m.get()[i*8+l*4+k*2]);

}
} // namespace detail
///@endcond

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
/** @class hide_implicit_rhs
 * @tparam ImplicitRHS The implicit (part of the) right hand side
 * is a functor type with no return value (subroutine)
 * of signature <tt> void operator()(value_type, const ContainerType&, ContainerType&)</tt>
 * The first argument is the time, the second is the input vector, which the
 * functor may \b not override, and the third is the output, i.e. y' = I(t, y)
 * translates to I(t, y, y').
 * The two ContainerType arguments never alias each other in calls to the functor.
 * The functor can throw to indicate failure. Exceptions should derive from
 * \c std::exception.
 */
/** @class hide_solver
 * @tparam Solver A functor type for the implicit right hand side.
 * Must solve the equation \f$ y - \alpha I(y,t) = y^*\f$ for \c y for  given
 * \c alpha, \c t and \c ys.
 * Alpha is always positive and non-zero.
 * Signature
 * <tt> void operator()( value_type alpha, value_type t, ContainerType& y, const ContainerType& ys); </tt>
 * The functor can throw. Any Exception should derive from \c std::exception.
  */
/*! @class hide_ode
 * @tparam ODE The ExplicitRHS or tuple type that corresponds to what is
 * inserted into the step member of the Stepper
 * @param ode rhs or tuple
 */
 /** @class hide_limiter
  * @tparam Limiter The filter or limiter class to use in connection with the
  * time-stepper has a member function \c operator()
  * of signature <tt> void operator()( ContainerType&)</tt>
  * The argument is the input vector that the function has to override
  * i.e. y = L( y) translates to L( y).
  */

/**
 * @brief A filter that does nothing
 * @ingroup time_utils
 */
struct IdentityFilter
{
    /**
     * @brief Do nothing
     *
     * @copydoc hide_ContainerType
     *
     * unnamed parameter (usually inout) remains unchanged
     */
    template<class ContainerType1>
    void operator()( ContainerType1&) const{ }
};
///@cond
template<class ContainerType>
struct FilteredERKStep;
///@endcond
//Should we try if filters inside RHS are equally usable?

///@addtogroup time
///@{


/**
* @brief Embedded Runge Kutta explicit time-step with error estimate
* \f$
 \begin{align}
    k_i = f\left( t^n + c_i \Delta t, u^n + \Delta t \sum_{j=1}^{i-1} a_{ij} k_j\right) \\
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j \\
    \tilde u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s \tilde b_j k_j \\
    \delta^{n+1} = u^{n+1} - \tilde u^{n+1} = \Delta t\sum_{j=1}^s (b_j - \tilde b_j) k_j
 \end{align}
\f$

The method is defined by its (extended explicit) ButcherTableau, given by
the coefficients \c a, \c b and \c c,  and \c s is the number
of stages. The embedding is given by the coefficients \c bt (tilde b).

You can provide your own coefficients or use one of the embedded methods
in the following table:
@copydoc hide_explicit_butcher_tableaus

* @note The name of this class is in reminiscence of the ARKode library https://sundials.readthedocs.io/en/latest/arkode/index.html
* @copydoc hide_ContainerType
*/
template< class ContainerType>
struct ERKStep
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@brief No memory allocation
    ERKStep() = default;
    /**
    * @brief Reserve internal workspace for the integration
    *
    * @param tableau Tableau, name or identifier that \c ConvertsToButcherTableau
    * @param copyable vector of the size that is later used in \c step (
     it does not matter what values \c copyable contains, but its size is important;
     the \c step method can only be called with vectors of the same size)
    */
    ERKStep( ConvertsToButcherTableau<value_type> tableau, const ContainerType&
        copyable): m_ferk(tableau, copyable)
    {
    }

    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = ERKStep( std::forward<Params>( ps)...);
    }
    ///@copydoc hide_copyable
    const ContainerType& copyable()const{ return m_ferk.copyable();}

    ///All subsequent calls to \c step method will ignore the first same as last property (useful if you want to implement an operator splitting)
    void ignore_fsal(){ m_ferk.ignore_fsal();}
    ///All subsequent calls to \c step method will enable the check for the first same as last property
    void enable_fsal(){ m_ferk.enable_fsal();}

    /// @brief Advance one step with error estimate
    ///@copydetails step(ExplicitRHS&,value_type,const ContainerType&,value_type&,ContainerType&,value_type)
    ///@param delta Contains error estimate (u1 - tilde u1) on return (must have equal size as \c u0)
    template<class ExplicitRHS>
    void step( ExplicitRHS& rhs, value_type t0, const ContainerType& u0, value_type& t1, ContainerType& u1, value_type dt, ContainerType& delta)
    {
        dg::IdentityFilter id;
        m_ferk.step( std::tie( rhs, id), t0, u0, t1, u1, dt, delta);
    }
    /**
    * @brief Advance one step ignoring error estimate and embedded method
    *
    * @copydoc hide_explicit_rhs
    * @param rhs right hand side subroutine
    * @param t0 start time
    * @param u0 value at \c t0
    * @param t1 (write only) end time ( equals \c t0+dt on return, may alias \c t0)
    * @param u1 (write only) contains result on return (may alias u0)
    * @param dt timestep
    * @note on return \c rhs(t1, u1) will be the last call to \c rhs (this is
    * useful if \c ExplicitRHS holds state, which is then updated to the current
    * timestep)
    * @note About the first same as last property (fsal): Some Butcher tableaus
    * (e.g. Dormand-Prince or Bogacki-Shampine) have the property that the last
    * value k_s of a timestep is the same as the first value k_0 of the next
    * timestep. This means that we can save one call to the right hand side.
    * This property is automatically activated if \c tableau.isFsal() returns
    * \c true and \c t0 equals \c t1 of the last call to \c step. You can
    * deactivate it by calling the \c ignore_fsal() method, which is useful for
    * splitting methods but increases the number of rhs calls by 1.
    * @attention On the rare occasion where you want to change the type of \c ExplicitRHS
    * from one step to the next the fsal property is interpreted wrongly and will
    * lead to wrong results. You will need to either re-construct the object or
    * set the ignore_fsal property before the next step.
    */
    template<class ExplicitRHS>
    void step( ExplicitRHS& rhs, value_type t0, const ContainerType& u0, value_type&
            t1, ContainerType& u1, value_type dt)
    {
        dg::IdentityFilter id;
        m_ferk.step( std::tie( rhs, id), t0, u0, t1, u1, dt);
    }
    ///global order of the method given by the current Butcher Tableau
    unsigned order() const {
        return m_ferk.order();
    }
    ///global order of the embedding given by the current Butcher Tableau
    unsigned embedded_order() const {
        return m_ferk.embedded_order();
    }
    ///number of stages of the method given by the current Butcher Tableau
    unsigned num_stages() const{
        return m_ferk.num_stages();
    }
  private:
    FilteredERKStep<ContainerType> m_ferk;
};

/**
* @brief EXPERIMENTAL: Filtered Embedded Runge Kutta explicit time-step with error estimate
* \f$
 \begin{align}
    k_i = f\left( t^n + c_i \Delta t, \Lambda\Pi \left[u^n + \Delta t \sum_{j=1}^{i-1} a_{ij} k_j\right]\right) \\
    u^{n+1} = \Lambda\Pi\left[u^{n} + \Delta t\sum_{j=1}^s b_j k_j\right] \\
    \delta^{n+1} = \Delta t\sum_{j=1}^s (\tilde b_j  - b_j) k_j
 \end{align}
\f$

@note
We compute \f$ \delta^{n+1} \f$ with the
unfiltered sum since with non-linear filters in the filtered sum, some error
components might not vanish and the timestepper crash. No filter is applied for
\f$ k_0\f$ since \f$ u^n\f$ is already filtered.
Even though it may look like it the filter **cannot** be absorbed into the
right hand side function f analytically. Also, the formulation is **not** equivalent
to that of the \c dg::ShuOsher class.

@copydetails ERKStep
*/
template< class ContainerType>
struct FilteredERKStep
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@copydoc ERKStep::ERKStep()
    FilteredERKStep() { m_k.resize(1); }
    ///@copydoc ERKStep::ERKStep(ConvertsToButcherTableau<value_type>,const ContainerType&)
    FilteredERKStep( ConvertsToButcherTableau<value_type> tableau, const
            ContainerType& copyable): m_rk(tableau), m_k(m_rk.num_stages(),
                copyable)
    {
        m_rkb.resize(m_k.size()), m_rkd.resize( m_k.size());
        for( unsigned i=0; i<m_k.size(); i++)
        {
            m_rkb[i] = m_rk.b(i);
            m_rkd[i] = m_rk.d(i);
        }
    }

    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = FilteredERKStep( std::forward<Params>( ps)...);
    }
    ///@copydoc hide_copyable
    const ContainerType& copyable()const{ return m_k[0];}

    ///All subsequent calls to \c step method will ignore the first same as last property (useful if you want to implement an operator splitting)
    void ignore_fsal(){ m_ignore_fsal = true;}
    ///All subsequent calls to \c step method will enable the check for the first same as last property
    void enable_fsal(){ m_ignore_fsal = false;}

    /// @brief Advance one step with error estimate
    ///@copydetails ERKStep::step(ExplicitRHS&,value_type,const ContainerType&,value_type&,ContainerType&,value_type)
    ///@param delta Contains error estimate (u1 - tilde u1) on return (must have equal size as \c u0)
    ///@copydoc hide_limiter
    template<class ExplicitRHS, class Limiter>
    void step( const std::tuple<ExplicitRHS,Limiter>& rhs, value_type t0, const ContainerType& u0, value_type& t1, ContainerType& u1, value_type dt, ContainerType& delta)
    {
        step ( rhs, t0, u0, t1, u1, dt, delta, true);
    }
    ///@copydoc ERKStep::step(ExplicitRHS&,value_type,const ContainerType&,value_type&,ContainerType&,value_type)
    ///@copydoc hide_limiter
    template<class ExplicitRHS, class Limiter>
    void step( const std::tuple<ExplicitRHS, Limiter>& rhs, value_type t0, const ContainerType& u0, value_type&
            t1, ContainerType& u1, value_type dt)
    {
        if( !m_tmp_allocated)
        {
            dg::assign( m_k[0], m_tmp);
            m_tmp_allocated = true;
        }
        step ( rhs, t0, u0, t1, u1, dt, m_tmp, false);
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
    template<class ExplicitRHS, class Limiter>
    void step( const std::tuple<ExplicitRHS, Limiter>& rhs, value_type t0, const ContainerType& u0, value_type& t1, ContainerType& u1, value_type dt, ContainerType& delta, bool);
    ButcherTableau<value_type> m_rk;
    std::vector<value_type> m_rkb, m_rkd;
    std::vector<ContainerType> m_k;
    value_type m_t1 = 1e300;//remember the last timestep at which ERK is called
    bool m_ignore_fsal = false;
    ContainerType m_tmp; //  only conditionally allocated
    bool m_tmp_allocated = false;
};

///@cond

template< class ContainerType>
template<class ExplicitRHS, class Limiter>
void FilteredERKStep<ContainerType>::step( const std::tuple<ExplicitRHS, Limiter>& ode, value_type t0, const ContainerType& u0, value_type& t1, ContainerType& u1, value_type dt, ContainerType& delta, bool compute_delta)
{
    unsigned s = m_rk.num_stages();
    std::vector<const ContainerType*> k_ptrs = dg::asPointers( m_k);
    //0 stage: probe
    value_type tu = t0;
    if( t0 != m_t1 || m_ignore_fsal)
        std::get<0>(ode)(t0, u0, m_k[0]); //freshly compute k_0
    //else take from last call
    for ( unsigned i=1; i<s; i++)
    {
        std::vector<value_type> rka( i);
        for( unsigned l=0; l<i; l++)
            rka[l] = m_rk.a(i,l);

        tu = DG_FMA( dt,m_rk.c(i),t0); //l=0
        dg::blas1::copy( u0, delta); // can't use u1 here cause u0 can alias
        dg::blas2::gemv( dt, dg::asDenseMatrix( k_ptrs, i), rka, 1., delta);

        std::get<1>(ode)( delta);
        std::get<0>(ode)( tu, delta, m_k[i]);
    }
    //Now add everything up to get solution and error estimate
    dg::blas1::copy( u0, u1);
    if( compute_delta)
        detail::gemm( {dt,dt}, dg::asDenseMatrix(k_ptrs), {&m_rkb, &m_rkd},
            {1.,0.}, {&u1, &delta});
    else
        blas2::gemv( dt, dg::asDenseMatrix(k_ptrs), m_rkb, 1., u1);
    std::get<1>(ode)( u1);
    //make sure (t1,u1) is the last call to f
    m_t1 = t1 = t0 + dt;
    if(!m_rk.isFsal() )
        std::get<0>(ode)( t1, u1, m_k[0]);
    else
    {
        using std::swap;
        swap( m_k[0], m_k[s-1]); //enable free swap functions
    }
}
///@endcond


/*!
 * @brief Additive Runge Kutta (semi-implicit) time-step with error estimate
 * following
 * <a href="https://sundials.readthedocs.io/en/latest/arkode/Mathematics_link.html#arkstep-additive-runge-kutta-methods">The ARKode library</a>
 *
 * Currently, the possible Butcher Tableaus for a fully implicit-explicit scheme
 * are the "Cavaglieri-3-1-2", "Cavaglieri-4-2-3", "ARK-4-2-3", "ARK-6-3-4" and "ARK-8-4-5" combinations.
 * @note All currently possible schemes enjoy the FSAL qualitiy in the sense
 * that only \c s-1 implicit solves and \c 1 evaluation of the implicit part
 * are needed per step; the Cavaglieri methods do not require
 * evaluations of the implicit part at all
 * @attention When you use the ARKStep in combination with the Adaptive time
 * step algorithm pay attention to solve the implicit part with sufficient
 * accuracy. Else, the error propagates into the time controller, which will
 * then choose the timestep as if the implicit part was explicit i.e. far too
 * small. This might have to do with stiffness-leakage [Kennedy and Carpenter, Appl. num. Math., (2003)]:
"An essential requirement for the viability of stiff/nonstiff IMEX schemes is that the stiffness remains
truely separable. If this were not the case then stiffness would leak out of the stiff terms and stiffen the
nonstiff terms. It would manifest itself as a loss in stability or a forced reduction in stepsize of the nonstiff
terms. A more expensive fully implicit approach might then be required, and hence, methods that leak
substantial stiffness might best be avoided".
 *
 * @copydoc hide_ContainerType
 */
template<class ContainerType>
struct ARKStep
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@copydoc ERKStep::ERKStep()
    ARKStep(){ m_kI.resize(1); }
    /*!@brief Construct with given name
     * @param name Currently, one of "Cavaglieri-3-1-2", "Cavaglieri-4-2-3", "ARK-4-2-3", "ARK-6-3-4" or "ARK-8-4-5"
     * @param copyable vector of the size that is later used in \c step (
      it does not matter what values \c copyable contains, but its size is important;
      the \c step method can only be called with vectors of the same size)
     */
    ARKStep( std::string name, const ContainerType& copyable)
    {
        std::string exp_name = name+" (explicit)";
        std::string imp_name = name+" (implicit)";
        *this = ARKStep( exp_name, imp_name, copyable);
    }
    /*!@brief Construct with two Butcher Tableaus
     *
     * The two Butcher Tableaus represent the parameters for the explicit
     * and implicit parts respectively. If both the explicit and implicit part
     * of your equations are nontrivial, they must be one of the "ARK-X-X-X" methods
     * listed in \c ConvertsToButcherTableau. Or you have your own tableaus of
     * course but both tableaus must have the same number of steps.
     *
     * @param ex_tableau Tableau for the explicit part
     * @param im_tableau Tableau for the implicit part (must have the same number of stages as \c ex_tableau )
     * @param copyable vector of the size that is later used in \c step (
      it does not matter what values \c copyable contains, but its size is important;
      the \c step method can only be called with vectors of the same size)
     */
    ARKStep( ConvertsToButcherTableau<value_type> ex_tableau,
             ConvertsToButcherTableau<value_type> im_tableau,
             const ContainerType& copyable
             ):
         m_rkE(ex_tableau),
         m_rkI(im_tableau),
         m_kE(m_rkE.num_stages(), copyable),
         m_kI(m_rkI.num_stages(), copyable)
    {
        assert( m_rkE.num_stages() == m_rkI.num_stages());
        // check fsal
        assert( m_rkI.a(0,0) == 0);
        assert( m_rkI.c(m_rkI.num_stages()-1) == 1);
        check_implicit_fsal();
        assign_coeffs();
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = ARKStep( std::forward<Params>( ps)...);
    }
    ///@copydoc hide_copyable
    const ContainerType& copyable()const{ return m_kI[0];}

    /**
    * @brief Advance one step
    *
    * @copydoc hide_explicit_rhs
    * @copydoc hide_implicit_rhs
    * @copydoc hide_solver
    * @param ode the <explicit rhs, implicit rhs, solver for the rhs> functor.
    * Typically \c std::tie(explicit_rhs, implicit_rhs, solver)
    * @param t0 start time
    * @param u0 value at \c t0
    * @param t1 (write only) end time ( equals \c t0+dt on return, may alias \c t0)
    * @param u1 (write only) contains result on return (may alias u0)
    * @param dt timestep
    * @param delta Contains error estimate (u1 - tilde u1) on return (must have equal size as \c u0)
    * @note the implementation is such that on return the last call is the
    * explicit part \c ex at the new (t1,u1).
    * This is useful if \c ex holds
    * state, which is then updated to the new timestep and/or if \c im changes
    * the state of \c ex
    * @note After a \c solve we immediately
    * call \c ex on the solution
    */
    template< class ExplicitRHS, class ImplicitRHS, class Solver>
    void step( const std::tuple<ExplicitRHS, ImplicitRHS, Solver>& ode, value_type t0, const ContainerType& u0, value_type& t1, ContainerType& u1, value_type dt, ContainerType& delta);
    ///@copydoc ERKStep::order()
    unsigned order() const {
        return m_rkE.order();
    }
    ///@copydoc ERKStep::embedded_order()
    unsigned embedded_order() const {
        return m_rkE.order();
    }
    ///@copydoc ERKStep::num_stages()
    unsigned num_stages() const{
        return m_rkE.num_stages();
    }
    private:
    ButcherTableau<value_type> m_rkE, m_rkI;
    std::vector<ContainerType> m_kE, m_kI;
    std::vector<value_type> m_rkb, m_rkd;
    value_type m_t1 = 1e300;
    bool m_implicit_fsal = false;
    void check_implicit_fsal(){
        m_implicit_fsal = true;
        for( unsigned i=0; i<m_rkI.num_stages(); i++)
            if( m_rkI.a(i,0) != 0)
                m_implicit_fsal = false;
    }
    void assign_coeffs()
    {
        m_rkb.resize( 2*m_rkI.num_stages());
        m_rkd.resize( 2*m_rkI.num_stages());
        for( unsigned i=0; i<m_rkI.num_stages(); i++)
        {
            m_rkb[2*i] = m_rkE.b(i);
            m_rkb[2*i+1] = m_rkI.b(i);
            m_rkd[2*i] = m_rkE.d(i);
            m_rkd[2*i+1] = m_rkI.d(i);
        }
    }
};

///@cond
template<class ContainerType>
template< class Explicit, class Implicit, class Solver>
void ARKStep<ContainerType>::step( const std::tuple<Explicit,Implicit,Solver>& ode, value_type t0, const ContainerType& u0, value_type& t1, ContainerType& u1, value_type dt, ContainerType& delta)
{
    unsigned s = m_rkE.num_stages();
    value_type tu = t0;
    //0 stage
    //!! Assume: a^E_00 = a^I_00 = 0
    if( t0 != m_t1)
        std::get<0>(ode)(t0, u0, m_kE[0]); //freshly compute k_0
    if( !m_implicit_fsal) // all a(i,0) == 0
        std::get<1>(ode)(t0, u0, m_kI[0]);
    // DO NOT HOLD POINTERS AS PRVIATE MEMBERS
    std::vector<const ContainerType*> k_ptrs( 2*m_rkI.num_stages());
    for( unsigned i=0; i<m_rkI.num_stages(); i++)
    {
        k_ptrs[2*i  ] = &m_kE[i];
        k_ptrs[2*i+1] = &m_kI[i];
    }

    for( unsigned i=1; i<s; i++)
    {
        std::vector<value_type> rka( 2*i);
        for(unsigned l=0; l<i; l++)
        {
            rka[2*l]   = m_rkE.a(i,l);
            rka[2*l+1] = m_rkI.a(i,l);
        }
        tu = DG_FMA( m_rkI.c(i),dt, t0);
        dg::blas1::copy( u0, m_kI[i]);
        dg::blas2::gemv( dt, dg::asDenseMatrix( k_ptrs, 2*i), rka, 1., m_kI[i]);
        value_type alpha = dt*m_rkI.a(i,i);
        std::get<2>(ode)( alpha, tu, delta, m_kI[i]);
        dg::blas1::axpby( 1./alpha, delta, -1./alpha, m_kI[i]);
        std::get<0>(ode)(tu, delta, m_kE[i]);
    }
    m_t1 = t1 = tu;
    //Now compute result and error estimate

    dg::blas1::copy( u0, u1);
    detail::gemm( {dt,dt}, dg::asDenseMatrix(k_ptrs), {&m_rkb, &m_rkd},
            {1.,0.}, {&u1, &delta});
    //make sure (t1,u1) is the last call to ex
    std::get<0>(ode)(t1,u1,m_kE[0]);
}
///@endcond

/*!
 * @brief Embedded diagonally implicit Runge Kutta time-step with error estimate
* \f$
 \begin{align}
    k_i = f\left( t^n + c_i \Delta t, u^n + \Delta t \sum_{j=1}^{i} a_{ij} k_j\right) \\
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j \\
    \tilde u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s \tilde b_j k_j \\
    \delta^{n+1} = u^{n+1} - \tilde u^{n+1} = \Delta t\sum_{j=1}^s (b_j-\tilde b_j) k_j
 \end{align}
\f$
 *
 * You can provide your own coefficients or use one of the methods
 * in the following table:
 * @copydoc hide_implicit_butcher_tableaus
 *
 * @copydoc hide_ContainerType
 */
template<class ContainerType>
struct DIRKStep
{
    //MW Dirk methods cannot have stage order greater than 1
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@copydoc ERKStep::ERKStep()
    DIRKStep(){ m_kI.resize(1); }

    /*!@brief Construct with a diagonally implicit Butcher Tableau
     *
     * The tableau may be one of the implict methods listed in
     * \c ConvertsToButcherTableau, or you provide your own tableau.
     *
     * @param im_tableau diagonally implicit tableau, name or identifier that \c ConvertsToButcherTableau
     * @param copyable vector of the size that is later used in \c step (
      it does not matter what values \c copyable contains, but its size is important;
      the \c step method can only be called with vectors of the same size)
     */
    DIRKStep( ConvertsToButcherTableau<value_type> im_tableau,
               const ContainerType& copyable):
         m_rkI(im_tableau),
         m_kI(m_rkI.num_stages(), copyable)
    {
        m_rkIb.resize(m_kI.size()), m_rkId.resize(m_kI.size());
        for( unsigned i=0; i<m_kI.size(); i++)
        {
            m_rkIb[i] = m_rkI.b(i);
            m_rkId[i] = m_rkI.d(i);
        }
    }

    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = DIRKStep( std::forward<Params>( ps)...);
    }
    ///@copydoc hide_copyable
    const ContainerType& copyable()const{ return m_kI[0];}

    /**
    * @brief Advance one step with error estimate
    *
    * @copydetails step(const std::tuple<ImplicitRHS,Solver>&,value_type,const ContainerType&,value_type&,ContainerType&,value_type)
    * @param delta Contains error estimate (u1 - tilde u1) on return (must have equal size as \c u0)
    */
    template< class ImplicitRHS, class Solver>
    void step( const std::tuple<ImplicitRHS,Solver>& ode, value_type t0, const
        ContainerType& u0, value_type& t1, ContainerType& u1, value_type dt,
        ContainerType& delta)
    {
        step( ode, t0, u0, t1, u1, dt, delta, true);
    }
    /**
    * @brief Advance one step ignoring error estimate and embedded method
    *
    * @copydoc hide_implicit_rhs
    * @copydoc hide_solver
    * @param ode the <right hand side, solver for the rhs> functors.
    * Typically \c std::tie(implicit_rhs, solver)
    * @attention Contrary to EDIRK methods DIRK and SDIRK methods (all diagonal
    * elements non-zero) never call \c implicit_rhs
    * @param t0 start time
    * @param u0 value at \c t0
    * @param t1 (write only) end time ( equals \c t0+dt on return
    *   may alias \c t0)
    * @param u1 (write only) contains result on return (may alias u0)
    * @param dt timestep
    */
    template< class ImplicitRHS, class Solver>
    void step( const std::tuple<ImplicitRHS, Solver>& ode, value_type t0, const
        ContainerType& u0, value_type& t1, ContainerType& u1, value_type dt)
    {
        if( !m_allocated)
        {
            dg::assign( m_kI[0], m_tmp);
            m_allocated = true;
        }
        step( ode, t0, u0, t1, u1, dt, m_tmp, false);
    }
    ///@copydoc ERKStep::order()
    unsigned order() const {
        return m_rkI.order();
    }
    ///@copydoc ERKStep::embedded_order()
    unsigned embedded_order() const {
        return m_rkI.order();
    }
    ///@copydoc ERKStep::num_stages()
    unsigned num_stages() const{
        return m_rkI.num_stages();
    }

    private:
    template< class ImplicitRHS, class Solver>
    void step( const std::tuple<ImplicitRHS, Solver>& ode, value_type t0, const
            ContainerType& u0, value_type& t1, ContainerType& u1, value_type
            dt, ContainerType& delta, bool compute_delta);
    ButcherTableau<value_type> m_rkI;
    std::vector<ContainerType> m_kI;
    ContainerType m_tmp;
    std::vector<value_type> m_rkIb, m_rkId;
    bool m_allocated = false;
};

///@cond
template<class ContainerType>
template< class ImplicitRHS, class Solver>
void DIRKStep<ContainerType>::step( const std::tuple<ImplicitRHS,Solver>& ode,  value_type t0, const ContainerType& u0, value_type& t1, ContainerType& u1, value_type dt, ContainerType& delta, bool compute_delta)
{
    unsigned s = m_rkI.num_stages();
    value_type tu = t0;
    //0 stage
    //rhs = u0
    tu = DG_FMA( m_rkI.c(0),dt, t0);
    value_type alpha;
    if( m_rkI.a(0,0) !=0 )
    {
        alpha = dt*m_rkI.a(0,0);
        std::get<1>(ode)( alpha, tu, delta, u0);
        dg::blas1::axpby( 1./alpha, delta, -1./alpha, u0, m_kI[0]);
    }
    else
        std::get<0>(ode)(tu, u0, m_kI[0]);
    std::vector<const ContainerType*> kIptr = dg::asPointers( m_kI);

    for( unsigned i=1; i<s; i++)
    {
        tu = DG_FMA( m_rkI.c(i),dt, t0);
        dg::blas1::copy( u0, m_kI[i]);
        std::vector<value_type> rkIa( i);
        for( unsigned l=0; l<i; l++)
            rkIa[l] = m_rkI.a(i,l);
        dg::blas2::gemv( dt, dg::asDenseMatrix(kIptr,i), rkIa, 1., m_kI[i]);
        if( m_rkI.a(i,i) !=0 )
        {
            alpha = dt*m_rkI.a(i,i);
            std::get<1>(ode)( alpha, tu, delta, m_kI[i]);
            dg::blas1::axpby( 1./alpha, delta, -1./alpha, m_kI[i]);
        }
        else
            std::get<0>(ode)(tu, delta, m_kI[i]);
    }
    t1 = t0 + dt;
    //Now compute result and error estimate
    dg::blas1::copy( u0, u1);
    if( compute_delta)
        detail::gemm( {dt,dt}, dg::asDenseMatrix(kIptr), {&m_rkIb, &m_rkId},
            {1.,0.}, {&u1, &delta});
    else
        blas2::gemv( dt, dg::asDenseMatrix(kIptr), m_rkIb, 1., u1);
}
///@endcond

/**
* @brief Runge-Kutta fixed-step explicit ODE integrator
* \f$
 \begin{align}
    k_i = f\left( t^n + c_i \Delta t, u^n + \Delta t \sum_{j=1}^{i-1} a_{ij} k_j\right) \\
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j
 \end{align}
\f$

The method is defined by its (explicit) ButcherTableau, given by
the coefficients \c a, \c b and \c c,  and \c s is the number
of stages.

You can provide your own coefficients or use one of our predefined methods (including the ones in Shu-Osher form):
@copydoc hide_explicit_butcher_tableaus
The following code snippet demonstrates how to use the class for the integration of
the harmonic oscillator:

@snippet runge_kutta_t.cpp function
@snippet runge_kutta_t.cpp doxygen
*
* @note Uses only \c dg::blas1 routines to integrate one step.
* @copydoc hide_ContainerType
*/
template<class ContainerType>
using RungeKutta = ERKStep<ContainerType>;

/**
* @brief Filtered Runge-Kutta fixed-step explicit ODE integrator
* \f$
 \begin{align}
    k_i = f\left( t^n + c_i \Delta t, \Lambda\Pi \left[u^n + \Delta t \sum_{j=1}^{i-1} a_{ij} k_j\right]\right) \\
    u^{n+1} = \Lambda\Pi\left[u^{n} + \Delta t\sum_{j=1}^s b_j k_j\right]
 \end{align}
\f$
@copydetails RungeKutta
*/
template<class ContainerType>
using FilteredRungeKutta = FilteredERKStep<ContainerType>;

/**
* @brief Shu-Osher fixed-step explicit ODE integrator with Slope Limiter / Filter
* \f$
 \begin{align}
    u_0 &= u_n \\
    u_i &= \Lambda\Pi \left(\sum_{j=0}^{i-1}\left[ \alpha_{ij} u_j + \Delta t \beta_{ij} f( t_j, u_j)\right]\right)\\
    u^{n+1} &= u_s
 \end{align}
\f$

where \f$ \Lambda\Pi\f$ is the limiter, \f$ i\in [1,s]\f$ and \c s is the number of stages (i.e. the number of times the right hand side is evaluated.

The method is defined by its (explicit) ShuOsherTableau, given by
the coefficients \c alpha and \c beta,  and \c s is the number
of stages.
@note the original reference for the scheme is
 * <a href="https://doi.org/10.1016/0021-9991(88)90177-5">
 Chi-Wang Shu, Stanley Osher,
Efficient implementation of essentially non-oscillatory shock-capturing schemes,
Journal of Computational Physics,
Volume 77, Issue 2,
1988,
Pages 439-471</a>
@note This struct can be used to implement the RKDG methods with slope-limiter described in
<a href ="https://doi.org/10.1023/A:1012873910884">Cockburn, B., Shu, CW. Runge–Kutta Discontinuous Galerkin Methods for Convection-Dominated Problems. Journal of Scientific Computing 16, 173–261 (2001) </a>

You can use one of our predefined methods (only the ones that are marked with "Shu-Osher-Form"):
@copydoc hide_explicit_butcher_tableaus
*
* @note Uses only \c dg::blas1 routines to integrate one step.
* @copydoc hide_ContainerType
*/
template<class ContainerType>
struct ShuOsher
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@copydoc ERKStep::ERKStep()
    ShuOsher(){m_u.resize(1);}
    /**
    * @brief Reserve internal workspace for the integration
    *
    * @param tableau Tableau, name or identifier that \c ConvertsToShuOsherTableau
    * @param copyable vector of the size that is later used in \c step (
     it does not matter what values \c copyable contains, but its size is important;
     the \c step method can only be called with vectors of the same size)
    */
    ShuOsher( dg::ConvertsToShuOsherTableau<value_type> tableau, const ContainerType& copyable): m_t( tableau), m_u(  m_t.num_stages(), copyable), m_k(m_u)
        { }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = ShuOsher( std::forward<Params>( ps)...);
    }
    ///@copydoc hide_copyable
    const ContainerType& copyable()const{ return m_u[0];}

    /**
    * @brief Advance one step
    *
    * @copydoc hide_explicit_rhs
    * @copydoc hide_limiter
    * @param ode right hand side subroutine and limiter to use.
    * Typically \c std::tie( rhs,limiter)
    * @param t0 start time
    * @param u0 value at \c t0
    * @param t1 (write only) end time ( equals \c t0+dt on return, may alias \c t0)
    * @param u1 (write only) contains result on return (may alias u0)
    * @param dt timestep
    * @note on return \c rhs(t1, u1) will be the last call to \c rhs (this is useful if \c ExplicitRHS holds state, which is then updated to the current timestep)
    */
    template<class ExplicitRHS, class Limiter>
    void step( const std::tuple<ExplicitRHS, Limiter>& ode, value_type t0, const ContainerType& u0, value_type& t1, ContainerType& u1, value_type dt){
        unsigned s = m_t.num_stages();
        std::vector<value_type> ts( m_t.num_stages()+1);
        ts[0] = t0;
        dg::blas1::copy(u0, m_u[0]);
        if( t0 != m_t1 ) //this is the first time we call step
            std::get<0>(ode)(ts[0], m_u[0], m_k[0]); //freshly compute k_0
        for( unsigned i=1; i<=s; i++)
        {
            dg::blas1::axpbypgz( m_t.alpha(i-1,0), m_u[0], dt*m_t.beta(i-1,0), m_k[0], 0., i==s ? u1 : m_u[i]);
            ts[i] = m_t.alpha(i-1,0)*ts[0] + dt*m_t.beta(i-1,0);
            for( unsigned j=1; j<i; j++)
            {
                //about the i-1: it is unclear to me how the ShuOsher tableau makes implicit schemes
                dg::blas1::axpbypgz( m_t.alpha(i-1,j), m_u[j], dt*m_t.beta(i-1,j), m_k[j], 1., i==s ? u1 : m_u[i]);
                ts[i] += m_t.alpha(i-1,j)*ts[j] + dt*m_t.beta(i-1,j);

            }
            std::get<1>(ode)( i==s ? u1 : m_u[i]);
            if(i!=s)
                std::get<0>(ode)(ts[i], m_u[i], m_k[i]);
            else
                //make sure (t1,u1) is the last call to f
                std::get<0>(ode)(ts[i], u1, m_k[0]);
        }
        m_t1 = t1 = ts[s];
    }
    ///@copydoc ERKStep::order
    unsigned order() const {
        return m_t.order();
    }
    ///@copydoc ERKStep::num_stages()
    unsigned num_stages() const{
        return m_t.num_stages();
    }
  private:
    ShuOsherTableau<value_type> m_t;
    std::vector<ContainerType> m_u, m_k;
    value_type m_t1 = 1e300;
};
/**
* @brief Runge-Kutta fixed-step implicit ODE integrator
* \f$
 \begin{align}
    k_i = f\left( t^n + c_i \Delta t, u^n + \Delta t \sum_{j=1}^{s} a_{ij} k_j\right) \\
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j
 \end{align}
\f$

The method is defined by its (implicit) ButcherTableau, given by
the coefficients \c a, \c b and \c c,  and \c s is the number
of stages.

You can provide your own coefficients or use one of our predefined methods:
@copydoc hide_implicit_butcher_tableaus
*
* @note Uses only \c dg::blas1 routines to integrate one step.
* @copydoc hide_ContainerType
*/
template<class ContainerType>
using ImplicitRungeKutta = DIRKStep<ContainerType>;


/// Checks if two number are equal within accuracy
/// @ingroup basics
inline bool is_same( double x, double y, double eps = 1e-15)
{
    return fabs(x - y) < eps * std::max(1.0, std::max( fabs(x), fabs(y)));
}
/// Checks if two number are equal within accuracy
/// @ingroup basics
inline bool is_same( float x, float y, float eps = 1e-6)
{
    return fabsf(x - y) < eps * std::max(1.0f, std::max( fabsf(x), fabsf(y)));
}

/// Alias for <tt> x == y </tt>
/// @ingroup basics
template<class T>
inline bool is_same( T x, T y)
{
    return x == y;
}

/// Checks if two number are integer divisable \f$a/b \in \mathbb{Z}\f$ within accuracy
/// @attention Does not check for equal sign!
/// @ingroup basics
inline bool is_divisable( double a, double b, double eps = 1e-15)
{
    return is_same( round(a/b)*b, a, eps);
}
/// Checks if two number are integer divisable \f$a/b \in \mathbb{Z}\f$ within accuracy
/// @attention Does not check for equal sign!
/// @ingroup basics
inline bool is_divisable( float a, float b, float eps = 1e-6)
{
    return is_same( (float)round(a/b)*b, (float)a, eps);
}

/*! @brief Integrate using a for loop and a fixed time-step
 *
 * The implementation (of integrate) is equivalent to
 * @code{.cpp}
  dg::blas1::copy( u0, u1);
  unsigned N = round((t1 - t0)/dt);
  for( unsigned i=0; i<N; i++)
      step( t0, u1, t0, u1, dt);
 * @endcode
 * where \c dt is a given constant. If \c t1 needs to be matched exactly, the last
 * timestep is shortened accordingly.
 * @ingroup time_utils
 * @sa AdaptiveTimeloop, MultistepTimeloop
 * @copydoc hide_ContainerType
 */
template<class ContainerType>
struct SinglestepTimeloop : public aTimeloop<ContainerType>
{
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>;
    /// no allocation
    SinglestepTimeloop( ) = default;

    /**
     * @brief Construct using a \c std::function
     *
     * @param step Called in the timeloop as <tt> step( t0, u1, t0, u1, dt) </tt>. Has to advance the ode in-place by \c dt
     * @param dt The constant timestep. Can be set later with \c set_dt. Can be negative.
     */
    SinglestepTimeloop( std::function<void ( value_type, const ContainerType&,
                value_type&, ContainerType&, value_type)> step, value_type dt = 0 )
        : m_step(step), m_dt(dt){}

    /**
     * @brief Bind the step function of a single step stepper
     *
     * Construct a lambda function that calls the step function of \c stepper
     * with given parameters and stores it internally in a \c std::function
     * @tparam Stepper possible steppers are for example dg::RungeKutta,
     * dg::ShuOsher and dg::ImplicitRungeKutta
     * @param stepper If constructed in-place (rvalue), will be copied into the
     * lambda. If an lvalue, then the lambda stores a reference
     * @attention If stepper is an lvalue then you need to make sure
     * that stepper remains valid to avoid a "dangling reference"
     * @copydoc hide_ode
     * @param dt The constant timestep. Can be set later with \c set_dt. Can be negative.
     */
    template<class Stepper, class ODE>
    SinglestepTimeloop( Stepper&& stepper, ODE&& ode, value_type dt = 0)
    {
        // Open/Close Principle (OCP) Software entities should be open for extension but closed for modification
        m_step = [=, cap = std::tuple<Stepper, ODE>(std::forward<Stepper>(stepper),
                std::forward<ODE>(ode))  ]( auto t0, const auto& y0, auto& t1, auto& y1,
                auto dtt) mutable
        {
            std::get<0>(cap).step( std::get<1>(cap), t0, y0, t1, y1, dtt);
        };
        m_dt = dt;
    }

    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = SinglestepTimeloop( std::forward<Params>( ps)...);
    }

    /**
     * @brief Set the constant timestep to be used in the integrate functions
     *
     * @param dt new timestep to use in integrate functions. Can be negative.
     */
    void set_dt( value_type dt){ m_dt = dt;}

    /*! @brief Integrate differential equation with a fixed number of steps
     *
     * Equivalent to
 * @code{.cpp}
  set_dt( (t1-t0)/(value_type)steps );
  integrate( t0, u0, t1, u1);
 * @endcode
     * @param t0 initial time
     * @param u0 initial condition
     * @param t1 final time
     * @param u1 (write-only) contains solution at \c t1 on return (may
     *      alias \c u0)
     * @param steps number of steps
     */
    void integrate_steps( value_type t0, const container_type& u0, value_type t1,
            container_type& u1, unsigned steps)
    {
        set_dt( (t1-t0)/(value_type)steps );
        this->integrate( t0, u0, t1, u1);
    }

    virtual SinglestepTimeloop* clone() const{ return new
        SinglestepTimeloop(*this);}
    private:
    virtual void do_integrate(value_type& t0, const container_type& u0, value_type t1, container_type& u1, enum to mode) const;
    std::function<void ( value_type, const ContainerType&, value_type&,
            ContainerType&, value_type)> m_step;
    virtual value_type do_dt( ) const { return m_dt;}
    value_type m_dt;
};

///@cond
template< class ContainerType>
void SinglestepTimeloop<ContainerType>::do_integrate(
        value_type&  t_begin, const ContainerType&
        begin, value_type t_end, ContainerType& end,
        enum to mode ) const
{
    bool forward = (t_end - t_begin > 0);
    if( (m_dt < 0 && forward) || ( m_dt > 0 && !forward) )
        throw dg::Error( dg::Message(_ping_)<<"Timestep has wrong sign! dt "<<m_dt);
    dg::blas1::copy( begin, end);
    if( m_dt == 0)
        throw dg::Error( dg::Message(_ping_)<<"Timestep may not be zero in SinglestepTimeloop!");
    if( is_divisable( t_end-t_begin, m_dt))
    {
        unsigned N = (unsigned)round((t_end - t_begin)/m_dt);
        for( unsigned i=0; i<N; i++)
            m_step( t_begin, end, t_begin, end, m_dt);
    }
    else
    {
        unsigned N = (unsigned)floor( (t_end-t_begin)/m_dt);
        for( unsigned i=0; i<N; i++)
            m_step( t_begin, end, t_begin, end, m_dt);
        if( dg::to::exact == mode)
        {
            value_type dt_final = t_end - t_begin;
            m_step( t_begin, end, t_begin, end, dt_final);
        }
        else
            m_step( t_begin, end, t_begin, end, m_dt);
    }
    return;
}
///@endcond

///@}

} //namespace dg

#endif //_DG_RK_
