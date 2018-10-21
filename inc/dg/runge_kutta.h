#ifndef _DG_RK_
#define _DG_RK_

#include <cassert>
#include <array>

#include "backend/exceptions.h"
#include "tableau.h"
#include "blas1.h"
#include "implicit.h"

/*! @file
 * @brief Runge-Kutta explicit time-integrators
 */

namespace dg{

 /** @class hide_rhs
  * @tparam RHS The right hand side
        is a functor type with no return value (subroutine)
        of signature <tt> void operator()(value_type, const ContainerType&, ContainerType&)</tt>
        The first argument is the time, the second is the input vector, which the functor may \b not override, and the third is the output,
        i.e. y' = f(t, y) translates to f(t, y, y').
        The two ContainerType arguments never alias each other in calls to the functor.
  */


/**
* @brief Struct for Embedded Runge Kutta explicit time-step with error estimate
* \f[
 \begin{align}
    k_i = f\left( t^n + c_i \Delta t, u^n + \Delta t \sum_{j=1}^{s-1} a_{ij} k_j\right) \\
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j \\
    \tilde u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s \tilde b_j k_j
 \end{align}
\f]

The method is defined by its (extended explicit) ButcherTableau, given by
the coefficients \c a, \c b and \c c,  and \c s is the number
of stages. The embedding is given by the coefficients \c bt (tilde b).

You can provide your own coefficients or use one of the embedded methods
in the following table:
@copydoc hide_explicit_butcher_tableaus

* @note The name of this class is in reminiscence of the ARKode library http://runge.math.smu.edu/arkode_dev/doc/guide/build/html/index.html
* @copydoc hide_ContainerType
* @ingroup time
*/
template< class ContainerType>
struct ERKStep
{
    using real_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container = ContainerType; //!< the type of the vector class in use
    ///@copydoc RungeKutta::RungeKutta()
    ERKStep(){
    }
    ///@copydoc RungeKutta::construct()
    ERKStep( const ContainerType& copyable, ConvertsToButcherTableau<real_type> tableau): m_rk(tableau), m_k(m_rk.num_stages(), copyable)
        { }
    ///@copydoc RungeKutta::construct()
    void construct( const ContainerType& copyable, ConvertsToButcherTableau<real_type> tableau){
        m_rk = tableau;
        m_k.assign(m_rk.num_stages(), copyable);
    }
    ///@copydoc RungeKutta::step()
    ///@param delta Contains error estimate on output (must have equal size as \c u0)
    template<class RHS>
    void step( RHS& rhs, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt, ContainerType& delta);
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
    ButcherTableau<real_type> m_rk;
    std::vector<ContainerType> m_k;
    real_type m_t1 = 1e300;//remember the last timestep at which ERK is called
};

template< class ContainerType>
template< class RHS>
void ERKStep<ContainerType>::step( RHS& f, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt, ContainerType& delta)
{
    unsigned s = m_rk.num_stages();
    //this behaviour must be documented
    //0 stage: probe fsal
    real_type tu = t0;
    if( t0 != m_t1)
        f(t0, u0, m_k[0]); //freshly compute k_0
    //else take from last call
    //1 stage
    if( s>1) {
        tu = DG_FMA( m_rk.c(1),dt, t0);
        blas1::axpby( 1., u0, dt*m_rk.a(1,0), m_k[0], delta);
        f( tu, delta, m_k[1]);
    }
    //2 stage
    if( s>2) {
        tu = DG_FMA( m_rk.c(2),dt, t0);
        blas1::evaluate( delta, dg::equals(), PairSum(), 1., u0,
                            dt*m_rk.a(2,0),m_k[0],
                            dt*m_rk.a(2,1),m_k[1]);
        f( tu, delta, m_k[2]);
    }
    //3 stage
    if( s> 3){
        tu = DG_FMA( m_rk.c(3),dt, t0);
        blas1::evaluate( delta, dg::equals(), PairSum(), 1., u0,
                             dt*m_rk.a(3,0),m_k[0],
                             dt*m_rk.a(3,1),m_k[1],
                             dt*m_rk.a(3,2),m_k[2]);
        f( tu, delta, m_k[3]);
    }
    //4 stage
    if( s>4){
        tu = DG_FMA( m_rk.c(4),dt, t0);
        blas1::evaluate( delta, dg::equals(), PairSum(), 1.        , u0,
                             dt*m_rk.a(4,0),m_k[0],  dt*m_rk.a(4,1),m_k[1],
                             dt*m_rk.a(4,2),m_k[2],  dt*m_rk.a(4,3),m_k[3]);
        f( tu, delta, m_k[4]);
    }
    //5 stage
    if( s>5) {
        tu = DG_FMA( m_rk.c(5),dt, t0);
        blas1::evaluate( delta, dg::equals(), PairSum(), 1., u0,
                 dt*m_rk.a(5,0),m_k[0], dt*m_rk.a(5,1),m_k[1],
                 dt*m_rk.a(5,2),m_k[2], dt*m_rk.a(5,3),m_k[3],
                 dt*m_rk.a(5,4),m_k[4]);
        f( tu, delta, m_k[5]);
    }
    //6 stage
    if( s>6)
    {
        tu = DG_FMA( m_rk.c(6),dt, t0);
        blas1::evaluate( delta, dg::equals(), PairSum(), 1., u0,
                           dt*m_rk.a(6,0),m_k[0], dt*m_rk.a(6,1),m_k[1],
                           dt*m_rk.a(6,2),m_k[2], dt*m_rk.a(6,3),m_k[3],
                           dt*m_rk.a(6,4),m_k[4], dt*m_rk.a(6,5),m_k[5]);
        f( tu, delta, m_k[6]);
        for ( unsigned i=7; i<s; i++)
        {
            blas1::axpby( 1.,u0, dt*m_rk.a(i,0),m_k[0], delta); //l=0
            tu = DG_FMA( dt,m_rk.c(i),t0); //l=0
            for( unsigned l=1; l<i; l++)
                blas1::axpby( dt*m_rk.a(i,l), m_k[l],1., delta);
            f( tu, delta, m_k[i]);
        }
    }
    //Now add everything up to get solution and error estimate
    switch( s)
    {
        //the first is for Euler
        case 1:
                blas1::subroutine( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0]); break;
        case 2: blas1::subroutine( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0],
                            dt*m_rk.b(1), dt*m_rk.d(1), m_k[1]); break;
        case 3: blas1::subroutine( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0],
                            dt*m_rk.b(1), dt*m_rk.d(1), m_k[1],
                            dt*m_rk.b(2), dt*m_rk.d(2), m_k[2]); break;
        case 4: blas1::subroutine( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0],
                            dt*m_rk.b(1), dt*m_rk.d(1), m_k[1],
                            dt*m_rk.b(2), dt*m_rk.d(2), m_k[2],
                            dt*m_rk.b(3), dt*m_rk.d(3), m_k[3]); break;
        case 5: blas1::subroutine( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0],
                            dt*m_rk.b(1), dt*m_rk.d(1), m_k[1],
                            dt*m_rk.b(2), dt*m_rk.d(2), m_k[2],
                            dt*m_rk.b(3), dt*m_rk.d(3), m_k[3],
                            dt*m_rk.b(4), dt*m_rk.d(4), m_k[4]); break;
        case 6: blas1::subroutine( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0],
                            dt*m_rk.b(1), dt*m_rk.d(1), m_k[1],
                            dt*m_rk.b(2), dt*m_rk.d(2), m_k[2],
                            dt*m_rk.b(3), dt*m_rk.d(3), m_k[3],
                            dt*m_rk.b(4), dt*m_rk.d(4), m_k[4],
                            dt*m_rk.b(5), dt*m_rk.d(5), m_k[5]); break;
        default: blas1::subroutine( dg::EmbeddedPairSum(),
                            u1, delta,
                            1., 0., u0,
                            dt*m_rk.b(0), dt*m_rk.d(0), m_k[0],
                            dt*m_rk.b(1), dt*m_rk.d(1), m_k[1],
                            dt*m_rk.b(2), dt*m_rk.d(2), m_k[2],
                            dt*m_rk.b(3), dt*m_rk.d(3), m_k[3],
                            dt*m_rk.b(4), dt*m_rk.d(4), m_k[4],
                            dt*m_rk.b(5), dt*m_rk.d(5), m_k[5],
                            dt*m_rk.b(6), dt*m_rk.d(6), m_k[6]);
            //sum the rest
            for( unsigned i=7; i<s; i++)
            {
                dg::blas1::axpby( dt*m_rk.b(i), m_k[i], 1., u1);
                dg::blas1::axpby( dt*m_rk.d(i), m_k[i], 1., delta);
            }
    }
    //make sure (t1,u1) is the last call to f
    m_t1 = t1 = t0 + dt;
    if(!m_rk.isFsal() )
        f(t1,u1,m_k[0]);
    else
        m_k[s-1].swap(m_k[0]);
}




/*!
 * @brief Struct for Additive Runge Kutta (semi-implicit) time-step with error estimate
 * following
 * <a href="http://runge.math.smu.edu/arkode_dev/doc/guide/build/html/Mathematics.html#arkstep-additive-runge-kutta-methods">The ARKode library</a>
 *
 * Currently, the possible Butcher Tableaus for a fully implicit-explicit scheme
 * are the "ARK-4-2-3", "ARK-6-3-4" and "ARK-8-4-5" combinations.
 * So far we did not implement the use of a mass matrix \c M.
 *
 * @copydoc hide_SolverType
 * @copydoc hide_ContainerType
 * @ingroup time
 */
template<class ContainerType, class SolverType = dg::DefaultSolver<ContainerType>>
struct ARKStep
{
    using real_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container = ContainerType; //!< the type of the vector class in use
    ///@copydoc RungeKutta::RungeKutta()
    ARKStep(){ }
    /*!@brief Construct with given name
     * @param copyable vector of the size that is later used in \c step (
      it does not matter what values \c copyable contains, but its size is important;
      the \c step method can only be called with vectors of the same size)
     * @param name Currently, one of "ARK-4-2-3", "ARK-6-3-4" or "ARK-8-4-5"
     * @param ps Parameters that, together with \c copyable as the first parameter,
     * are forwarded to the constructor of \c SolverType
     * @tparam SolverParams Type of parameters (deduced by the compiler)
     */
    template<class ...SolverParams>
    ARKStep( const ContainerType& copyable,
             std::string name,
             SolverParams&& ...ps
             )
    {
        if( name == "ARK-4-2-3" )
            construct( copyable, "ARK-4-2-3 (explicit)", "ARK-4-2-3 (implicit)", std::forward<SolverParams>(ps)...);
        else if( name == "ARK-6-3-4" )
            construct( copyable, "ARK-6-3-4 (explicit)", "ARK-6-3-4 (implicit)", std::forward<SolverParams>(ps)...);
        else if( name == "ARK-8-4-5" )
            construct( copyable, "ARK-8-4-5 (explicit)", "ARK-8-4-5 (implicit)", std::forward<SolverParams>(ps)...);
        else
            throw dg::Error( dg::Message()<<"Unknown name");
    }

    ///@copydoc construct()
    template<class ...SolverParams>
    ARKStep( const ContainerType& copyable,
             ConvertsToButcherTableau<real_type> ex_tableau,
             ConvertsToButcherTableau<real_type> im_tableau,
             SolverParams&& ...ps
             ):
         m_rhs( copyable),
         m_rkE(ex_tableau),
         m_rkI(im_tableau),
         m_kE(m_rkE.num_stages(), copyable),
         m_kI(m_rkI.num_stages(), copyable),
         m_solver( copyable, std::forward<SolverParams>(ps)...)
    {
        assert( m_rkE.num_stages() == m_rkI.num_stages());
    }

    /*!@brief Construct with two Butcher Tableaus
     *
     * The two Butcher Tableaus represent the parameters for the explicit
     * and implicit parts respectively. If both the explicit and implicit part
     * of your equations are nontrivial, they must be one of the "ARK-X-X-X" methods
     * listed in \c ConvertsToButcherTableau. Or you have your own tableaus of
     * course but both tableaus must have the same number of steps.
     *
     * @param copyable vector of the size that is later used in \c step (
      it does not matter what values \c copyable contains, but its size is important;
      the \c step method can only be called with vectors of the same size)
     * @param ex_tableau Tableau for the explicit part
     * @param im_tableau Tableau for the implicit part (must have the same number of stages as \c ex_tableau )
     * @param ps Parameters that, together with \c copyable as the first parameter,
     * are forwarded to the constructor of \c SolverType
     * @tparam SolverParams Type of parameters (deduced by the compiler)
     */
    template<class ...SolverParams>
    void construct( const ContainerType& copyable,
             ConvertsToButcherTableau<real_type> ex_tableau,
             ConvertsToButcherTableau<real_type> im_tableau,
             SolverParams&& ...ps
             )
    {
        m_rhs = copyable;
        m_rkE = ex_tableau;
        m_rkI = im_tableau;
        assert( m_rkE.num_stages() == m_rkI.num_stages());
        m_kE.assign(m_rkE.num_stages(), copyable);
        m_kI.assign(m_rkI.num_stages(), copyable);
        m_solver = SolverType( copyable, std::forward<SolverParams>(ps)...);
    }

    /**
    * @brief Advance one step
    *
    * @copydoc hide_explicit_implicit
    * @param t0 start time
    * @param u0 value at \c t0
    * @param t1 (write only) end time ( equals \c t0+dt on output, may alias \c t0)
    * @param u1 (write only) contains result on output (may alias u0)
    * @param dt timestep
    * @param delta Contains error estimate on output (must have equal size as \c u0)
    * @note on return \c ex(t1, u1) will be the last call to \c ex (this is useful if \c Explicit holds state, which is then updated to the current timestep)
    */
    template< class Explicit, class Implicit>
    void step( Explicit& ex, Implicit& im, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt, ContainerType& delta);
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
    ContainerType m_rhs, m_u1;
    ButcherTableau<real_type> m_rkE, m_rkI;
    std::vector<ContainerType> m_kE, m_kI;
    SolverType m_solver;
    real_type m_t1 = 1e300;
};

template<class ContainerType, class SolverType>
template< class Explicit, class Implicit>
void ARKStep<ContainerType, SolverType>::step( Explicit& ex, Implicit& im, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt, ContainerType& delta)
{
    unsigned s = m_rkE.num_stages();
    real_type tu = t0;
    //0 stage
    //a^E_00 = a^I_00 = 0
    if( t0 != m_t1)
        ex(t0, u0, m_kE[0]); //freshly compute k_0
    im(t0, u0, m_kI[0]);

    //1 stage
    blas1::evaluate( m_rhs, dg::equals(), PairSum(), 1., u0,
            dt*m_rkE.a(1,0), m_kE[0],
            dt*m_rkI.a(1,0), m_kI[0]);
    tu = DG_FMA( m_rkI.c(1),dt, t0);
    //store solution in delta, init with last solution
    //blas1::copy( u0, delta); //if Implicit is zero this leads to unnecessary iterations
    blas1::copy( m_rhs, delta); //better init with rhs
    m_solver.solve( -dt*m_rkI.a(1,1), im, tu, delta, m_rhs);
    ex(tu, delta, m_kE[1]);
    im(tu, delta, m_kI[1]);

    //2 stage
    blas1::evaluate( m_rhs, dg::equals(), PairSum(), 1., u0,
             dt*m_rkE.a(2,0), m_kE[0],
             dt*m_rkE.a(2,1), m_kE[1],
             dt*m_rkI.a(2,0), m_kI[0],
             dt*m_rkI.a(2,1), m_kI[1]);
    tu = DG_FMA( m_rkI.c(2),dt, t0);
    //just take last solution as init
    blas1::copy( m_rhs, delta); //better init with rhs
    m_solver.solve( -dt*m_rkI.a(2,2), im, tu, delta, m_rhs);
    ex(tu, delta, m_kE[2]);
    im(tu, delta, m_kI[2]);
    //3 stage
    blas1::evaluate( m_rhs, dg::equals(), PairSum(), 1., u0,
             dt*m_rkE.a(3,0), m_kE[0],
             dt*m_rkE.a(3,1), m_kE[1],
             dt*m_rkE.a(3,2), m_kE[2],
             dt*m_rkI.a(3,0), m_kI[0],
             dt*m_rkI.a(3,1), m_kI[1],
             dt*m_rkI.a(3,2), m_kI[2]);
    tu = DG_FMA( m_rkI.c(3),dt, t0);
    blas1::copy( m_rhs, delta); //better init with rhs
    m_solver.solve( -dt*m_rkI.a(3,3), im, tu, delta, m_rhs);
    ex(tu, delta, m_kE[3]);
    im(tu, delta, m_kI[3]);
    //higher stages
    for( unsigned i=4; i<s; i++)
    {
        dg::blas1::copy( u0, m_rhs);
        for( unsigned j=0; j<s; j++)
            dg::blas1::axpbypgz( dt*m_rkE.a(i,j), m_kE[j],
                                 dt*m_rkI.a(i,j), m_kI[j], 1., m_rhs);
        tu = DG_FMA( m_rkI.c(i),dt, t0);
        blas1::copy( m_rhs, delta); //better init with rhs
        m_solver.solve( -dt*m_rkI.a(i,i), im, tu, delta, m_rhs);
        ex(tu, delta, m_kE[i]);
        im(tu, delta, m_kI[i]);
    }
    m_t1 = t1 = tu;
    // do up to 8 stages for ARK-8-4-5
    //Now compute result and error estimate
    blas1::subroutine( dg::EmbeddedPairSum(),
            u1, delta,
             1., 0., u0,
            dt*m_rkE.b(0), dt*m_rkE.d(0),m_kE[0],
            dt*m_rkE.b(1), dt*m_rkE.d(1),m_kE[1],
            dt*m_rkE.b(2), dt*m_rkE.d(2),m_kE[2],
            dt*m_rkE.b(3), dt*m_rkE.d(3),m_kE[3],
            dt*m_rkI.b(0), dt*m_rkI.d(0),m_kI[0],
            dt*m_rkI.b(1), dt*m_rkI.d(1),m_kI[1],
            dt*m_rkI.b(2), dt*m_rkI.d(2),m_kI[2],
            dt*m_rkI.b(3), dt*m_rkI.d(3),m_kI[3]);
    //sum the rest
    for( unsigned i=4; i<s; i++)
    {
        dg::blas1::axpbypgz( dt*m_rkE.b(i), m_kE[i],
                             dt*m_rkI.b(i), m_kI[i], 1., u1);
        dg::blas1::axpbypgz( dt*m_rkE.d(i), m_kE[i],
                             dt*m_rkI.d(i), m_kI[i], 1., delta);
    }
    //make sure (t1,u1) is the last call to ex
    ex(t1,u1,m_kE[0]);
}

/**
* @brief Struct for Runge-Kutta fixed-step explicit time-integration
* \f[
 \begin{align}
    k_i = f\left( t^n + c_i \Delta t, u^n + \Delta t \sum_{j=1}^{s-1} a_{ij} k_j\right) \\
    u^{n+1} = u^{n} + \Delta t\sum_{j=1}^s b_j k_j
 \end{align}
\f]

The method is defined by its (explicit) ButcherTableau, given by
the coefficients \c a, \c b and \c c,  and \c s is the number
of stages.

You can provide your own coefficients or use one of our predefined methods:
@copydoc hide_explicit_butcher_tableaus
The following code snippet demonstrates how to use the class for the integration of
the harmonic oscillator:

@snippet runge_kutta_t.cu function
@snippet runge_kutta_t.cu doxygen
* @ingroup time
*
* @note Uses only \c dg::blas1 routines to integrate one step.
* @copydoc hide_ContainerType
*/
template<class ContainerType>
struct RungeKutta
{
    using real_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container = ContainerType; //!< the type of the vector class in use
    ///@brief No memory allocation, Call \c construct before using the object
    RungeKutta(){}
    ///@copydoc construct()
    RungeKutta( const ContainerType& copyable, ConvertsToButcherTableau<real_type> tableau): m_erk( copyable, tableau), m_delta( copyable)
        { }
    /**
    * @brief Reserve internal workspace for the integration
    *
    * @param copyable vector of the size that is later used in \c step (
     it does not matter what values \c copyable contains, but its size is important;
     the \c step method can only be called with vectors of the same size)
    * @param tableau Tableau, name or identifier that \c ConvertsToButcherTableau
    */
    void construct( const ContainerType& copyable, ConvertsToButcherTableau<real_type> tableau){
        m_erk = ERKStep<ContainerType>( copyable, tableau);
        m_delta = copyable;
    }
    /**
    * @brief Advance one step
    *
    * @copydoc hide_rhs
    * @param rhs right hand side subroutine
    * @param t0 start time
    * @param u0 value at \c t0
    * @param t1 (write only) end time ( equals \c t0+dt on output, may alias \c t0)
    * @param u1 (write only) contains result on output (may alias u0)
    * @param dt timestep
    * @note on return \c rhs(t1, u1) will be the last call to \c rhs (this is useful if \c RHS holds state, which is then updated to the current timestep)
    */
    template<class RHS>
    void step( RHS& rhs, real_type t0, const ContainerType& u0, real_type& t1, ContainerType& u1, real_type dt){
        m_erk.step( rhs, t0, u0, t1, u1, dt, m_delta);
    }
    ///@copydoc ERKStep::order
    unsigned order() const {
        return m_erk.order();
    }
    ///@copydoc ERKStep::num_stages()
    unsigned num_stages() const{
        return m_erk.num_stages();
    }
  private:
    ERKStep<ContainerType> m_erk;
    ContainerType m_delta;
};

///@addtogroup time
///@{

/**
 * @brief Integrate differential equation with an explicit Runge-Kutta scheme and a fixed number of steps
 *
 * @copydoc hide_rhs
 * @copydoc hide_ContainerType
 * @param tableau Tableau, name or identifier that \c ConvertsToButcherTableau
 * @param rhs The right-hand-side
 * @param t_begin initial time
 * @param begin initial condition
 * @param t_end final time
 * @param end (write-only) contains solution at \c t_end on output (may alias begin)
 * @param N number of steps
 */
template< class RHS, class ContainerType>
void stepperRK(ConvertsToButcherTableau<get_value_type<ContainerType>> tableau, RHS& rhs, get_value_type<ContainerType>  t_begin, const ContainerType& begin, get_value_type<ContainerType> t_end, ContainerType& end, unsigned N )
{
    using real_type = get_value_type<ContainerType>;
    RungeKutta<ContainerType > rk( begin, tableau);
    if( t_end == t_begin){ end = begin; return;}
    const real_type dt = (t_end-t_begin)/(real_type)N;
    dg::blas1::copy( begin, end);
    real_type t0 = t_begin;
    for( unsigned i=0; i<N; i++)
        rk.step( rhs, t0, end, t0, end, dt);
}


///@}

} //namespace dg

#endif //_DG_RK_
