#include <cmath>
#include "dg/algorithm.h"

#include "functors.h"
#include "lanczos.h"
#include "sqrt_cauchy.h"
#include "sqrt_ode.h"

#ifdef DG_BENCHMARK
#include "backend/timer.h"
#endif //DG_BENCHMARK

namespace dg {
namespace mat {

/**
 * @brief Operator that integrates an ODE from 0 to 1 with an  adaptive ERK class as timestepper
 *
 * The intended use is to integrate Matrix equations
 * @code{.cpp}
 * dg::Helmholtz<Geometry, Matrix, Container> A( g, alpha, dg::centered);
 * auto sqrt_ode = dg::mat::make_sqrtode( A, 1., A.weights(), eps);
 * unsigned number = 0;
 * auto sqrtA = dg::mat::make_directODESolve( sqrt_ode, "Dormand-Prince-7-4-5",1e-5, 1e-7, number);
 * dg::apply ( sqrtA , x , b);
 * @endcode
 * The call \f$ b = f(x) \f$ corresponds to integrating \f$ \dot y = F(y)\f$ with \f$ y(0 ) = x\f$ to \f$ b = y(1)\f$
 * @param ode The differential equation to integrate (forwarded to \c dg::Adaptive<dg::ERKStep>)
 * @param tableau The tableau for \c dg::ERKStep
 * @param epsTimerel relative accuracy of adaptive ODE solver
 * @param epsTimeabs absolute accuracy of adaptive ODE solver
 * @param number Is linked to the lambda. Contains the number of steps the
 *  adaptive timestepper used to completion
 * @param t0 Change starting time to \c t0
 * @param t1 Change end time to \c t1
 *
 * @sa \c dg::make_sqrtode \c dg::make_expode, \c dg::make_besselI0ode
 * @ingroup matrixfunctionapproximation
 */
template<class value_type, class ExplicitRHS>
auto make_directODESolve( ExplicitRHS&& ode,
        std::string tableau, value_type epsTimerel, value_type epsTimeabs,
        unsigned& number, value_type t0 = 0., value_type t1 = 1.)
{
    return [=, cap = std::tuple<ExplicitRHS>(std::forward<ExplicitRHS>(ode)),
            rtol = epsTimerel, atol = epsTimeabs]
            ( const auto& x, auto& b) mutable
        {
            value_type reject_limit = 2;
            dg::Adaptive<dg::ERKStep<std::decay_t<decltype(b)>>> adapt( tableau, x);
            dg::AdaptiveTimeloop<std::decay_t<decltype(b)>> loop( adapt,
                    std::get<0>(cap), dg::pid_control, dg::l2norm, rtol, atol,
                    reject_limit);
            loop.integrate( t0, x, t1, b);
            number = adapt.nsteps();
        };
}

/** @brief Shortcut for \f$b \approx \sqrt{A} x \f$ solve directly via sqrt Cauchy integral solve
 * @ingroup matrixfunctionapproximation
*/
template< class Container>
struct DirectSqrtCauchy
{
   public:
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    DirectSqrtCauchy() {}
    /**
     * @brief Construct DirectSqrtCauchy
     *
     * @param A The matrix (stored by reference)
     * @param weights
     * @param epsCG
     * @param iterCauchy maximum number of Cauchy iterations
     * @param EVmin
     * @param EVmax
     */
    template<class MatrixType>
    DirectSqrtCauchy(
            MatrixType& A,
            const Container& weights,
            value_type epsCG,
            unsigned iterCauchy,
            value_type EVmin, value_type EVmax, int exp)
    {
        m_pcg.construct( weights, 10000);
        Container m_temp = weights;
        m_iterCauchy = iterCauchy;
        m_cauchysqrtint.construct(weights);
        m_EVmin = EVmin, m_EVmax = EVmax;
        m_A = [&]( const Container& x, Container& y){
            dg::blas2::symv( A, x, y);
        };
        m_op = m_cauchysqrtint.make_denominator(A);
        m_wAinv = [&, eps = epsCG, w = weights]
            ( const Container& x, Container& y){
                m_pcg.solve( m_op, y, x, 1., w, eps);
        };
        m_exp = exp;
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = DirectSqrtCauchy( std::forward<Params>( ps)...);
    }
    /**
     * @brief Compute \f$x \approx \sqrt{A} b \f$ via sqrt Cauchy integral solve
     *
     * @param b input vector
     * @param x output vector. Is approximating \f$x \approx \sqrt{A} b  \f$
     * @return number of integration steps of sqrt cauchy solve
     */
    unsigned operator()(const Container& b, Container& x)
    {
        m_cauchysqrtint(m_A, m_wAinv, b, x, m_EVmin, m_EVmax, m_iterCauchy, m_exp);
        return m_iterCauchy;
    }
  private:
    unsigned m_iterCauchy;
    std::function<void ( const Container&, Container&)> m_A, m_wAinv, m_op;
    dg::PCG<Container> m_pcg;
    dg::mat::SqrtCauchyInt<Container> m_cauchysqrtint;
    value_type m_EVmin, m_EVmax;
    int m_exp;
};

/*!
 * @brief Shortcut for \f$x \approx \sqrt{A}^{\pm 1} b \f$ via exploiting first
 *  a Krylov projection of \c A b and and secondly a sqrt cauchy solve
 *
 * @ingroup matrixfunctionapproximation
 *
 * @note The approximation relies on Projection
 * \f$x = \sqrt{A}^{\pm 1} b  \approx R \sqrt{T}^{\pm 1}e_1\f$,
 * where \f$T\f$ and \f$V\f$ is the tridiagonal and
 * orthogonal matrix of the Krylov projection and \f$e_1\f$ is the normalized unit
 * vector. The vector \f$\sqrt{T^{\pm 1}} e_1\f$ is computed via the sqrt Cauchy
 * solve.
 */
template<class Container>
class KrylovSqrtCauchy
{
  public:
    using value_type = dg::get_value_type<Container>; //!< value type of the ContainerType class
    using HVec = dg::HVec;
    ///@brief Allocate nothing, Call \c construct method before usage
    KrylovSqrtCauchy(){}
    ///@copydoc construct()
 /**
     * @brief Construct KrylovSqrtCauchy
     *
     * @param A self-adjoint operator (stored as reference)
     * @param exponent if +1 compute \f$ \sqrt(A)\f$, if -1 compute \f$ 1/\sqrt(A)\f$
     * @param weights in which A is self-adjoint
     * @param minEV minimum Eigenvalue of A
     * @param maxEV maximum Eigenvalue of A
     * @param iterCauchy iterations of cauchy integral
     * @param eps accuracy of MCG method
     * @param max_iterations of MCG method
     */
    template<class MatrixType>
    KrylovSqrtCauchy( MatrixType& A, int exponent, const Container& weights,
            value_type minEV, value_type maxEV,
            unsigned iterCauchy, value_type eps, unsigned max_iterations
            ) : m_weights(weights), m_eps(eps), m_EVmin(minEV), m_EVmax(maxEV),
                m_exp(exponent)
    {
        m_A = [&]( const Container& x, Container& y){
            dg::blas2::symv( A, x, y);
        };
        m_iterCauchy = iterCauchy;
        m_mcg.construct(weights, max_iterations);
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = KrylovSqrtCauchy( std::forward<Params>( ps)...);
    }
    /**
     * @brief Solve the system \f$ x = \sqrt{A}^{\pm 1} b \f$ using Krylov
     *  projection of A and sqrt Cauchy solve
     *
     * @param b The right hand side vector.
     * @param x Contains an initial value
     * @return number of iterations in Krylov subspace solve
     *
     * @note So far only ordinary convergence criterium of MCG method.
     *  Should be adapted to square root criterium.
     */
    unsigned operator()(const Container& b, Container& x)
    {
        value_type bnorm = sqrt(dg::blas2::dot(b, m_weights, b));
        if( bnorm == 0)
        {
            dg::blas1::copy( b, x);
            return 0;
        }
        //Compute TH
        auto TH = m_mcg.tridiag(m_A, b, m_weights, m_eps, 1.);
        auto e1H = m_mcg.make_e1(), yH(e1H);
        HVec one(e1H.size(), 1.);
        DirectSqrtCauchy<HVec> cauchy( TH, one, 1e-8, m_iterCauchy, m_EVmin,
                m_EVmax, m_exp);
        cauchy( e1H, yH);

        //dg::mat::SqrtCauchyInt<HVec> cauchysqrtH( e1H);
        //auto wTinv = [&w = cauchysqrtH.w(), &TH = TH]( const auto& y, auto& x)
        //{
        //    auto wTH = TH;
        //    for( unsigned u=0; u<wTH.num_rows; u++)
        //        wTH.values( u,1) += w*w;
        //    cusp::coo_matrix<int, value_type, cusp::host_memory> THinvH;
        //    dg::mat::invert( wTH, THinvH);
        //    dg::blas2::symv( THinvH, y, x);
        //};
        //cauchysqrtH(TH, wTinv, e1H, yH, m_EVmin, m_EVmax,
        //    m_iterCauchy, m_exp); //(minEV, maxEV) estimated // y= T^(-1/2) e_1

        m_mcg.normMbVy(m_A, TH, yH, x, b, bnorm); // x =  R T^(-1/2) e_1

        return m_mcg.get_iter();
    }
  private:
    unsigned m_iterCauchy;
    std::function<void ( const Container&, Container&)> m_A;
    Container m_weights;
    value_type m_eps, m_EVmin, m_EVmax;
    dg::mat::Lanczos< Container > m_mcg;
    int m_exp;
};


/*!
 * @brief Shortcut for \f$x \approx \sqrt{A}^{\pm 1} b \f$ via exploiting first
 *  a Krylov projection of \c A b and and secondly a sqrt cauchy solve
 *
 * @ingroup matrixfunctionapproximation
 *
 * @note The approximation relies on Projection
 * \f$x = \sqrt{A}^{\pm 1} b  \approx R \sqrt{T}^{\pm 1}e_1\f$,
 * where \f$T\f$ and \f$V\f$ is the tridiagonal and
 * orthogonal matrix of the Krylov projection and \f$e_1\f$ is the normalized unit
 * vector. The vector \f$\sqrt{T^{\pm 1}} e_1\f$ is computed via the sqrt ODE
 * solve.
 */
template<class Container>
class KrylovSqrtODE
{
  public:
    using value_type = dg::get_value_type<Container>; //!< value type of the ContainerType class
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HVec = dg::HVec;
    ///@brief Allocate nothing, Call \c construct method before usage
    KrylovSqrtODE(){}
    ///@copydoc construct()
 /**
     * @brief Construct KrylovSqrtSolve
     *
     * @param A self-adjoint operator (stored by reference)
     * @param exponent if +1 compute \f$ \sqrt(A)\f$, if -1 compute \f$ 1/\sqrt(A)\f$
     * @param weights in which A is self-adjoint
     * @param tableau
     * @param rtol
     * @param atol
     * @param max_iterations
     * @param eps accuracy of MCG method
     */
    template<class MatrixType>
    KrylovSqrtODE( MatrixType& A, int exponent, const Container& weights,
            std::string tableau, value_type rtol,
            value_type atol, unsigned max_iterations, value_type eps ) :
        m_tableau(tableau),
        m_epsTimerel( rtol), m_epsTimeabs( atol),
        m_weights(weights), m_eps(eps), m_exp(exponent)
    {
        m_A = [&]( const Container& x, Container& y){
            dg::blas2::symv( A, x, y);
        };
        m_mcg.construct(weights, max_iterations);
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = KrylovSqrtODE( std::forward<Params>( ps)...);
    }
    /**
     * @brief Solve the system \f$ x = \sqrt{A}^{\pm 1} b \f$ using Krylov
     *  projection of A and sqrt ODE solve
     *
     * @param b The right hand side vector.
     * @param x Contains an initial value
     * @return number of iterations in Krylov subspace solve
     *
     * @note So far only ordinary convergence criterium of MCG method.
     *  Should be adapted to square root criterium.
     */
    std::array<unsigned,2> operator()(const Container& b, Container& x)
    {
        value_type bnorm = sqrt(dg::blas2::dot(b, m_weights, b));
        if( bnorm == 0)
        {
            dg::blas1::copy( b, x);
            return {0,0};
        }
        //Compute TH
        auto TH = m_mcg(m_A, b, m_weights, m_eps, 1.);
        auto e1H = m_mcg.make_e1(), yH(e1H), yyH(e1H);

        unsigned number = 0;
        auto inv_sqrt = make_inv_sqrtodeTri( TH, e1H);
        auto sqrtHSolve =  make_directODESolve( inv_sqrt,
            m_tableau, m_epsTimerel, m_epsTimeabs, number);
        dg::apply( sqrtHSolve, e1H, yH);
        if( m_exp >= 0 )
        {
            dg::apply( TH, yH, yyH);
            yyH.swap(yH);
        }

        m_mcg.Ry(m_A, TH, yH, x, b); // x =  R T^(-1/2) e_1

        return {m_mcg.get_iter(), number};
    }
  private:
    unsigned m_iterCauchy;
    std::function<void ( const Container&, Container&)> m_A;
    std::string m_tableau;
    value_type m_epsTimerel, m_epsTimeabs;
    Container m_weights;
    value_type m_eps;
    dg::mat::MCG< Container > m_mcg;
    int m_exp;
};


} //namespace mat
} //namespace dg
