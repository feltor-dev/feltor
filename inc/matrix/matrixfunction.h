#pragma once
#include <cmath>

#include <boost/math/special_functions.hpp> // has to be included before lapack in certain versions
#include <cusp/transpose.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
//#include <cusp/print.h>

#include <cusp/lapack/lapack.h>
#include "dg/algorithm.h"

#include "functors.h"
#include "sqrt_cauchy.h"
#include "sqrt_ode.h"


namespace dg {
namespace mat {
///@addtogroup tridiagfunction
///@{

/**
 * @brief Create a functor that uses Eigenvalue decomposition to compute
 * \f$ f(T)\vec e_1 = E f(\Lambda) E^T \vec e_1 \f$ for symmetric tridiagonal \c T
 *
 * @note This is a general purpose solution. Very fast for small sizes (<40) of T,
 * but scales badly for larger sizes. Use
 * more specialized solutions if the number of iterations becomes high
 * @param f the matrix function (e.g. dg::SQRT<double> or dg::EXP<double>)
 * @return an operator to use in \c UniversalLanczos solve method
 * @sa \c UniversalLanczos
 */
template<class UnaryOp>
auto make_FuncEigen_Te1( UnaryOp f)
{
    return [f]( const auto& T)
    {
        using value_type = typename std::decay_t<decltype(T)>::value_type;
        unsigned iter = T.num_rows;
        cusp::array2d< value_type, cusp::host_memory> evecs(iter,iter);
        cusp::array1d< value_type, cusp::host_memory> evals(iter);
        dg::HVec e1H(iter,0.), yH(e1H);
        e1H[0] = 1.;
        yH.resize( iter);
        //Compute Eigendecomposition
        //MW !! the subdiagonal entries start at 0 in lapack, the n-th element
        // is used as workspace (from lapack docu)
        cusp::lapack::stev(T.values.column(1), T.values.column(2),
                evals, evecs, 'V');
        //for( unsigned u=0; u<iter; u++)
        //    std::cout << u << " "<<evals[u]<<std::endl;
        cusp::coo_matrix<int, value_type, cusp::host_memory> EH, EHt;
        cusp::convert(evecs, EH);
        cusp::transpose(EH, EHt);
        //Compute f(T) e1 = E f(Lambda) E^t e1
        dg::blas2::symv(EHt, e1H, yH);
        dg::blas1::transform(evals, e1H, [f] (double x){
            try{
                return f(x);
            }
            catch(boost::exception& e) //catch boost overflow error
            {
                return 0.;
            }
        });
        dg::blas1::pointwiseDot(e1H, yH, e1H);
        dg::blas2::symv(EH, e1H, yH);
        return yH;
    };
}

/**
 * @brief Create a functor that computes \f$ \sqrt{T^{\pm 1}} \vec e_1\f$ using SqrtCauchyInt
 *
 * @note The Eigenvalues can be estimated from a few lanczos iterations (which
 *  is at least more reliable than doing it semi-analytically)
 * @code{.cpp}
 *  dg::mat::UniversalLanczos lanczos( A.weights(), 20);
 *  auto T = lanczos.tridiag( A, A.weights(), A.weights());
 *  auto EVs = dg::mat::compute_extreme_EV( T);
 *  auto make_SqrtCauchy_Te1( -1, EVs, 40);
 * @endcode
 *
 * @param exp exponent if +1 compute \f$ \sqrt(T)\f$, if -1 compute \f$ 1/\sqrt(T)\f$
 * @param EVs {minimum Eigenvalue of A, maximum Eigenvalue of A}
 * @param stepsCauchy iterations of cauchy integral
 * @return an operator to use in \c UniversalLanczos solve method
 * @sa \c SqrtCauchyInt \c UniversalLanczos
 */
template< class value_type>
auto make_SqrtCauchy_Te1( int exp, std::array<value_type,2> EVs, unsigned stepsCauchy)
{
    return [=]( const auto& T)
    {
        unsigned size = T.num_rows;
        thrust::host_vector<value_type> e1H(size, 0.), yH(e1H);
        e1H[0] = 1.;

        dg::mat::SqrtCauchyInt<HVec> cauchysqrtH( e1H);
        auto wTinv = [&w = cauchysqrtH.w(), &T = T]( const auto& y, auto& x)
        {
            // invert 1*T + w*wI
            dg::mat::compute_Tinv_y( T, x, y, 1., w*w);
        };
        cauchysqrtH(T, wTinv, e1H, yH, EVs, stepsCauchy, exp);
        return yH;
    };
}

/**
 * @brief Create a functor that computes \f$ \sqrt{T^{\pm 1}} \vec e_1\f$ using either Eigen or SqrtCauchy solve based on whichever is fastest for given size
 *
 * @note The Eigenvalues can be estimated from a few lanczos iterations (which
 *  is at least more reliable than doing it semi-analytically)
 * @code{.cpp}
 *  dg::mat::UniversalLanczos lanczos( A.weights(), 20);
 *  auto T = lanczos.tridiag( A, A.weights(), A.weights());
 *  auto EVs = dg::mat::compute_extreme_EV( T);
 *  auto make_SqrtCauchyEigen_Te1( -1, EVs, 40);
 * @endcode
 * This function uses an Eigen decomposition for small sizes of  T and a
 * SqrtCauchyInt solve for larger sizes to optimize execution times
 * @param exp exponent if +1 compute \f$ \sqrt{T}\f$, if -1 compute \f$ 1/\sqrt{T}\f$
 * @param EVs {minimum Eigenvalue of A, maximum Eigenvalue of A}
 * @param stepsCauchy iterations of cauchy integral
 * @return an operator to use in \c UniversalLanczos solve method
 * @sa \c SqrtCauchyInt \c UniversalLanczos
 */
template< class value_type>
auto make_SqrtCauchyEigen_Te1( int exp, std::array<value_type,2> EVs, unsigned stepsCauchy)
{
    std::function< value_type(value_type)> func = dg::SQRT<value_type>();
    if( exp < 0)
        func = [](value_type x){return 1./sqrt(x);};

    auto eigen = make_FuncEigen_Te1( func);
    auto cauchy = make_SqrtCauchy_Te1( exp, EVs, stepsCauchy);
    return [=]( const auto& T)
    {
        unsigned size = T.num_rows;
        dg::HVec yH;
        if ( size < 40)
            yH = eigen( T);
        else
            yH = cauchy(T);
        return yH;
    };
}


/**
 * @brief Create a functor that computes \f$ \sqrt{T^{\pm 1}} \vec e_1\f$ using ODE solve
 * @param exp exponent if +1 compute \f$ \sqrt{T}\f$, if -1 compute \f$ 1/\sqrt{T}\f$
 * @param tableau Tableau of time integrator
 * @param rtol relative tolerance of time integrator
 * @param atol absolute tolerance of time integrator
 * @param number links to number of steps in time integrator
 * @return an operator to use in \c UniversalLanczos solve method
 * @sa \c make_directODESolve \c UniversalLanczos
 */
template< class value_type>
auto make_SqrtODE_Te1( int exp, std::string tableau, value_type rtol,
        value_type atol, unsigned& number)
{
    return [=, &num = number](const auto& T)
    {
        unsigned size = T.num_rows;
        HVec e1H(size, 0), yH(e1H);
        e1H[0] = 1.;
        auto inv_sqrt = make_inv_sqrtodeTri( T, e1H);
        auto sqrtHSolve =  make_directODESolve( inv_sqrt,
            tableau, rtol, atol, num);
        dg::apply( sqrtHSolve, e1H, yH);
        if( exp >= 0 )
        {
            dg::apply( T, yH, e1H);
            return e1H;
        }
        return yH;
    };
}

/**
 * @brief Create a functor that computes \f$ T^{\pm 1} \vec e_1\f$ directly
 * @param exp exponent if +1 compute \f$ Te_1\f$, if -1 compute \f$ T^{-1} e_1\f$
 * @return an operator to use in \c UniversalLanczos solve method
 * @sa \c UniversalLanczos
 */
inline auto make_Linear_Te1( int exp)
{
    return [= ](const auto& T)
    {
        unsigned size = T.num_rows;
        HVec e1H(size, 0), yH(e1H);
        e1H[0] = 1.;
        if( exp < 0)
            compute_Tinv_y( T, yH, e1H);
        else
            dg::blas2::symv( T, e1H, yH);
        return yH;
    };
}

///@}
} //namespace mat
} //namespace dg
