#pragma once

#include "dg/algorithm.h"
#include "lanczos.h"

namespace dg{
namespace mat{


/**
 * @brief Fast computation of \f$ \vec x = A^{\pm 1/2}\vec b\f$
 *
 * Convenience wrapper that
 * uses \c dg::mat::UniversalLanczos combined with
 * \c dg::mat::make_SqrtCauchyEigen_Te1 in its
 * "universal" stopping criterion
 * @note This is the fastest method to compute matrix square roots vector
 *  multiplications that we found to date
 * @ingroup matrixfunctionapproximation
 */
template<class Container>
struct MatrixSqrt
{
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;

    /// Construct empty
    MatrixSqrt() = default;


    /**
     * @brief Construct from matrix
     *
     * @tparam MatrixType
     * @param A self-adjoint matrix; is stored by reference
     * @param exp exponent, if < 0 then 1/sqrt(A) is computed, else sqrt(A)
     * @param weights the weights in which A is self-adjoint
     * @param eps_rel relative accuracy of solution
     * @param nrmb_correction absolute accuracy in units of \c eps_rel
     * @param max_iter Maximum number of iterations in Lanczos tridiagonalization
     * @param cauchy_steps number of cells in the Cauchy integral
     */
    template<class MatrixType>
    MatrixSqrt(  MatrixType& A, int exp,
            const Container& weights, value_type eps_rel,
            value_type nrmb_correction  = 1.,
            unsigned max_iter = 500, unsigned cauchy_steps = 40
            ) : m_weights(weights),
    m_exp(exp), m_cauchy( cauchy_steps), m_eps(eps_rel),
        m_abs(nrmb_correction)
    {
        m_A = [&]( const Container& x, Container& y){
            return dg::apply( A, x, y);
        };
        m_lanczos.construct( weights, max_iter);
        dg::mat::UniversalLanczos<Container> eigen( weights, 20);
        auto T = eigen.tridiag( A, weights, weights);
        m_EVs = dg::mat::compute_extreme_EV( T);
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = MatrixSqrt( std::forward<Params>( ps)...);
    }
    /// Get the number of Lanczos iterations in latest call to \c operator()
    unsigned get_iter() const{return m_number;}
    template<class ContainerType0, class ContainerType1>

    /**
     * @brief Apply matrix sqrt
     *
     * @param b input vector
     * @param x output vector, contains \f$ x = A^{\pm 1/2} \vec b\f$
     */
    void operator()( const ContainerType0 b, ContainerType1& x)
    {
        auto func = make_SqrtCauchyEigen_Te1( m_exp, m_EVs, m_cauchy);
        m_number = m_lanczos.solve( x, func, m_A, b, m_weights, m_eps, m_abs,
                "universal", 1., 2);
    }
    private:
    UniversalLanczos<Container> m_lanczos;
    Container m_weights;
    std::function< void( const Container&, Container&)> m_A;
    std::array<value_type, 2> m_EVs;
    int m_exp;
    unsigned m_number, m_cauchy;
    value_type m_eps, m_abs;

};
}//namespace mat
}//namespace dg
