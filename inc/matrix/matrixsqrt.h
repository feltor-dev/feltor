#pragma once

#include "dg/algorithm.h"
#include "lanczos.h"

namespace dg{
namespace mat{


/**
 * @brief Fast computation of \f$ \vec x = A^{\pm 1/2}\vec b\f$ for self-adjoint positive definite \f$A\f$
 *
 * Convenience wrapper that
 * uses \c dg::mat::UniversalLanczos combined with
 * \c dg::mat::make_SqrtCauchyEigen_Te1 in its
 * "universal" stopping criterion
 * @note This is the fastest method to compute matrix square roots vector
 *  multiplications that we found to date
 * @ingroup matrixfunctionapproximation
 * @attention Just as in the Lanczos or PCG methods the matrix \f$ A\f$ needs to be positive-definite (i.e. it won't work for negative definite)
 */
template<class ContainerType>
struct MatrixSqrt
{
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>;

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
            const ContainerType& weights, value_type eps_rel,
            value_type nrmb_correction  = 1.,
            unsigned max_iter = 500, unsigned cauchy_steps = 40
            ) : m_weights(weights),
    m_exp(exp), m_cauchy( cauchy_steps), m_eps(eps_rel),
        m_abs(nrmb_correction)
    {
        m_A = [&]( const ContainerType& x, ContainerType& y){
            return dg::apply( A, x, y);
        };
        m_lanczos.construct( weights, max_iter);
        dg::mat::UniversalLanczos<ContainerType> eigen( weights, 20);
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

    /**
     *@brief Set or unset performance timings during iterations
     *@param benchmark If true, additional output will be written to \c std::cout during solution
     *@param message An optional identifier that is printed together with the
     * benchmark (intended use is to distinguish different messages
     * in the output)
    */
    void set_benchmark( bool benchmark, std::string message = "SQRT"){
        m_benchmark = benchmark;
        m_message = message;
    }

    /**
     * @brief Apply matrix sqrt
     *
     * @param b input vector
     * @param x output vector, contains \f$ x = A^{\pm 1/2} \vec b\f$
     */
    template<class ContainerType0, class ContainerType1>
    void operator()( const ContainerType0 b, ContainerType1& x)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        dg::Timer t;
        t.tic();
        auto func = make_SqrtCauchyEigen_Te1( m_exp, m_EVs, m_cauchy);
        m_number = m_lanczos.solve( x, func, m_A, b, m_weights, m_eps, m_abs,
                "universal", 1., 2);
        t.toc();
        if( m_benchmark)
            DG_RANK0 std::cout << "# `"<<m_message<<"` solve with {"<<m_number<<","<<m_cauchy<<"} iterations took "<<t.diff()<<"s\n";
    }
    private:
    UniversalLanczos<ContainerType> m_lanczos;
    ContainerType m_weights;
    std::function< void( const ContainerType&, ContainerType&)> m_A;
    std::array<value_type, 2> m_EVs;
    int m_exp;
    unsigned m_number, m_cauchy;
    value_type m_eps, m_abs;
    bool m_benchmark = true;
    std::string m_message = "SQRT";

};

// The following is only indirectly tested in diffusion project but should appear formally in _t file here as well
/**
 * @brief Computation of \f$ \vec x = f(A)\vec b\f$ for self-adjoint positive definite \f$ A\f$
 *
 * where \f$ f(x) = f_{outer}(f_{inner}(x))\f$ is composed of an inner pre-factor function
 * \f$f_{inner}(x)\f$ and an outer \f$ f_{outer}(x)\f$ function.
 * The rational for this design choice is maintain
 * flexibility when using this class in one of our exponential integrators where the
 * outer function is set by the time integrator itself while \f$ f_{inner}\f$ can be
 * set by the user. Outside the exponential time integrator \f$ f_{inner}\f$ has no use
 * and left as the default identity.
 *
 * The class is a convenience wrapper that
 * uses \c dg::mat::UniversalLanczos combined with
 * \c dg::mat::make_FuncEigen_Te1( f) in its
 * "universal" stopping criterion
 * @ingroup matrixfunctionapproximation
 * @attention Just as in the Lanczos or PCG methods the matrix \f$ A\f$ needs to be positive-definite (i.e. it won't work for negative definite)
 */
template<class ContainerType>
struct MatrixFunction
{
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>;

    /// Construct empty
    MatrixFunction() = default;

    /**
     * @brief Construct from matrix
     *
     * @tparam MatrixType
     * @param A self-adjoint matrix; is stored by reference
     * @param weights the weights in which A is self-adjoint
     * @param eps_rel relative accuracy of solution
     * @param nrmb_correction absolute accuracy in units of \c eps_rel
     * @param max_iter Maximum number of iterations in Lanczos tridiagonalization
     * @param f_inner the inner "pre-factor" function (useful only in connection with
     * a exponential integrator where f_outer is set by the integrator)
     */
    template<class MatrixType>
    MatrixFunction(  MatrixType& A,
            const ContainerType& weights, value_type eps_rel,
            value_type nrmb_correction  = 1.,
            unsigned max_iter = 500,
            std::function<value_type(value_type)> f_inner = [](value_type x){return x;}
            ) : m_weights(weights),
        m_f_inner(f_inner), m_eps(eps_rel),
        m_abs(nrmb_correction)
    {
        m_A = [&]( const ContainerType& x, ContainerType& y){
            return dg::apply( A, x, y);
        };
        m_lanczos.construct( weights, max_iter);
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = MatrixFunction( std::forward<Params>( ps)...);
    }
    /// Get the number of Lanczos iterations in latest call to \c operator()
    unsigned get_iter() const{return m_number;}

    /**
     *@brief Set or unset performance timings during iterations
     *@param benchmark If true, additional output will be written to \c std::cout during solution
     *@param message An optional identifier that is printed together with the
     * benchmark (intended use is to distinguish different messages
     * in the output)
    */
    void set_benchmark( bool benchmark, std::string message = "Function"){
        m_benchmark = benchmark;
        m_message = message;
    }

    /**
     * @brief Apply matrix function
     *
     * @param f_outer Matrix function to apply together with \c f_inner from constructor
     * @param b input vector
     * @param x output vector, contains \f$ x = f(A) \vec b\f$
     */
    template<class UnaryOp, class ContainerType0, class ContainerType1>
    void operator()( UnaryOp f_outer, const ContainerType0 b, ContainerType1& x)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        dg::Timer t;
        t.tic();
        auto func = make_FuncEigen_Te1( [&](value_type x) {return f_outer(m_f_inner(x));});
        m_number = m_lanczos.solve( x, func, m_A, b, m_weights, m_eps, m_abs,
                "universal", 1., 2);
        t.toc();
        if( m_benchmark)
            DG_RANK0 std::cout << "# `"<<m_message<<"` solve with {"<<m_number<<"} iterations took "<<t.diff()<<"s\n";
    }
    private:
    UniversalLanczos<ContainerType> m_lanczos;
    ContainerType m_weights;
    std::function< void( const ContainerType&, ContainerType&)> m_A;
    std::array<value_type, 2> m_EVs;
    std::function<value_type(value_type)> m_f_inner;
    unsigned m_number;
    value_type m_eps, m_abs;
    bool m_benchmark = true;
    std::string m_message = "Function";

};


}//namespace mat
}//namespace dg
