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

/**
 * @brief Computation of \f$ \vec x = f(A,\vec d)\vec b\f$ and \f$ \vec x = f(\vec d, A)\vec b\f$
 * for self-adjoint positive definite \f$ A\f$
 *
 * @ingroup matrixfunctionapproximation
 * @attention Just as in the Lanczos or PCG methods the matrix \f$ A\f$ needs to be positive-definite (i.e. it won't work for negative definite)
 */
template<class ContainerType>
struct ProductMatrixFunction
{
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>;
    /// Construct empty
    ProductMatrixFunction() = default;

    /**
     * @brief Allocate memory for the method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    ProductMatrixFunction( const ContainerType& copyable, unsigned max_iterations)
    {
        m_lanczos.construct( copyable, max_iterations);
        m_v = m_vp = m_vm = m_f = copyable;
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = ProductMatrixFunction( std::forward<Params>( ps)...);
    }

    ///@copydoc MatrixFunction::set_benchmark(bool,std::string)
    void set_benchmark( bool benchmark, std::string message = "Function"){
        m_benchmark = benchmark;
        m_message = message;
    }

    /**
     * @brief Compute \f$ \vec x = f(\vec d, A) \vec b = (E_{A} \odot F ) E^T_{A}M^T b\f$
     *
     * where \f$ E_A := V_A E_T \f$ and \f$ F_{ai} := f( d_a, \lambda_i)\f$
     * and \f$ T\f$ and \f$ V_A\f$  are the tridiagonal matrix and vectors that
     * come out of a Lanczos iteration on \f$ A\f$, \f$ W\f$, \f$ \vec b\f$; \f$ \vec d\f$ is a vector and \f$ A \f$ is
     * a positive definite matrix self-adjoint in the weights \f$ W\f$ .
     * @note The stopping criterion used on the Lanczos iteration is the
     * universal one applied to \f$ f(1, x) \f$
     * @param x output-vector, contains result on output, ignored on input
     * @param op a  binary Operator representing the product matrix function
     * @param diag the diagonal vector
     * @param A A self-adjoint, positive definit matrix
     * @param b The initial vector that starts orthogonalization
     * @param weights Weights that define the scalar product in which \c A is
     *  self-adjoint and in which the error norm is computed.
     * @param eps relative accuracy of residual in Lanczos iteration
     * @param nrmb_correction the absolute error \c C in units of \c eps to be
     * respected
     * @return The number of Lanczos iterations used
     */
    template<class ContainerType0, class BinaryOp, class ContainerType1,
        class MatrixType, class ContainerType2, class ContainerType3>
    unsigned apply(
            ContainerType0& x,
            BinaryOp op,
            const ContainerType1& diag,
            MatrixType&& A,
            const ContainerType2& b,
            const ContainerType3& weights,
            value_type eps,
            value_type nrmb_correction = 1.)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        dg::Timer t;
        t.tic();
        auto func = make_FuncEigen_Te1( [&](value_type x) {return op(1., x);});
        auto T = m_lanczos.tridiag( func, std::forward<MatrixType>(A),
                b, weights, eps, nrmb_correction,
                "universal", 1.0, 2);
        compute_vlcl( op, diag, std::forward<MatrixType>(A), T, x, b,
                    m_lanczos.get_bnorm());
        t.toc();
        if( m_benchmark)
            DG_RANK0 std::cout << "# `"<<m_message<<"` solve with {"<<T.num_rows<<"} iterations took "<<t.diff()<<"s\n";
        return T.num_rows;
    }
    /**
     * @brief Compute \f$ \vec x = f(A, \vec d) \vec b = E_{A} (F^T \odot   E^T_{A}M^T) b\f$
     *
     * where \f$ E_A := V_A E_T \f$ and \f$ F_{ai} := f( d_a, \lambda_i)\f$
     * and \f$ T\f$ and \f$ V_A\f$  are the tridiagonal matrix and vectors that
     * come out of a Lanczos iteration on \f$ A\f$, \f$ W\f$, \f$ \vec b\f$; \f$ \vec d\f$ is a vector and \f$ A \f$ is
     * a positive definite matrix self-adjoint in the weights \f$ W\f$ .
     * @note \f$ f(A, \vec d)\f$ is the adjoint operation to \f$ f( \vec d, A)\f$
     *  since both \f$ \vec d\f$ and \f$ A\f$ are self-adjoint.
     * @note The stopping criterion used on the Lanczos iteration is the
     * universal one applied to \f$ f(x, 1) \f$
     * @param x output-vector, contains result on output, ignored on input
     * @param op a  binary Operator representing the product matrix function
     * @param diag the diagonal vector
     * @param A A self-adjoint, positive definit matrix
     * @attention The order of \c A and \c diag is reversed compared to the
     * \c apply method
     * @param b The initial vector that starts orthogonalization
     * @param weights Weights that define the scalar product in which \c A is
     *  self-adjoint and in which the error norm is computed.
     * @param eps relative accuracy of residual in Lanczos iteration
     * @param nrmb_correction the absolute error \c C in units of \c eps to be
     * respected
     * @return The number of Lanczos iterations used
     */
    template<class ContainerType0, class BinaryOp, class MatrixType,
        class ContainerType1, class ContainerType2, class ContainerType3>
    unsigned apply_adjoint(
            ContainerType0& x,
            BinaryOp op,
            MatrixType&& A,
            const ContainerType1& diag,
            const ContainerType2& b,
            const ContainerType3& weights,
            value_type eps,
            value_type nrmb_correction = 1.)
    {
        // Should this be another class?
        // if A does not change Lanczos iterations could be reused from apply function!?
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        dg::Timer t;
        t.tic();
        auto func = make_FuncEigen_Te1( [&](value_type x) {return op( x, 1.);});
        auto T = m_lanczos.tridiag( func, std::forward<MatrixType>(A),
                b, weights, eps, nrmb_correction,
                "universal", 1.0, 2);
        compute_vlcl_adjoint( op, std::forward<MatrixType>(A), diag, T, x, b,
                weights, m_lanczos.get_bnorm());

        t.toc();
        if( m_benchmark)
            DG_RANK0 std::cout << "# `"<<m_message<<"` solve with {"<<T.num_rows<<"} iterations took "<<t.diff()<<"s\n";
        return T.num_rows;
    }

    private:
    template< class BinaryOp, class ContainerType0, class MatrixType,
        class DiaMatrixType, class ContainerType1,
        class ContainerType2>
    void compute_vlcl( BinaryOp op, const ContainerType0& diag,
            MatrixType&& A,
            const DiaMatrixType& T,
            ContainerType1& x,
            const ContainerType2& b,
            value_type bnorm)
    {
        dg::blas1::copy(0., x);
        if( 0 == bnorm )
        {
            return;
        }
        unsigned iter = T.num_rows;
        cusp::array2d< value_type, cusp::host_memory> evecs(iter,iter);
        cusp::array1d< value_type, cusp::host_memory> evals(iter);
        cusp::lapack::stev(T.values.column(1), T.values.column(2),
                evals, evecs, 'V');
        dg::blas1::axpby(1./bnorm, b, 0.0, m_v); //m_v[1] = b/||b||
        dg::blas1::copy(0., m_vm);
        // compute c_1 v_1
        for ( unsigned k=0; k<iter; k++)
        {
            dg::blas1::evaluate( m_f, dg::equals(), op, diag, evals[k]);
            dg::blas1::pointwiseDot( bnorm*evecs(0, k)*evecs(0,k), m_f, m_v, 1.,
                    x);
        }
        for ( unsigned i=0; i<iter-1; i++)
        {
            dg::blas2::symv( std::forward<MatrixType>(A), m_v, m_vp);
            dg::blas1::axpbypgz(
                    -T.values(i,0)/T.values(i,2), m_vm,
                    -T.values(i,1)/T.values(i,2), m_v,
                               1.0/T.values(i,2), m_vp);
            m_vm.swap( m_v);
            m_v.swap( m_vp);
            // compute c_l v_l
            for ( unsigned k=0; k<iter; k++)
            {
                dg::blas1::evaluate( m_f, dg::equals(), op, diag, evals[k]);
                dg::blas1::pointwiseDot( bnorm*evecs(0, k)*evecs(i+1,k), m_f, m_v,
                        1., x);
            }
        }
    }
    template< class BinaryOp, class MatrixType, class ContainerType0,
        class DiaMatrixType, class ContainerType1,
        class ContainerType2, class ContainerType3>
    void compute_vlcl_adjoint( BinaryOp op,
            MatrixType&& A,
            const ContainerType0& diag,
            const DiaMatrixType& T,
            ContainerType1& x,
            const ContainerType2& b,
            const ContainerType3& weights,
            value_type bnorm)
    {
        dg::blas1::copy(0., x);
        if( 0 == bnorm )
        {
            return;
        }
        unsigned iter = T.num_rows;
        cusp::array2d< value_type, cusp::host_memory> evecs(iter,iter);
        cusp::array1d< value_type, cusp::host_memory> evals(iter);
        cusp::lapack::stev(T.values.column(1), T.values.column(2),
                evals, evecs, 'V');
        dg::blas1::axpby(1./bnorm, b, 0.0, m_v); //m_v[1] = b/||b||
        dg::blas1::copy(0., m_vm);
        // compute alpha_i1
        cusp::array2d< value_type, cusp::host_memory> alpha(iter,iter);
        for ( unsigned k=0; k<iter; k++)
        {
            dg::blas1::evaluate( m_f, dg::equals(), op, evals[k], diag);
            dg::blas1::pointwiseDot( m_f, m_v, m_f);
            alpha( k,0) = dg::blas2::dot( m_f, weights, b);
        }
        for ( unsigned i=0; i<iter-1; i++)
        {
            dg::blas2::symv( std::forward<MatrixType>(A), m_v, m_vp);
            dg::blas1::axpbypgz(
                    -T.values(i,0)/T.values(i,2), m_vm,
                    -T.values(i,1)/T.values(i,2), m_v,
                               1.0/T.values(i,2), m_vp);
            m_vm.swap( m_v);
            m_v.swap( m_vp);
            for ( unsigned k=0; k<iter; k++)
            {
                dg::blas1::evaluate( m_f, dg::equals(), op, evals[k], diag);
                dg::blas1::pointwiseDot( m_f, m_v, m_f);
                alpha( k,i+1) = dg::blas2::dot( m_f, weights, b);
            }
        }
        // compute E_li E_ki alpha_ik v_l
        std::vector<double> cl( iter, 0.0);
        for( unsigned l=0; l<iter; l++)
            for( unsigned i=0; i<iter; i++)
                for( unsigned k=0; k<iter; k++)
                    cl[l] += evecs(i,k)*alpha(k,i)*evecs(l,k);
        // 3rd Lanczos iteration
        dg::blas1::axpby(1./bnorm, b, 0.0, m_v); //m_v[1] = b/||b||
        dg::blas1::copy(0., m_vm);
        dg::blas1::axpby( cl[0], m_v, 1., x);
        for ( unsigned i=0; i<iter-1; i++)
        {
            dg::blas2::symv( std::forward<MatrixType>(A), m_v, m_vp);
            dg::blas1::axpbypgz(
                    -T.values(i,0)/T.values(i,2), m_vm,
                    -T.values(i,1)/T.values(i,2), m_v,
                               1.0/T.values(i,2), m_vp);
            m_vm.swap( m_v);
            m_v.swap( m_vp);
            dg::blas1::axpby( cl[i+1], m_v, 1., x);
        }
    }

    UniversalLanczos<ContainerType> m_lanczos;
    bool m_benchmark = true;
    std::string m_message = "Function";
    ContainerType  m_v, m_vp, m_vm, m_f;
};

}//namespace mat
}//namespace dg
