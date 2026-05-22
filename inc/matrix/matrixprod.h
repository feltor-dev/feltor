
#pragma once

#include "dg/algorithm.h"
#include "lanczos.h"
#include "contours.h"
#include "optimise.h"

namespace dg{
namespace mat{

/**
 * @brief Computation of \f$ \vec x = f(A,\vec d)\vec b\f$ and \f$ \vec x = f(\vec d, A)\vec b\f$
 * where \f$ A \f$ is
 * a positive definite matrix self-adjoint in the weights \f$ W\f$ .
 *
 * The first identity is computed via \f$ \vec x = f(\vec d, A) \vec b = (E_{A} \odot F ) E^T_{A}M^T b\f$
 * where \f$ E_A := V_A E_T \f$ and \f$ F_{ai} := f( d_a, \lambda_i)\f$
 * and \f$ T\f$ and \f$ V_A\f$  are the tridiagonal matrix and vectors that
 * come out of a Lanczos iteration on \f$ A\f$, \f$ W\f$, \f$ \vec b\f$; \f$ \vec d\f$ is a vector.
 *
 * The second identity is computed via \f$ \vec x = f(A, \vec d) \vec b = E_{A} (F^T \odot   E^T_{A}M^T) b\f$
 * where \f$ E_A := V_A E_T \f$ and \f$ F_{ai} := f( d_a, \lambda_i)\f$
 * and \f$ T\f$ and \f$ V_A\f$  are the tridiagonal matrix and vectors that
 * come out of a Lanczos iteration on \f$ A\f$, \f$ W\f$, \f$ \vec b\f$; \f$ \vec d\f$ is a vector
 *
 * @ingroup matrixfunctionapproximation
 * @attention Just as in the Lanczos or PCG methods the matrix \f$ A\f$ needs to be positive-definite (i.e. it won't work for negative definite)
 * @note The \c apply and \c apply_adjoint methods are just abbreviations. If one wants full control, e.g. to reuse a tridiagonalisation one has to manually code:
 *
 * @code{.cpp}
 *  double max = dg::blas1::reduce( diag, -1e308, thrust::maximum<double>());
    auto func = dg::mat::make_FuncEigen_Te1( [&](value_type x) {return op( max, x);});
    dg::mat::ProductMatrixFunction<ContainerType> prod( x, 100);
    auto T = prod.lanczos().tridiag( func, A,
                b, weights, eps, nrmb_correction,
                "universal", 1.0, 1);
    prod.compute_vlcl( op, diag, A, T, x, b, prod.lanczos().get_bnorm());
    // or
    prod.compute_vlcl_adjoint( op, A, diag, T, x, b,
                weights, prod.lanczos().get_bnorm());
 * @endcode
 * @attention The adjoint methods unfortunately do not converge so use cautiously!
 * @sa dg::mat::UniversalLanczos dg::mat::CauchyMatrixProduct
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
     * @param max_iterations Maximum number of iterations in Lanczos to be used
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
    void set_benchmark( bool benchmark, std::string message = "ProductFunction"){
        m_benchmark = benchmark;
        m_message = message;
    }

    /**
     * @brief Compute \f$ \vec x = f(\vec d, A) \vec b = (E_{A} \odot F ) E^T_{A}M^T b\f$
     *
     * This function is equivalent to:
     * @code{.cpp}
        auto func = dg::mat::make_FuncEigen_Te1( [&](value_type x) {return op(1., x);});
        auto T = m_lanczos.tridiag( func, std::forward<MatrixType>(A),
                b, weights, eps, nrmb_correction,
                "universal", 1.0, 2);
        compute_vlcl( op, diag, std::forward<MatrixType>(A), T, x, b,
                    m_lanczos.get_bnorm());
        return T.num_rows;
     * @endcode
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
     * @attention The adjoint methods unfortunately do not converge so use cautiously!
     *
     * This function is equivalent to:
     * @code{.cpp}
        auto func = make_FuncEigen_Te1( [&](value_type x) {return op( x, 1.);});
        auto T = m_lanczos.tridiag( func, std::forward<MatrixType>(A),
                b, weights, eps, nrmb_correction,
                "universal", 1.0, 2);
        compute_vlcl_adjoint( op, std::forward<MatrixType>(A), diag, T, x, b,
                weights, m_lanczos.get_bnorm());
        return T.num_rows;
     * @endcode
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

    /**
     * @brief Compute \f$ \vec x = f(\vec d, A) \vec b = (E_{A} \odot F ) E^T_{A}M^T b\f$
     *
     * where \f$ E_A := V_A E_T \f$ and \f$ F_{ai} := f( d_a, \lambda_i)\f$
     * and \f$ T\f$ and \f$ V_A\f$  are the tridiagonal matrix and vectors that
     * come out of a Lanczos iteration on \f$ A\f$, \f$ W\f$, \f$ \vec b\f$; \f$ \vec d\f$ is a vector.
     *
     * This function takes a previously computed tridiagonalisation of \c A,
     * called \c T and computes the Eigendecomposition of \c T and then
     * re-creates the Eigenvectors in \c V to compute the above result.
     *
     * @note the Tridiagonalisation \c T can thus be reused to compute various matrix functions
     * of the same right hand side. It can be computed using
     * @code{.cpp}
     * double max = dg::blas1::reduce( diag, -1e308, thrust::maximum<double>());
       auto func = dg::mat::make_FuncEigen_Te1( [&](value_type x) {return op( max, x);});
       dg::mat::ProductMatrixFunction<ContainerType> prod( x, 100);
       auto T = prod.lanczos().tridiag( func, A,
                   b, weights, eps, nrmb_correction,
                   "universal", 1.0, 1);
       prod.compute_vlcl( op, diag, A, T, x, b, prod.lanczos().get_bnorm());
       // or
       prod.compute_vlcl_adjoint( op, A, diag, T, x, b,
                   weights, prod.lanczos().get_bnorm());
     * @endcode
     * @param op a  binary Operator representing the product matrix function
     * @param diag the diagonal vector
     * @param A A self-adjoint, positive definit matrix
     * @param T The tridiagonalisation of \c A
     * @param x output-vector, contains result on output, ignored on input
     * @param b The initial vector that starts orthogonalization
     * @param bnorm the norm of \c b
     */
    template< class BinaryOp, class ContainerType0, class MatrixType,
        class ContainerType1, class ContainerType2>
    void compute_vlcl( BinaryOp op, const ContainerType0& diag,
            MatrixType&& A,
            const TriDiagonal<thrust::host_vector<value_type>>& T,
            ContainerType1& x,
            const ContainerType2& b,
            value_type bnorm)
    {
        dg::blas1::copy(0., x);
        if( 0 == bnorm )
        {
            return;
        }
        unsigned iter = T.O.size();
        thrust::host_vector<value_type> evals = T.O, plus  = T.P;
        thrust::host_vector<value_type> work (2*iter-2);
        dg::SquareMatrix<value_type> EHt(iter);
        //Compute Eigendecomposition
        lapack::stev('V', evals, plus, EHt.data(), work);
        dg::blas1::axpby(1./bnorm, b, 0.0, m_v); //m_v[1] = b/||b||
        dg::blas1::copy(0., m_vm);
        // compute c_1 v_1
        for ( unsigned k=0; k<iter; k++)
        {
            dg::blas1::evaluate( m_f, dg::equals(), op, diag, evals[k]);
            dg::blas1::pointwiseDot( bnorm*EHt(k, 0)*EHt(k,0), m_f, m_v, 1.,
                    x);
        }
        for ( unsigned i=0; i<iter-1; i++)
        {
            dg::blas2::symv( std::forward<MatrixType>(A), m_v, m_vp);
            dg::blas1::axpbypgz(
                    -T.M[i]/T.P[i], m_vm,
                    -T.O[i]/T.P[i], m_v,
                        1.0/T.P[i], m_vp);
            m_vm.swap( m_v);
            m_v.swap( m_vp);
            // compute c_l v_l
            for ( unsigned k=0; k<iter; k++)
            {
                dg::blas1::evaluate( m_f, dg::equals(), op, diag, evals[k]);
                dg::blas1::pointwiseDot( bnorm*EHt(k,0)*EHt(k,i+1), m_f, m_v,
                        1., x);
            }
        }
    }

    /**
     * @brief Compute \f$ \vec x = f(A, \vec d) \vec b = E_{A} (F^T \odot   E^T_{A}M^T) b\f$
     *
     * where \f$ E_A := V_A E_T \f$ and \f$ F_{ai} := f( d_a, \lambda_i)\f$
     * and \f$ T\f$ and \f$ V_A\f$  are the tridiagonal matrix and vectors that
     * come out of a Lanczos iteration on \f$ A\f$, \f$ W\f$, \f$ \vec b\f$; \f$ \vec d\f$ is a vector
     *
     * @attention The adjoint methods unfortunately do not converge so use cautiously!
     *
     * This function takes a previously computed tridiagonalisation of \c A,
     * called \c T and computes the Eigendecomposition of \c T and then
     * re-creates the Eigenvectors in \c V to compute the above result.
     *
     * @note \f$ f(A, \vec d)\f$ is the adjoint operation to \f$ f( \vec d, A)\f$
     *  since both \f$ \vec d\f$ and \f$ A\f$ are self-adjoint.
     * @param op a  binary Operator representing the product matrix function
     * @param A A self-adjoint, positive definit matrix
     * @param diag the diagonal vector
     * @attention The order of \c A and \c diag is reversed compared to the
     * \c apply method
     * @param T The tridiagonalisation of \c A
     * @param x output-vector, contains result on output, ignored on input
     * @param b The initial vector that starts orthogonalization
     * @param weights Weights that define the scalar product in which \c A is
     *  self-adjoint and in which the error norm is computed.
     * @param bnorm the norm of \c b
     */
    template< class BinaryOp, class MatrixType, class ContainerType0,
        class ContainerType1, class ContainerType2, class ContainerType3>
    void compute_vlcl_adjoint( BinaryOp op,
            MatrixType&& A,
            const ContainerType0& diag,
            const TriDiagonal<thrust::host_vector<value_type>>& T,
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
        unsigned iter = T.O.size();
        thrust::host_vector<value_type> evals = T.O, plus  = T.P;
        thrust::host_vector<value_type> work (2*iter-2);
        dg::SquareMatrix<value_type> EHt(iter);
        //Compute Eigendecomposition
        lapack::stev('V', evals, plus, EHt.data(), work);
        dg::blas1::axpby(1./bnorm, b, 0.0, m_v); //m_v[1] = b/||b||
        dg::blas1::copy(0., m_vm);
        // compute alpha_i1
        dg::SquareMatrix<value_type> alpha(iter);
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
                    -T.M[i]/T.P[i], m_vm,
                    -T.O[i]/T.P[i], m_v,
                        1.0/T.P[i], m_vp);
            m_vm.swap( m_v);
            m_v.swap( m_vp);
            for ( unsigned k=0; k<iter; k++)
            {
                dg::blas1::evaluate( m_f, dg::equals(), op, evals[k], diag);
                dg::blas1::pointwiseDot( m_f, m_v, m_f);
                alpha( k,i+1) = dg::blas2::dot( m_f, weights, b);
            }
        }
        // Observation: With an exponential function the lines of alpha get extremely small (because exp(lambda) gets very small ... so maybe one can save a few scalar products
        // compute E_li E_ki alpha_ik v_l
        std::vector<double> cl( iter, 0.0);
        for( unsigned l=0; l<iter; l++)
            for( unsigned i=0; i<iter; i++)
                for( unsigned k=0; k<iter; k++)
                    cl[l] += EHt(k,i)*alpha(k,i)*EHt(k,l);
        // 3rd Lanczos iteration
        dg::blas1::axpby(1./bnorm, b, 0.0, m_v); //m_v[1] = b/||b||
        dg::blas1::copy(0., m_vm);
        dg::blas1::axpby( cl[0], m_v, 1., x);
        for ( unsigned i=0; i<iter-1; i++)
        {
            dg::blas2::symv( std::forward<MatrixType>(A), m_v, m_vp);
            dg::blas1::axpbypgz(
                    -T.M[i]/T.P[i], m_vm,
                    -T.O[i]/T.P[i], m_v,
                        1.0/T.P[i], m_vp);
            m_vm.swap( m_v);
            m_v.swap( m_vp);
            dg::blas1::axpby( cl[i+1], m_v, 1., x);
        }
    }

    /**
     * @brief Access the Lanczos class that is constructed with the constructor parameters
     */
    UniversalLanczos<ContainerType>& lanczos() { return m_lanczos;}
    private:

    UniversalLanczos<ContainerType> m_lanczos;
    bool m_benchmark = true;
    std::string m_message = "ProductFunction";
    ContainerType  m_v, m_vp, m_vm, m_f;
};

/*!
 * @brief Computation of \f$ \vec x = f(A,\vec d)\vec b\f$ or \f$ \vec x = f( \vec d, A) \vec b\f$ where \f$ A \f$ is a
 * positive (semi)-definite matrix self-adjoint in the weights \f$ W\f$ .
 *
 * This class implements the %Cauchy contour integral method
 * \f[
 * \begin{align}
 * f( A, D) \vec b \approx \sum_{k=1}^{N} \frac{1}{z_k 1 -  A} w_kf(z_k, D) \vec b\\
 * f( D, A) \vec b \approx \sum_{k=1}^{N} w_kf(z_k, D)\frac{1}{z_k 1 -  A}  \vec b\\
 * \end{align
 * \f]
 *
 * The complex nodes and weights \f$ z_k\f$ and \f$ w_k\f$ are found by applying
 * the Levenberg-Marquardt optimization to an initial Talbot curve. The number of nodes is
 * such that the given error tolerance is fulfilled.
 * The individual complex Helmholtz type equations are solved using a \c dg::MultigridCG2d COCG algorithm
 * and we store the previous result at every timestep
 *
 * The class automatically caches the extreme Eigenvalues of the matrix A.
 * Furthermore, the previous solution(s) to the Helmholtz equations are stored.
 * The \c solve method automatically recognises a change in D or the matrix
 * function f and accordingly recomputes nodes and weighs and clears the
 * previous solution cache. However, changes in A are not automatically
 * recognised. If the matrix needs to change the \c clear_cache member function
 * must be called before the next solve call in order to trigger a
 * re-computation of the Eigenvalues.
 *
 * @tparam Geometry The Geometry type in \c dg::MultigridCG2d
 * @tparam Matrix The (real) derviative class for projection / interpolation in Multigrid
 * @tparam ComplexContainer A complex Container type
 * @ingroup matrixfunctionapproximation
 */
template<class Geometry, class Matrix, class ComplexContainer>
struct CauchyMatrixProduct
{
    CauchyMatrixProduct() = default;
    CauchyMatrixProduct( double lm_eps, const Geometry& grid, unsigned stages, bool adjoint = true )
    : m_eps( lm_eps),
    m_multi( grid, stages),
    m_previous( 2, {1, m_multi.copyable()}),
    m_z( m_multi.copyable()),
    m_rhs( m_multi.copyable()),
    m_grid_points(grid.size()),
    m_cauchy_opt(),
    m_adjoint(adjoint)
    {
    }

    /// Access the internal multigrid method to be able to construct matrices
    const dg::MultigridCG2d<Geometry, Matrix, ComplexContainer,
        dg::complex_symmetric>& multigrid() const { return m_multi;}

    /*!
     * @brief Clear the cached Eigenvalues of the matrix in the \c solve method
     *
     * Call if the matrix in the next call to the \c solve method changes
     * (which typically should not happen);
     * ignore otherwise.
     */
    void clear_cache(){
        m_EV_up2date = false;
    }

    /*!
     * @brief Verbose output to \c std::cout
     * @param verbose If true output more information to \c std::cout
     */
    void set_verbose( bool verbose) {
        m_verbose = verbose;
        m_cauchy_opt.set_verbose(verbose);
    }

    /*!
     * @brief Number of (complex) nodes used in the latest call to \c solve
     * @return Number of complex nodes
     */
    unsigned num_nodes() const { return m_cauchy_opt.num_nodes();}

    /*!
     * @brief Determine if adjoint or direct bivariate matrix function is computed
     *
     * @attention Changing this parameter resets the solution cache (i.e. previous stored solutions are zeroed)
     * @param adjoint If true compute the matrix function \f$ f(A, d)b\f$, else compute \f$ f(d, A)b\f$
     */
    void set_adjoint( bool adjoint) {
        // reset solution cache
        if( m_adjoint != adjoint)
        {
            m_previous.assign( m_previous.size(), {1, m_multi.copyable()});
            m_adjoint = adjoint;
        }
    }

    /// Current value of the adjoint parameter
    bool get_adjoint() const { return m_adjoint;}

    /*!
     * @brief Compute the bivariate matrix function
     *
     * In the first call the extreme Eigenvalues of \c ops[0] are computed and
     * stored.  In the following calls the cached Eigenvalues are used unless
     * \c clear_cache is called beforehand.
     *
     * In a first step we then determine if the previously used nodes and weights
     * are still sufficient for the given parameters and optionally re-compute them.
     * (The method tries to avoid recomputing the nodes if possible because of
     * how long it may take).
     * In a second step the num_nodes complex Helmholtz problems are solved using
     * multigrid methods and initial guesses from previous solves.
     * @param x (write-only) Contains solution on output
     * @param func The bivariate matrix function
     * @param dxfunc The derivative of the bivariate matrix function
     * @param ops The matrix discretized on the grid used in \c multigrid()
     * @param d The diagonal vector
     * @param b the right hand side
     * @param eps the error tolerance forwarded to the \c multigrid().solve method
     */
    template<class MatrixType, class UnaryFunc, class UnaryFuncD,
        class ContainerType0, class ContainerType1, class ContainerType2>
    void solve( ContainerType0& x, UnaryFunc func, UnaryFuncD dxfunc, std::vector<MatrixType>& ops,
        const ContainerType1& d, const ContainerType2& b, std::vector<double> eps)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        // 1. (re)compute current nodes and weights
        if( !m_EV_up2date)
        {
            // 1. Compute extreme Eigenvalues
            update_extremeEVs( ops);
        }
        double dmin = dg::blas1::reduce( d, +1e300, thrust::minimum());
        double dmax = dg::blas1::reduce( d, -1e300, thrust::maximum());
        bool changed = false;
        if( m_verbose )
        {
            DG_RANK0 std::cout << "# "<<dmin<<" < D < "<<dmax<<"\n";
            DG_RANK0 std::cout << "# "<<m_lmin<<" < Lambda < "<<m_lmax<<"\n";
        }
        const auto& zkwk = m_cauchy_opt.update_zkwk( changed, func, dxfunc,
            m_lmin, m_lmax, dmin, dmax, m_with_zero, m_eps);
        if( changed) // if the nodes change we need to re-alloced solution space.
            m_previous.assign( m_cauchy_opt.num_nodes(), {1, m_multi.copyable()});

        if( m_verbose )
        {
            DG_RANK0 std::cout << "# Current number of nodes "<<m_cauchy_opt.num_nodes()<<"\n";
        }

        thrust::complex<double> zk, wk;
        ///////////////
        struct ShiftedOp
        {
            ShiftedOp( MatrixType& mat, const thrust::complex<double>& zk)
            : m_zk(zk), m_mat(mat){}
            void operator()( const ComplexContainer& x, ComplexContainer& y)
            {
                // Question: does COCG not care if matrix is positive/negative definite?
                // maybe not, since matrix does not have real EV anyways?
                dg::blas2::symv( m_mat, x, y);
                dg::blas1::axpby( m_zk, x, -1., y);
            }
            auto weights() const { return m_mat.weights();}
            auto precond() const { return m_mat.precond();}
            private:
            const thrust::complex<double>& m_zk;
            MatrixType& m_mat;
        };
        ///////////////
        std::vector<ShiftedOp > shifted_ops;
        for( unsigned u=0; u<m_multi.stages(); u++)
            shifted_ops.push_back( ShiftedOp{ ops[u], zk});

        dg::blas1::copy( 0., x);
        for( unsigned k=0; k<m_cauchy_opt.num_nodes(); k++)
        //for( int k=m_cauchy_opt.num_nodes()-1; k>=0; k--)
        {
            zk = zkwk.first[k];
            wk = zkwk.second[k];
            // std::cout << "Zk wk "<<zk<<" "<<wk<<"\n";
            // The very first m_z is zero: this should work in COCG as an allowed initial guess
            m_previous[k].extrapolate( m_z);

            if( m_adjoint)
            {
                dg::blas1::axpby( zk, d, 0., m_rhs);
                dg::blas1::transform ( m_rhs, m_rhs, func);
                dg::blas1::pointwiseDot( wk, m_rhs, b, 0., m_rhs);
                m_multi.solve( shifted_ops, m_z, m_rhs, eps);
                m_previous[k].update( m_z);
            }
            else
            {
                dg::blas1::axpby( wk, b, 0., m_rhs);
                m_multi.solve( shifted_ops, m_z, m_rhs, eps);
                m_previous[k].update( m_z);
                dg::blas1::axpby( zk, d, 0., m_rhs);
                dg::blas1::transform ( m_rhs, m_rhs, func);
                dg::blas1::pointwiseDot( m_rhs, m_z, m_z);
            }
            dg::blas1::subroutine([]DG_DEVICE( thrust::complex<double> z, double& x) {
                x += 2*z.real();}, m_z, x );
        }
        //std::cout << "SOL\n";
        //for( unsigned u=0; u<10; u++)
        //    std::cout << x[u]<<" ";
        //std::cout << std::endl;
    }

    private:

    template<class MatrixType>
    void update_extremeEVs(std::vector<MatrixType>& ops )
    {
        dg::mat::UniversalLanczos<ComplexContainer> lanczos( ops[0].weights(), 2000);
        if( m_verbose)
            lanczos.set_verbose(true);
        auto rnd = ops[0].weights();
        dg::blas1::transform( rnd, rnd, dg::RandomNumbers<double>(0.0,1.0));
        auto T = lanczos.tridiag( ops[0], rnd, ops[0].weights(), 1e-4, 1., "compute_extreme_EV");
        //auto T = lanczos.tridiag( ops[0], rnd, ops[0].weights());
        auto EVs = dg::mat::compute_extreme_EV( T);
        // Let's use 10% safety range here (maybe test more but initial test show a slight improvement)
        m_lmax = 1.1*EVs[1];
        m_lmin = 0.9*EVs[0];
        m_with_zero = false;
        if( m_lmin < 1e-10*m_lmax)
        {
            if( m_verbose) DG_RANK0 std::cout << "# Found zero EV!\n";
            m_lmin = m_lmax/ m_grid_points;
            m_with_zero = true;
        }
        m_EV_up2date = true;
    }


    double m_eps;
    MultigridCG2d<Geometry, Matrix, ComplexContainer, dg::complex_symmetric> m_multi; // does not remember any solutions
    std::vector<dg::Extrapolation<ComplexContainer, double>> m_previous; // previous solutions for every zk
    ComplexContainer m_z, m_rhs; // complex vectors
    unsigned m_grid_points;
    CauchyOptimizer m_cauchy_opt;
    bool m_adjoint = true;

    bool m_with_zero = false;
    double m_lmin = 0, m_lmax = 0;
    bool m_EV_up2date = false;
    bool m_verbose = false;
};

}//namespace mat
}//namespace dg
