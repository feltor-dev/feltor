
#pragma once

#include "dg/algorithm.h"
#include "lanczos.h"
#include "contours.h"

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
 * @sa dg::mat::UniversalLanczos dg::mat::CauchyMatrixProductAdj
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
 * @brief Computation of \f$ \vec x = f(\alpha A,\vec d)\vec b\f$ where \f$ A \f$ is a
 * positive definite matrix self-adjoint in the weights \f$ W\f$ .
 *
 * This class implements the %Cauchy contour integral method
 * \f[ f( \alpha A, D) \vec b \approx \sum_{k=1}^{N} \frac{w_k}{z_k 1 - \alpha A} f(z_k, D) \vec b \f]
 *
 * The complex nodes and weights \f$ z_k\f$ and \f$ w_k\f$ are found by applying
 * the Levenberg-Marquardt optimization to an initial Talbot curve. The number of nodes is
 * a constructor parameter that cannot be changed afterwards.
 * The individual complex Helmholtz type equations are solved using a dg::MultigridCG2d COCG algorithm
 * and we store the previous result at every timestep
 *
 * The class automatically keeps a solution cache, which contains the extreme Eigenvalues of the matrix
 * A and the diagonal matrix D together with the optimal complex nodes and weights. Furthermore, the
 * previous solution(s) to the Helmholtz equations are stored. The class automatically recognises a change
 * in D but not in A or the matrix function f. If either of those two change the \c clear_cache member
 * function must be called before a solve call.
 *
 * @tparam Geometry The Geometry type in MultigridCG2d
 * @tparam Matrix The (real) derviative class for projection / interpolation in Multigrid
 * @tparam ComplexContainer A complex Container type
 * @ingroup matrixfunctionapproximation
 */
template<class Geometry, class Matrix, class ComplexContainer>
struct CauchyMatrixProductAdj
{
    CauchyMatrixProductAdj() = default;
    CauchyMatrixProductAdj( unsigned num_nodes, const Geometry& grid, unsigned stages )
    : m_num_nodes( num_nodes), m_multi( grid, stages), m_previous( num_nodes,
        {1, m_multi.copyable()}), m_z( m_multi.copyable()), m_rhs( m_multi.copyable())
    {
    }

    const dg::MultigridCG2d<Geometry, Matrix, ComplexContainer, dg::complex_symmetric>& multigrid() { return m_multi;}
    template<class MatrixType, class UnaryFunc, class UnaryFuncD,
        class ContainerType0, class ContainerType1, class ContainerType2>
    void solve( ContainerType0& x, UnaryFunc func, UnaryFuncD dxlnfunc, double alpha, std::vector<MatrixType>& ops,
        const ContainerType1& d, const ContainerType2& b, std::vector<double> eps)
    {
        bool zeroInit = false;
        if( !m_up2date)
        {
            zeroInit = true;
            init_cache( ops, func, dxlnfunc, d);
        }

        double dmin = dg::blas1::reduce( d, +1e300, thrust::minimum());
        double dmax = dg::blas1::reduce( d, -1e300, thrust::maximum());
        if( dmin < m_dmin || dmax > m_dmax)
            update_zkwk( func, dxlnfunc, dmin, dmax);

        thrust::complex<double> zk, wk;
        ///////////////
        struct ShiftedOp
        {
            ShiftedOp( MatrixType& mat, const thrust::complex<double>& z, double alpha)
            : m_z(z), m_mat(mat), m_alpha(alpha){}
            void operator()( const ComplexContainer& x, ComplexContainer& y)
            {
                // Question: does COCG not care if matrix is positive/negative definite?
                // maybe not, since matrix does not have real EV anyways?
                dg::blas2::symv( m_mat, x, y);
                dg::blas1::axpby( -m_z, x, -m_alpha, y);
            }
            auto weights() const { return m_mat.weights();}
            auto precond() const { return m_mat.precond();}
            private:
            const thrust::complex<double>& m_z;
            MatrixType& m_mat;
            double m_alpha;
        };
        ///////////////
        std::vector<ShiftedOp > shifted_ops;
        for( unsigned u=0; u<m_multi.stages(); u++)
            shifted_ops.push_back( ShiftedOp{ ops[u], zk, alpha});

        dg::blas1::copy( 0., x);
        for( unsigned k=0; k<m_num_nodes; k++)
        {
            zk = m_zk[k];
            wk = m_wk[k];
            // std::cout << "Zk wk "<<zk<<" "<<wk<<"\n";
            // The very first m_z is zero: this should work in COCG as an allowed initial guess
            if( zeroInit && k == 0)
                dg::blas1::copy( 0., m_z);
            else if( zeroInit && k > 0)
                m_previous[k-1].extrapolate( m_z);
            else
                m_previous[k].extrapolate( m_z);
            dg::blas1::axpby( zk, d, 0., m_rhs);
            dg::blas1::transform ( m_rhs, m_rhs, func);
            dg::blas1::pointwiseDot( wk, m_rhs, b, 0., m_rhs);
            m_multi.solve( shifted_ops, m_z, m_rhs, eps);
            m_previous[k].update( m_z);
            dg::blas1::subroutine([]DG_DEVICE( thrust::complex<double> z, double& x) {
                x += 2*z.real();}, m_z, x );
        }
        //std::cout << "SOL\n";
        //for( unsigned u=0; u<10; u++)
        //    std::cout << x[u]<<" ";
        //std::cout << std::endl;
    }

    void clear_cache(){
        m_up2date = false;
    }

    void set_verbose( bool verbose) { m_verbose = verbose;}
    private:
    template<class MatrixType, class UnaryFunc, class UnaryFuncD,
        class ContainerType1>
    void init_cache( std::vector<MatrixType>& ops, UnaryFunc func, UnaryFuncD dxlnfunc,
        const ContainerType1& d)
    {
        // 1. Compute extreme Eigenvalues and min/max of d
        update_extremeEVs( ops[0]);
        double dmin = dg::blas1::reduce( d, +1e300, thrust::minimum());
        double dmax = dg::blas1::reduce( d, -1e300, thrust::maximum());
        update_zkwk( func, dxlnfunc, dmin, dmax);
        m_up2date = true;
    }

    template<class UnaryFunc, class UnaryFuncD>
    void update_zkwk( UnaryFunc func, UnaryFuncD dxlnfunc, double dmin, double dmax)
    {
        m_dmin = dmin, m_dmax = dmax;
        auto rrs = dg::mat::generate_range( dmin, dmax);
        auto lls = dg::mat::generate_range( m_lmin, m_lmax);
        std::vector<double> results( lls.size()*rrs.size());
        if( m_verbose)
        {
            std::cout << "# Extreme EVs are "<<m_lmin<<" "<<m_lmax<<"\n";
            std::cout << "# Extreme d's are "<<m_dmin<<" "<<m_dmax<<"\n";
        }

        std::vector<double> params = {0.5017,0.6122,0.2645,dg::mat::finv_alpha(0.6407)};
        // 2. Levenberg-Marquardt algorithm
        for( unsigned n = 2; n <= m_num_nodes; n++)
        {
            dg::mat::LeastSquaresCauchyError
                 cauchy( 2*n, dg::mat::weights_and_nodes_talbot, func, rrs, lls);
            dg::mat::LeastSquaresCauchyJacobian
                 jac( 2*n, dg::mat::weights_and_nodes_talbot, dg::mat::jacobian_talbot, func, dxlnfunc, rrs, lls);
            // One can play between 1 and 2 here
            cauchy.set_order(1);
            jac.set_order(1);

            unsigned steps = levenberg_marquardt( cauchy, jac, params, results, 1e-4, 1000);
            if( m_verbose && n == m_num_nodes)
            {
                std::cout << "# Num steps in Levenberg Marquardt "<<steps<<"\n";
                cauchy.error( params, results);
                std::cout << "# Cauchy error "<<dg::blas1::dot( results, results)<<" ";
                std::cout << "#  with params "<<params[0]<<" "<<params[1]<<" "<<params[2]<<" "<<params[3]<<"\n";
                std::cout << "# Abs max error "<<dg::blas1::reduce( results, -1e300, thrust::maximum<double>(), dg::ABS<double>())<<"\n";
            }
        }
        dg::mat::LeastSquaresCauchyError
            Icauchy( 2*m_num_nodes, dg::mat::weights_and_nodes_identity, func, rrs, lls);
        dg::mat::LeastSquaresCauchyJacobian
            Ijac( 2*m_num_nodes, dg::mat::weights_and_nodes_identity, dg::mat::jacobian_identity, func, dxlnfunc, rrs, lls);
        Icauchy.set_order(1);
        Ijac.set_order(1);
        auto zkwk = dg::mat::weights_and_nodes_talbot( 2*m_num_nodes, params);
        auto paramsI = dg::mat::weights_and_nodes2params( zkwk);
        unsigned steps = levenberg_marquardt( Icauchy, Ijac, paramsI, results, 1e-6, 1000);
        Icauchy.error( paramsI, results);
        if( m_verbose)
        {
            std::cout << "# Num steps in Levenberg Marquardt Id "<<steps<<"\n";
            std::cout << "# Cauchy I error "<<dg::blas1::dot( results, results)<<"\n";
            std::cout << "# Abs max I error "<<dg::blas1::reduce( results, -1e300, thrust::maximum<double>(), dg::ABS<double>())<<"\n";
        }
        zkwk = dg::mat::weights_and_nodes_identity( 2*m_num_nodes, paramsI);
        m_zk = zkwk.first;
        m_wk = zkwk.second;
    }

    template<class MatrixType>
    void update_extremeEVs( MatrixType&& A)
    {
        dg::mat::UniversalLanczos<ComplexContainer> lanczos( A.weights(), 20);
        auto T = lanczos.tridiag( A, A.weights(), A.weights());
        auto EVs = dg::mat::compute_extreme_EV( T);
        m_lmin = EVs[0], m_lmax = EVs[1];
    }

    unsigned m_num_nodes;
    double m_lmin, m_lmax, m_dmin, m_dmax;
    std::vector<thrust::complex<double>> m_zk, m_wk;
    MultigridCG2d<Geometry, Matrix, ComplexContainer, dg::complex_symmetric> m_multi; // does not remember any solutions
    std::vector<dg::Extrapolation<ComplexContainer, double>> m_previous; // previous solutions for every zk
    ComplexContainer m_z, m_rhs; // complex vectors
    bool m_up2date = false;
    bool m_verbose = false;
};

}//namespace mat
}//namespace dg
