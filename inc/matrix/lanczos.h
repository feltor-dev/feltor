#pragma once

#include "dg/algorithm.h"
#include "tridiaginv.h"
#include "matrixfunction.h"


/**
* @brief Classes for Krylov space approximations of a Matrix-Vector product
*/

namespace dg{
namespace mat{

/** @class hide_lanczos_params_doc
 *
 * @param A A self-adjoint, positive definit matrix
 * @param b The initial vector that starts orthogonalization.
 * @attention If tridiagonalisation is used to determine Eigenvalues, \c b
 * should be initialised with random values to make misconvergence and breakdown due to
 * beta = 0 less likely.
 * @param weights Weights that define the scalar product in which \c A is
 *  self-adjoint and in which the error norm is computed.
 * @param eps relative accuracy of residual
 * @param nrmb_correction the absolute error \c C in units of \c eps to be respected (ignored for "compute_*_EV" error norms)
 * @param error_norm Either "residual" or "universal" or
 * "compute_extreme_EV" or "compute_max_EV". In the latter two cases the
 * iteration terminates if the extreme ( = maximum and minimum) /maximum
 * Eigenvalue(s) of the tridiagonal matrix converge i.e. \f$ | \lambda_{\max,
 * i} - \lambda_{\max, i-1}|< \epsilon \lambda_{\max,i}\f$ (analogous for
 * minimum). In the Eigenvalue cases \c nrmb_correction, \c res_fac and \c q are ignored.
 * In the Eigenvalue case the algorithm trows if breakdown occurs.
 * The error conditions is checked every iteration for the first 10 iterations
 * and then only every 10th iteration.
 * @param res_fac factor \f$ \tau\f$ in the "universal" and "resisdual" \c
 * error_norm that is multiplied to the norm of the residual. Used to account
 * for specific matrix function and operator in the convergence criterium
 * @param q The q-number in the "universal" \c error_norm (ignored otherwise)
 * @sa https://www.jstor.org/stable/2007471?seq=1
 */


/**
* @brief Tridiagonalize \f$A\f$ and approximate \f$f(A)b \approx |b|_W V f(T) e_1\f$
*  via Lanczos algorithm. A is self-adjoint in the weights \f$ W\f$
*
* @ingroup matrixfunctionapproximation
*
* The M-Lanczos method is based on the paper <a
* href="https://doi.org/10.1137/100800634"> Novel Numerical Methods for Solving
* the Time-Space Fractional Diffusion Equation in Two Dimensions</a>  by Q.
* Yang et al, but adopts a more efficient implementation similar to that in the
* PCG method. Further also the conventional Lanczos method can be found there
* and also in text books such as
* <a href="https://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf">Iteratvie
* Methods for Sparse Linear Systems 2nd edition by Yousef Saad</a>.
*
* The Lanczos algorithm's main purpose is to find (approximate)
* Eigenvalues and Eigenvectors of a symmetric definite matrix. It does so by
* computing Eigenvalues of the (symmetric) tridiagonal matrix \f$ T\f$ (the
* "Ritz" or "Rayleigh-Ritz" values) that the Lanczos algorithm produces. The
* extreme values i.e. the maximum and minimum Eigenvalue of \f$ T\f$
* converge fairly quickly to the extreme Eigenvalues of \f$A\f$.
*
* @note We have several stopping criteria.
* The **residual** criterion stops when \f$ \tau ||r_i||_W = \tau ||\vec b||_W \beta_i (T^{-1})_{1m} \leq \epsilon_{rel} ||\vec b||_W + \epsilon_{aps} \f$ where \f$ \tau \f$ is a residual factor accounting for the condition of various matrix functions
*
* @note The **universal** stopping criterion is based on the paper
* <a href="https://doi.org/10.1007/s10444-021-09882-7"> Estimating the error in matrix function approximations</a>  by Q.
* Eshghi, N. and Reichel L.,
* The iteration stops when \f[
||\vec{e}_{f,m}||_W = ||\vec{b}||_W ||\left(\check{T} - f(\bar{T})\right)\vec{e}_1||_2 \leq \epsilon_{rel} ||\vec{b}||_W ||f(T)\vec e_1||_2 + \epsilon_{abs}
\f]
with
\f[
\bar{T} =
   \begin{pmatrix}
   T & \beta_m \vec{e}_m & & \\
    \beta_m \vec{e}_m^T &  \alpha_{m-1}  & \beta_{m-2} &  \\
    & \beta_{m-2} & \alpha_{m-2} & \beta_{m-1-q} \\
    & & \beta_{n-1-q} & \alpha_{n-q}
   \end{pmatrix}
\f]
\f[
   \check{T} =
   \begin{pmatrix}
   f(T) & 0  \\
    0 & 0
   \end{pmatrix}
\f]

* @note The "compute_extreme_EV" and "compute_max_EV" stopping criteria
* are based on tracking the convergence progress of \f$\lambda^T_\min\f$ and
* \f$ \lambda^T_\max\f$ or the tridiagonal matrix \f$ T\f$.
* \f[
* |\lambda^T_{\min, 10(i-1)} - \lambda^T_{\min , 10i}| < 10(\epsilon\lambda_\min  + 10^{-12}\lambda_\max)\\
* |\lambda^T_{\max, 10(i-1)} - \lambda^T_{\max , 10i}| < 10\epsilon\lambda_\max
* \f]
* We test convergence only every 10th iteration for various reasons
* - The Eigenvalue decomposition of \f$ T\f$ may become performance relevant
*   for very large \f$ i\f$. Testing only every 10th iterations amortises the cost
* - Lanczos is at risk of misconvergence which means the algorithm may
*   "converge" to a \f$\lambda^T\f$ that is not an extreme Eigenvalue because
*   for several iterations no larger/smaller Eigenvalue appears (this happens
*   not infrequently). Testing only every 10th iteration decreases this risk
*   such that we do not see it in practise (but it does not vanish entirely)
*   .
*   Also here note that theoretically the minimum and maximum Eigenvalue should converge
*   at the same rate, however the error constant for the minimum Eigenvalue is much worse
*   (by the condition number \f$\kappa\f$ of the matrix!). This is because all Eigenvalues that are found
*   by \f$ T\f$ lie within the spectrum of \f$A\f$, i.e.
*   \f[
*   \lambda^A_\min < \lambda^T_\min < \lambda^T_\max < \lambda^T_\max
*   \\
*   \frac{\lambda^A_\max - \lambda^T_\max }{\lambda^A_\max}  <  1
*   \\
*   \frac{\lambda^T_\min - \lambda^A_\min}{\lambda^A_\min}  = \kappa\frac{\lambda^T_\min - \lambda^A_\min}{\lambda^A_\max} <  \kappa
*   \f]
*
*
* @note THIS METHOD DOES NOT WORK (but is still instructive:
* <a href="https://www.jstor.org/stable/2007471?seq=1">On estimating the largest Eigenvalue with the Lanczos algorithm</a>  by B. N. Parlett, H. Simon and L. M. Stringer
* The proposed stopping criterion consists of checking if the
* found (approximate) Eigenvector is indeed an Eigenvector of \f$A\f$.
* Suppose we find
* \f[
* T_m s = \lambda s
* \f]
* Then, we accept the corresponding vector \f$ y = V_m s\f$ as an Eigenvector of \f$ A\f$
* if
* \f[
* || Ay - \lambda y ||_W  \leq \epsilon || y||_W
* \f]
* Since the \f$ V_m\f$ are normalised  we esimate \f$ || y||_W > 0.9\f$
* and we have \f$ A y= A V_m s = V_m T_m s + \beta_m \vec v_{m+1} \vec e_m^T s = \lambda V_m s + \beta_m \vec v_{m+1} s_m\f$ where \f$ s_m\f$ is the last component of \f$s\f$.
* Thus, we have
* \f$
* A y - \lambda y = \beta_m \vec v_{m+1} s_m
* \f$
* and therefore
* \f[
* \beta_m s_m  \leq 0.9 \epsilon
* \f]
* Unfortunately in practise \f$ \beta_m s_m \f$ is rather large and the convergence is extremely slow.
* Even if the Eigenvalues converge rather quickly, the Eigenvectors do not!
*
*
* The common Lanczos method (and M-Lanczos) method are prone to loss of
* orthogonality for finite precision.
* @copydoc hide_ContainerType
*/
template< class ContainerType >
class UniversalLanczos
{
  public:
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    UniversalLanczos(){}
    /**
     * @brief Allocate memory for the method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    UniversalLanczos( const ContainerType& copyable, unsigned max_iterations)
    {
        m_v = m_vp = m_vm = copyable;
        m_max_iter = max_iterations;
        m_iter = max_iterations;
        //sub matrix and vector
        set_iter( max_iterations);
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = UniversalLanczos( std::forward<Params>( ps)...);
    }

    ///@brief Set the maximum number of iterations
    ///@param new_max New maximum number
    void set_max( unsigned new_max) {
        m_max_iter = new_max;
        set_iter( new_max);
    }

    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_max() const {return m_max_iter;}

    ///@brief Set or unset debugging output during iterations
    ///@param verbose If true, additional output will be written to \c std::cout during solution
    void set_verbose( bool verbose){ m_verbose = verbose;}

    ///@brief Norm of \c b from last call to \c operator()
    ///@return bnorm
    double get_bnorm() const{return m_bnorm;}

    ///@brief Get the number of iterations in the last call to \c tridiag or \c solve
    /// (same as T.num_rows)
    ///@return the number of iterations in the last call to \c tridiag or \c solve
    unsigned get_iter() const {return m_iter;}

    /**
    * @brief \f$ x = f(A)b \approx ||b||_W V f(T) e_1 \f$
    *  via Lanczos and matrix function computation.
    *  A is self-adjoint in the weights \f$ W\f$.
    *
    * Tridiagonalize \f$A\f$ using Lanczos algorithm with a residual or
    * universal stopping criterion
    *
     * @param x output vector
     * @param f the matrix function that is called like \c yH = f( T) where T
     * is the tridiagonal matrix and returns the result of \f$ f(T)\vec e_1\f$
     *  (for example \c dg::mat::make_FuncEigen_Te1( dg::SQRT<double>()) )
     * @sa matrixfunctionapproximation
     * @copydoc hide_lanczos_params_doc
     *
     * @return number of iterations of M-Lanczos routine
    */
    template < class MatrixType, class ContainerType0, class ContainerType1,
             class ContainerType2, class FuncTe1>
    unsigned solve(ContainerType0& x, FuncTe1 f,
            MatrixType&& A, const ContainerType1& b,
            const ContainerType2& weights, double eps,
            double nrmb_correction = 1.,
            std::string error_norm = "universal",
            double res_fac = 1.,
            unsigned q = 1 )
    {
        tridiag( f, std::forward<MatrixType>(A), b, weights, eps,
                nrmb_correction, error_norm, res_fac, q);
        if( "residual" == error_norm)
            m_yH = f( m_TH);
        //Compute x = |b|_M V f(T) e1
        normMbVy(std::forward<MatrixType>(A), m_TH, m_yH, x, b,
                m_bnorm);
        return m_iter;
    }

    /**
     * @brief Tridiagonalization of A using Lanczos method with \f$ f(x) = x^{-1} \f$
     *
     * @note Just calls the general \c tridiag method with the function \f$
     * f(x) = x^{-1}\f$. Useful if one wants to compute the extreme Eigenvalues
     * of \f$A\f$
     * @note The maximum Eigenvalue converges much faster than the minimum Eigenvalue.
     *
     * @code{.cpp}
     *  dg::mat::UniversalLanczos lanczos( A.weights(), 2000);
     *  auto T = lanczos.tridiag( A, A.weights(), A.weights());
     *  auto EV = dg::mat::compute_extreme_EV( T);
     *  // EV[0] is the minimum, EV[1] the maximum Eigenvalue
     * @endcode
     * @note The Eigenvalues of \f$ T\f$ are called the **Rayleigh-Ritz values**
     * @copydoc hide_lanczos_params_doc
     *
     * @return returns the tridiagonal matrix T. Note that \f$ T = (MV)^T A V \f$.
     *  The number of iterations is given by \c T.num_rows
      */
    template< class MatrixType, class ContainerType0, class ContainerType1>
    const dg::TriDiagonal<thrust::host_vector<double>>& tridiag( MatrixType&& A, const ContainerType0& b,
            const ContainerType1& weights, double eps = 1e-4,
            double nrmb_correction = 1.,
            std::string error_norm = "compute_extreme_EV",
            double res_fac = 1.,
            unsigned q = 1 )
    {
        auto op = make_Linear_Te1( -1);
        tridiag( op, std::forward<MatrixType>(A), b, weights, eps,
                nrmb_correction, error_norm, res_fac, q);
        return m_TH;
    }

    /** @brief compute \f$ x = |b|_W V y \f$ from a given tridiagonal matrix T
     * and in-place re-computation of V
     *
     * We avoid explicit storage of the large matrix V
     * @param A A self-adjoint positive definit matrix
     * @param T Tridiagonal matrix
     * @param y a (host) vector e.g y= T e_1 or y= f(T) e_1, must have size of
     *  \c T.num_rows
     * @param x The result vector (output)
     * @param b Contains the initial value of lanczos method
     * @param bnorm the norm of b in weights, \c get_bnorm()
     */
    template< class MatrixType, class ContainerType0,
        class ContainerType1,class ContainerType2>
    void normMbVy( MatrixType&& A,
            const dg::TriDiagonal<thrust::host_vector<double>>& T,
            const ContainerType0& y,
            ContainerType1& x,
            const ContainerType2& b, double bnorm)
    {
        dg::blas1::copy(0., x);
        if( 0 == bnorm )
        {
            return;
        }
        dg::blas1::axpby(1./bnorm, b, 0.0, m_v); //m_v[1] = b/||b||
        dg::blas1::copy(0., m_vm);
        // check if (potentially) all higher elements in y are zero
        unsigned less_iter = 0;
        for( unsigned i=0; i<y.size(); i++)
            if( y[i] != 0)
                less_iter = i+1;
        dg::blas1::axpby( y[0]*bnorm, m_v, 1., x); //Compute b= |b| V y

        for ( unsigned i=0; i<less_iter-1; i++)
        {
            dg::blas2::symv( std::forward<MatrixType>(A), m_v, m_vp);
            dg::blas1::axpbypgz(
                    -T.M[i]/T.P[i], m_vm,
                    -T.O[i]/T.P[i], m_v,
                               1.0/T.P[i], m_vp);
            dg::blas1::axpby( y[i+1]*bnorm, m_vp, 1., x); //Compute b= |b| V y
            m_vm.swap( m_v);
            m_v.swap( m_vp);
        }
    }

    /**
     * @brief Tridiagonalization of A using Lanczos method.
     *
     * Tridiagonalize \f$A\f$ using Lanczos algorithm with a residual or
     * universal stopping criterion on the function \f$ f(x) \f$
     * @param f Unary function (ignored if \c error_norm is not "universal")
     * @copydoc hide_lanczos_params_doc
     *
     * @return returns the tridiagonal matrix T. Note that \f$ T = (MV)^T A V \f$.
     *  The number of iterations is given by \c T.num_rows
      */
    template < class UnaryOp, class MatrixType,
             class ContainerType1, class ContainerType2>
    const dg::TriDiagonal<thrust::host_vector<double>>& tridiag(UnaryOp f,
            MatrixType&& A, const ContainerType1& b,
            const ContainerType2& weights, double eps,
            double nrmb_correction,
            std::string error_norm = "residual",
            double res_fac = 1.,
            unsigned q = 1 )
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        m_bnorm = nrm( weights, b);
        if( m_verbose)
        {
            DG_RANK0 std::cout << "# Norm of b  "<<m_bnorm <<"\n";
            if ( "compute_extreme_EV" == error_norm || "compute_max_EV" == error_norm)
            {
                DG_RANK0 std::cout << "# Rel EV errors: \n";
            }
            else
            {
                DG_RANK0 std::cout << "# Res factor "<<res_fac <<"\n";
                DG_RANK0 std::cout << "# Residual errors: \n";
            }
        }
        if( m_bnorm == 0)
        {
            set_iter(1);
            return m_TH;
        }
        double residual = 1e300;
        dg::blas1::axpby(1./m_bnorm, b, 0.0, m_v); //m_v[1] = x/||x||
        double betaip = 0.;
        double alphai = 0.; // For Hermitian matrices all dot products in CG are real
        double lmin = -1e300, lmax = 1e300;
        for( unsigned i=0; i<m_max_iter; i++)
        {
            m_TH.M[i] =  betaip; // -1 diagonal
            dg::blas2::symv(std::forward<MatrixType>(A), m_v, m_vp);
            dg::blas1::axpby(-betaip, m_vm, 1.0, m_vp);  // only - if i>0, therefore no if (i>0)
            alphai  = dot(m_vp, weights, m_v);
            m_TH.O[i] = alphai;
            dg::blas1::axpby(-alphai, m_v, 1.0, m_vp);
            betaip = nrm( weights, m_vp);
            if (fabs( betaip) <= 1e-12) // what this means is that x = A^{-1}b is exactly in the Krylov subspace
            {
                set_iter(i+1);
                if ( "compute_extreme_EV" == error_norm || "compute_max_EV" == error_norm)
                {
                    // If we did continue iterating after betaip = 0 the results are rather random
                    throw dg::Error( dg::Message(_ping_) << "betaip = 0 found in tridiagonalisation \n");
                }
                else if( m_verbose)
                    DG_RANK0 std::cout << "beta["<<i+1 <<"]=0 encountered\n";
                break;
            }
            m_TH.P[i] = betaip;  // +1 diagonal

            double xnorm = 0.;
            if( "residual" == error_norm)
            {
                residual = compute_residual_error( m_TH, i)*m_bnorm;
                xnorm = m_bnorm;
            }
            else if ( "compute_extreme_EV" == error_norm || "compute_max_EV" == error_norm)
            {
                if(i%10 == 0)
                {
                    // We use P as "subdiagonal" because it is symmetric and the first element must be on 0 index
                    // (last is ignored)
                    thrust::host_vector<double> evals( i+1), subdiagonal( i+1), Z, work;
                    for( unsigned u=0; u<i+1; u++)
                    {
                        evals[u] = m_TH.O[u];
                        subdiagonal[u] = m_TH.P[u];
                    }
                    lapack::stev('N', evals,  subdiagonal, Z, work);
                    if( m_verbose)
                        DG_RANK0 std::cout << "# lmin = " << evals[0] << " lmax = "<<evals[i]<<"\tat i = " << i << "\n";// << betaip * evecs(i,i)<<" " <<betaip * evecs(0,i)<<"\n";
                    //if( fabs(betaip * evecs(i,i)) < 0.9*eps) // CONVERGES SUPER SLOW
                    if( fabs(lmax - evals[i]) < 10*eps*evals[i])
                    {
                        if( ("compute_extreme_EV" == error_norm &&
                            //fabs( betaip * evecs(0,i)) < eps) // CONVERGES EVEN SLOWER
                            fabs( lmin - evals[0]) < 10 * (eps*fabs(lmin) + 1e-12*lmax))
                            || "compute_max_EV" == error_norm)
                        {
                            set_iter(i+1);
                            break;
                        }
                    }
                    lmin = evals[0], lmax = evals[i];
                }
            }
            else
            {
                if( i>=q &&(  (i<=10) || (i>10 && i%10 == 0) ))
                {
                    residual = compute_universal_error( m_TH, i, q, f,
                            m_yH)*m_bnorm;
                    xnorm = dg::fast_l2norm( m_yH)*m_bnorm;
                }
                else
                {
                    residual = 1e10;
                    xnorm = m_bnorm;
                }
            }
            if( m_verbose && "compute_extreme_EV" != error_norm && "compute_max_EV" != error_norm)
                DG_RANK0 std::cout << "# ||r||_W = " << residual << "\tat i = " << i << "\n";
            if (res_fac*residual< eps*(xnorm + nrmb_correction) )
            {
                set_iter(i+1);
                break;
            }
            dg::blas1::scal(m_vp, 1./betaip);
            m_vm.swap(m_v);
            m_v.swap( m_vp);
            set_iter( m_max_iter);
        }
        return m_TH;
    }
    private:
    // !! Calling this "norm" would shadow std::norm in NORM
    template<class ContainerType1, class ContainerType2>
    auto nrm( const ContainerType1& w, const ContainerType2& x)
    {
        using value_type_x = dg::get_value_type<ContainerType2>;
        constexpr bool is_complex = dg::is_scalar_v<value_type_x, dg::ComplexTag>;
        if constexpr (is_complex)
            return sqrt( blas1::vdot( dg::detail::NORM(), w, x));
        else
            return sqrt( blas2::dot( w, x));
    }
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    auto dot( const ContainerType0& x0, const ContainerType1& w, const ContainerType2& x1)
    {
        using value_type_x1 = dg::get_value_type<ContainerType2>;
        constexpr bool is_complex = dg::is_scalar_v<value_type_x1, dg::ComplexTag>;
        // There is no complex_symmetric in Lanczos...
        if constexpr (not is_complex ) // or complex_mode == dg::complex_symmetric)
            return blas2::dot( x0, w, x1); // this returns value_type
        else // For Hermitian matrices all dot products in CG are real
            return blas1::vdot( dg::detail::DOT(), x0, w, x1); // this returns floating point
    }
    double compute_residual_error( const dg::TriDiagonal<thrust::host_vector<double>>& TH, unsigned iter)
    {
        double T1 = compute_Tinv_m1( TH, iter+1);
        return TH.P[iter]*fabs(T1); //Tinv_i1
    }
    template<class UnaryOp>
    double compute_universal_error( const dg::TriDiagonal<thrust::host_vector<double>>& TH, unsigned iter,
            unsigned q, UnaryOp f, HVec& yH)
    {
        unsigned new_iter = iter + 1 + q;
        set_iter( iter+1);
        dg::TriDiagonal<thrust::host_vector<double>> THtilde( new_iter);
        for( unsigned u=0; u<iter+1; u++)
        {
            THtilde.M[u] = TH.M[u];
            THtilde.O[u] = TH.O[u];
            THtilde.P[u] = TH.P[u];
        }
        for( unsigned u=1; u<=q; u++)
        {
            THtilde.M[ iter+u] = u==1 ? TH.P[iter] :
                TH.O[iter+1-u];
            THtilde.O[ iter+u] = TH.O[ iter-u];
            THtilde.P[ iter+u] = TH.M[ iter-u];
        }
        yH = f( TH);
        HVec yHtilde = f( THtilde);
        for( unsigned u=0; u<yH.size(); u++)
            yHtilde[u] -= yH[u];
        double norm = dg::fast_l2norm( yHtilde);
        return norm;
    }

    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        m_TH.resize(new_iter);
        m_iter = new_iter;
    }
    ContainerType  m_v, m_vp, m_vm;
    dg::TriDiagonal<thrust::host_vector<double>> m_TH;
    HVec m_yH;
    unsigned m_iter, m_max_iter;
    bool m_verbose = false;
    double m_bnorm = 0.;
};


} //namespace mat
} //namespace dg

