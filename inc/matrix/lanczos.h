#pragma once

#include "dg/algorithm.h"
#include "tridiaginv.h"
#include "matrixfunction.h"


/**
* @brief Classes for Krylov space approximations of a Matrix-Vector product
*/

namespace dg{
namespace mat{

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
* @note We have two stopping criteria.
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
*
* The common Lanczos method (and M-Lanczos) method are prone to loss of
* orthogonality for finite precision. Here, only the basic Paige fix is used.
* Thus the iterations should be kept as small as possible. Could be fixed via
* full, partial or selective reorthogonalization strategies, but so far no
* problems occured due to this.
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
    value_type get_bnorm() const{return m_bnorm;}

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
     * @param A self-adjoint and semi-positive definit matrix
     * @param b input vector
     * @param weights weights in which A is self-adjoint
     * @param eps relative accuracy of M-Lanczos method
     * @param nrmb_correction the absolute error in units of \c eps to be
     *  respected
     * @param error_norm Either "residual" or "universal"
     * @param res_fac factor \f$ \tau\f$ that is multiplied to the norm of the
     *  residual. Used to account for specific matrix function and operator in
     *  the convergence criterium
     * @param q The q-number in the "universal stopping criterion
     *
     * @return number of iterations of M-Lanczos routine
    */
    template < class MatrixType, class ContainerType0, class ContainerType1,
             class ContainerType2, class FuncTe1>
    unsigned solve(ContainerType0& x, FuncTe1 f,
            MatrixType&& A, const ContainerType1& b,
            const ContainerType2& weights, value_type eps,
            value_type nrmb_correction = 1.,
            std::string error_norm = "universal",
            value_type res_fac = 1.,
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
     * @brief Tridiagonalization of A using Lanczos method with \f$ f(x) = x^{-1}\f$.
     *
     * @note Just calls the general \c tridiag method with the function \f$ f(x) = x^{-1}\f$. Useful if one wants to compute the extreme Eigenvalues of \f$A\f$
 * @code{.cpp}
 *  dg::mat::UniversalLanczos lanczos( A.weights(), 20);
 *  auto T = lanczos.tridiag( A, A.weights(), A.weights());
 *  auto EV = dg::mat::compute_extreme_EV( T);
 *  // EV[0] is the minimum, EV[1] the maximum Eigenvalue
 * @endcode
      */
    template< class MatrixType, class ContainerType0, class ContainerType1>
    const dg::TriDiagonal<thrust::host_vector<value_type>>& tridiag( MatrixType&& A, const ContainerType0& b,
            const ContainerType1& weights, value_type eps = 1e-12,
            value_type nrmb_correction = 1.,
            std::string error_norm = "universal",
            value_type res_fac = 1.,
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
            const dg::TriDiagonal<thrust::host_vector<value_type>>& T,
            const ContainerType0& y,
            ContainerType1& x,
            const ContainerType2& b, value_type bnorm)
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
     * @param f Unary function
     * @param A A self-adjoint, positive definit matrix
     * @param b The initial vector that starts orthogonalization
     * @param weights Weights that define the scalar product in which \c A is
     *  self-adjoint and in which the error norm is computed.
     * @param eps relative accuracy of residual
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     * @param error_norm Either "residual" or "universal"
     * @param res_fac factor \f$ \tau\f$ that is multiplied to the norm of the
     *  residual. Used to account for specific matrix function and operator in
     *  the convergence criterium
     * @param q The q-number in the "universal stopping criterion
     *
     * @return returns the tridiagonal matrix T. Note that \f$ T = (MV)^T A V \f$.
     *  The number of iterations is given by \c T.num_rows
      */
    template < class UnaryOp, class MatrixType,
             class ContainerType1, class ContainerType2>
    const dg::TriDiagonal<thrust::host_vector<value_type>>& tridiag(UnaryOp f,
            MatrixType&& A, const ContainerType1& b,
            const ContainerType2& weights, value_type eps,
            value_type nrmb_correction,
            std::string error_norm = "residual",
            value_type res_fac = 1.,
            unsigned q = 1 )
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        m_bnorm = sqrt(dg::blas2::dot(b, weights, b));
        if( m_verbose)
        {
            DG_RANK0 std::cout << "# Norm of b  "<<m_bnorm <<"\n";
            DG_RANK0 std::cout << "# Res factor "<<res_fac <<"\n";
            DG_RANK0 std::cout << "# Residual errors: \n";
        }
        if( m_bnorm == 0)
        {
            set_iter(1);
            return m_TH;
        }
        value_type residual;
        dg::blas1::axpby(1./m_bnorm, b, 0.0, m_v); //m_v[1] = x/||x||
        value_type betaip = 0.;
        value_type alphai = 0.;
        for( unsigned i=0; i<m_max_iter; i++)
        {
            m_TH.M[i] =  betaip; // -1 diagonal
            dg::blas2::symv(std::forward<MatrixType>(A), m_v, m_vp);
            dg::blas1::axpby(-betaip, m_vm, 1.0, m_vp);  // only - if i>0, therefore no if (i>0)
            alphai  = dg::blas2::dot(m_vp, weights, m_v);
            m_TH.O[i] = alphai;
            dg::blas1::axpby(-alphai, m_v, 1.0, m_vp);
            betaip = sqrt(dg::blas2::dot(m_vp, weights, m_vp));
            if (betaip == 0)
            {
                if( m_verbose)
                    DG_RANK0 std::cout << "beta["<<i+1 <<"]=0 encountered\n";
                set_iter(i+1);
                break;
            }
            m_TH.P[i] = betaip;  // +1 diagonal

            value_type xnorm = 0.;
            if( "residual" == error_norm)
            {
                residual = compute_residual_error( m_TH, i)*m_bnorm;
                xnorm = m_bnorm;
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
            if( m_verbose)
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
    value_type compute_residual_error( const dg::TriDiagonal<thrust::host_vector<value_type>>& TH, unsigned iter)
    {
        value_type T1 = compute_Tinv_m1( TH, iter+1);
        return TH.P[iter]*fabs(T1); //Tinv_i1
    }
    template<class UnaryOp>
    value_type compute_universal_error( const dg::TriDiagonal<thrust::host_vector<value_type>>& TH, unsigned iter,
            unsigned q, UnaryOp f, HVec& yH)
    {
        unsigned new_iter = iter + 1 + q;
        set_iter( iter+1);
        dg::TriDiagonal<thrust::host_vector<value_type>> THtilde( new_iter);
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
        value_type norm = dg::fast_l2norm( yHtilde);
        return norm;
    }

    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        m_TH.resize(new_iter);
        m_iter = new_iter;
    }
    ContainerType  m_v, m_vp, m_vm;
    dg::TriDiagonal<thrust::host_vector<value_type>> m_TH;
    HVec m_yH;
    unsigned m_iter, m_max_iter;
    bool m_verbose = false;
    value_type m_bnorm = 0.;
};


} //namespace mat
} //namespace dg

