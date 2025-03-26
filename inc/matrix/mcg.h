#pragma once

#include "dg/algorithm.h"
#include "tridiaginv.h"
#include "matrixfunction.h"

/**
* @brief Classes for Krylov space approximations of a Matrix-Vector product
*/

namespace dg{
namespace mat{

/*!
* @brief EXPERIMENTAL Tridiagonalize \f$A\f$ and approximate
* \f$f(A)b \approx R f(\tilde T) e_1\f$
*  via CG algorithm. A is self-adjoint in the weights \f$ W\f$
 *
 * @attention This class has the exact same result as the corresponding
 *  UniversalLanczos class. Its use is for mere experimental purposes.
 *
 * @ingroup matrixfunctionapproximation
 *
 * This class is based on the approach of the paper <a href="https://doi.org/10.1016/0377-0427(87)90020-3)" > An iterative solution method for solving f(A)x = b, using Krylov subspace information obtained for the symmetric positive definite matrix A</a> by H. A. Van Der Vorst
 *
 * @note The approximation relies on Projection
 * \f$x = f(A) b  \approx  R  f(\tilde T) e_1\f$,
 * where \f$\tilde T\f$ and \f$R\f$ are the tridiagonal and orthogonal
 * matrix of the CG solve respectively and \f$e_1\f$ is the normalized unit
 * vector.
 */
template< class ContainerType>
class MCG
{
  public:
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    MCG(){}
    /**
     * @brief Allocate memory for the mcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    MCG( const ContainerType& copyable, unsigned max_iterations)
    {
        m_ap = m_p = m_r = copyable;
        m_max_iter = max_iterations;
        set_iter( max_iterations);
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

    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = MCG( std::forward<Params>( ps)...);
    }
    ///@brief Get the number of iterations in the last call to \c operator()
    /// (size of T)
    ///@return the number of iterations in the last call to \c operator()
    unsigned get_iter() const {return m_iter;}
    /**
     * @brief Compute x = R y
     * @param A A self-adjoint, positive definit matrix
     * @param T T non-symmetric tridiagonal Matrix from MCG tridiagonalization
     * @param y (host) vector with v.size() = iter. Must have size of \c T.num_rows.
     *  Typically \f$ T^{(-1)} e_1 \f$ or \f$ f(T^{(-1)}) e_1 \f$
     * @param x Contains the matrix approximation \f$x = Ry \f$ (output)
     * @param b The right hand side vector.
     */
    template< class MatrixType, class DiaMatrixType, class ContainerType0,
        class ContainerType1, class ContainerType2>
    void Ry( MatrixType&& A, const DiaMatrixType& T,
            const ContainerType0& y, ContainerType1& x,
            const ContainerType2& b)
    {
        dg::blas1::copy(0., x);

        dg::blas1::copy( b, m_r);

        dg::blas1::copy( m_r, m_p );

        dg::blas1::axpby( y[0], m_r, 1., x); //Compute x= R y
        for ( unsigned i=0; i<y.size()-1; i++)
        {
            dg::blas2::symv( std::forward<MatrixType>(A), m_p, m_ap);
            value_type alphainv = i==0 ? T.O[i] :
                T.O[i] + T.P[i-1];
            value_type beta = -T.P[i]/alphainv;
            dg::blas1::axpby( -1./alphainv, m_ap, 1., m_r);
            dg::blas1::axpby(1., m_r, beta, m_p );
            dg::blas1::axpby( y[i+1], m_r, 1., x); //Compute x= R y
        }
    }
    /**
     * @brief Tridiagonalize the system \f$ A\vec b\f$ using CG
     *
     * @param A A self-adjoint, positive definit matrix
     * @param b The right hand side vector.
     * @param weights Weights that define the scalar product in which \c A is
     *  self-adjoint and in which the error norm is computed.
     * @param eps relative accuracy of residual
     * @param nrmb_correction the absolute error \c C in units of \c eps to be
     *  respected
     * @param res_fac factor that is multiplied to the norm of the residual.
     *  Used to account for specific matrix function and operator in the
     *  convergence criterium
     *
     * @return The tridiagonal matrix \f$ T\f$ with size \c get_iter()
     * @note So far only ordinary convergence criterium of CG method, in
     * particular for \f$ A x  = b \f$. If used for matrix function
     * computation, \f$ f(A) x  = b \f$, the parameter eps should be
     * multiplied with appropriate factors to account for the different
     * convergence criteria.
     * The Matrix R and T of the tridiagonalization are
     * further used for computing matrix functions. The x vector must be
     * initialized with 0 if used for tridiagonalization.
      */
    template< class MatrixType, class ContainerType0, class ContainerType1>
    const dg::TriDiagonal<thrust::host_vector<value_type>>& operator()(
            MatrixType&& A, const ContainerType0& b,
            const ContainerType1& weights, value_type eps = 1e-12,
            value_type nrmb_correction = 1., value_type res_fac = 1.)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        value_type nrmzr_old = dg::blas2::dot( b, weights, b);
        value_type nrmb = sqrt(nrmzr_old);
        m_bnorm = nrmb;
        if( m_verbose)
        {
            DG_RANK0 std::cout << "# Norm of b  "<<nrmb <<"\n";
            DG_RANK0 std::cout << "# Res factor "<<res_fac <<"\n";
            DG_RANK0 std::cout << "# Residual errors: \n";
        }
        if( nrmb == 0)
        {
            set_iter(1);
            return m_TH;
        }
        dg::blas1::copy( b, m_r);
        dg::blas1::copy( m_r, m_p );

        value_type alpha = 0, beta = 0, nrmzr_new = 0, alpha_old = 0., beta_old = 0.;
        for( unsigned i=0; i<m_max_iter; i++)
        {
            alpha_old = alpha, beta_old = beta;
            dg::blas2::symv( std::forward<MatrixType>(A), m_p, m_ap);
            alpha = nrmzr_old /dg::blas2::dot( m_p, weights, m_ap);
            dg::blas1::axpby( -alpha, m_ap, 1., m_r);
            nrmzr_new = dg::blas2::dot( m_r, weights, m_r);
            beta = nrmzr_new/nrmzr_old;
            if(m_verbose)
            {
                DG_RANK0 std::cout << "# ||r||_W = " << sqrt(nrmzr_new) << "\tat i = " << i << "\n";
            }
            if( i == 0)
            {
                m_TH.M[i] = 0.;
                m_TH.O[i] = 1./alpha;
                m_TH.P[i] = -beta/alpha;
            }
            else
            {
                m_TH.M[i] = -1./alpha_old;
                m_TH.O[i] =  1./alpha + beta_old/alpha_old;
                m_TH.P[i] = -beta/alpha;
            }
            if( res_fac*sqrt( nrmzr_new)
                    < eps*(nrmb + nrmb_correction))
            {
                set_iter(i+1);
                break;
            }
            dg::blas1::axpby(1., m_r, beta, m_p );
            nrmzr_old=nrmzr_new;
        }

        return m_TH;
    }
    /**
     * @brief Return the vector \f$ \vec e_1\f$ with size \c get_iter()
     *
     * @return e_1
     */
    thrust::host_vector<value_type> make_e1( ) {
        thrust::host_vector<value_type> e1H(m_iter, 0.);
        e1H[0] = 1.;
        return e1H;
    }
  private:
    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        // The alignment (which is the pitch of the underlying values)
        // of m_max_iter preserves the existing elements
        m_TH.resize(new_iter),
        m_iter = new_iter;
    }
    ContainerType m_r, m_ap, m_p;
    unsigned m_max_iter, m_iter;
    dg::TriDiagonal<thrust::host_vector<value_type>> m_TH;
    bool m_verbose = false;
    value_type m_bnorm = 0.;
};

/*!
 * @brief EXPERIMENTAL Shortcut for \f$x \approx f(A) b  \f$ solve via exploiting first a Krylov projection achieved by the M-CG method and a matrix function computation via Eigen-decomposition
 *
 * @attention This class has the exact same result as the corresponding
 *  UniversalLanczos class. Its use is for mere experimental purposes.
 * @ingroup matrixfunctionapproximation
 * A is self-adjoint in the weights \f$ W\f$
 *
 * The approximation relies on Projection
 * \f$x = f(A) b  \approx  R f(\tilde T)^{-1} e_1\f$,
 * where \f$\tilde T\f$ and \f$R\f$ are the tridiagonal and orthogonal matrix
 * of the M-CG solve and \f$e_1\f$ is the normalized unit vector. The vector
 * \f$f(\tilde T) e_1\f$ is computed via the Eigen decomposition and similarity
 * transform of \f$ \tilde T\f$.
 * @note Since MCG and Lanczos are equivalent the result of this class is the
 *  same within round-off errors as a Lanczos solve with the "residual" error
 *  norm
 */
template<class Container>
class MCGFuncEigen
{
  public:
    using value_type = dg::get_value_type<Container>;
    using HVec = dg::HVec;
    ///@brief Allocate nothing, Call \c construct method before usage
    MCGFuncEigen(){}
    /**
     * @brief Construct MCGFuncEigen
     *
     * @param copyable a copyable container
     * @param max_iterations Max iterations of Lanczos method
     */
    MCGFuncEigen( const Container& copyable, unsigned max_iterations)
    {
        m_e1H.assign(max_iterations, 0.);
        m_e1H[0] = 1.;
        m_yH.assign(max_iterations, 1.);
        m_mcg.construct(copyable, max_iterations);
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = MCGFuncEigen( std::forward<Params>( ps)...);
    }
    /**
     * @brief Compute \f$x \approx f(A)*b \f$ via M-CG method and Eigen decomposition
     *
     * @param b input vector
     * @param x output vector
     * @param f the matrix function (e.g. dg::SQRT<double> or dg::EXP<double>)
     * @param A self-adjoint and semi-positive definit matrix
     * @param weights weights
     * @param eps relative accuracy of M-CG method
     * @param nrmb_correction the absolute error \c C in units of \c eps to be
     *  respected
     * @param res_fac residual factor for stopping criterium of M-CG method
     *
     * @return number of iterations of M-CG routine
     */
    template < class MatrixType, class ContainerType0, class ContainerType1 ,
             class ContainerType2, class UnaryOp>
    unsigned operator()(ContainerType0& x, UnaryOp f,
            MatrixType&& A, const ContainerType1& b,
            const ContainerType2& weights, value_type eps,
            value_type nrmb_correction, value_type res_fac )
    {
        if( sqrt(dg::blas2::dot(b, weights, b)) == 0)
        {
            dg::blas1::copy( b, x);
            return 0;
        }
        //Compute x (with initODE with gemres replacing cg invert)
        m_TH = m_mcg(std::forward<MatrixType>(A), b, weights, eps,
                nrmb_correction, res_fac);
        unsigned iter = m_mcg.get_iter();
        //resize
        m_alpha.resize(iter);
        m_delta.resize(iter,1.);
        m_beta.resize(iter-1);
        m_EHt.resize(iter);
        m_e1H.resize(iter, 0.);
        m_e1H[0] = 1.;
        m_yH.resize(iter);
        m_work.resize( 2*iter-2);
        //fill diagonal entries of similarity transformed T matrix (now symmetric)
        for(unsigned i = 0; i<iter; i++)
        {
            m_alpha[i] = m_TH.O[i];
            if (i<iter-1) {
                if      (m_TH.P[i] > 0.) m_beta[i] =  sqrt(m_TH.P[i]*m_TH.M[i+1]); // sqrt(b_i * c_i)
                else if (m_TH.P[i] < 0.) m_beta[i] = -sqrt(m_TH.P[i]*m_TH.M[i+1]); //-sqrt(b_i * c_i)
                else m_beta[i] = 0.;
            }
            if (i>0) m_delta[i] = m_delta[i-1]*sqrt(m_TH.M[i]/m_TH.P[i-1]);
        }
        //Compute Eigendecomposition
        lapack::stev(LAPACK_COL_MAJOR, 'V', m_alpha, m_beta, m_EHt.data(), m_work);
        m_EH = m_EHt.transpose();
        //Compute f(T) e1 = D E f(Lambda) E^t D^{-1} e1
        dg::blas1::pointwiseDivide(m_e1H, m_delta, m_e1H);
        dg::blas2::symv(m_EHt, m_e1H, m_yH);
        dg::blas1::transform(m_alpha, m_e1H, [f] (double x){
            try{
                return f(x);
            }
            catch(boost::exception& e) //catch boost overflow error
            {
                return 0.;
            }
        });
        dg::blas1::pointwiseDot(m_e1H, m_yH, m_e1H);
        dg::blas2::symv(m_EH, m_e1H, m_yH);
        dg::blas1::pointwiseDot(m_yH, m_delta, m_yH);
        //Compute x = R f(T) e_1
        m_mcg.Ry(std::forward<MatrixType>(A), m_TH, m_yH, x, b);

        return iter;
    }
  private:
    HVec m_e1H, m_yH;
    dg::TriDiagonal<thrust::host_vector<value_type>> m_TH;
    dg::SquareMatrix<value_type> m_EH, m_EHt;
    thrust::host_vector<value_type> m_alpha, m_beta, m_delta, m_work;
    dg::mat::MCG< Container > m_mcg;

};
} //namespace mat
} //namespace dg

