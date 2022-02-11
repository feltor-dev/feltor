#pragma once
#include <cusp/dia_matrix.h>
#include <cusp/coo_matrix.h>

//#include <cusp/print.h>
#include "dg/algorithm.h"
#include "tridiaginv.h"


/**
* @brief Classes for Krylov space approximations of a Matrix-Vector product
*/

namespace dg{
namespace mat{

/**
* @brief Tridiagonalize \f$A\f$ and approximate \f$f(A)b \approx |b|_W V f(T) e_1\f$
*. A is self-adjoint in the weights \f$ W\f$
*
* @ingroup matrixapproximation
*
* The M-Lanczos method is based on the paper <a href="https://doi.org/10.1137/100800634"> Novel Numerical Methods for Solving the Time-Space Fractional Diffusion Equation in Two Dimensions</a>  by Q. Yang et al, but adopts a more efficient implementation similar to that in the PCG method. Further also the conventional Lanczos method can be found there and also in text books such as <a href="https://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf">Iteratvie Methods for Sparse Linear Systems" 2nd edition by Yousef Saad </a>
*
* The iteration stops when \f$ \tau ||r_i||_W = \tau ||\vec b||_W \beta_i (T^{-1})_{1m} \leq \eps \f$
* @note The common lanczos method (and M-Lanczos) method are prone to loss of orthogonality for finite precision. Here, only the basic Paige fix is used. Thus the iterations should be kept as small as possible. Could be fixed via full, partial or selective reorthogonalization strategies, but so far no problems occured due to this.
*/
template< class ContainerType >
class Lanczos
{
  public:
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HVec = dg::HVec;
    ///@brief Allocate nothing, Call \c construct method before usage
    Lanczos(){}
    /**
     * @brief Allocate memory for the method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    Lanczos( const ContainerType& copyable, unsigned max_iterations)
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
        *this = Lanczos( std::forward<Params>( ps)...);
    }

    ///@brief Set the maximum number of iterations
    ///@param new_max New maximum number
    void set_max( unsigned new_max) {m_max_iter = new_max;}

    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_max() const {return m_max_iter;}

    ///@brief Get the number of iterations in the last call to \c operator()
    /// (same as T.num_rows)
    ///@return the number of iterations in the last call to \c operator()
    unsigned get_iter() const {return m_iter;}

    /** @brief compute \f$ x = |b|_W V y \f$ from a given tridiagonal matrix T
     * and in-place re-computation of V
     *
     * We avoid explicit storage of the large matrix V
     * @param A A self-adjoint positive definit matrix
     * @param T Tridiagonal matrix (cusp::dia_matrix format)
     * @param y a (host) vector e.g y= T e_1 or y= f(T) e_1, must have size of
     *  \c T.num_rows
     * @param x The result vector (output)
     * @param b Contains the initial value of lanczos method
     * @param bnorm the norm of b in weights, \c get_bnorm()
     */
    template< class MatrixType, class DiaMatrixType, class ContainerType0,
        class ContainerType1,class ContainerType2>
    void normMbVy( MatrixType&& A,
            const DiaMatrixType& T,
            const ContainerType0& y,
            ContainerType1& x,
            const ContainerType2& b, value_type bnorm)
    {
        dg::blas1::axpby(1./bnorm, b, 0.0, m_v); //m_v[1] = b/||b||
        dg::blas1::copy(0., x);
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
                    -T.values(i,0)/T.values(i,2), m_vm,
                    -T.values(i,1)/T.values(i,2), m_v,
                               1.0/T.values(i,2), m_vp);
            dg::blas1::axpby( y[i+1]*bnorm, m_vp, 1., x); //Compute b= |b| V y
            m_vm.swap( m_v);
            m_v.swap( m_vp);

        }
    }
    ///@brief Set or unset debugging output during iterations
    ///@param verbose If true, additional output will be written to \c std::cout during solution
    void set_verbose( bool verbose){ m_verbose = verbose;}


    ///@brief Norm of \c b from last call to \c operator()
    ///@return bnorm
    value_type get_bnorm() const{return m_bnorm;}
    /**
     * @brief Tridiagonalization of A using Lanczos method.
     *
     * The iteration stops when \f$ \tau ||r_i||_W = \tau ||\vec b||_W \beta_i (T^{-1})_{1i} \leq \eps (||b||_W + C) \f$
     * @param A A self-adjoint, positive definit matrix
     * @param b The initial vector that starts orthogonalization
     * @param weights Weights that define the scalar product in which \c A is
     *  self-adjoint and in which the error norm is computed.
     * @param eps accuracy of residual
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     * @param res_fac factor \f$ \tau\f$ that is multiplied to the norm of the
     * residual.  Used to account for specific matrix function and operator in
     * the convergence criterium
     *
     * @return returns the tridiagonal matrix T. Note that \f$ T = (MV)^T A V \f$.
     *  The number of iterations is given by \c T.num_rows
     *
     * @note So far only ordinary convergence criterium (residuum) of Lanczos
     * method is used, in particular for \f$   x = A^{-1} b \f$. If used for matrix
     * function computation \f$  x = f( A) b \f$, the parameter eps should be
     * multiplied with appropriate factors to account for the different
     * convergence criteria.
      */
    template< class MatrixType, class ContainerType0, class ContainerType1>
    const HDiaMatrix& operator()( MatrixType&& A, const ContainerType0& b,
            const ContainerType1& weights, value_type eps = 1e-12,
            value_type nrmb_correction = 1., value_type res_fac = 1.)
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
            m_TH.values(i,0) =  betaip; // -1 diagonal
            dg::blas2::symv(std::forward<MatrixType>(A), m_v, m_vp);
            dg::blas1::axpby(-betaip, m_vm, 1.0, m_vp);  // only - if i>0, therefore no if (i>0)
            alphai  = dg::blas2::dot(m_vp, weights, m_v);
            m_TH.values(i,1) = alphai;
            dg::blas1::axpby(-alphai, m_v, 1.0, m_vp);
            betaip = sqrt(dg::blas2::dot(m_vp, weights, m_vp));
            if (betaip == 0)
            {
                if( m_verbose)
                    DG_RANK0 std::cout << "beta["<<i+1 <<"]=0 encountered\n";
                set_iter(i+1);
                break;
            }
            m_TH.values(i,2) = betaip;  // +1 diagonal
            double T1 = compute_Tinv_m1( m_TH, i+1);
            residual = m_bnorm*betaip*fabs(T1); //Tinv_i1
            if( m_verbose)
                DG_RANK0 std::cout << "# ||r||_W = " << residual << "\tat i = " << i << "\n";
            if (res_fac*residual< eps*(m_bnorm + nrmb_correction) )
            {
                set_iter(i+1);
                break;
            }
            dg::blas1::scal(m_vp, 1./betaip);
            m_vm.swap(m_v);
            m_v.swap( m_vp);
        }
        return m_TH;
    }

    /**
     * @brief Return the vector \f$ \vec e_1\f$ with size \c get_iter()
     *
     * @param iter size
     * @return e_1
     */
    HVec make_e1( ) {
        HVec e1H(m_iter, 0.);
        e1H[0] = 1.;
        return e1H;
    }
  private:

    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        // The alignment (which is the pitch of the underlying values)
        // of m_max_iter preserves the existing elements
        m_TH.resize(new_iter, new_iter, 3*new_iter-2, 3, m_max_iter);
        m_TH.diagonal_offsets[0] = -1;
        m_TH.diagonal_offsets[1] =  0;
        m_TH.diagonal_offsets[2] =  1;
        m_iter = new_iter;
    }
    ContainerType  m_v, m_vp, m_vm;
    HDiaMatrix m_TH;
    unsigned m_iter, m_max_iter;
    bool m_verbose = false;
    value_type m_bnorm = 0.;
};

/*!
 * @brief Class for approximating \f$x = R f(\tilde T)\vec e_1 \approx f(A) b \f$ via exploiting a Krylov projection achieved by the CG method
 *
 * @ingroup matrixapproximation
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
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HVec = dg::HVec;
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
    void set_max( unsigned new_max) {m_max_iter = new_max;}

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
            value_type alphainv = i==0 ? T.values( i,1) :
                T.values(i,1) + T.values( i-1,2);
            value_type beta = -T.values( i,2)/alphainv;
            dg::blas1::axpby( -1./alphainv, m_ap, 1., m_r);
            dg::blas1::axpby(1., m_r, beta, m_p );
            dg::blas1::axpby( y[i+1], m_r, 1., x); //Compute x= R y
        }
    }
    /**
     * @brief Solve the system \f$A*x = b \f$ for x using CG method
     *
     * @param A A self-adjoint, positive definit matrix
     * @param x Contains the initial value (\f$x\equiv 0\f$ if used for
     * tridiagonalization) and the matrix approximation \f$x = A^{-1} b\f$ as
     * output if \c compute_x is set to true
     * @param b The right hand side vector.
     * @param weights Weights that define the scalar product in which \c A is
     *  self-adjoint and in which the error norm is computed.
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be
     *  respected
     * @param res_fac factor that is multiplied to the norm of the residual.
     *  Used to account for specific matrix function and operator in the
     *  convergence criterium
     *
     * @return The tridiagonal matrix \f$ T\f$
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
    const HDiaMatrix& operator()( MatrixType&& A, const ContainerType0& b,
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
                m_TH.values(i,0) = 0.;
                m_TH.values(i,1) = 1./alpha;
                m_TH.values(i,2) = -beta/alpha;
            }
            else
            {
                m_TH.values(i,0) = -1./alpha_old;
                m_TH.values(i,1) =  1./alpha + beta_old/alpha_old;
                m_TH.values(i,2) = -beta/alpha;
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
     * @param iter size
     * @return e_1
     */
    HVec make_e1( ) {
        HVec e1H(m_iter, 0.);
        e1H[0] = 1.;
        return e1H;
    }
  private:
    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        // The alignment (which is the pitch of the underlying values)
        // of m_max_iter preserves the existing elements
        m_TH.resize(new_iter, new_iter, 3*new_iter-2, 3, m_max_iter);
        m_TH.diagonal_offsets[0] = -1;
        m_TH.diagonal_offsets[1] =  0;
        m_TH.diagonal_offsets[2] =  1;
        m_iter = new_iter;
    }
    ContainerType m_r, m_ap, m_p;
    unsigned m_max_iter, m_iter;
    HDiaMatrix m_TH;
    bool m_verbose = false;
    value_type m_bnorm = 0.;
};

} //namespace mat
} //namespace dg

