#pragma once
#include <cusp/dia_matrix.h>
#include <cusp/coo_matrix.h>

#include <cusp/print.h>
#include "dg/algorithm.h"
#include "tridiaginv.h"


/**
* @brief Classes for Krylov space approximations of a Matrix-Vector product
*/

namespace dg{

/**
* @brief Tridiagonalize \f$A\f$ and approximate \f$f(A)x \approx |x|_M V f(T) e_1\f$
*. A is self-adjoint in the weights \f$ M\f$
*
* @ingroup matrixapproximation
*
* The M-Lanczos method is based on the paper <a href="https://doi.org/10.1137/100800634"> Novel Numerical Methods for Solving the Time-Space Fractional Diffusion Equation in Two Dimensions</a>  by Q. Yang et al, but adopts a more efficient implementation similar to that in the PCG method. Further also the conventional Lanczos method can be found there and also in text books such as <a href="https://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf">Iteratvie Methods for Sparse Linear Systems" 2nd edition by Yousef Saad </a>
*
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
        m_TH.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_TH.diagonal_offsets[0] = -1;
        m_TH.diagonal_offsets[1] =  0;
        m_TH.diagonal_offsets[2] =  1;
    }
    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = Lanczos( std::forward<Params>( ps)...);
    }
    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        m_TH.resize(new_iter, new_iter, 3*new_iter-2, 3, m_max_iter);
        m_TH.diagonal_offsets[0] = -1;
        m_TH.diagonal_offsets[1] =  0;
        m_TH.diagonal_offsets[2] =  1;
        m_iter = new_iter;
    }

    ///@brief Get the current  number of iterations
    ///@return the current number of iterations
    unsigned get_iter() const {return m_iter;}

    /** @brief compute \f$ b = |x|_M V y \f$ from a given tridiagonal matrix T
     * and in-place re-computation of V
     *
     * We avoid explicit storage of the large matrix V
     * @param A A self-adjoint positive definit matrix
     * @param T Tridiagonal matrix (cusp::dia_matrix format)
     * @param y a (host) vector e.g y= T e_1 or y= f(T) e_1
     * @param x Contains the initial value of lanczos method
     * @param b The right hand side vector (output)
     * @param xnorm M-norm of x
     * @param iter size of tridiagonal matrix
     */
    template< class MatrixType, class DiaMatrixType, class ContainerType0, class ContainerType1,class ContainerType2>
    void normMxVy( MatrixType&& A, const DiaMatrixType& T, const ContainerType0& y, ContainerType1& b, const ContainerType2& x, value_type xnorm,  unsigned iter)
    {
        dg::blas1::axpby(1./xnorm, x, 0.0, m_v); //m_v[1] = x/||x||
        dg::blas1::copy(0., b);
        dg::blas1::copy(0., m_vm);
        // check if (potentially) all higher elements in y are zero
        unsigned less_iter = iter;
        for( unsigned i=0; i<iter; i++)
            if( y[i] != 0)
                less_iter = i+1;
        for ( unsigned i=0; i<less_iter; i++)
        {
            dg::blas1::axpby( y[i]*xnorm, m_v, 1., b); //Compute b= V y

            dg::blas2::symv( std::forward<MatrixType>(A), m_v, m_vp);
            dg::blas1::axpbypgz(
                    -T.values(i,0)/T.values(i,2), m_vm,
                    -T.values(i,1)/T.values(i,2), m_v,
                               1.0/T.values(i,2), m_vp);
            m_vm.swap( m_v);
            m_v.swap( m_vp);
        }
    }
    ///@brief Set or unset debugging output during iterations
    ///@param verbose If true, additional output will be written to \c std::cout during solution
    void set_verbose( bool verbose){ m_verbose = verbose;}
    /**
     * @brief Tridiagonalization of A using Lanczos method. Optionally compute
     * \f$ b= A x \approx || x ||_2 V T e_1\f$
     *
     * @param A A self-adjoint, positive definit matrix
     * @param x Contains an initial value
     * @param b on output contains \f$b = || x ||_M V T e_1 \approx A x\f$ if
     *  \c compute_b is set to \c true, else is not used
     * @param weights Weights that define the scalar product in which \c A is
     *  self-adjoint and in which the error norm is computed.
     * @param eps accuracy of residual
     * @param compute_b Specify if \f$b = || x ||_M V T e_1 \f$ should be
     *  computed or if only T should be computed
     * @param res_fac factor that is multiplied to the norm of the residual.
     *  Used to account for specific matrix function and operator in the
     *  convergence criterium
     *
     * @return returns the tridiagonal matrix T. Note that  \f$ T = (MV)^T A V \f$.
     *
     * @note So far only ordinary convergence criterium (residuum) of Lanczos
     * method is used, in particular for \f$ A x  = b \f$. If used for matrix
     * function computation \f$ f( A) x  = b \f$, the parameter eps should be
     * multiplied with appropriate factors to account for the different
     * convergence criteria.
      */
    template< class MatrixType, class ContainerType0, class ContainerType1, class ContainerType2>
    const HDiaMatrix& operator()( MatrixType&& A, const ContainerType0& x, ContainerType1& b, const ContainerType2& weights, value_type eps, bool compute_b = false, value_type res_fac = 1.)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        value_type xnorm = sqrt(dg::blas2::dot(x, weights, x));
        value_type residual;
        dg::blas2::symv(std::forward<MatrixType>(A),x, m_v);
        value_type r0norm = sqrt(dg::blas2::dot(m_v, weights, m_v));

        dg::blas1::axpby(1./xnorm, x, 0.0, m_v); //m_v[1] = x/||x||
        value_type betaip = 0.;
        value_type alphai = 0.;
        if( compute_b)
            dg::blas1::copy( 0., b);
        for( unsigned i=0; i<m_max_iter; i++)
        {
            m_TH.values(i,0) =  betaip; // -1 diagonal
            dg::blas2::symv(std::forward<MatrixType>(A), m_v, m_vp);
            dg::blas1::axpby(-betaip, m_vm, 1.0, m_vp);  // only - if i>0, therefore no if (i>0)
            alphai  = dg::blas2::dot(m_vp, weights, m_v);
            if( compute_b && i==0)
                dg::blas1::axpby( alphai*xnorm, m_v, 1., b); //Compute b= V y
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
            if( compute_b && i == 1)
                dg::blas1::axpby( betaip*xnorm, m_v, 1., b); //Compute b= V y
            m_TH.values(i,2) = betaip;  // +1 diagonal
            m_tridiaginvH.resize(i+1);
            m_tridiaginvH(m_TH, m_TinvH);
            // The first row of Tinv is the same as the first column (symmetry)
            residual = r0norm*betaip*abs(m_TinvH.values[i]); //Tinv_i1
            //residual = betaip*abs(m_TinvH.values[i]); //Tinv_i1
            if( m_verbose)
                DG_RANK0 std::cout << "# ||r||_M =  " << residual << " at i = " << i << "\n";
            if (res_fac*residual< eps )
            {
                set_iter(i+1);
                break;
            }
            dg::blas1::scal(m_vp, 1./betaip);
            m_vm.swap(m_v);
            m_v.swap( m_vp);
        }
        if (compute_b == true)
        {
            HVec e1H(get_iter(), 0.), yH(e1H);
            e1H[0] = 1.;
            dg::blas2::symv (m_TH, e1H, yH); //y= T e_1
            normMxVy(A, m_TH, yH, b, x, xnorm, get_iter()); //b= |x| V T e_1 
        }
        return m_TH;
    }
  private:
    ContainerType  m_v, m_vp, m_vm;
    HDiaMatrix m_TH;
    HCooMatrix m_TinvH;
    unsigned m_iter, m_max_iter;
    dg::TridiagInvDF<HVec, HDiaMatrix, HCooMatrix> m_tridiaginvH;
    bool m_verbose = false;
};

/*!
 * @brief Class for approximating \f$x \approx A^{-1} b  \f$ solve via exploiting a Krylov projection achieved by the CG method
 *
 * @ingroup matrixapproximation
 *
 * This class is based on the approach of the paper <a href="https://doi.org/10.1016/0377-0427(87)90020-3)" > An iterative solution method for solving f(A)x = b, using Krylov subspace information obtained for the symmetric positive definite matrix A</a> by H. A. Van Der Vorst
 *
 * @note The approximation relies on Projection \f$x = A^{-1} b  \approx  R T^{-1} e_1\f$, where \f$T\f$ and \f$R\f$ is the tridiagonal and orthogonal matrix of the PCG solve and \f$e_1\f$ is the normalized unit vector. The vector \f$T^{-1} e_1\f$ can be further processed for matrix function approximation
 \f$f(T^{-1}) e_1\f$  */
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
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    MCG( const ContainerType& copyable, unsigned max_iterations)
    {
        m_ap = m_p = m_r = copyable;
        m_max_iter = max_iterations;
        m_TH.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_TH.diagonal_offsets[0] = -1;
        m_TH.diagonal_offsets[1] =  0;
        m_TH.diagonal_offsets[2] =  1;
        m_iter = max_iterations;
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

    ///@copydoc hide_construct
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = MCG( std::forward<Params>( ps)...);
    }
    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        m_TH.resize(new_iter, new_iter, 3*new_iter-2, 3, m_max_iter);
        m_TH.diagonal_offsets[0] = -1;
        m_TH.diagonal_offsets[1] =  0;
        m_TH.diagonal_offsets[2] =  1;
        m_iter = new_iter;
    }
    ///@brief Get the current  number of iterations
    ///@return the current number of iterations
    unsigned get_iter() const {return m_iter;}
    /** @brief Compte x = R y
     * @param A A self-adjoint, positive definit matrix
     * @param T T non-symmetric tridiagonal Matrix from MCG tridiagonalization
     * @param weights Weights used to compute the norm for the error condition
     * @param y (host) vector with v.size() = iter.
     *  Typically \f$ T^(-1) e_1 \f$ or \f$ f(T^(-1)) e_1 \f$
     * @param x Contains the matrix approximation \f$x = A^{-1} b\f$ as output
     * @param b The right hand side vector.
     * @param iter number of iterations (size of T)
     */
    template< class MatrixType, class DiaMatrixType, class ContainerType0,
        class ContainerType1, class ContainerType2>
    void Ry( MatrixType&& A,
            DiaMatrixType& T,
            const ContainerType& weights,
            const ContainerType0& y, ContainerType1& x, const ContainerType2& b,
            unsigned iter)
    {
        dg::blas1::copy(0., x); //could be removed if x is correctly initialized

        dg::blas1::copy( b, m_r);

        dg::blas1::copy( m_r, m_p );

        for ( unsigned i=0; i<iter; i++)
        {
            dg::blas1::axpby( y[i], m_r, 1., x); //Compute x=0 + R y
            dg::blas2::symv( std::forward<MatrixType>(A), m_p, m_ap);
            value_type alphainv = i==0 ? T.values( i,1) :
                T.values(i,1) + T.values( i-1,2);
            value_type beta = -T.values( i,2)/alphainv;
            dg::blas1::axpby( -1./alphainv, m_ap, 1., m_r);
            dg::blas1::axpby(1., m_r, beta, m_p );
        }
    }
    /**
     * @brief Solve the system \f$A*x = b \f$ for x using PCG method
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
     * @param compute_x Specify if \f$x = R T^{-1} e_1 \f$ should be computed
     *  or if only T should be computed
     * @param res_fac factor that is multiplied to the norm of the residual.
     *  Used to account for specific matrix function and operator in the
     *  convergence criterium
     *
     * @return Number of iterations used to achieve desired precision
     * @note So far only ordinary convergence criterium of CG method, in
     * particular for \f$ A x  = b \f$. If used for matrix function
     * computation, \f$ f(A) x  = b \f$, the parameter eps should be
     * multiplied with appropriate factors to account for the different
     * convergence criteria.
     * The Matrix R and T of the tridiagonalization are
     * further used for computing matrix functions. The x vector must be
     * initialized with 0 if used for tridiagonalization.
      */
    template< class MatrixType, class ContainerType0, class ContainerType1,
        class ContainerType2>
    const HDiaMatrix& operator()( MatrixType&& A, ContainerType0& x,
            const ContainerType1& b, const ContainerType2& weights,
            value_type eps = 1e-12, value_type nrmb_correction = 1,
            bool compute_x = false, value_type res_fac = 1.)
    {
        value_type nrmb = sqrt( dg::blas2::dot( weights, b));
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        if( m_verbose)
        {
            DG_RANK0 std::cout << "# Norm of b "<<nrmb <<"\n";
            DG_RANK0 std::cout << "# Residual errors: \n";
        }
        if( nrmb == 0)
        {
            dg::blas1::copy( b, x);
            set_iter(1);
            return m_TH;
        }
        dg::blas2::symv( std::forward<MatrixType>(A), x, m_r);
        dg::blas1::axpby( 1., b, -1., m_r);
        dg::blas1::copy( m_r, m_p );

        value_type nrmzr_old = dg::blas2::dot( m_r, weights, m_r);
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
                value_type temp = dg::blas2::dot( m_r, weights, m_r);
                DG_RANK0 std::cout << "# Absolute res_fac*||r||_M "<<res_fac*sqrt( temp) <<"\t ";
                DG_RANK0 std::cout << "#  < Critical "<<eps*(nrmb + nrmb_correction)<<"\t ";
                DG_RANK0 std::cout << "# (Relative "<<res_fac*sqrt( temp )/nrmb << ")\n";
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
            if( res_fac*sqrt( dg::blas2::dot( m_r, weights, m_r))
                    < eps*(nrmb + nrmb_correction))
            {
                set_iter(i+1);
                break;
            }
            dg::blas1::axpby(1., m_r, beta, m_p );
            nrmzr_old=nrmzr_new;
        }

        //Compute inverse of tridiagonal matrix
        if (compute_x == true)
        {
            HVec e1H(get_iter(), 0.);
            HVec yH(get_iter(), 0.);
            e1H[0] = 1.;
            dg::TridiagInvDF<HVec, HDiaMatrix, HCooMatrix> tridiaginv(yH);
            auto TinvH = tridiaginv(m_TH); //Compute on Host!
            dg::blas2::symv(TinvH, e1H, yH);  // m_y= T^(-1) e_1
            Ry(std::forward<MatrixType>(A), m_TH, weights, yH, x, b,  get_iter());  // x = 0 + R T^(-1) e_1
        }
        return m_TH;
    }
  private:
    ContainerType m_r, m_ap, m_p;
    unsigned m_max_iter, m_iter;
    HDiaMatrix m_TH;
    bool m_verbose = false;
};

} //namespace dg

