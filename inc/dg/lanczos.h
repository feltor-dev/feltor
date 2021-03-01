#pragma once
#include "blas.h"
#include "functors.h"
#include "backend/timer.h"

/**
* @brief Classes for Krylov space approximations of a Matrix-Vector product
*/

namespace dg{
   

/**
* @brief Functor class for computing the inverse of a general tridiagonal matrix 
* 
* @ingroup invert
*/
template< class ContainerType, class DiaMatrix, class CooMatrix>
class InvTridiag
{
  public:
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    InvTridiag(){}
    //Constructor
    InvTridiag(const std::vector<value_type>& copyable) 
    {
        phi.assign(copyable.size()+1,0.);
        theta.assign(copyable.size()+1,0.);
        Tinv.resize(copyable.size(), copyable.size(),  copyable.size()* copyable.size());
        temp = 0.;
    }
    /**
     * @brief Resize inverse tridiagonal matrix and helper vectors
     * 
     * @param new_size new size of square matrix
    */
    void resize(unsigned new_size) {
        phi.resize(new_size+1);
        theta.resize(new_size+1);
        Tinv.resize(new_size, new_size, new_size*new_size);
    }

    int sign(unsigned i)
    {  
        if (i%2==0) return 1;
        else return -1;
    }
 /**
     * @brief Compute the inverse of a tridiagonal matrix T
     * 
     * @param T tridiagonal matrix
     * 
     * @return the inverse of the tridiagonal matrix (coordinate format)
     */
    CooMatrix operator()(const DiaMatrix& T)
    {
        ContainerType alpha(theta.size()-1,0.);
        ContainerType beta(theta.size()-1,0.);
        ContainerType gamma(theta.size()-1,0.);
        for(unsigned i = 0; i<theta.size()-1; i++)
        {
            alpha[i] = T.values(i,1);    // 0 diagonal
            beta[i]  = T.values(i,2)  ;  // +1 diagonal //dia_rows entry works since its outside of matrix
            gamma[i] = T.values(i+1,0);  // -1 diagonal            
        }
        Tinv = this->operator()(alpha, beta, gamma);
        return Tinv;
    }
     /**
     * @brief Compute the inverse of a tridiagonal matrix with diagonal vectors a,b,c
     * 
     * @param a  "0" diagonal vector
     * @param b "+1" diagonal vector (starts with index 0 to (size of b)-1)
     * @param c "-1" diagonal vector (starts with index 0 to (size of c)-1)
     * 
     * @return the inverse of the tridiagonal matrix (coordinate format)
     */
    template<class ContainerType0>
    CooMatrix operator()(const ContainerType0& a, const ContainerType0& b,  const ContainerType0& c)
    {
        //Compute theta and phi
        unsigned is=0;
        unsigned vector_size = a.size();
        for( unsigned i = 0; i<theta.size(); i++)
        {   
            is = (theta.size() - 1) - i;
            if (i==0) 
            {   
                theta[0] = 1.; 
                phi[is]  = 1.;
            }
            else if (i==1) 
            {
                theta[1] = a[0]; 
                phi[is]  = a[is];
            }
            else
            {
                theta[i] = a[i-1] * theta[i-1] - b[i-2] * c[i-2] * theta[i-2];
                phi[is]  = a[is]  * phi[is+1]  - b[is]  * c[is]  * phi[is+2];
            }
        }
        //Compute inverse tridiagonal matrix elements
        for( unsigned j=0; j<vector_size; j++)
        {   
            for( unsigned i=0; i<vector_size; i++)
            {   
                Tinv.row_indices[i*vector_size+j]    = i;
                Tinv.column_indices[i*vector_size+j] = j; 
                if (i<=j) {
                    temp = std::accumulate(std::next(b.begin(),i), std::next(b.begin(),j), 1., std::multiplies<value_type>());
                    Tinv.values[i*vector_size+j] =temp*sign(i+j) * theta[i] * phi[j+1]/theta[vector_size];

                }   
                else // if (i>j)
                {
                    temp = std::accumulate(std::next(c.begin(),j), std::next(c.begin(),i), 1., std::multiplies<value_type>());
                    Tinv.values[i*vector_size+j] =temp*sign(i+j) * theta[j] * phi[i+1]/theta[vector_size];
                }
            }
        }
        return Tinv;
    }
  private:
    std::vector<value_type> phi, theta;
    CooMatrix Tinv;    
    value_type temp;
};

/**
* @brief Functor class for the Lanczos method to approximate \f$b = Ax\f$ or \f$b = M^{-1} A x\f$
* for b. A is a symmetric and \f$M^{-1}\f$ are typically the inverse weights.
*
* @ingroup matrixapproximation
* 
* 
* The M-Lanczos method is based on the paper <a href="https://doi.org/10.1137/100800634"> Novel Numerical Methods for Solving the Time-Space Fractional Diffusion Equation in Two Dimensions</a>  by Q. Yang et al, but adopts a more efficient implementation similar to that in the PCG method. Further also the conventional Lanczos method can be found there and also in text books such as <a href="https://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf">Iteratvie Methods for Sparse Linear Systems" 2nd edition by Yousef Saad </a>
* 
* @note The common lanczos method (and M-Lanczos) method are prone to loss of orthogonality for finite precision. Here, only the basic Paige fix is used. Thus the iterations should be kept as small as possible. Could be fixed via full, partial or selective reorthogonalization strategies, but so far no problems occured due to this.
*/
template< class ContainerType, class SubContainerType, class DiaMatrix, class CooMatrix>
class Lanczos
{
  public:
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory> ;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    ///@brief Allocate nothing, Call \c construct method before usage
    Lanczos(){}
    ///@copydoc construct()
    Lanczos( const ContainerType& copyable, unsigned max_iterations) 
    {
          construct(copyable, max_iterations);
    }
    /**
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    void construct( const ContainerType& copyable, unsigned max_iterations) {
        m_vi = m_vip = m_vim = m_wi = m_wim = m_wip = copyable;
        dg::assign(copyable, m_b);
        m_max_iter = max_iterations;
        m_iter = max_iterations;
        //sub matrix and vector
        m_T.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_T.diagonal_offsets[0] = -1;
        m_T.diagonal_offsets[1] =  0;
        m_T.diagonal_offsets[2] =  1;
        m_TH.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_TH.diagonal_offsets[0] = -1;
        m_TH.diagonal_offsets[1] =  0;
        m_TH.diagonal_offsets[2] =  1;
        m_TinvH.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_e1.assign(m_max_iter, 0.);
        m_y.assign(m_max_iter, 0.);
        m_e1[0] = 1.;
    }
    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        m_T.resize(new_iter, new_iter, 3*new_iter-2, 3, m_max_iter);
        m_T.diagonal_offsets[0] = -1;
        m_T.diagonal_offsets[1] =  0;
        m_T.diagonal_offsets[2] =  1;
        m_TH.resize(new_iter, new_iter, 3*new_iter-2, 3, m_max_iter);
        m_TH.diagonal_offsets[0] = -1;
        m_TH.diagonal_offsets[1] =  0;
        m_TH.diagonal_offsets[2] =  1;
        m_e1.assign(new_iter, 0.);
        m_y.assign(new_iter, 0.);
        m_e1[0] = 1.;
        m_iter = new_iter;
    }
    ///@brief Get the current  number of iterations
    ///@return the current number of iterations
    unsigned get_iter() const {return m_iter;}

    /** @brief compute b = |x| V y from a given tridiagonal matrix T 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param T Tridiagonal matrix (cusp::dia_matrix format)
     * @param y a vector e.g y= T e_1 or y= f(T) e_1
     * @param x Contains an initial value of lanczos method
     * @param b The right hand side vector (output)
     * @param xnorm 2-norm of x
     * @param iter size of tridiagonal matrix
     */
    template< class MatrixType, class DiaMatrixType, class ContainerType0, class ContainerType1,class ContainerType2>
    void norm2xVy( MatrixType& A, DiaMatrixType& T, ContainerType0& y, ContainerType1& b, ContainerType2& x, value_type xnorm,  unsigned iter)
    {
        dg::blas1::axpby(1./xnorm, x, 0.0, m_vi); //m_v[1] = x/||x||
        dg::blas1::scal(b, 0.);
        for ( unsigned i=0; i<iter; i++)
        {
            dg::blas1::axpby( y[i], m_vi, 1., b); //Compute b= V y

            dg::blas2::symv( A, m_vi, m_vip);                    
            dg::blas1::axpby(-T.values(i,0), m_vim, 1.0, m_vip);  
            dg::blas1::axpby(-T.values(i,1), m_vi, 1.0, m_vip);  
            dg::blas1::scal(m_vip, 1./T.values(i,2));  
            m_vim = m_vi;
            m_vi = m_vip;
        }
        dg::blas1::scal(b, xnorm ); 
    }
    /** @brief compute b = |x|_M V y from a given tridiagonal matrix T 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param T Tridiagonal matrix (cusp::dia_matrix format)
     * @param M the weights 
     * @param Minv the inverse of M - the inverse weights
     * @param y a vector e.g y= T e_1 or y= f(T) e_1
     * @param x Contains an initial value of lanczos method
     * @param b The right hand side vector (output)
     * @param xnorm M-norm of x
     * @param iter size of tridiagonal matrix
     */
    template< class MatrixType, class DiaMatrixType, class SquareNorm1, class SquareNorm2, class ContainerType0, class ContainerType1,class ContainerType2>
    void normMxVy( MatrixType& A, DiaMatrixType& T, SquareNorm1& M, SquareNorm2& Minv, ContainerType0& y, ContainerType1& b, ContainerType2& x, value_type xnorm,  unsigned iter)
    {
        dg::blas1::axpby(1./xnorm, x, 0.0, m_vi); //m_v[1] = x/||x||
        dg::blas2::symv(M, m_vi, m_wi);
        dg::blas1::scal(b, 0.);
        for( unsigned i=0; i<iter; i++)
        {
            dg::blas1::axpby( y[i], m_vi, 1., b); //Compute b= V y

            dg::blas2::symv(A, m_vi, m_wip); 
            dg::blas1::axpby(-T.values(i,0), m_wim, 1.0, m_wip);    //only - if i>0, therefore no if (i>0)
            dg::blas1::axpby(-T.values(i,1), m_wi, 1.0, m_wip);
            dg::blas2::symv(Minv,m_wip,m_vip);
            dg::blas1::scal(m_vip, 1./T.values(i,2));
            dg::blas1::scal(m_wip, 1./T.values(i,2));
            m_vi  = m_vip;
            m_wim = m_wi;
            m_wi  = m_wip;
        }
        dg::blas1::scal(b, xnorm );
    }
    /**
     * @brief Solve the system \f$ b= A x \approx || x ||_2 V T e_1\f$ using Lanczos method. Useful for tridiagonalization of A (cf return statement).
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains an initial value
     * @param b The right hand side vector
     * @param eps accuracy of residual
     * 
     * @return returns the tridiagonal matrix T. Note that  \f$ T = V^T A V \f$.
      */
    template< class MatrixType, class ContainerType0, class ContainerType1>
    DiaMatrix operator()( MatrixType& A, const ContainerType0& x, ContainerType1& b, value_type eps )
    {
        value_type xnorm = sqrt(dg::blas1::dot(x, x));
        value_type residual;
#ifdef DG_BENCHMARK
        dg::Timer t;
        value_type invtime=0.;
#endif //DG_BENCHMARK

        dg::blas1::axpby(1./xnorm, x, 0.0, m_vi); //m_v[1] = x/||x||
        value_type betaip = 0.;
        value_type alphai = 0.;
        for( unsigned i=0; i<m_max_iter; i++)
        {
            m_TH.values(i,0) =  betaip; // -1 diagonal            
            dg::blas2::symv(A, m_vi, m_vip);                    
            dg::blas1::axpby(-betaip, m_vim, 1.0, m_vip);  // only - if i>0, therefore no if (i>0)  
            alphai  = dg::blas1::dot(m_vip, m_vi);
            m_TH.values(i,1) = alphai;
            dg::blas1::axpby(-alphai, m_vi, 1.0, m_vip);      
            betaip = sqrt(dg::blas1::dot(m_vip, m_vip));     
            if (betaip == 0) {
#ifdef DG_DEBUG
                std::cout << "beta["<<i+1 <<"]=0 encountered\n";
#endif //DG_DEBUG
                set_iter(i+1); 
                break;
            } 
            dg::blas1::scal(m_vip, 1./betaip);  
            m_vim = m_vi; //swap instead?
            m_vi = m_vip;
            m_TH.values(i,2) = betaip;  // +1 diagonal

            if (i>0) {
                m_invtridiagH.resize(i);
#ifdef DG_BENCHMARK
                t.tic();
#endif //DG_BENCHMARK
                m_TinvH = m_invtridiagH(m_TH); //TODO slow -> criterium without inversion ? 
#ifdef DG_BENCHMARK
                t.toc();
                invtime+=t.diff();
#endif //DG_BENCHMARK
                residual = xnorm*betaip*betaip*abs(m_TinvH.values[i-1]);
#ifdef DG_DEBUG
                std::cout << "||r||_2 =  " << residual << "  # of iterations = " << i+1 << "\n";
#endif //DG_DEBUG
                if (residual< eps ) {
                    set_iter(i+1); 
                    break;
                }
            }
        }
#ifdef DG_BENCHMARK
        std::cout << get_iter() << " T inversions took " << invtime << "s\n";
#endif //DG_BENCHMARK
        m_T = m_TH;
        dg::blas2::symv(m_T, m_e1, m_y); //y= T e_1
        norm2xVy(A, m_T, m_y, b, x, xnorm, get_iter()); //b= |x| V T e_1 
        return m_T;
    }
    /**
     * @brief Solve the system \f$b= M^{-1} A x \approx || x ||_M V T e_1\f$ for b using M-Lanczos method. Useful for the fast computatin of matrix functions of \f$ M^{-1} A\f$.
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains an initial value
     * @param b The right hand side vector. 
     * @param M the weights 
     * @param Minv the inverse of M - the inverse weights
     * @param eps accuracy of residual
     * 
     * @return returns the tridiagonal matrix T and orthonormal basis vectors contained in the matrix V matrix. Note that  \[f T = V^T A V \f$
     */
    template< class MatrixType, class ContainerType0, class ContainerType1, class SquareNorm1, class SquareNorm2>
    DiaMatrix operator()( MatrixType& A, const ContainerType0& x, ContainerType1& b,  SquareNorm1& M, SquareNorm2& Minv, value_type eps)
    {
        value_type xnorm = sqrt(dg::blas2::dot(x, M, x));
        value_type residual;
#ifdef DG_BENCHMARK
        dg::Timer t;
        value_type invtime=0.;

#endif //DG_BENCHMARK
        
        dg::blas1::axpby(1./xnorm, x, 0.0, m_vi); //m_v[1] = x/||x||
        value_type betaip = 0.;
        value_type alphai = 0.;
        dg::blas2::symv(M, m_vi, m_wi);
        
        for( unsigned i=0; i<m_max_iter; i++)
        { 
            m_TH.values(i,0) =  betaip;  // -1 diagonal
            dg::blas2::symv(A, m_vi, m_wip); 
            dg::blas1::axpby(-betaip, m_wim, 1.0, m_wip);    //only - if i>0, therefore no if (i>0)
            alphai = dg::blas1::dot(m_wip, m_vi);  
            m_TH.values(i,1) = alphai;
            dg::blas1::axpby(-alphai, m_wi, 1.0, m_wip);     
            dg::blas2::symv(Minv,m_wip,m_vip);
            betaip = sqrt(dg::blas1::dot(m_wip, m_vip)); 
            if (betaip == 0) {
#ifdef DG_DEBUG
                std::cout << "beta["<<i+1 <<"]=0 encountered\n";
#endif //DG_DEBUG
                set_iter(i+1); 
                break;
            } 
            dg::blas1::scal(m_vip, 1./betaip);     
            dg::blas1::scal(m_wip, 1./betaip);  
            m_vi  = m_vip;
            m_wim = m_wi;
            m_wi  = m_wip;
            m_TH.values(i,2) =  betaip;  // +1 diagonal
            if (i>0) {
                m_invtridiagH.resize(i);
#ifdef DG_BENCHMARK
                t.tic();
#endif //DG_BENCHMARK
                m_TinvH = m_invtridiagH(m_TH); //TODO slow -> criterium without inversion ? 
#ifdef DG_BENCHMARK
                t.toc();
                invtime+=t.diff();
#endif //DG_BENCHMARK
                residual = xnorm*betaip*betaip*abs(m_TinvH.values[i-1]);
#ifdef DG_DEBUG
                std::cout << "||r||_M =  " << residual << "  # of iterations = " << i+1 << "\n";
#endif //DG_DEBUG
                if (residual< eps ) {
                    set_iter(i+1); 
                    break;
                }
            }
        }
#ifdef DG_BENCHMARK
        std::cout << get_iter() << " T inversions took " << invtime << "s\n";
#endif //DG_BENCHMARK
        m_T = m_TH;
        dg::blas2::symv(m_T, m_e1, m_y); //T e_1
        normMxVy(A, m_T, M, Minv, m_y, b, x, xnorm, get_iter()); //b= |x| V T e_1 
        return m_T;
    }
  private:
    ContainerType  m_vi, m_vip, m_vim, m_wi, m_wip, m_wim;
    std::vector<dg::HVec> m_vH;
    std::vector<ContainerType> m_v;
    dg::HVec m_viH;
    HDiaMatrix m_TH;
    HCooMatrix m_TinvH;
    SubContainerType m_e1, m_y, m_b;
    unsigned m_iter, m_max_iter;
    DiaMatrix m_T;  
    InvTridiag<dg::HVec, HDiaMatrix, HCooMatrix> m_invtridiagH;
};

/*! 
 * @brief Class for approximating \f$x \approx A^{-1} b  \f$ solve via exploiting a Krylov projection achieved by the PCG method 
 * 
 * @ingroup matrixapproximation
 * 
 * This class is based on the approach of the paper <a href="https://doi.org/10.1016/0377-0427(87)90020-3)" > An iterative solution method for solving f(A)x = b, using Krylov subspace information obtained for the symmetric positive definite matrix A</a> by H. A. Van Der Vorst
 * 
 * @note The approximation relies on Projection \f$x = A^{-1} b  \approx  R T^{-1} e_1\f$, where \f$T\f$ and \f$R\f$ is the tridiagonal and orthogonal matrix of the PCG solve and \f$e_1\f$ is the normalized unit vector. The vector \f$T^{-1} e_1\f$ can be further processed for matrix function approximation 
 \f$f(T^{-1}) e_1\f$  */
template< class ContainerType, class SubContainerType, class DiaMatrix, class CooMatrix>
class CGtridiag
{
  public:
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    CGtridiag(){}
    ///@copydoc construct()
    CGtridiag( const ContainerType& copyable, unsigned max_iterations)
    {
          construct(copyable, max_iterations);
    }
    ///@brief Set the maximum number of iterations
    ///@param new_max New maximum number
    void set_max( unsigned new_max) {m_max_iter = new_max;}
    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_max() const {return m_max_iter;}
    /**
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    void construct( const ContainerType& copyable, unsigned max_iterations)
    {
        m_v0.assign(max_iterations,0.);
        m_vp.assign(max_iterations,0.);
        m_vm.assign(max_iterations,0.);
        m_e1.assign(max_iterations,0.);
        m_y.assign(max_iterations,0.);
        m_ap = m_p = m_r = copyable;
        m_max_iter = max_iterations;
        m_R.resize(m_r.size(), max_iterations, max_iterations*m_r.size());
        m_Tinv.resize(m_r.size(), max_iterations, max_iterations*m_r.size());
        m_invtridiag.resize(max_iterations);
        m_iter = max_iterations;
    }
    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        m_v0.resize(new_iter);
        m_vp.resize(new_iter);
        m_vm.resize(new_iter);
        m_e1.resize(new_iter, 0.);
        m_e1[0] = 1.;
        m_y.resize(new_iter,0.);
        m_R.resize(m_r.size(), new_iter, new_iter*m_r.size());
        m_Tinv.resize(new_iter, new_iter, new_iter*new_iter);
        m_invtridiag.resize(new_iter);
        m_iter = new_iter;

    }
    ///@brief Get the current  number of iterations
    ///@return the current number of iterations
    unsigned get_iter() const {return m_iter;}
    /**
     * @brief Solve the system \f$A*x = b \f$ for x using PCG method 
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains the initial value (\f[x!=0\f] if used for tridiagonalization) and the matrix approximation \f[x = A^{-1} b\f] as output
     * @param b The right hand side vector. 
     * @param P The preconditioner to be used
     * @param M (Inverse) Weights used to compute the norm for the error condition
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     * 
     * @return Number of iterations used to achieve desired precision
     * @note So far only ordinary convergence criterium of CG method. Should be adapted to square root criterium. 
     * Note that the preconditioner must be \f$P = M^{-1} \f$ if the Matrix R and T of the tridiagonalization are further used for computing matrix functions. Then the x vector must be initialized with 0.
     * 
     * 
      */
    template< class MatrixType, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm>
    std::pair<CooMatrix, CooMatrix> operator()( MatrixType& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& M, value_type eps = 1e-12, value_type nrmb_correction = 1)
    {
        value_type nrmb = sqrt( dg::blas2::dot( M, b));
#ifdef DG_DEBUG
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif //MPI
        {
        std::cout << "# Norm of b "<<nrmb <<"\n";
        std::cout << "# Residual errors: \n";
        }
#endif //DG_DEBUG
        if( nrmb == 0)
        {
            dg::blas1::copy( b, x);
            set_iter(1);
            return m_TinvRpair;
        }
        dg::blas2::symv( A, x, m_r);
        dg::blas1::axpby( 1., b, -1., m_r);
        
        if( sqrt( dg::blas2::dot( M, m_r)) < eps*(nrmb + nrmb_correction)) //TODO replace with appropriate criterium
        {
            set_iter(1);
            return m_TinvRpair;
        }
        dg::blas2::symv( P, m_r, m_p );
        dg::blas1::copy( m_p, m_ap);

        value_type nrm2r_old = dg::blas1::dot( m_ap, m_r);
        value_type nrm2r_new;
        
        unsigned counter = 0;
        for( unsigned i=0; i<m_max_iter; i++)
        {
            for( unsigned j=0; j<m_ap.size(); j++)
            {            
                m_R.row_indices[counter]    = j;
                m_R.column_indices[counter] = i; 
                m_R.values[counter]         = m_ap.data()[j];
                counter++;
            }
            dg::blas2::symv( A, m_p, m_ap);
            m_alpha = nrm2r_old /dg::blas1::dot( m_p, m_ap);
            dg::blas1::axpby( -m_alpha, m_ap, 1., m_r);
#ifdef DG_DEBUG
#ifdef MPI_VERSION
            if(rank==0)
#endif //MPI
            {
                std::cout << "# Absolute r*M*r "<<sqrt( dg::blas2::dot(M, m_r)) <<"\t ";
                std::cout << "#  < Critical "<<eps*nrmb + eps <<"\t ";
                std::cout << "# (Relative "<<sqrt( dg::blas2::dot(M, m_r) )/nrmb << ")\n";
            }
#endif //DG_DEBUG
            if( sqrt( dg::blas2::dot( M, m_r)) < eps*(nrmb + nrmb_correction)) //TODO change this criterium  for square root matrix
            {
                dg::blas2::symv(P, m_r, m_ap);
                nrm2r_new = dg::blas1::dot( m_ap, m_r);
                m_vp[i]   = -nrm2r_new/nrm2r_old/m_alpha;
                m_vm[i]   = -1./m_alpha;
                m_v0[i+1] = -m_vp[i];
                m_v0[i]  -= m_vm[i];
                set_iter(i+1);
                break;
            }
            dg::blas2::symv(P, m_r, m_ap);
            nrm2r_new = dg::blas1::dot( m_ap, m_r);
            dg::blas1::axpby(1., m_ap, nrm2r_new/nrm2r_old, m_p );
                       
            m_vp[i]   = -nrm2r_new/nrm2r_old/m_alpha;
            m_vm[i]   = -1./m_alpha;
            m_v0[i+1] = -m_vp[i];
            m_v0[i]  -= m_vm[i];
            nrm2r_old=nrm2r_new;
        }
        //Compute inverse of tridiagonal matrix
        m_Tinv = m_invtridiag(m_v0, m_vp, m_vm);

        dg::blas2::symv(m_Tinv, m_e1, m_y);  //  T^(-1) e_1   
        
#ifdef MPI_VERSION
        dg::blas2::symv(m_R, m_y, x.data());  // x =  R T^(-1) e_1  
#else
        dg::blas2::symv(m_R, m_y, x);  // x =  R T^(-1) e_1  
#endif
        m_TinvRpair = std::make_pair(m_Tinv, m_R);
        return m_TinvRpair;
    }
  private:
    value_type m_alpha;
    std::vector<value_type> m_v0, m_vp, m_vm;
    ContainerType m_r, m_ap, m_p;
    SubContainerType m_e1, m_y;
    unsigned m_max_iter, m_iter;
    std::pair<CooMatrix, CooMatrix> m_TinvRpair;
    CooMatrix m_R, m_Tinv;      
    dg::InvTridiag<SubContainerType, DiaMatrix, CooMatrix> m_invtridiag;
};

} //namespace dg

