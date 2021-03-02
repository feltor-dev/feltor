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
        m_size = copyable.size();
        phi.assign(m_size+1,0.);
        theta.assign(m_size+1,0.);
        Tinv.resize(m_size, m_size,  m_size* m_size);
        temp = 0.;
    }
    /**
     * @brief Resize inverse tridiagonal matrix and helper vectors
     * 
     * @param new_size new size of square matrix
    */
    void resize(unsigned new_size) {
        m_size = new_size;
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
        ContainerType alpha(m_size,0.);
        ContainerType beta(m_size,0.);
        ContainerType gamma(m_size,0.);
        for(unsigned i = 0; i<m_size; i++)
        {
            alpha[i] = T.values(i,1);    // 0 diagonal
            beta[i]  = T.values(i,2)  ;  // +1 diagonal //dia_rows entry works since its outside of matrix
            gamma[i] = T.values(i+1,0);  // -1 diagonal            
        }
        Tinv = this->operator()(alpha, beta, gamma);
        return Tinv;
    }
//     value_type theta_func(unsigned i, const ContainerType& a, const ContainerType& b,  const ContainerType& c)
//     {
//         if (i==0) return 1.;
//         else if (i==1) return a[0];
//         else a[i-1] * theta_func(i-1, a,b,c) - b[i-2] * c[i-2] * theta_func(i-2, a,b,c);
//     }
//     value_type phi_func(unsigned i, const ContainerType& a, const ContainerType& b,  const ContainerType& c)
//     {
//         if (i==m_size) return 1.;
//         else if (i==m_size - 1) return a[m_size-1];
//         else a[i]  * phi_func(i+1, a,b,c)  - b[i]  * c[i]  * phi_func(i+2, a,b,c);
//     }
     /**
     * @brief Compute the inverse of a tridiagonal matrix with diagonal vectors a,b,c
     * 
     * @param a  "0" diagonal vector
     * @param b "+1" diagonal vector (starts with index 0 to (size of b)-1)
     * @param c "-1" diagonal vector (starts with index 0 to (size of c)-1)
     * 
     * @return the inverse of the tridiagonal matrix (coordinate format)
     */
    CooMatrix operator()(const ContainerType& a, const ContainerType& b,  const ContainerType& c)
    {
        unsigned is=0;
        for( unsigned i = 0; i<m_size+1; i++)
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
//             std::cout << theta[i] << " " << phi[is] << "\n";
//             std::cout << theta_func(i, a,b,c) << " " << phi_func(is, a,b,c) << "\n";
        }
        //Compute inverse tridiagonal matrix elements
        for( unsigned j=0; j<m_size; j++)
        {   
            for( unsigned i=0; i<m_size; i++)
            {   
                Tinv.row_indices[i*m_size+j]    = i;
                Tinv.column_indices[i*m_size+j] = j; 
                if (i<=j) {
                    temp = std::accumulate(std::next(b.begin(),i), std::next(b.begin(),j), 1., std::multiplies<value_type>());
                    Tinv.values[i*m_size+j] =temp*sign(i+j) * theta[i] * phi[j+1]/theta[m_size];

                }   
                else // if (i>j)
                {
                    temp = std::accumulate(std::next(c.begin(),j), std::next(c.begin(),i), 1., std::multiplies<value_type>());
                    Tinv.values[i*m_size+j] =temp*sign(i+j) * theta[j] * phi[i+1]/theta[m_size];
                }
            }
        }
//         //Compute inverse tridiagonal matrix elements
//         for( unsigned j=0; j<m_size; j++)
//         {   
//                 std::cout << theta_func(j, a,b,c) << "  " << phi_func(j, a,b,c) << "\n";
// 
//             for( unsigned i=0; i<m_size; i++)
//             {   
//                 Tinv.row_indices[i*m_size+j]    = i;
//                 Tinv.column_indices[i*m_size+j] = j; 
//                 if (i<=j) {
//                     temp = std::accumulate(std::next(b.begin(),i), std::next(b.begin(),j), 1., std::multiplies<value_type>());
//                     Tinv.values[i*m_size+j] =temp*sign(i+j) * theta_func(i, a,b,c) * phi_func(j+1, a,b,c)/theta_func(m_size, a,b,c);
// 
//                 }   
//                 else // if (i>j)
//                 {
//                     temp = std::accumulate(std::next(c.begin(),j), std::next(c.begin(),i), 1., std::multiplies<value_type>());
//                     Tinv.values[i*m_size+j] =temp*sign(i+j) * theta_func(j, a,b,c) * phi_func(i+1, a,b,c)/theta_func(m_size, a,b,c);
//                 }
//             }
//         }
        return Tinv;
    }
  private:
    std::vector<value_type> phi, theta;
    CooMatrix Tinv;    
    value_type temp;
    unsigned m_size;
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
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HVec = dg::HVec;
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
        m_v = m_vp = m_w = m_wm = m_wp = copyable;
        m_max_iter = max_iterations;
        m_iter = max_iterations;
        //sub matrix and vector
        m_TH.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_TH.diagonal_offsets[0] = -1;
        m_TH.diagonal_offsets[1] =  0;
        m_TH.diagonal_offsets[2] =  1;
        m_TinvH.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
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
        dg::blas1::axpby(1./xnorm, x, 0.0, m_v); //m_v[1] = x/||x||
        dg::blas1::scal(b, 0.);
        for ( unsigned i=0; i<iter; i++)
        {
            dg::blas1::axpby( y[i], m_v, 1., b); //Compute b= V y

            dg::blas2::symv( A, m_v, m_vp);                    
            dg::blas1::axpbypgz(-T.values(i,0), m_wm, -T.values(i,1), m_v, 1.0, m_vp);  
            dg::blas1::scal(m_vp, 1./T.values(i,2));  
            m_wm = m_v;
            m_v = m_vp;
        }
        dg::blas1::scal(b, xnorm ); 
    }
    /** @brief compute b = |x|_M V y from a given tridiagonal matrix T 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param T Tridiagonal matrix (cusp::dia_matrix format)
     * @param Minv the inverse of M - the inverse weights
     * @param M the weights 
     * @param y a vector e.g y= T e_1 or y= f(T) e_1
     * @param x Contains an initial value of lanczos method
     * @param b The right hand side vector (output)
     * @param xnorm M-norm of x
     * @param iter size of tridiagonal matrix
     */
    template< class MatrixType, class DiaMatrixType, class SquareNorm1, class SquareNorm2, class ContainerType0, class ContainerType1,class ContainerType2>
    void normMxVy( MatrixType& A, DiaMatrixType& T, SquareNorm1& Minv, SquareNorm2& M,  ContainerType0& y, ContainerType1& b, ContainerType2& x, value_type xnorm,  unsigned iter)
    {
        dg::blas1::axpby(1./xnorm, x, 0.0, m_v); //m_v[1] = x/||x||
        dg::blas2::symv(M, m_v, m_w);
        dg::blas1::scal(b, 0.);
        for( unsigned i=0; i<iter; i++)
        {
            dg::blas1::axpby( y[i], m_v, 1., b); //Compute b= V y

            dg::blas2::symv(A, m_v, m_wp); 
            dg::blas1::axpbypgz(-T.values(i,0), m_wm, -T.values(i,1), m_w,  1.0, m_wp);
            dg::blas2::symv(Minv, m_wp, m_vp);
            dg::blas1::scal(m_vp, 1./T.values(i,2));
            dg::blas1::scal(m_wp, 1./T.values(i,2));
            m_v  = m_vp;
            m_wm = m_w;
            m_w  = m_wp;
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
     * @param compute_b Specify if \f$b = || x ||_2 V T e_1 \f$ should be computed or if only T should be computed
     * 
     * @return returns the tridiagonal matrix T. Note that  \f$ T = V^T A V \f$.
      */
    template< class MatrixType, class ContainerType0, class ContainerType1>
    DiaMatrix operator()( MatrixType& A, const ContainerType0& x, ContainerType1& b, value_type eps, bool compute_b = false)
    {
        value_type xnorm = sqrt(dg::blas1::dot(x, x));
        value_type residual;
#ifdef DG_BENCHMARK
        dg::Timer t;
        value_type invtime=0.;
#endif //DG_BENCHMARK

        dg::blas1::axpby(1./xnorm, x, 0.0, m_v); //m_v[1] = x/||x||
        value_type betaip = 0.;
        value_type alphai = 0.;
        for( unsigned i=0; i<m_max_iter; i++)
        {
            m_TH.values(i,0) =  betaip; // -1 diagonal            
            dg::blas2::symv(A, m_v, m_vp);                    
            dg::blas1::axpby(-betaip, m_wm, 1.0, m_vp);  // only - if i>0, therefore no if (i>0)  
            alphai  = dg::blas1::dot(m_vp, m_v);
            m_TH.values(i,1) = alphai;
            dg::blas1::axpby(-alphai, m_v, 1.0, m_vp);      
            betaip = sqrt(dg::blas1::dot(m_vp, m_vp));     
            if (betaip == 0) {
#ifdef DG_DEBUG
                std::cout << "beta["<<i+1 <<"]=0 encountered\n";
#endif //DG_DEBUG
                set_iter(i+1); 
                break;
            } 
            dg::blas1::scal(m_vp, 1./betaip);  
            m_wm = m_v; //swap instead? wim stands for vim here
            m_v = m_vp;
            m_TH.values(i,2) = betaip;  // +1 diagonal

            if (i>0) {
#ifdef DG_BENCHMARK
                t.tic();
#endif //DG_BENCHMARK                
                m_invtridiagH.resize(i);
                m_TinvH = m_invtridiagH(m_TH); //TODO slow -> criterium without inversion ? 
#ifdef DG_BENCHMARK
                t.toc();
                invtime+=t.diff();
#endif //DG_BENCHMARK
                residual = xnorm*betaip*abs(m_TinvH.values[i-1]);
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
        if (compute_b == true)
        {
            SubContainerType e1(get_iter(), 0.), y(e1);
            e1[0] = 1.;
            dg::blas2::symv (m_T, e1, y); //y= T e_1
            norm2xVy(A, m_T, y, b, x, xnorm, get_iter()); //b= |x| V T e_1 
        }
        return m_T;
    }
    /**
     * @brief Solve the system \f$b= M^{-1} A x \approx || x ||_M V T e_1\f$ for b using M-Lanczos method. Useful for the fast computatin of matrix functions of \f$ M^{-1} A\f$.
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains an initial value
     * @param b The right hand side vector. 
     * @param Minv the inverse of M - the inverse weights
     * @param M the weights 
     * @param eps accuracy of residual
     * @param compute_b Specify if \f$b = || x ||_M  V T e_1 \f$ should be computed or if only T should be computed
     * @return returns the tridiagonal matrix T and orthonormal basis vectors contained in the matrix V matrix. Note that  \[f T = V^T A V \f$
     */
    template< class MatrixType, class ContainerType0, class ContainerType1, class SquareNorm1, class SquareNorm2>
    DiaMatrix operator()( MatrixType& A, const ContainerType0& x, ContainerType1& b,  SquareNorm1& Minv, SquareNorm2& M,  value_type eps, bool compute_b = false)
    {
        value_type xnorm = sqrt(dg::blas2::dot(x, M, x));
        value_type residual;
#ifdef DG_BENCHMARK
        dg::Timer t;
        value_type invtime=0.;

#endif //DG_BENCHMARK
        
        dg::blas1::axpby(1./xnorm, x, 0.0, m_v); //m_v[1] = x/||x||
        value_type betaip = 0.;
        value_type alphai = 0.;
        dg::blas2::symv(M, m_v, m_w);
        
        for( unsigned i=0; i<m_max_iter; i++)
        { 
            m_TH.values(i,0) =  betaip;  // -1 diagonal
            dg::blas2::symv(A, m_v, m_wp); 
            dg::blas1::axpby(-betaip, m_wm, 1.0, m_wp);    //only - if i>0, therefore no if (i>0)
            alphai = dg::blas1::dot(m_wp, m_v);  
            m_TH.values(i,1) = alphai;
            dg::blas1::axpby(-alphai, m_w, 1.0, m_wp);     
            dg::blas2::symv(Minv,m_wp,m_vp);
            betaip = sqrt(dg::blas1::dot(m_wp, m_vp)); 
            if (betaip == 0) {
#ifdef DG_DEBUG
                std::cout << "beta["<<i+1 <<"]=0 encountered\n";
#endif //DG_DEBUG
                set_iter(i+1); 
                break;
            } 
            dg::blas1::scal(m_vp, 1./betaip);     
            dg::blas1::scal(m_wp, 1./betaip);  
            m_v  = m_vp;
            m_wm = m_w;
            m_w  = m_wp;
            m_TH.values(i,2) =  betaip;  // +1 diagonal
            if (i>0) {
#ifdef DG_BENCHMARK
                t.tic();
#endif //DG_BENCHMARK
                m_invtridiagH.resize(i);
                m_TinvH = m_invtridiagH(m_TH); //TODO slow -> criterium without inversion ? 
#ifdef DG_BENCHMARK
                t.toc();
                invtime+=t.diff();
#endif //DG_BENCHMARK
                residual = xnorm*betaip*abs(m_TinvH.values[i-1]);
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
        if (compute_b == true)
        {
            SubContainerType e1(get_iter(), 0.), y(e1);
            e1[0] = 1.;
            dg::blas2::symv(m_T, e1, y); //T e_1
            normMxVy(A, m_T, Minv, M, y, b, x, xnorm, get_iter()); //b= |x| V T e_1 
        }
        return m_T;
    }
  private:
    ContainerType  m_v, m_vp, m_w, m_wp, m_wm;
    HDiaMatrix m_TH;
    HCooMatrix m_TinvH;
    unsigned m_iter, m_max_iter;
    DiaMatrix m_T;  
    InvTridiag<HVec, HDiaMatrix, HCooMatrix> m_invtridiagH;
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
class MCG
{
  public:
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HVec = dg::HVec;
    ///@brief Allocate nothing, Call \c construct method before usage
    MCG(){}
    ///@copydoc construct()
    MCG( const ContainerType& copyable, unsigned max_iterations)
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
        m_ap = m_p = m_r = copyable;
        m_max_iter = max_iterations;
        m_TH.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_TH.diagonal_offsets[0] = -1;
        m_TH.diagonal_offsets[1] =  0;
        m_TH.diagonal_offsets[2] =  1;
        m_iter = max_iterations;
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
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param T T non-symmetric tridiagonal Matrix from MCG tridiagonalization     
     * @param Minv The inverse weights
     * @param M Weights used to compute the norm for the error condition
     * @param y vector of with v.size() = iter. Typically \f$ T^(-1) e_1 \f$ or \f$ f(T^(-1)) e_1 \f$ 
     * @param x Contains the initial value (\f$x!=0\f$ if used for tridiagonalization) and the matrix approximation \f$x = A^{-1} b\f$ as output
     * @param b The right hand side vector. 
     * @param iter number of iterations (size of T)
     */
    template< class MatrixType, class DiaMatrixType, class SquareNorm1, class SquareNorm2, class ContainerType0, class ContainerType1, class ContainerType2>
    void Ry( MatrixType& A, DiaMatrixType& T,  SquareNorm1& Minv, SquareNorm2& M, ContainerType0& y, ContainerType1& x, ContainerType2& b,  unsigned iter)
    {
        dg::blas2::symv( A, x, m_r);
        dg::blas1::axpby( 1., b, -1., m_r);

        dg::blas2::symv( Minv, m_r, m_p );
        dg::blas1::copy( m_p, m_ap);
        dg::blas1::scal(x, 0.); //could be removed if x is correctly initialized

        for ( unsigned i=0; i<iter; i++)
        {
            dg::blas1::axpby( y[i], m_ap, 1.,x); //Compute b= R y
            dg::blas2::symv( A, m_p, m_ap);
            dg::blas1::axpby( 1./T.values(i,0) , m_ap, 1., m_r);
            dg::blas2::symv(Minv, m_r, m_ap);
            dg::blas1::axpby(1., m_ap, T.values(i,2)/T.values(i,0) , m_p );
        }
    }
    /**
     * @brief Solve the system \f$A*x = b \f$ for x using PCG method 
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains the initial value (\f$x!=0\f$ if used for tridiagonalization) and the matrix approximation \f$x = A^{-1} b\f$ as output
     * @param b The right hand side vector. 
     * @param Minv The preconditioner to be used
     * @param M (Inverse) Weights used to compute the norm for the error condition
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     * @param compute_x Specify if \f$x = R T^{-1} e_1 \f$ should be computed or if only T should be computed 
     * @return Number of iterations used to achieve desired precision
     * @note So far only ordinary convergence criterium of CG method. Should be adapted to square root criterium or any other matrix function. 
     * Note that the method is similar to the PCG method with  preconditioner \f$P = M^{-1} \f$. The Matrix R and T of the tridiagonalization are further used for computing matrix functions. The x vector must be initialized with 0 if used for tridiagonalization.
     * 
     * 
      */
    template< class MatrixType, class ContainerType0, class ContainerType1, class SquareNorm1, class SquareNorm2>
    DiaMatrix operator()( MatrixType& A, ContainerType0& x, const ContainerType1& b, SquareNorm1& Minv, SquareNorm2& M, value_type eps = 1e-12, value_type nrmb_correction = 1, bool compute_x = false)
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
            return m_T;
        }
        dg::blas2::symv( A, x, m_r);
        dg::blas1::axpby( 1., b, -1., m_r);
        
        if( sqrt( dg::blas2::dot( M, m_r)) < eps*(nrmb + nrmb_correction)) //TODO replace with appropriate criterium
        {
            set_iter(1);
            return m_T;
        }
        dg::blas2::symv( Minv, m_r, m_p );
        dg::blas1::copy( m_p, m_ap);

        value_type nrm2r_old = dg::blas1::dot( m_ap, m_r);
        value_type nrm2r_new;
        value_type alpha;
        for( unsigned i=0; i<m_max_iter; i++)
        {
            dg::blas2::symv( A, m_p, m_ap);
            alpha = nrm2r_old /dg::blas1::dot( m_p, m_ap);
            dg::blas1::axpby( -alpha, m_ap, 1., m_r);
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
                dg::blas2::symv(Minv, m_r, m_ap);
                nrm2r_new = dg::blas1::dot( m_ap, m_r);
                m_TH.values(i,2)   = -nrm2r_new/nrm2r_old/alpha;
                m_TH.values(i,0)   = -1./alpha;
                m_TH.values(i+1,1) = -m_TH.values(i,2);
                m_TH.values(i,1)  -= m_TH.values(i,0);
                set_iter(i+1);
                break;
            }
            dg::blas2::symv(Minv, m_r, m_ap);
            nrm2r_new = dg::blas1::dot( m_ap, m_r);
            dg::blas1::axpby(1., m_ap, nrm2r_new/nrm2r_old, m_p );
                       
            m_TH.values(i,2)   = -nrm2r_new/nrm2r_old/alpha;
            m_TH.values(i,0)   = -1./alpha;
            m_TH.values(i+1,1) = -m_TH.values(i,2);
            m_TH.values(i,1)  -= m_TH.values(i,0);
            nrm2r_old=nrm2r_new;
        }
        //Compute inverse of tridiagonal matrix
        m_T = m_TH;
        if (compute_x == true)
        {
            SubContainerType e1(get_iter(), 0.), y(e1);
            e1[0] = 1.;
            dg::InvTridiag<SubContainerType, DiaMatrix, CooMatrix> invtridiag;
            invtridiag.resize(get_iter());
            CooMatrix Tinv = invtridiag(m_T);
            dg::blas2::symv(Tinv, e1, y);  // m_y= T^(-1) e_1   
            Ry(A, m_T, Minv, M, y, x, b,  get_iter());  // x =  R T^(-1) e_1  
        }
        return m_T;
    }
  private:
    ContainerType m_r, m_ap, m_p;
    unsigned m_max_iter, m_iter;
    HDiaMatrix m_TH;
    DiaMatrix m_T;
};

} //namespace dg

