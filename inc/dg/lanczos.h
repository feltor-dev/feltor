#pragma once

#include "blas.h"
#include "functors.h"



namespace dg{
    
/**
* @brief Functor class for computing the inverse of a general tridiagonal matrix 
*/
template< class ContainerType, class DiaMatrix, class CooMatrix>
class InvTridiag
{
  public:
    using container_type = ContainerType;
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
        Tinv = this->operator()(alpha,beta,gamma);
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
        unsigned counter = 0;
        for( unsigned i=0; i<a.size(); i++)
        {   
            for( unsigned j=0; j<a.size(); j++)
            {   
                Tinv.row_indices[counter]    = j;
                Tinv.column_indices[counter] = i; 
                temp=1.;
                if (i<j) {
                    for (unsigned k=i; k<j; k++) temp*=b[k];
                    Tinv.values[counter] =temp*pow(-1,i+j) * theta[i] * phi[j+1]/theta[a.size()];
                    
                }
                else if (i==j)
                {
                    Tinv.values[counter] =theta[i] * phi[j+1]/theta[a.size()];
                }   
                else // if (i>j)
                {
                    for (unsigned k=j; k<i; k++) temp*=c[k];           
                    Tinv.values[counter] =temp*pow(-1,i+j) * theta[j] * phi[i+1]/theta[a.size()];

                }
                counter++;
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
* @brief Functor class for the Lanczos method to solve \f[b = Ax\f] or \f[b = S^{-1} A x\f]
* for b. A is a symmetric and \f[S^{-1}\f] are typically the inverse weights.
*
* 
* @note The common lanczos method (and M-Lanczos) method are prone to loss of orthogonality for finite precision. Here, only the basic Paige fix is used. Thus the iterations should be kept as small as possible. Could be fixed via full, partial or selective reorthogonalization strategies, but so far no problems occured due to this.
*/
template< class ContainerType, class SubContainerType, class DiaMatrix, class CooMatrix>
class Lanczos
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
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
        m_alpha.assign(max_iterations,0.);
        m_beta.assign(max_iterations,0.);
        m_v.assign(max_iterations,copyable);
        m_w.assign(max_iterations,copyable);
        m_vi = m_vip = m_wi = m_wim = m_wip= copyable;
        m_max_iter = max_iterations;
        m_iter = max_iterations;
        m_T.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_T.diagonal_offsets[0] = -1;
        m_T.diagonal_offsets[1] =  0;
        m_T.diagonal_offsets[2] =  1;
        m_Tinv.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_V.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_e1.assign(m_max_iter, 0.);
        m_temp.assign(m_max_iter, 0.);
        m_e1[0]=1.;
    }
    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        m_T.resize(new_iter, new_iter, 3*new_iter-2, 3, m_max_iter);
        m_T.diagonal_offsets[0] = -1;
        m_T.diagonal_offsets[1] =  0;
        m_T.diagonal_offsets[2] =  1;
        m_V.resize(m_vi.size(), new_iter, new_iter*m_vi.size()); 
        m_e1.assign(new_iter, 0.);
        m_temp.assign(new_iter, 0.);
        m_e1[0]=1.;
        m_iter = new_iter;
    }
    ///@brief Get the current  number of iterations
    ///@return the current number of iterations
    unsigned get_iter() const {return m_iter;}
    /**
     * @brief Solve the system \f[ b= A x \approx || x ||_2 V T e_1\f] using Lanczos method. Useful for tridiagonalization of A (cf return statement).
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains an initial value
     * @param b The right hand side vector. 
     * 
     * @return returns the tridiagonal matrix T and orthonormal basis vectors contained in the matrix V matrix. Note that  \f[ T = V^T A V \f].
      */
    template< class MatrixType, class ContainerType0, class ContainerType1>
    std::pair<DiaMatrix, CooMatrix> operator()( MatrixType& A, const ContainerType0& x, ContainerType1& b)
    {
        get_value_type<ContainerType> xnorm = sqrt(dg::blas1::dot(x, x));

        //Initial vector
        dg::blas1::axpby(1./xnorm, x, 0.0, m_v[0]); //m_v[1] = x/||x||
        m_beta[0] = 0.;

        //Algorithm for i=1
        dg::blas2::symv(A, m_v[0], m_v[1]);  
        m_alpha[0] = dg::blas1::dot(m_v[1], m_v[0]);      
        dg::blas1::axpby(-m_alpha[0], m_v[0], 1.0, m_v[1]);            
        m_beta[1] = sqrt(dg::blas1::dot(m_v[1], m_v[1]));  

        dg::blas1::scal(m_v[1], 1./m_beta[1]);
        //Algorithm for i>1
        for( unsigned i=1; i<m_max_iter-1; i++)
        {
            dg::blas2::symv(A, m_v[i], m_v[i+1]);                    
            dg::blas1::axpby(-m_beta[i], m_v[i-1], 1.0, m_v[i+1]);     
            m_alpha[i] = dg::blas1::dot(m_v[i+1], m_v[i]);            
            dg::blas1::axpby(-m_alpha[i], m_v[i], 1.0, m_v[i+1]);      
            m_beta[i+1] = sqrt(dg::blas1::dot(m_v[i+1], m_v[i+1]));     
    //         if (m_beta[i+1] == 0) break;
//             std::cout << m_beta[i+1]  << "\n";
            dg::blas1::scal(m_v[i+1], 1./m_beta[i+1]);  
            
        }
        //Last m_alpha
        dg::blas2::symv(A, m_v[m_max_iter-1], m_vi);

        dg::blas1::axpby(-m_beta[m_max_iter-1], m_v[m_max_iter-2], 1.0, m_vi); 
        m_alpha[m_max_iter-1] = dg::blas1::dot(m_vi, m_v[m_max_iter-1]);
        
        //Fill T and V Matrix
        unsigned counter = 0;
        for( unsigned i=0; i<m_max_iter; i++)
        {
            m_T.values(i,0) =  m_beta[i];  // -1 diagonal
            m_T.values(i,1) =  m_alpha[i];  // 0 diagonal
            m_T.values(i,2) =  m_beta[i+1];  // +1 diagonal //dia_rows entry works since its outside of matrix
            
            for( unsigned j=0; j<m_v[0].size(); j++)
            {            
                m_V.row_indices[counter]    = j;
                m_V.column_indices[counter] = i; 
                m_V.values[counter]         = m_v[i][j];
                counter++;
            }
        }     
        
        //Check implementation: b=||x|| V T e_1 = || x || (m_v[0] m_alpha[01] + m_v[1] beta[1])
        dg::blas1::axpby(m_alpha[0], m_v[0], m_beta[1], m_v[1], b);
        dg::blas1::scal(b, xnorm ); 
        
        m_TVpair = std::make_pair(m_T, m_V);
        return m_TVpair;
    }
    /**
     * @brief Solve the system \f[b= S^{-1} A x \approx || x ||_S V T e_1\f] for b using M-Lanczos (in this case S-Lanczos) method. Useful for the fast computatin of matrix functions of \f[ S^{-1} A\f].
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains an initial value
     * @param b The right hand side vector. 
     * @param S the weights 
     * @param Sinv the inverse of S - the inverse weights
     * @param eps accuracy of residual
     * 
     * @return returns the tridiagonal matrix T and orthonormal basis vectors contained in the matrix V matrix. Note that  \[f T = V^T A V \f]
     */
    template< class MatrixType, class ContainerType0, class ContainerType1, class SquareNorm1, class SquareNorm2>
    std::pair<DiaMatrix, CooMatrix> operator()( MatrixType& A, const ContainerType0& x, ContainerType1& b,  SquareNorm1& S, SquareNorm2& Sinv, value_type eps)
    {
        value_type xnorm = sqrt(dg::blas2::dot(x, S, x));
        value_type residual;
#ifdef DG_BENCHMARK
        Timer t;
        value_type invtime=0.;
#endif //DG_BENCHMARK
   /*//     Implementation #1 (naiv implementation, however slightly faster than 2 and 3)
        dg::blas1::axpby(1./xnorm, x, 0.0, m_v[0]); //m_v[1] = x/||x||
        m_beta[0] = 0.;
        dg::blas2::symv(A, m_v[0], m_vi);  
        dg::blas2::symv(Sinv, m_vi, m_v[1]);
        m_alpha[0] = dg::blas2::dot(m_v[1], S, m_v[0]);      
        dg::blas1::axpby(-m_alpha[0], m_v[0], 1.0, m_v[1]);            
        m_beta[1] = sqrt(dg::blas2::dot( S, m_v[1]));   
        dg::blas1::scal(m_v[1], 1./m_beta[1]);
 
        for( unsigned i=1; i<m_max_iter-1; i++)
        {
            dg::blas2::symv(A, m_v[i],m_vi);                    
            dg::blas2::symv(Sinv,m_vi,m_v[i+1]);                
            dg::blas1::axpby(-m_beta[i], m_v[i-1], 1.0, m_v[i+1]);   
            m_alpha[i] = dg::blas2::dot(m_v[i+1], S, m_v[i]);          
            dg::blas1::axpby(-m_alpha[i], m_v[i], 1.0, m_v[i+1]);      
            m_beta[i+1] = sqrt(dg::blas2::dot(S, m_v[i+1]));       
            dg::blas1::scal(m_v[i+1], 1./m_beta[i+1]);       
            
        }
        dg::blas2::symv(A, m_v[m_max_iter-1], m_vi);
        dg::blas2::symv(Sinv,m_vi, m_vi); 
        dg::blas1::axpby(-m_beta[m_max_iter-1], m_v[m_max_iter-2], 1.0, m_vi);
        m_alpha[m_max_iter-1] = dg::blas2::dot(m_vi, S, m_v[m_max_iter-1]);

//         //Implementation #2 (with w and blas1 dots)
        dg::blas1::axpby(1./xnorm, x, 0.0, m_v[0]); //m_v[1] = x/||x||
        m_beta[0] = 0.;
        dg::blas2::symv(S, m_v[0], m_w[0]);
//         dg::blas2::symv(A, m_v[0], m_w[1]); 
//         m_alpha[0] = dg::blas1::dot(m_w[1], m_v[0]); 
//         dg::blas1::axpby(-m_alpha[0], m_w[0], 1.0, m_w[1]);   
//         dg::blas2::symv(Sinv,m_w[1],m_v[1]);
//         m_beta[1] = sqrt(dg::blas1::dot( m_w[1], m_v[1]));  
//         dg::blas1::scal(m_v[1], 1./m_beta[1]);              
//         dg::blas1::scal(m_w[1], 1./m_beta[1]);

        for( unsigned i=0; i<m_max_iter-1; i++)
        {    
            dg::blas2::symv(A, m_v[i], m_w[i+1]);                    
            if (i>0) dg::blas1::axpby(-m_beta[i], m_w[i-1], 1.0, m_w[i+1]);    
            m_alpha[i] = dg::blas1::dot(m_w[i+1], m_v[i]);          
            dg::blas1::axpby(-m_alpha[i], m_w[i], 1.0, m_w[i+1]);       
            dg::blas2::symv(Sinv,m_w[i+1],m_v[i+1]);
            m_beta[i+1] = sqrt(dg::blas1::dot(m_w[i+1], m_v[i+1]));       
            dg::blas1::scal(m_v[i+1], 1./m_beta[i+1]);              
            dg::blas1::scal(m_w[i+1], 1./m_beta[i+1]);  
            
            //TODO convergence criterium for sqrt operator
        }
        dg::blas2::symv(A, m_v[m_max_iter-1], m_vi);                    
        dg::blas1::axpby(-m_beta[m_max_iter-1], m_w[m_max_iter-2], 1.0, m_vi);    
        m_alpha[m_max_iter-1] = dg::blas1::dot(m_vi, m_v[m_max_iter-1]);   
        Fill T and V Matrix for #1 and #2
       unsigned counter = 0;
        for( unsigned i=0; i<m_max_iter; i++)
        {
            m_T.values(i,0) =  m_beta[i];  // -1 diagonal
            m_T.values(i,1) =  m_alpha[i];  // 0 diagonal
            m_T.values(i,2) =  m_beta[i+1];  // +1 diagonal //dia_rows entry works since its outside of matrix
            
            for( unsigned j=0; j<m_v[0].size(); j++)
            {            
                V.row_indices[counter]    = j;
                V.column_indices[counter] = i; 
                V.values[counter]         = m_v[i][j];
                counter++;
            }
        } */      
        //Implementation #3 (with w and blas1 dots and without vectors)
        dg::blas1::axpby(1./xnorm, x, 0.0, m_vi); //m_v[1] = x/||x||
        value_type betaip = 0;
        dg::blas2::symv(S, m_vi, m_wi);
        unsigned counter = 0;
        for( unsigned i=0; i<m_max_iter; i++)
        {  
            for( unsigned j=0; j<m_v[0].size(); j++)
            {            
                m_V.row_indices[counter]    = j;
                m_V.column_indices[counter] = i; 
                m_V.values[counter]         = m_vi.data()[j];
                counter++;
            }
            m_T.values(i,0) =  betaip;  // -1 diagonal
            dg::blas2::symv(A, m_vi, m_wip); 
            dg::blas1::axpby(-betaip, m_wim, 1.0, m_wip);    //only -= if i>0, therefore no if (i>0)
            m_T.values(i,1) = dg::blas1::dot(m_wip, m_vi);    
            dg::blas1::axpby(-m_T.values(i,1), m_wi, 1.0, m_wip);     
            dg::blas2::symv(Sinv,m_wip,m_vip);
            betaip = sqrt(dg::blas1::dot(m_wip, m_vip)); 
            if (betaip == 0) {
#ifdef DG_DEBUG
                std::cout << "beta[i+1]=0 encountered\n";
#endif //DG_DEBUG
                set_iter(i+1); 
                break;
            } 
            dg::blas1::scal(m_vip, 1./betaip);     
            dg::blas1::scal(m_wip, 1./betaip);  
            m_vi=m_vip;
            m_wim=m_wi;
            m_wi=m_wip;
            m_T.values(i,2) =  betaip;  // +1 diagonal
            if (i>0) {
                m_invtridiag.resize(i);
#ifdef DG_BENCHMARK
                t.tic();
#endif //DG_BENCHMARK
                m_Tinv = m_invtridiag(m_T); //Compute inverse of T //TODO is slow - criterium without inversion ? 
#ifdef DG_BENCHMARK
                t.toc();
                invtime+=t.diff();
#endif //DG_BENCHMARK
                residual = xnorm*betaip*betaip*abs(m_Tinv.values[i-1]);
#ifdef DG_DEBUG
                std::cout << "||r||_S =  " << residual << "  # of iterations = " << i+1 << "\n";
#endif //DG_DEBUG
                if (residual< eps ) {
                    set_iter(i+1); //update iteration number and resize matrix V and T
                    break;
                }
            }
        }   
#ifdef DG_BENCHMARK
        std::cout << get_iter() << " T inversions took " << invtime << "s\n";
#endif //DG_BENCHMARK

//         Check implementation: b=||x|| V T e_1 = || x || (m_v[0] m_alpha[01] + m_v[1] m_beta[1])
//         Note that it depends only on first two vectors and alpha,beta (changes if square root is applied upon T)
//         dg::blas1::axpby(m_alpha[0], m_v[0], m_beta[1], m_v[1], b); //faster alternative

        dg::blas2::symv(m_T, m_e1, m_temp); //T e_1
        dg::blas2::symv(m_V, m_temp, b); // V T e_1
        dg::blas1::scal(b, xnorm ); 
        m_TVpair = std::make_pair(m_T,m_V);
        return m_TVpair;
    }
  private:
    std::vector<value_type> m_alpha, m_beta;
    std::vector<ContainerType> m_v, m_w;
    ContainerType  m_vi, m_vip, m_wi, m_wip, m_wim;
    SubContainerType m_e1, m_temp;
    unsigned m_iter, m_max_iter;
    std::pair<DiaMatrix, CooMatrix> m_TVpair; 
    DiaMatrix m_T;
    CooMatrix m_V, m_Tinv;    
    InvTridiag<SubContainerType, DiaMatrix, CooMatrix> m_invtridiag;

};

/*! 
 * @brief Shortcut for \f[x \approx \sqrt{A}^{-1} b  \f] solve via exploiting first a Krylov projection achieved by the PCG method and and secondly a sqrt ODE solve with the adaptive ERK class as timestepper. 
 * 
 * @note The approximation relies on Projection \f[x = \sqrt{A}^{-1} b  \approx  R \sqrt{T^{-1}} e_1\f], where \f[T\f] and \f[V\f] is the tridiagonal and orthogonal matrix of the PCG solve and \f[e_1\f] is the normalized unit vector. The vector \f[\sqrt{T^{-1}} e_1\f] is computed via the sqrt ODE solve.
 */
template< class ContainerType, class SubContainerType, class DiaMatrix, class CooMatrix>
class CGtridiag
{
  public:
    using container_type = ContainerType;
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
        m_ap = m_p = m_r = m_rh = copyable;
        m_max_iter = max_iterations;
        m_iter = max_iterations;
        m_R.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_Tinv.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
    }
    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        m_v0.resize(new_iter);
        m_vp.resize(new_iter);
        m_vm.resize(new_iter);
        m_e1.assign(new_iter,0.);
        m_e1[0]=1.;
        m_y.assign(new_iter,0.);
        m_R.resize(m_rh.size(), new_iter, new_iter*m_rh.size());
        m_Tinv.resize(new_iter, new_iter, new_iter*new_iter);
        m_invtridiag.resize(new_iter);
        m_iter = new_iter;

    }
    ///@brief Get the current  number of iterations
    ///@return the current number of iterations
    unsigned get_iter() const {return m_iter;}
    /**
     * @brief Solve the system \f[\sqrt{A}*x = b \f] for x using PCG method and sqrt ODE solve
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains an initial value
     * @param b The right hand side vector. 
     * @param P The preconditioner to be used
     * @param S (Inverse) Weights used to compute the norm for the error condition
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     * 
     * @return Number of iterations used to achieve desired precision
     * @note So far only ordinary convergence criterium of CG method. Should be adapted to square root criterium.
      */
    template< class MatrixType, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm>
    std::pair<CooMatrix, CooMatrix> operator()( MatrixType& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps = 1e-12, value_type nrmb_correction = 1)
    {
        //Do CG iteration do get R and T matrix
        value_type nrmb = sqrt( dg::blas2::dot( S, b));
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
            return m_TinvRpair;
        }
        dg::blas2::symv( A, x, m_r);
        dg::blas1::axpby( 1., b, -1., m_r);
        dg::blas2::symv( P, m_r, m_p );//<-- compute p_0
        dg::blas1::copy( m_p, m_rh);

        //note that dot does automatically synchronize
        value_type nrm2r_old = dg::blas1::dot( m_rh, m_r); //and store the norm of it
        value_type nrm2r_new;
        
        unsigned counter = 0;
        for( unsigned i=0; i<m_max_iter; i++)
        {

            for( unsigned j=0; j<m_rh.size(); j++)
            {            
                m_R.row_indices[counter]    = j;
                m_R.column_indices[counter] = i; 
                m_R.values[counter]         = m_rh.data()[j];
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
                std::cout << "# Absolute r*S*r "<<sqrt( dg::blas2::dot(S,m_r)) <<"\t ";
                std::cout << "#  < Critical "<<eps*nrmb + eps <<"\t ";
                std::cout << "# (Relative "<<sqrt( dg::blas2::dot(S,m_r) )/nrmb << ")\n";
            }
    #endif //DG_DEBUG
            if( sqrt( dg::blas2::dot( S, m_r)) < eps*(nrmb + nrmb_correction)) //TODO change this criterium  for square root matrix
            {
                dg::blas2::symv(P, m_r, m_rh);
                nrm2r_new = dg::blas1::dot( m_rh, m_r);
                m_vp[i] = -nrm2r_new/nrm2r_old/m_alpha;
                m_vm[i] = -1./m_alpha;
                m_v0[i+1] = -m_vp[i];
                m_v0[i] -= m_vm[i];
                
                m_max_iter=i+1;
                break;
            }
            dg::blas2::symv(P, m_r, m_rh);
            nrm2r_new = dg::blas1::dot( m_rh, m_r);
            dg::blas1::axpby(1., m_rh, nrm2r_new/nrm2r_old, m_p );
                       
            m_vp[i] = -nrm2r_new/nrm2r_old/m_alpha;
            m_vm[i] = -1./m_alpha;
            m_v0[i+1] = -m_vp[i];
            m_v0[i] -= m_vm[i];
            nrm2r_old=nrm2r_new;
        }

        set_iter(m_max_iter);
        //Compute inverse of tridiagonal matrix
        m_Tinv = m_invtridiag(m_v0,m_vp,m_vm);

        dg::blas2::symv(m_Tinv, m_e1, m_y);  //  T^(-1) e_1   
        dg::blas2::symv(m_R, m_y, x);  // x =  R T^(-1) e_1   
        m_TinvRpair = std::make_pair(m_Tinv, m_R);
        return m_TinvRpair;
    }
  private:
    value_type m_alpha;
    std::vector<value_type> m_v0, m_vp, m_vm;
    ContainerType m_r, m_ap, m_p, m_rh;
    SubContainerType m_e1, m_y;
    unsigned m_max_iter, m_iter;
    std::pair<CooMatrix, CooMatrix> m_TinvRpair;
    CooMatrix m_R, m_Tinv;      
    dg::InvTridiag<SubContainerType, DiaMatrix, CooMatrix> m_invtridiag;
};

} //namespace dg

