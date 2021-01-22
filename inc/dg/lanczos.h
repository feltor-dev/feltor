// #include <cmath>
#pragma once

#include "blas.h"
#include "functors.h"

#include <cusp/dia_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>

namespace dg{
    
/**
* @brief Functor class for computing the inverse of a general tridiagonal matrix 
*/
template< class ContainerType>
class InvTridiag
{
  public:
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    using coo_type =  cusp::coo_matrix<int, value_type, cusp::device_memory>;
    using dia_type =  cusp::dia_matrix<int, value_type, cusp::device_memory>;
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
    coo_type operator()(const dia_type& T)
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
    coo_type operator()(const ContainerType0& a, const ContainerType0& b,  const ContainerType0& c)
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
    coo_type Tinv;    
    value_type temp;
};
/**
* @brief Functor class for the Lanczos method to solve
* \f[ Ax=b\f]
* for b. A is a symmetric 
*
* 
* @note The common lanczos method (and M-Lanczos) method are prone to loss of orthogonality for finite precision. Here, only the basic Paige fix is used. Thus the iterations should be kept as small as possible. Could be fixed via full, partial or selective reorthogonalization strategies, but so far no problems occured due to this.
*/
template< class ContainerType>
class Lanczos
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    using dia_type =  cusp::dia_matrix<int, value_type, cusp::device_memory>;
    using coo_type =  cusp::coo_matrix<int, value_type, cusp::device_memory>;
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
        alpha.assign(max_iterations,0.);
        beta.assign(max_iterations,0.);
        v.assign(max_iterations,copyable);
        w.assign(max_iterations,copyable);
        vi = vip = wi = wim = wip= copyable;
        max_iter = max_iterations;
        iter = max_iterations;
        T.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        T.diagonal_offsets[0] = -1;
        T.diagonal_offsets[1] =  0;
        T.diagonal_offsets[2] =  1;
        Tinv.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        V.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
    }
    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        T.resize(new_iter, new_iter, 3*new_iter-2, 3, max_iter);
        T.diagonal_offsets[0] = -1;
        T.diagonal_offsets[1] =  0;
        T.diagonal_offsets[2] =  1;
        V.resize(vi.size(), new_iter, new_iter*vi.size()); 
        iter = new_iter;
    }
    ///@brief Get the current  number of iterations
    ///@return the current number of iterations
    unsigned get_iter() const {return iter;}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return w;}
    /**
     * @brief Solve the system A*x = b for b using Lanczos method
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains an initial value
     * @param b The right hand side vector. 
      */
    template< class MatrixType, class ContainerType0, class ContainerType1>
    std::pair<dia_type, coo_type> operator()( MatrixType& A, const ContainerType0& x, ContainerType1& b)
    {
        get_value_type<ContainerType> xnorm = sqrt(dg::blas1::dot(x, x));

        //Initial vector
        dg::blas1::axpby(1./xnorm, x, 0.0, v[0]); //v[1] = x/||x||
        beta[0] = 0.;

        //Algorithm for i=1
        dg::blas2::symv(A, v[0], v[1]);  
        alpha[0] = dg::blas1::dot(v[1], v[0]);      
        dg::blas1::axpby(-alpha[0], v[0], 1.0, v[1]);            
        beta[1] = sqrt(dg::blas1::dot(v[1], v[1]));  

        dg::blas1::scal(v[1], 1./beta[1]);
        //Algorithm for i>1
        for( unsigned i=1; i<max_iter-1; i++)
        {
            dg::blas2::symv(A, v[i], v[i+1]);                    
            dg::blas1::axpby(-beta[i], v[i-1], 1.0, v[i+1]);     
            alpha[i] = dg::blas1::dot(v[i+1], v[i]);            
            dg::blas1::axpby(-alpha[i], v[i], 1.0, v[i+1]);      
            beta[i+1] = sqrt(dg::blas1::dot(v[i+1], v[i+1]));     
    //         if (beta[i+1] == 0) break;
//             std::cout << beta[i+1]  << "\n";
            dg::blas1::scal(v[i+1], 1./beta[i+1]);  
            
        }
        //Last alpha
        dg::blas2::symv(A, v[max_iter-1], vi);

        dg::blas1::axpby(-beta[max_iter-1], v[max_iter-2], 1.0, vi); 
        alpha[max_iter-1] = dg::blas1::dot(vi, v[max_iter-1]);
        
        //Fill T and V Matrix
        unsigned counter = 0;
        for( unsigned i=0; i<max_iter; i++)
        {
            T.values(i,0) =  beta[i];  // -1 diagonal
            T.values(i,1) =  alpha[i];  // 0 diagonal
            T.values(i,2) =  beta[i+1];  // +1 diagonal //dia_rows entry works since its outside of matrix
            
            for( unsigned j=0; j<v[0].size(); j++)
            {            
                V.row_indices[counter]    = j;
                V.column_indices[counter] = i; 
                V.values[counter]         = v[i][j];
                counter++;
            }
        }     
        
        //Check implementation: b=||x|| V T e_1 = || x || (v[0] alpha[01] + v[1] beta[1])
        dg::blas1::axpby(alpha[0], v[0], beta[1], v[1], b);
        dg::blas1::scal(b, xnorm ); 
        
        TVpair = std::make_pair(T,V);
        return TVpair;
    }
    /**
     * @brief Solve the system A*x = b for b using M-Lanczos method
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains an initial value
     * @param b The right hand side vector. 
     * @param S should be the weights 
     * @param Sinv should be the inverse of S, the inverse weights
     * @param eps accuracy of residual
     */
    template< class MatrixType, class ContainerType0, class ContainerType1, class SquareNorm1, class SquareNorm2>
    std::pair<dia_type, coo_type> operator()( MatrixType& A, const ContainerType0& x, ContainerType1& b,  SquareNorm1& S, SquareNorm2& Sinv, value_type eps)
    {
        value_type xnorm = sqrt(dg::blas2::dot(x, S, x));
        value_type residual;
#ifdef DG_BENCHMARK
        Timer t;
        value_type invtime=0.;
#endif //DG_BENCHMARK
   /*//     Implementation #1 (naiv implementation, however slightly faster than 2 and 3)
        dg::blas1::axpby(1./xnorm, x, 0.0, v[0]); //v[1] = x/||x||
        beta[0] = 0.;
        dg::blas2::symv(A, v[0], vi);  
        dg::blas2::symv(Sinv, vi, v[1]);
        alpha[0] = dg::blas2::dot(v[1], S, v[0]);      
        dg::blas1::axpby(-alpha[0], v[0], 1.0, v[1]);            
        beta[1] = sqrt(dg::blas2::dot( S, v[1]));   
        dg::blas1::scal(v[1], 1./beta[1]);
 
        for( unsigned i=1; i<max_iter-1; i++)
        {
            dg::blas2::symv(A, v[i],vi);                    
            dg::blas2::symv(Sinv,vi,v[i+1]);                
            dg::blas1::axpby(-beta[i], v[i-1], 1.0, v[i+1]);   
            alpha[i] = dg::blas2::dot(v[i+1], S, v[i]);          
            dg::blas1::axpby(-alpha[i], v[i], 1.0, v[i+1]);      
            beta[i+1] = sqrt(dg::blas2::dot(S, v[i+1]));       
            dg::blas1::scal(v[i+1], 1./beta[i+1]);       
            
        }
        dg::blas2::symv(A, v[max_iter-1], vi);
        dg::blas2::symv(Sinv,vi, vi); 
        dg::blas1::axpby(-beta[max_iter-1], v[max_iter-2], 1.0, vi);
        alpha[max_iter-1] = dg::blas2::dot(vi, S, v[max_iter-1]);

//         //Implementation #2 (with w and blas1 dots)
        dg::blas1::axpby(1./xnorm, x, 0.0, v[0]); //v[1] = x/||x||
        beta[0] = 0.;
        dg::blas2::symv(S, v[0], w[0]);
//         dg::blas2::symv(A, v[0], w[1]); 
//         alpha[0] = dg::blas1::dot(w[1], v[0]); 
//         dg::blas1::axpby(-alpha[0], w[0], 1.0, w[1]);   
//         dg::blas2::symv(Sinv,w[1],v[1]);
//         beta[1] = sqrt(dg::blas1::dot( w[1], v[1]));  
//         dg::blas1::scal(v[1], 1./beta[1]);              
//         dg::blas1::scal(w[1], 1./beta[1]);

        for( unsigned i=0; i<max_iter-1; i++)
        {    
            dg::blas2::symv(A, v[i], w[i+1]);                    
            if (i>0) dg::blas1::axpby(-beta[i], w[i-1], 1.0, w[i+1]);    
            alpha[i] = dg::blas1::dot(w[i+1], v[i]);          
            dg::blas1::axpby(-alpha[i], w[i], 1.0, w[i+1]);       
            dg::blas2::symv(Sinv,w[i+1],v[i+1]);
            beta[i+1] = sqrt(dg::blas1::dot(w[i+1], v[i+1]));       
            dg::blas1::scal(v[i+1], 1./beta[i+1]);              
            dg::blas1::scal(w[i+1], 1./beta[i+1]);  
            
            //TODO convergence criterium for sqrt operator
        }
        dg::blas2::symv(A, v[max_iter-1], vi);                    
        dg::blas1::axpby(-beta[max_iter-1], w[max_iter-2], 1.0, vi);    
        alpha[max_iter-1] = dg::blas1::dot(vi, v[max_iter-1]);   
        Fill T and V Matrix for #1 and #2
       unsigned counter = 0;
        for( unsigned i=0; i<max_iter; i++)
        {
            T.values(i,0) =  beta[i];  // -1 diagonal
            T.values(i,1) =  alpha[i];  // 0 diagonal
            T.values(i,2) =  beta[i+1];  // +1 diagonal //dia_rows entry works since its outside of matrix
            
            for( unsigned j=0; j<v[0].size(); j++)
            {            
                V.row_indices[counter]    = j;
                V.column_indices[counter] = i; 
                V.values[counter]         = v[i][j];
                counter++;
            }
        } */      
        //Implementation #3 (with w and blas1 dots and without vectors)
        dg::blas1::axpby(1./xnorm, x, 0.0, vi); //v[1] = x/||x||
        value_type betaip = 0;
        dg::blas2::symv(S, vi, wi);
        unsigned counter = 0;
        for( unsigned i=0; i<max_iter; i++)
        {  
            for( unsigned j=0; j<v[0].size(); j++)
            {            
                V.row_indices[counter]    = j;
                V.column_indices[counter] = i; 
                V.values[counter]         = vi[j];
                counter++;
            }
            T.values(i,0) =  betaip;  // -1 diagonal
            dg::blas2::symv(A, vi, wip); 
            dg::blas1::axpby(-betaip, wim, 1.0, wip);    //only -= if i>0, therefore no if (i>0)
            T.values(i,1) = dg::blas1::dot(wip, vi);    
            dg::blas1::axpby(-T.values(i,1), wi, 1.0, wip);     
            dg::blas2::symv(Sinv,wip,vip);
            betaip = sqrt(dg::blas1::dot(wip, vip)); 
            if (betaip == 0) {
#ifdef DG_DEBUG
                std::cout << "beta[i+1]=0 encountered\n";
#endif //DG_DEBUG
                set_iter(i+1); 
                break;
            } 
            dg::blas1::scal(vip, 1./betaip);     
            dg::blas1::scal(wip, 1./betaip);  
            vi=vip;
            wim=wi;
            wi=wip;
            T.values(i,2) =  betaip;  // +1 diagonal
            if (i>0) {
                invtridiag.resize(i);
#ifdef DG_BENCHMARK
                t.tic();
#endif //DG_BENCHMARK
                Tinv = invtridiag(T); //Compute inverse of T
#ifdef DG_BENCHMARK
                t.toc();
                invtime+=t.diff();
#endif //DG_BENCHMARK
                residual = xnorm*betaip*betaip*abs(Tinv.values[i-1]);
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

//         Check implementation: b=||x|| V T e_1 = || x || (v[0] alpha[01] + v[1] beta[1])
//         Note that it depends only on first two vectors and alpha,beta (changes if square root is applied upon T)
//         dg::blas1::axpby(alpha[0], v[0], beta[1], v[1], b); //faster alternative
        
        ContainerType e1( get_iter(), 0.), temp(e1);
        e1[0]=1.;
        dg::blas2::symv(T, e1, temp); //T e_1
        dg::blas2::symv(V, temp, b); // V T e_1
        dg::blas1::scal(b, xnorm ); 
        TVpair = std::make_pair(T,V);
        return TVpair;
    }
  private:
    std::vector<value_type> alpha, beta;
    std::vector<ContainerType> v, w;
    ContainerType  vi, vip, wi, wip, wim;
    unsigned iter, max_iter;
    std::pair<dia_type, coo_type> TVpair; 
    dia_type T;
    coo_type V, Tinv;    
    InvTridiag<ContainerType> invtridiag;

};


} //namespace dg

