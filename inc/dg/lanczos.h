#include <cmath>

#include "blas.h"
#include "functors.h"

#include <cusp/dia_matrix.h>
#include <cusp/coo_matrix.h>

namespace dg{
    
    
/**
* @brief Functor class for computing the inverse of a general tridiagonal matrix 
*/
template< class ContainerType>
class InvTridiag
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    using coo_type =  cusp::coo_matrix<int, value_type, cusp::host_memory>;
    ///@brief Allocate nothing, Call \c construct method before usage
    InvTridiag(){}
    //Constructor
    InvTridiag(const ContainerType& copyable) : 
        phi(copyable.size()+1, 0.),
        theta(copyable.size()+1, 0.)
    {
        Tinv.resize(copyable.size(), copyable.size(),  copyable.size()* copyable.size());
        temp = 0.;
    }

     /**
     * @brief Compute the inverse of a tridiagonal matrix with diagonal vectors a,b,c
     * 
     * @param a  "0" diagonal vector
     * @param b "+1" diagonal vector 
     * @param c "-1" diagonal vector
     * 
     * @return the inverse of the tridiagonal matrix (coordinate format)
     * 
     * @Note Be careful with indexing of off diagonal vector entries
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
                phi[is]  = a[is-1];
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
                    Tinv.values[counter] =temp*pow(-1,i+j) * theta[i] * phi[j+1]/theta[theta.size()-1];
                    
                }
                else if (i>j)
                {
                    for (unsigned k=j; k<i; k++) temp*=c[k];           
                    Tinv.values[counter] =temp*pow(-1,i+j) * theta[j] * phi[i+1]/theta[theta.size()-1];

                }
                else // if (i==j)
                {
                    Tinv.values[counter] =theta[i] * phi[j+1]/theta[theta.size()-1];
                }   
                counter++;
            }
        }
        return Tinv;
    }
  private:
    ContainerType phi, theta;
    coo_type Tinv;    
    value_type temp;
};

/**
* @brief Functor class for the Lanczos method to solve
* \f[ Ax=b\f]
* for b. A is a symmetric 
*
* 
* @note The common lanczos method (and M-Lanczos) method are prone to loss of orthogonality for finite precision. Here, only the basic Paige fix is used. Thus the iterations should be kept as small as possible. Could be fixed via reorthogonalization strategies, but so far the required iteration numbers are low enough. 
*/
template< class ContainerType>
class Lanczos
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    using dia_type =  cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using coo_type =  cusp::coo_matrix<int, value_type, cusp::host_memory>;
    ///@brief Allocate nothing, Call \c construct method before usage
    Lanczos(){}
    ///@copydoc construct()
    Lanczos( const ContainerType& copyable, unsigned max_iterations) : 
        max_iter(max_iterations)
    {
          construct(copyable, max_iterations);
    }
    ///@brief Set the maximum number of iterations
    ///@param new_max New maximum number
    void set_max( unsigned new_max) {max_iter = new_max;}
    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_max() const {return max_iter;}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return w;}

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
        w = copyable;
        temp = copyable;
        max_iter = max_iterations;
        T.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        T.diagonal_offsets[0] = -1;
        T.diagonal_offsets[1] =  0;
        T.diagonal_offsets[2] =  1;
        V.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
    }
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
            dg::blas2::symv(A, v[i], v[i+1]);                     //v[i+1]= A v[i]
            dg::blas1::axpby(-beta[i], v[i-1], 1.0, v[i+1]);      //v[i+1]= A v[i] - beta[i] v[i-1]
            alpha[i] = dg::blas1::dot(v[i+1], v[i]);             //alpha[i]= W v[i+1].v[i]
            dg::blas1::axpby(-alpha[i], v[i], 1.0, v[i+1]);       //v[i+1]= A v[i] - beta[i] v[i-1] - \alpha[i] v[i]
            beta[i+1] = sqrt(dg::blas1::dot(v[i+1], v[i+1]));       //beta[i+1] = ||v[i+1]||
    //         if (beta[i+1] == 0) break;
            dg::blas1::scal(v[i+1], 1./beta[i+1]);                //v[i+1] = (A v[i] - beta[i] v[i-1] - \alpha[i] v[i])/beta[i+1]
        }
        //Last alpha
        dg::blas2::symv(A, v[max_iter-1], w);

        dg::blas1::axpby(-beta[max_iter-1], v[max_iter-2], 1.0, w); //w= A v[i] - beta[i] v[i-1]
        alpha[max_iter-1] = dg::blas1::dot(w, v[max_iter-1]);
        
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
     */
    template< class MatrixType, class ContainerType0, class ContainerType1, class SquareNorm1, class SquareNorm2>
    std::pair<dia_type, coo_type> operator()( MatrixType& A, const ContainerType0& x, ContainerType1& b,  SquareNorm1& S, SquareNorm2& Sinv)
    {
        get_value_type<ContainerType> xnorm = sqrt(dg::blas2::dot(x, S, x));

        //Initial vector
        dg::blas1::axpby(1./xnorm, x, 0.0, v[0]); //v[1] = x/||x||
        beta[0] = 0.;

        //Algorithm for i=1
        dg::blas2::symv(A, v[0], temp);  
        dg::blas2::symv(Sinv, temp, v[1]);
        alpha[0] = dg::blas2::dot(v[1], S, v[0]);      
        dg::blas1::axpby(-alpha[0], v[0], 1.0, v[1]);            
        beta[1] = sqrt(dg::blas2::dot( S, v[1]));  

        dg::blas1::scal(v[1], 1./beta[1]);
        //Algorithm for i>1
        for( unsigned i=1; i<max_iter-1; i++)
        {
            dg::blas2::symv(A, v[i],temp);                    
            dg::blas2::symv(Sinv,temp,v[i+1]);                   //v[i+1]= S^(-1) A v[i]

            dg::blas1::axpby(-beta[i], v[i-1], 1.0, v[i+1]);     //v[i+1]= A v[i] - beta[i] v[i-1]
            alpha[i] = dg::blas2::dot(v[i+1], S, v[i]);          //alpha[i]= < v[i+1],v[i]>_S

            dg::blas1::axpby(-alpha[i], v[i], 1.0, v[i+1]);       //v[i+1]= A v[i] - beta[i] v[i-1] - \alpha[i] v[i]
            beta[i+1] = sqrt(dg::blas2::dot(S, v[i+1]));       //beta[i+1] = ||v[i+1]||_S
    //         if (beta[i+1] == 0) break;
            dg::blas1::scal(v[i+1], 1./beta[i+1]);                //v[i+1] = (A v[i] - beta[i] v[i-1] - \alpha[i] v[i])/beta[i+1]
        }
        //Last alpha
        dg::blas2::symv(A, v[max_iter-1], temp);
        dg::blas2::symv(Sinv,temp, w);

        dg::blas1::axpby(-beta[max_iter-1], v[max_iter-2], 1.0, w); //w= A v[i] - beta[i] v[i-1]
        alpha[max_iter-1] = dg::blas2::dot(w, S, v[max_iter-1]);
        
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
        
    //     Check implementation: b=||x|| V T e_1 = || x || (v[0] alpha[01] + v[1] beta[1])
    //     Note that it depends only on first two vectors and alpha,beta -> this changes if square root is applied upon T. It is thus only a basic check
        dg::blas1::axpby(alpha[0], v[0], beta[1], v[1], b);
        dg::blas1::scal(b, xnorm ); 
        
        TVpair = std::make_pair(T,V);
        return TVpair;
    }
  private:
    std::vector<value_type> alpha, beta;
    std::vector<ContainerType> v;
    ContainerType w, temp;
    unsigned max_iter;
    std::pair<dia_type, coo_type> TVpair; 
    dia_type T;
    coo_type V;    
};
// TODO

template< class ContainerType>
class CGsqrt
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    using dia_type =  cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using coo_type =  cusp::coo_matrix<int, value_type, cusp::host_memory>;
    ///@brief Allocate nothing, Call \c construct method before usage
    CGsqrt(){}
    ///@copydoc construct()
    CGsqrt( const ContainerType& copyable, unsigned max_iterations) : 
        max_iter(max_iterations)
    {
          construct(copyable, max_iterations);
    }
  private:
    max_iter
      
};

} //namespace dg

