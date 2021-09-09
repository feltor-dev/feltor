#pragma once
#include "blas.h"
#include "functors.h"
#include "backend/timer.h"
#include <cusp/dia_matrix.h>
#include <cusp/coo_matrix.h>

/**
* @brief Classes for Krylov space approximations of a Matrix-Vector product
*/

namespace dg{
   

/**
* @brief Functor class for computing the inverse of a general tridiagonal matrix. 
*        NOTE: HMGTI algorithm of "Inversion of general tridiagonal matrices" by             Moawwad El-Mikkawy and Abdelrahman Karawi
* 
*       Is unstable for matrix size of roughly > 150. Fails for certain tridiagonal matrix forms. Not tested thoroughly!
* @ingroup invert
*/
template< class ContainerType, class DiaMatrix, class CooMatrix>
class TridiagInvHMGTI
{
  public:
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    TridiagInvHMGTI(){}
    /**
     * @brief Construct from vector
     * 
     * @param copyable vector
     */
    TridiagInvHMGTI(const ContainerType& copyable) 
    {
        m_size = copyable.size();
        m_alphas.assign(m_size+1,0.);
        m_betas.assign(m_size+1,0.);
        m_Tinv.resize(m_size, m_size,  m_size* m_size);
    }
    /**
     * @brief Construct from size of vector
     * 
     * @param size size of vector
     */
    TridiagInvHMGTI(unsigned size) 
    {
        m_size = size;
        m_alphas.assign(m_size+1,0.);
        m_betas.assign(m_size+1,0.);
        m_Tinv.resize(m_size, m_size,  m_size* m_size);
    }
    /**
     * @brief Resize inverse tridiagonal matrix and helper vectors
     * 
     * @param new_size new size of square matrix
    */
    void resize(unsigned new_size) {
        m_size = new_size;
        m_alphas.resize(m_size+1,0.);
        m_betas.resize(m_size+1,0.);
        m_Tinv.resize(m_size, m_size,  m_size* m_size);
    }
    /**
     * @brief Compute the sign function 
     * 
     * @param i an unsigned integer parameter
     * 
     * @return returns the sign function \f$ (-1)^i\f$
     */
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
     **/
    CooMatrix operator()(const DiaMatrix& T)
    {
        ContainerType alpha(m_size, 0.);
        ContainerType beta(m_size, 0.);
        ContainerType gamma(m_size, 0.);
        for(unsigned i = 0; i<m_size; i++)
        {
            alpha[i] = T.values(i,1);    // 0 diagonal
            beta[i]  = T.values(i,2)  ;  // +1 diagonal //dia_rows entry works since its outside of matrix
            gamma[i] = T.values(i+1,0);  // -1 diagonal            
        }
        return this->operator()(alpha, beta, gamma);
    }
     /**
     * @brief Compute the inverse of a tridiagonal matrix with diagonal vectors a,b,c
     * 
     * @param a  "0" diagonal vector ( 0...(size-1) )
     * @param b "+1" diagonal vector ( 0...(size-2) )
     * @param c "-1" diagonal vector ( 0...(size-2) )
     * 
     * @return the inverse of the tridiagonal matrix (coordinate format)
     */
    CooMatrix operator()(const ContainerType& a, const ContainerType& b,  const ContainerType& c)
    {
        //fill alphas
        m_alphas[0]=1.0;
        m_alphas[1]=a[0];
        for( unsigned i = 2; i<m_size+1; i++)
        {
            m_alphas[i] = a[i-1]*m_alphas[i-1] - c[i-2]*b[i-2]*m_alphas[i-2];
            if (m_alphas[i] ==0 && i<m_size) {
                std::cout << "# Failure in alpha["<<i<<"] !" << std::endl;
            } 
        }
        if (m_alphas[m_size] ==0) 
            std::cout << "# No Inverse of tridiagonal matrix exists !" << std::endl;
        
        //fill betas
        m_betas[m_size]=1.0;
        m_betas[m_size-1]=a[m_size-1];
        m_betas[0] = m_alphas[m_size];
        for( unsigned i = m_size-2; i>0; i--)
        {
            m_betas[i] = a[i]*m_betas[i+1] - c[i]*b[i]*m_betas[i+2];
            if (m_betas[i] ==0) 
            {
                std::cout << "# Failure in beta !" << std::endl;
            }
        }
        
        //Diagonal entries
        m_Tinv.row_indices[0*m_size+0]    = 0;
        m_Tinv.row_indices[0*m_size+0]    = 0; 
        m_Tinv.row_indices[(m_size-1)*m_size+(m_size-1)]    = (m_size-1);
        m_Tinv.row_indices[(m_size-1)*m_size+(m_size-1)]    = (m_size-1);
        m_Tinv.values[0*m_size+0] = 1.0/(a[0]-c[0]*b[0]*m_betas[2]/m_betas[1]);
        m_Tinv.values[(m_size-1)*m_size+(m_size-1)] = 1.0/(a[m_size-1] - c[m_size-2]*b[m_size-2]*m_alphas[m_size-2]/m_alphas[m_size-1]);        
        for( unsigned i=1; i<m_size-1; i++) 
        {   
            m_Tinv.row_indices[i*m_size+i]    = i;
            m_Tinv.column_indices[i*m_size+i] = i;
            m_Tinv.values[i*m_size+i] = 1.0/(a[i]-c[i-1]*b[i-1]*m_alphas[i-1]/m_alphas[i]-c[i]*b[i]*m_betas[i+2]/m_betas[i+1]);
        }
        //Off-diagonal entries
        for( unsigned i=0; i<m_size; i++) 
        {   
            for( unsigned j=0; j<m_size; j++) 
            {  
                m_Tinv.row_indices[i*m_size+j]    = j;
                m_Tinv.column_indices[i*m_size+j] = i;
                if (i<j) {
                    m_Tinv.values[i*m_size+j] = sign(j-i)*std::accumulate(std::next(b.begin(),i), std::next(b.begin(),j), 1., std::multiplies<value_type>())*
                                                m_alphas[i]/m_alphas[j]*m_Tinv.values[j*m_size+j];
                }
                else if (i>j)
                {
                    m_Tinv.values[i*m_size+j] = sign(i-j)*std::accumulate(std::next(c.begin(),j), std::next(c.begin(),i), 1., std::multiplies<value_type>())*
                                                m_betas[i+1]/m_betas[j+1]*m_Tinv.values[j*m_size+j];
                }
            }
        }
        return m_Tinv;
    }
  private:
    ContainerType m_alphas, m_betas;
    CooMatrix m_Tinv;
    unsigned m_size;
};


/**
* @brief Functor class for computing the inverse of a general tridiagonal matrix. The algorithm does not rely on the determinant.
* @Note For some special cases division by zero occurs (can be fixed if necessary cf. 2nd paper)
*       This is the algorihm of "On the inverses of general tridiagonal matrices" by Hou-Biao Li, Ting-Zhu Huang, Xing-Ping Liu, Hong Li
*       Appears to be the same than the algorithm in "ON AN INVERSE FORMULA OF A TRIDIAGONAL MATRIX" by Tomoyuki Sugimoto
* 
* @ingroup invert
*/
template< class ContainerType, class DiaMatrix, class CooMatrix>
class TridiagInvDF
{
  public:
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    TridiagInvDF(){}
    /**
     * @brief Construct from vector
     * 
     * @param copyable vector
     */
    TridiagInvDF(const ContainerType& copyable) 
    {
        m_size = copyable.size();
        m_phi.assign(m_size,0.);
        m_theta.assign(m_size,0.);
        m_Tinv.resize(m_size, m_size,  m_size* m_size);
    }
    /**
     * @brief Construct from size of vector
     * 
     * @param size size of vector
     */
    TridiagInvDF(unsigned size) 
    {
        m_size = size;
        m_phi.assign(m_size,0.);
        m_theta.assign(m_size,0.);
        m_Tinv.resize(m_size, m_size,  m_size* m_size);
    }
    /**
     * @brief Resize inverse tridiagonal matrix and helper vectors
     * 
     * @param new_size new size of square matrix
    */
    void resize(unsigned new_size) {
        m_size = new_size;
        m_phi.resize(m_size,0.);
        m_theta.resize(m_size,0.);
        m_Tinv.resize(m_size, m_size,  m_size* m_size);
    }
    /**
     * @brief Compute the inverse of a tridiagonal matrix T
     * 
     * @param T tridiagonal matrix
     * 
     * @return the inverse of the tridiagonal matrix (coordinate format)
     **/
    CooMatrix operator()(const DiaMatrix& T)
    {
        ContainerType alpha(m_size, 0.);
        ContainerType beta(m_size, 0.);
        ContainerType gamma(m_size, 0.);
        for(unsigned i = 0; i<m_size; i++)
        {
            alpha[i] = T.values(i,1);    // 0 diagonal
            beta[i]  = T.values(i,2)  ;  // +1 diagonal //dia_rows entry works since its outside of matrix
            gamma[i] = T.values(i+1,0);  // -1 diagonal            
        }
        return this->operator()(alpha, beta, gamma);
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
    CooMatrix operator()(const ContainerType& a, const ContainerType& b,  const ContainerType& c)
    {
        value_type helper = 0.0;
        //fill phi values
        m_phi[0] = - b[0]/a[0];
        for( unsigned i = 1; i<m_size; i++)
        {   
            helper = m_phi[i-1]* c[i-1] + a[i];
            if (helper==0) std::cout << "Failure: Division by zero\n" << std::endl;
            else m_phi[i]   = -b[i]/helper;
        }
//         m_phi[m_size] = 0.0;

        //fill theta values
        if (m_size == 1)  m_theta[m_size-1]  = 0.0;
        else 
        {
            m_theta[m_size-1]  = - c[m_size-2]/a[m_size-1]; 
            for( int i = m_size-2; i>=0; i--)
            {
                helper = m_theta[i+1]*b[i] + a[i];
                if (helper==0) std::cout << "Failure: Division by zero\n" << std::endl;
                else m_theta[i]  = -c[i-1]/helper;
            }
        }
//         m_theta[0] = 0.0; 
        
        //Diagonal entries
        m_Tinv.row_indices[0*m_size+0]    = 0;
        m_Tinv.row_indices[0*m_size+0]    = 0; 
        helper = a[0] + b[0]* m_theta[1];
        if (helper==0) std::cout << "Failure: No inverse exists\n" << std::endl;
        else m_Tinv.values[0*m_size+0] = 1.0/helper;        

        m_Tinv.row_indices[(m_size-1)*m_size+(m_size-1)]    = (m_size-1);
        m_Tinv.row_indices[(m_size-1)*m_size+(m_size-1)]    = (m_size-1);        
        if (m_size == 1) helper = a[m_size-1];
        else helper = a[m_size-1] + c[m_size-2]*m_phi[m_size-2]; 
        
        if (helper==0) std::cout << "Failure: No inverse exists\n" << std::endl;
        else m_Tinv.values[(m_size-1)*m_size+m_size-1] = 1.0/helper;     
        
        for( unsigned i=1; i<m_size-1; i++) 
        {   
            m_Tinv.row_indices[i*m_size+i]    = i;
            m_Tinv.column_indices[i*m_size+i] = i;
            helper = a[i] + c[i-1]*m_phi[i-1] + b[i]* m_theta[i+1];
            if (helper==0) std::cout << "Failure: No inverse exists\n" << std::endl;
            else m_Tinv.values[i*m_size+i] = 1.0/helper;
        }
        
        //Off-diagonal entries
        for( unsigned j=0; j<m_size-1; j++) //row index
        {   
            for (unsigned i=j+1; i<m_size; i++)
            {
                m_Tinv.row_indices[i*m_size+j]    = i;
                m_Tinv.column_indices[i*m_size+j] = j; 
                m_Tinv.values[i*m_size+j] = m_theta[i]*m_Tinv.values[(i-1)*m_size+j];
            }
        }
        for( unsigned j=1; j<m_size; j++) //row index
        {   
            for (int i=j-1; i>=0; i--)
            {
                m_Tinv.row_indices[i*m_size+j]    = i;
                m_Tinv.column_indices[i*m_size+j] = j; 
                m_Tinv.values[i*m_size+j] = m_phi[i]*m_Tinv.values[(i+1)*m_size+j];
            }
        }

        return m_Tinv;
    }
  private:
    ContainerType m_phi, m_theta;
    CooMatrix m_Tinv;
    unsigned m_size;
};

/**
* @brief Functor class for computing the inverse of a general tridiagonal matrix. 
*        NOTE: If roughly the matrix size m>150 the algorithm is unstable. However, it performs extremely fast if it stays below this value.
*              This is the algorihm of "Inversion of a Tridiagonal Jacobi Matrix" by Riaz A. Usmani
* 
* @ingroup invert
*/
template< class ContainerType, class DiaMatrix, class CooMatrix>
class TridiagInvD
{
  public:
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    TridiagInvD(){}
    /**
     * @brief Construct from vector
     * 
     * @param copyable vector
     */
    TridiagInvD(const ContainerType& copyable) 
    {
        m_size = copyable.size();
        m_phi.assign(m_size+1,0.);
        m_theta.assign(m_size+1,0.);
        m_Tinv.resize(m_size, m_size,  m_size* m_size);
    }
    /**
     * @brief Construct from size of vector
     * 
     * @param size size of vector
     */
    TridiagInvD(unsigned size) 
    {
        m_size = size;
        m_phi.assign(m_size+1,0.);
        m_theta.assign(m_size+1,0.);
        m_Tinv.resize(m_size, m_size,  m_size* m_size);
    }
    /**
     * @brief Resize inverse tridiagonal matrix and helper vectors
     * 
     * @param new_size new size of square matrix
    */
    void resize(unsigned new_size) {
        m_size = new_size;
        m_phi.resize(m_size+1,0.);
        m_theta.resize(m_size+1,0.);
        m_Tinv.resize(m_size, m_size,  m_size* m_size);
    }
   /**
     * @brief Compute the sign function 
     * 
     * @param i an unsigned integer parameter
     * 
     * @return returns the sign function \f$ (-1)^i\f$
     */
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
     **/
    CooMatrix operator()(const DiaMatrix& T)
    {
        ContainerType alpha(m_size, 0.);
        ContainerType beta(m_size, 0.);
        ContainerType gamma(m_size, 0.);
        for(unsigned i = 0; i<m_size; i++)
        {
            alpha[i] = T.values(i,1);    // 0 diagonal
            beta[i]  = T.values(i,2)  ;  // +1 diagonal //dia_rows entry works since its outside of matrix
            gamma[i] = T.values(i+1,0);  // -1 diagonal            
        }
        return this->operator()(alpha, beta, gamma);
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
    CooMatrix operator()(const ContainerType& a, const ContainerType& b,  const ContainerType& c)
    {
        unsigned is=0;
        for( unsigned i = 0; i<m_size+1; i++)
        {   
            is = m_size - i;
            if (i==0) 
            {   
                m_theta[0] = 1.; 
                m_phi[is]  = 1.;
            }
            else if (i==1) 
            {
                m_theta[1] = a[0]; 
                m_phi[is]  = a[is];
            }
            else
            {
                m_theta[i] = a[i-1] * m_theta[i-1] - b[i-2] * c[i-2] * m_theta[i-2];
                m_phi[is]  = a[is]  * m_phi[is+1]  - b[is]  * c[is]  * m_phi[is+2];
            }
        }
//         std::cout << " m_theta[m_size]" << m_theta[m_size] << std::endl; //      catch m_theta[m_size] == -nan;

        //Compute inverse tridiagonal matrix elements
        for( unsigned i=0; i<m_size; i++) //row index
        {   
            for( unsigned j=0; j<m_size; j++) //column index
            {   
                m_Tinv.row_indices[i*m_size+j]    = i;
                m_Tinv.column_indices[i*m_size+j] = j; 
                if (i<j) {
                    m_Tinv.values[i*m_size+j] = std::accumulate(std::next(b.begin(),i), std::next(b.begin(),j), 1., std::multiplies<value_type>())*sign(i+j) * m_theta[i] * m_phi[j+1]/m_theta[m_size];
                }   
                else if (i==j)
                {
                    m_Tinv.values[i*m_size+j] =  m_theta[i] * m_phi[i+1]/m_theta[m_size];
                }
                else // if (i>j)
                {
                    m_Tinv.values[i*m_size+j] = std::accumulate(std::next(c.begin(),j), std::next(c.begin(),i), 1., std::multiplies<value_type>())*sign(i+j) * m_theta[j] * m_phi[i+1]/m_theta[m_size];
                }
            }
        }
        return m_Tinv;
    }
  private:
    ContainerType m_phi, m_theta;
    CooMatrix m_Tinv;
    unsigned m_size;
};
}
