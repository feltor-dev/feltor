#pragma once

#include <boost/math/special_functions.hpp> // has to be included before lapack in certain versions
#include <cusp/dia_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/lapack/lapack.h>
#include "dg/algorithm.h"

#include "functors.h"
/**
* @brief Classes for Krylov space approximations of a Matrix-Vector product
*/

namespace dg{
namespace mat{

///@addtogroup matrixinvert
///@{

/**
 * @brief Computes the value of \f$ (T^{-1})_{m1} = \langle \vec e_m, T^{-1}\vec e_1\rangle\f$ via a Thomas algorithm
 *
 * @note This is extremely fast (timings can bearly be measured;
 *  for size = 100 <1e-6s)
 * @tparam value_type real type
 * @param T The tridiagonal matrix
 * @param size the parameter \c m
 *
 * @return \f$ (T^{-1})_{1m}\f$
 */
template<class value_type>
value_type compute_Tinv_m1( const cusp::dia_matrix<int, value_type,
        cusp::host_memory>& T, unsigned size)
{
    value_type ci = T.values(0,2)/T.values(0,1), ciold = 0.;
    value_type di = 1./T.values(0,1), diold = 0.;
    for( unsigned i=1; i<size; i++)
    {
        ciold = ci, diold = di;
        ci = T.values(i,2)/ ( T.values(i,1)-T.values(i,0)*ciold);
        di = -T.values( i, 0)*diold/(T.values(i,1)-T.values(i,0)*ciold);
    }
    return di;
}
/**
 * @brief Computes the value of \f$ x = ((aT+dI)^{-1})y \f$ via Thomas algorithm
 *
 * @note This is extremely fast (timings can bearly be measured;
 *  for size = 100 <1e-6s)
 * @tparam value_type real type
 * @param T The tridiagonal matrix
 * @param x contains the solution (resized if necessary)
 * @param y the right hand side
 * @param a optional scaling factor of T
 * @param d optional addition to diagonal of T
 */
template<class value_type>
void compute_Tinv_y( const cusp::dia_matrix<int, value_type,
        cusp::host_memory>& T,
        thrust::host_vector<value_type>& x,
        const thrust::host_vector<value_type>& y, value_type a  = 1.,
        value_type d = 0.)
{
    unsigned size = y.size();
    x.resize(size);
    thrust::host_vector<value_type> ci(size), di(size);
    ci[0] = a*T.values(0,2)/( a*T.values(0,1) + d);
    di[0] = y[0]/( a*T.values(0,1) + d);
    for( unsigned i=1; i<size; i++)
    {
        ci[i] = a*T.values(i,2)/ ( a*T.values(i,1) + d -a*T.values(i,0)*ci[i-1]);
        di[i] = (y[i]-a*T.values( i, 0)*di[i-1])/(a*T.values(i,1) + d
                -a*T.values(i,0)*ci[i-1]);
    }
    x[size-1] = di[size-1];
    for( int i=size-2; i>=0; i--)
        x[i] = di[i] - ci[i]*x[i+1];
}


/**
* @brief Compute the inverse of a general tridiagonal matrix.
*
* @note HMGTI algorithm of "Inversion of general tridiagonal matrices" by
*  Moawwad El-Mikkawy and Abdelrahman Karawi
*  Is unstable for matrix size of roughly > 150. Fails for certain
*  tridiagonal matrix forms.
* @attention Not tested thoroughly!
* @tparam ContainerType \c thrust::host_vector<value_type> or similar shared memory host vector
* @tparam CooMatrix \c cusp::coo_matrix<int, value_type, cusp::host_memory>;
* @tparam DiaMatrix \c cusp::dia_matrix<int, value_type, cusp::host_memory>;
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
    }
    /**
     * @brief Compute the inverse of a tridiagonal matrix T
     *
     * @param T tridiagonal matrix
     * @param Tinv the inverse of the tridiagonal matrix (coordinate format)
     *  gets resized if necessary
     **/
    void operator()(const DiaMatrix& T, CooMatrix& Tinv)
    {
        this->operator()(
                T.values.column(1), // 0 diagonal
                T.values.column(2), // +1 diagonal
                T.values.column(0), // -1 diagonal
                Tinv);
    }
    /**
     * @brief Compute the inverse of a tridiagonal matrix T
     *
     * @param T tridiagonal matrix
     * @return the inverse of the tridiagonal matrix (coordinate format)
     **/
    CooMatrix operator()(const DiaMatrix& T)
    {
        CooMatrix Tinv;
        this->operator()( T, Tinv);
        return Tinv;
    }
     /**
     * @brief Compute the inverse of a tridiagonal matrix with diagonal vectors a,b,c
     *
     * The diagonal vectors are given as in the cusp dia_matrix format
     * @param a  "0" diagonal vector (index 0 is on row 0)
     * @param b "+1" diagonal vector (index 0 is on row 0, last index outside)
     * @param c "-1" diagonal vector (index 0 is on row 0, outside of matrix)
     * @param Tinv the inverse of the tridiagonal matrix (coordinate format)
     *  gets resized if necessary
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void operator()(const ContainerType0& a, const ContainerType1& b,
            const ContainerType2& c, CooMatrix& Tinv)
    {
        unsigned ss = m_size;
        Tinv.resize(ss, ss,  ss* ss);
        if( ss == 1)
        {
            Tinv.row_indices[0] = 0;
            Tinv.column_indices[0] = 0;
            Tinv.values[0] = 1./b[0];
            return;
        }
        //fill alphas
        m_alphas[0]=1.0;
        m_alphas[1]=a[0];
        for( unsigned i = 2; i<ss+1; i++)
        {
            m_alphas[i] = a[i-1]*m_alphas[i-1] - c[i-1]*b[i-2]*m_alphas[i-2];
            if (m_alphas[i] ==0 && i<ss) {
                throw dg::Error( dg::Message(_ping_) << "# Failure in alpha["<<i<<"] !");
            }
        }
        if (m_alphas[ss] ==0)
            throw dg::Error( dg::Message(_ping_) << "# No Inverse of tridiagonal matrix exists !");

        //fill betas
        m_betas[ss]=1.0;
        m_betas[ss-1]=a[ss-1];
        m_betas[0] = m_alphas[ss];
        for( int i = ss-2; i>0; i--)
        {
            m_betas[i] = a[i]*m_betas[i+1] - c[i+1]*b[i]*m_betas[i+2];
            if (m_betas[i] ==0)
            {
                throw dg::Error( dg::Message(_ping_) << "# Failure in beta["<<i<<"] !");
            }
        }
        //Diagonal entries
        Tinv.row_indices[0*ss+0]    = 0;
        Tinv.column_indices[0*ss+0]    = 0;
        Tinv.row_indices[(ss-1)*ss+(ss-1)]    = (ss-1);
        Tinv.column_indices[(ss-1)*ss+(ss-1)]    = (ss-1);
        Tinv.values[0*ss+0] = 1.0/(a[0]-c[1]*b[0]*m_betas[2]/m_betas[1]);
        Tinv.values[(ss-1)*ss+(ss-1)] = 1.0/(a[ss-1] -
                c[ss-1]*b[ss-2]*m_alphas[ss-2]/m_alphas[ss-1]);
        for( unsigned i=1; i<ss-1; i++)
        {
            Tinv.row_indices[i*ss+i]    = i;
            Tinv.column_indices[i*ss+i] = i;
            Tinv.values[i*ss+i] =
                1.0/(a[i]-c[i]*b[i-1]*m_alphas[i-1]/m_alphas[i]
                         -c[i+1]*b[i]*m_betas[i+2]/m_betas[i+1]);
        }
        //Off-diagonal entries
        for( unsigned i=0; i<ss; i++)
        {
            for( unsigned j=0; j<ss; j++)
            {
                Tinv.row_indices[i*ss+j]    = j;
                Tinv.column_indices[i*ss+j] = i;
                if (i<j) {
                    Tinv.values[i*ss+j] =
                        sign(j-i)*std::accumulate(std::next(b.begin(),i),
                                std::next(b.begin(),j), 1.,
                                std::multiplies<value_type>())*
                        m_alphas[i]/m_alphas[j]*Tinv.values[j*ss+j];
                }
                else if (i>j)
                {
                    Tinv.values[i*ss+j] =
                        sign(i-j)*std::accumulate(std::next(c.begin(),j+1),
                                std::next(c.begin(),i+1), 1.,
                                std::multiplies<value_type>())*
                        m_betas[i+1]/m_betas[j+1]*Tinv.values[j*ss+j];
                }
            }
        }
    }
  private:
    /// \f$ (-1)^i\f$
    int sign(unsigned i)
    {
        if (i%2==0) return 1;
        else return -1;
    }
    ContainerType m_alphas, m_betas;
    unsigned m_size;
};


/**
* @brief USE THIS ONE Compute the inverse of a general tridiagonal matrix. The algorithm does not rely on the determinant.
* @note For some special cases division by zero occurs (can be fixed if necessary cf. 2nd paper)
*  This is the algorihm of "On the inverses of general tridiagonal matrices" by Hou-Biao Li, Ting-Zhu Huang, Xing-Ping Liu, Hong Li
*  Appears to be the same as the algorithm in "ON AN INVERSE FORMULA OF A TRIDIAGONAL MATRIX" by Tomoyuki Sugimoto
*
* @tparam ContainerType \c thrust::host_vector<value_type> or similar shared memory host vector
* @tparam CooMatrix \c cusp::coo_matrix<int, value_type, cusp::host_memory>;
* @tparam DiaMatrix \c cusp::dia_matrix<int, value_type, cusp::host_memory>;
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
    }
    /**
     * @brief Compute the inverse of a tridiagonal matrix T
     *
     * @param T tridiagonal matrix
     * @param Tinv the inverse of the tridiagonal matrix (coordinate format),
     *  gets resized if necessary
     **/
    void operator()(const DiaMatrix& T, CooMatrix& Tinv)
    {
        this->operator()(
                T.values.column(1), // 0 diagonal
                T.values.column(2), // +1 diagonal
                T.values.column(0), // -1 diagonal
                Tinv);
    }
    /**
     * @brief Compute the inverse of a tridiagonal matrix T
     *
     * @param T tridiagonal matrix
     * @return the inverse of the tridiagonal matrix (coordinate format)
     **/
    CooMatrix operator()(const DiaMatrix& T)
    {
        CooMatrix Tinv;
        this->operator()( T, Tinv);
        return Tinv;
    }
     /**
     * @brief Compute the inverse of a tridiagonal matrix with diagonal vectors a,b,c
     *
     * The diagonal vectors are given as in the cusp dia_matrix format
     * @param a  "0" diagonal vector (index 0 is on row 0)
     * @param b "+1" diagonal vector (index 0 is on row 0, last index outside)
     * @param c "-1" diagonal vector (index 0 is on row 0, outside of matrix)
     * @param Tinv the inverse of the tridiagonal matrix (coordinate format)
     *  gets resized if necessary
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void operator()(const ContainerType0& a, const ContainerType1& b,
            const ContainerType2& c, CooMatrix& Tinv)
    {
        Tinv.resize(m_size, m_size,  m_size* m_size);
        value_type helper = 0.0;
        //fill phi values
        m_phi[0] = - b[0]/a[0];
        for( unsigned i = 1; i<m_size; i++)
        {
            helper = m_phi[i-1]* c[i] + a[i];
            if (helper==0) throw dg::Error( dg::Message(_ping_)<< "Failure: Division by zero\n");
            else m_phi[i]   = -b[i]/helper;
        }
//         m_phi[m_size] = 0.0;

        //fill theta values
        if (m_size == 1)  m_theta[m_size-1]  = 0.0;
        else
        {
            m_theta[m_size-1]  = - c[m_size-1]/a[m_size-1];
            for( int i = m_size-2; i>=0; i--)
            {
                helper = m_theta[i+1]*b[i] + a[i];
                if (helper==0) throw dg::Error( dg::Message(_ping_)<< "Failure: Division by zero\n");
                else m_theta[i]  = -c[i]/helper;
            }
        }
//         m_theta[0] = 0.0;
        //Diagonal entries
        Tinv.row_indices[0*m_size+0]    = 0;
        Tinv.column_indices[0*m_size+0]    = 0;
        helper = a[0] + b[0]* m_theta[1];
        if (helper==0) throw dg::Error( dg::Message(_ping_)<< "Failure: No inverse exists\n");
        else Tinv.values[0*m_size+0] = 1.0/helper;

        Tinv.row_indices[(m_size-1)*m_size+(m_size-1)]    = (m_size-1);
        Tinv.column_indices[(m_size-1)*m_size+(m_size-1)]    = (m_size-1);
        if (m_size == 1) helper = a[m_size-1];
        else helper = a[m_size-1] + c[m_size-1]*m_phi[m_size-2];

        if (helper==0) throw dg::Error( dg::Message(_ping_)<< "Failure: No inverse exists\n");
        else Tinv.values[(m_size-1)*m_size+m_size-1] = 1.0/helper;

        for( unsigned i=1; i<m_size-1; i++)
        {
            Tinv.row_indices[i*m_size+i]    = i;
            Tinv.column_indices[i*m_size+i] = i;
            helper = a[i] + c[i]*m_phi[i-1] + b[i]* m_theta[i+1];
            if (helper==0) throw dg::Error( dg::Message(_ping_)<< "Failure: No inverse exists\n");
            else Tinv.values[i*m_size+i] = 1.0/helper;
        }
        //Off-diagonal entries
        for( unsigned j=0; j<m_size-1; j++) //row index
        {
            for (unsigned i=j+1; i<m_size; i++)
            {
                Tinv.row_indices[i*m_size+j]    = i;
                Tinv.column_indices[i*m_size+j] = j;
                Tinv.values[i*m_size+j] = m_theta[i]*Tinv.values[(i-1)*m_size+j];
            }
        }
        for( unsigned j=1; j<m_size; j++) //row index
        {
            for (int i=j-1; i>=0; i--)
            {
                Tinv.row_indices[i*m_size+j]    = i;
                Tinv.column_indices[i*m_size+j] = j;
                Tinv.values[i*m_size+j] = m_phi[i]*Tinv.values[(i+1)*m_size+j];
            }
        }
    }
  private:
    ContainerType m_phi, m_theta;
    unsigned m_size;
};

/**
* @brief Compute the inverse of a general tridiagonal matrix.
*
* @attention If roughly the matrix size m>150 the algorithm is unstable. However,
*  it performs extremely fast if it stays below this value.  This is the
*  algorihm of "Inversion of a Tridiagonal Jacobi Matrix" by Riaz A. Usmani
*
* @tparam ContainerType \c thrust::host_vector<value_type> or similar shared memory host vector
* @tparam CooMatrix \c cusp::coo_matrix<int, value_type, cusp::host_memory>;
* @tparam DiaMatrix \c cusp::dia_matrix<int, value_type, cusp::host_memory>;
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
    }
    /**
     * @brief Compute the inverse of a tridiagonal matrix T
     *
     * @param T tridiagonal matrix
     * @param Tinv the inverse of the tridiagonal matrix (coordinate format)
     *  gets resized if necessary
     **/
    void operator()(const DiaMatrix& T, CooMatrix& Tinv)
    {
        this->operator()(
                T.values.column(1), // 0 diagonal
                T.values.column(2), // +1 diagonal
                T.values.column(0), // -1 diagonal
                Tinv);
    }
    /**
     * @brief Compute the inverse of a tridiagonal matrix T
     *
     * @param T tridiagonal matrix
     * @return the inverse of the tridiagonal matrix (coordinate format)
     **/
    CooMatrix operator()(const DiaMatrix& T)
    {
        CooMatrix Tinv;
        this->operator()( T, Tinv);
        return Tinv;
    }
     /**
     * @brief Compute the inverse of a tridiagonal matrix with diagonal vectors a,b,c
     *
     * The diagonal vectors are given as in the cusp dia_matrix format
     * @param a  "0" diagonal vector (index 0 is on row 0)
     * @param b "+1" diagonal vector (index 0 is on row 0, last index outside)
     * @param c "-1" diagonal vector (index 0 is on row 0, outside of matrix)
     * @param Tinv the inverse of the tridiagonal matrix (coordinate format)
     *  gets resized if necessary
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void operator()(const ContainerType0& a, const ContainerType1& b,
            const ContainerType2& c, CooMatrix& Tinv)
    {
        Tinv.resize( m_size, m_size, m_size*m_size);
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
                m_theta[i] = a[i-1] * m_theta[i-1] - b[i-2] * c[i-1]  * m_theta[i-2];
                m_phi[is]  = a[is]  * m_phi[is+1]  - b[is]  * c[is+1] * m_phi[is+2];
            }
        }

        //Compute inverse tridiagonal matrix elements
        for( unsigned i=0; i<m_size; i++) //row index
        {
            for( unsigned j=0; j<m_size; j++) //column index
            {
                Tinv.row_indices[i*m_size+j]    = i;
                Tinv.column_indices[i*m_size+j] = j;
                if (i<j) {
                    Tinv.values[i*m_size+j] =
                        std::accumulate(std::next(b.begin(),i),
                                std::next(b.begin(),j), 1.,
                                std::multiplies<value_type>())*sign(i+j) *
                        m_theta[i] * m_phi[j+1]/m_theta[m_size];
                }
                else if (i==j)
                {
                    Tinv.values[i*m_size+j] =  m_theta[i] * m_phi[i+1]/m_theta[m_size];
                }
                else // if (i>j)
                {
                    Tinv.values[i*m_size+j] =
                        std::accumulate(std::next(c.begin(),j+1),
                                std::next(c.begin(),i+1), 1.,
                                std::multiplies<value_type>())*sign(i+j) *
                        m_theta[j] * m_phi[i+1]/m_theta[m_size];
                }
            }
        }
    }
  private:
    /// \f$ (-1)^i\f$
    int sign(unsigned i)
    {
        if (i%2==0) return 1;
        else return -1;
    }
    ContainerType m_phi, m_theta;
    unsigned m_size;
};

/**
 * @brief Invert a tridiagonal matrix
 *
 * This is a convenience shortcut for
 * @code{.cpp}
 *   return dg::TridiagInvDF( size)(T, Tinv);
 * @endcode
 * @tparam value_type real type
 * @param T
 * @param Tinv (gets resized if necessary)
 */
template<class value_type>
void invert(const cusp::dia_matrix<int,value_type,cusp::host_memory>& T,
        cusp::coo_matrix<int,value_type,cusp::host_memory>& Tinv)
{
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HVec = dg::HVec;
    return TridiagInvDF<HVec,HDiaMatrix,HCooMatrix>( T.num_rows)(T, Tinv);
}
/**
 * @brief Invert a tridiagonal matrix
 *
 * This is a convenience shortcut for
 * @code{.cpp}
 *   return dg::TridiagInvDF( size)(T);
 * @endcode
 * @tparam value_type real type
 * @param T
 * @return Tinv
 */
template<class value_type>
cusp::coo_matrix<int,value_type,cusp::host_memory> invert(
        const cusp::dia_matrix<int,value_type,cusp::host_memory>& T)
{
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HVec = dg::HVec;
    return TridiagInvDF<HVec,HDiaMatrix,HCooMatrix>( T.num_rows)(T);
}

/**
 * @brief Compute extreme Eigenvalues of a symmetric tridiangular matrix
 *
 * @code{.cpp}
 *  dg::mat::UniversalLanczos lanczos( A.weights(), 20);
 *  auto T = lanczos.tridiag( A, A.weights(), A.weights());
 *  auto EV = dg::mat::compute_extreme_EV( T);
 *  // EV[0] is the minimum, EV[1] the maximum Eigenvalue
 * @endcode
 * @tparam value_type real type
 * @param T symmetric tridiangular matrix
 * @return {EVmin, EVmax}
 */
template<class value_type>
std::array<value_type, 2> compute_extreme_EV( const cusp::dia_matrix<int,value_type, cusp::host_memory>& T)
{
    unsigned size = T.num_rows;
    cusp::array2d< value_type, cusp::host_memory> evecs;
    cusp::array1d< value_type, cusp::host_memory> evals(size);
    cusp::lapack::stev(T.values.column(1), T.values.column(2),
            evals, evecs, 'N');
    value_type EVmax = dg::blas1::reduce( evals, 0., dg::AbsMax<value_type>());
    value_type EVmin = dg::blas1::reduce( evals, EVmax, dg::AbsMin<value_type>());
    return std::array<value_type, 2>{EVmin, EVmax};
}


///@}

} // namespace mat
} // namespace dg
