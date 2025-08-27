#pragma once

#include <boost/math/special_functions.hpp> // has to be included before lapack in certain versions
#include "dg/algorithm.h"

#include "functors.h"
/**
* @brief Classes for Krylov space approximations of a Matrix-Vector product
*/

namespace dg{
namespace mat{
///@cond
namespace lapack
{
// MW: Update 27.5.2025
// Unfortunately, the CMake support for LAPACKE is rather not straightforward
// while LAPACK is supported out of the box with find_package( LAPACK)
// so we rather call the fortran functions from C directly ourselves.
//
// This is how you add new routines:
// 1. Go find relevant Fortran routine ( usually there are separate routine for
// each value type; single, double  ,complex )
// FORTRAN     https://www.netlib.org/lapack/explore-html/index.html
//
// 2. Call the fortran function from C packaged in a nice C++ interface!
// We here follow
// https://scicomp.stackexchange.com/questions/26395/how-to-start-using-lapack-in-c
// 2.1 Add the extern "C" binding below, where all parameters are pointers
// 2.2 When matrices are involved note that the Fortran ordering is "column
// major", which is the transpose of how e.g. our SquareMatrix is ordered)
// 2.3. Replace all arrays with a ContainerType in our interface
// 2.4. Use C++-17 if constexpr to dispatch value type
//
extern "C" {
extern void dstev_(char*,int*,double*,double*,double*,int*,double*,int*);
extern void sstev_(char*,int*,float*,float*,float*,int*,float*,int*);
extern void dsygv_(int*,char*,char*,int*,double*,int*,double*,int*,double*,double*,int*,int*);
extern void ssygv_(int*,char*,char*,int*,float*,int*,float*,int*,float*,float*,int*,int*);
}
// Compute Eigenvalues and, optionally, Eigenvectors of a real symmetric tridiagonal matrix A
template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3>
void stev(
    char job,          // 'N' Compute Eigenvalues only, 'V' Compute Eigenvalues and Eigenvectors
    ContainerType0& D, // diagonal of T on input, Eigenvalues (in ascending order) on output
    ContainerType1& E, // subdiagonal of T on input |size D.size()-1 ; in E[0] - E[D.size()-2]|; destroyed on output
    ContainerType2& Z, // IF job = 'V' && column major then the i-th column contains i-th EV, if job = 'N' not referenced
    ContainerType3& work // If job = 'V' needs size max( 1, 2*D.size() - 2), else not referenced
    )
{
    using value_type = dg::get_value_type<ContainerType0>;
    static_assert( std::is_same_v<value_type, double> or std::is_same_v<value_type, float>,
                   "Value type must be either float or double");
    static_assert( std::is_same_v<dg::get_value_type<ContainerType1>, value_type> &&
                   std::is_same_v<dg::get_value_type<ContainerType2>, value_type> &&
                   std::is_same_v<dg::get_value_type<ContainerType3>, value_type>,
                   "All Vectors must have same value type");
    static_assert( std::is_same_v<dg::get_execution_policy<ContainerType0>, dg::SerialTag> &&
                   std::is_same_v<dg::get_execution_policy<ContainerType1>, dg::SerialTag> &&
                   std::is_same_v<dg::get_execution_policy<ContainerType2>, dg::SerialTag> &&
                   std::is_same_v<dg::get_execution_policy<ContainerType3>, dg::SerialTag>,
                   "All Vectors must have serial execution policy");

    // job = 'N' Compute Eigenvalues only
    // job = 'V' Compute Eigenvalues and Eigenvectors
    int N = D.size();
    value_type * D_ptr = thrust::raw_pointer_cast( &D[0]);
    value_type * E_ptr = thrust::raw_pointer_cast( &E[0]);
    value_type * Z_ptr = nullptr;
    int ldz = N;
    value_type * work_ptr = nullptr;
    if( job == 'V')
    {
        Z_ptr = thrust::raw_pointer_cast( &Z[0]);
        work_ptr = thrust::raw_pointer_cast( &work[0]);
    }

    int info;
    if constexpr ( std::is_same_v<value_type, double>)
        dstev_( &job, &N, D_ptr, E_ptr, Z_ptr, &ldz, work_ptr, &info);
    else if constexpr ( std::is_same_v<value_type, float>)
        sstev_( &job, &N, D_ptr, E_ptr, Z_ptr, ldz, work_ptr, &info);
    if( info != 0)
    {
        throw dg::Error( dg::Message(_ping_) << "stev failed with error code "<<info<<"\n");
    }
}


// Look for dsygv on Lapack docu!
// Compute Eigenvalues and, optionally, Eigenvectors of a real symmetric matrix system A x = lambda B x
template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3>
void sygv(
    int itype, // 1: A*x = (\Lambda)*B*x, 2: A*B*x = (\Lambda)*x, 3: B*A*x = (\Lambda)*x
    char jobz, // 'N' Compute Eigenvalues only, 'V' Compute Eigenvalues and Eigenvectors
    char uplo, // 'U' Upper triangles of A and B are stored; 'L' Lower triangles of A and B
    int N,
    ContainerType0& A, // matrix A [LDA rows, N cols], contains Eigenvecs on output if requested
    int lda,
    ContainerType1& B, // matrix B [LDB rows, N cols], destroyed on output
    int ldb,
    ContainerType2& W,   // [out] Eigenvalues in ascending order
    ContainerType3& work // Workspace Size 3*N-1
    )
{
    using value_type = dg::get_value_type<ContainerType0>;
    static_assert( std::is_same_v<value_type, double> or std::is_same_v<value_type, float>,
                   "Value type must be either float or double");
    static_assert( std::is_same_v<dg::get_value_type<ContainerType1>, value_type> &&
                   std::is_same_v<dg::get_value_type<ContainerType2>, value_type> &&
                   std::is_same_v<dg::get_value_type<ContainerType3>, value_type>,
                   "All Vectors must have same value type");
    static_assert( std::is_same_v<dg::get_execution_policy<ContainerType0>, dg::SerialTag> &&
                   std::is_same_v<dg::get_execution_policy<ContainerType1>, dg::SerialTag> &&
                   std::is_same_v<dg::get_execution_policy<ContainerType2>, dg::SerialTag> &&
                   std::is_same_v<dg::get_execution_policy<ContainerType3>, dg::SerialTag>,
                   "All Vectors must have serial execution policy");

    // jobz = 'N' Compute Eigenvalues only
    // jobz = 'V' Compute Eigenvalues and Eigenvectors
    value_type * A_ptr = thrust::raw_pointer_cast( &A[0]);
    value_type * B_ptr = thrust::raw_pointer_cast( &B[0]);
    value_type * W_ptr = thrust::raw_pointer_cast( &W[0]);
    value_type * work_ptr = thrust::raw_pointer_cast( &work[0]);
    int work_size = (int)work.size();

    int info;
    if constexpr ( std::is_same_v<value_type, double>)
        dsygv_( &itype, &jobz, &uplo, &N, A_ptr, &lda, B_ptr, &ldb, W_ptr, work_ptr, &work_size, &info);
    else if constexpr ( std::is_same_v<value_type, float>)
        ssygv_( &itype, &jobz, &uplo, &N, A_ptr, &lda, B_ptr, &ldb, W_ptr, work_ptr, &work.size, &info);
    if( info != 0)
    {
        throw dg::Error( dg::Message(_ping_) << "sygv failed with error code "<<info<<"\n");
    }
}




}
///@endcond

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
value_type compute_Tinv_m1( const dg::TriDiagonal<thrust::host_vector<value_type>>
        & T, unsigned size)
{
    // P = Plus diagonal
    // O = zerO diagonal
    // M = Minus diagonal
    value_type ci = T.P[0]/T.O[0], ciold = 0.;
    value_type di = 1./T.O[0], diold = 0.;
    for( unsigned i=1; i<size; i++)
    {
        ciold = ci, diold = di;
        ci = T.P[i]/ ( T.O[i]-T.M[i]*ciold);
        di = -T.M[i]*diold/(T.O[i]-T.M[i]*ciold);
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
void compute_Tinv_y( const dg::TriDiagonal<thrust::host_vector<value_type>>
        & T,
        thrust::host_vector<value_type>& x,
        const thrust::host_vector<value_type>& y, value_type a  = 1.,
        value_type d = 0.)
{
    unsigned size = y.size();
    x.resize(size);
    thrust::host_vector<value_type> ci(size), di(size);
    ci[0] = a*T.P[0]/( a*T.O[0] + d);
    di[0] = y[0]/( a*T.O[0] + d);
    for( unsigned i=1; i<size; i++)
    {
        ci[i] = a*T.P[i]/ ( a*T.O[i] + d -a*T.M[i]*ci[i-1]);
        di[i] = (y[i]-a*T.M[i]*di[i-1])/(a*T.O[i] + d
                -a*T.M[i]*ci[i-1]);
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
* @tparam real_type float or double
*/
template< class real_type>
class TridiagInvHMGTI
{
  public:
    using value_type = real_type; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    TridiagInvHMGTI(){}
    /**
     * @brief Construct from vector
     *
     * @param copyable vector
     */
    TridiagInvHMGTI(const thrust::host_vector<real_type>& copyable)
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
    void operator()(const dg::TriDiagonal<thrust::host_vector<real_type>>& T, dg::SquareMatrix<real_type>& Tinv)
    {
        this->operator()(
                T.O, // 0 diagonal
                T.P, // +1 diagonal
                T.M, // -1 diagonal
                Tinv);
    }
    /**
     * @brief Compute the inverse of a tridiagonal matrix T
     *
     * @param T tridiagonal matrix
     * @return the inverse of the tridiagonal matrix (coordinate format)
     **/
    dg::SquareMatrix<real_type> operator()(const dg::TriDiagonal<thrust::host_vector<real_type>>& T)
    {
        dg::SquareMatrix<real_type> Tinv;
        this->operator()( T, Tinv);
        return Tinv;
    }
     /**
     * @brief Compute the inverse of a tridiagonal matrix with diagonal vectors a,b,c
     *
     * The diagonal vectors are given as in the \c dg::TriDiagonal matrix format
     * @param a  "0" diagonal vector (index 0 is on row 0, \c O in \c dg::TriDiagonal)
     * @param b "+1" diagonal vector (index 0 is on row 0, last index outside, \c P in \c dg::TriDiagonal)
     * @param c "-1" diagonal vector (index 0 is on row 0, outside of matrix, \c M in \c dg::TriDiagonal)
     * @param Tinv the inverse of the tridiagonal matrix (coordinate format)
     *  gets resized if necessary
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void operator()(const ContainerType0& a, const ContainerType1& b,
            const ContainerType2& c, dg::SquareMatrix<real_type>& Tinv)
    {
        unsigned ss = m_size;
        Tinv.resize(ss);
        if( ss == 1)
        {
            Tinv(0,0) = 1./b[0];
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
        Tinv(0, 0) = 1.0/(a[0]-c[1]*b[0]*m_betas[2]/m_betas[1]);
        Tinv(ss-1, ss-1) = 1.0/(a[ss-1] -
                c[ss-1]*b[ss-2]*m_alphas[ss-2]/m_alphas[ss-1]);
        for( unsigned i=1; i<ss-1; i++)
        {
            Tinv( i,i) =
                1.0/(a[i]-c[i]*b[i-1]*m_alphas[i-1]/m_alphas[i]
                         -c[i+1]*b[i]*m_betas[i+2]/m_betas[i+1]);
        }
        //Off-diagonal entries
        for( unsigned i=0; i<ss; i++)
        {
            for( unsigned j=0; j<ss; j++)
            {
                if (i<j) {
                    Tinv(i, j) =
                        sign(j-i)*std::accumulate(std::next(b.begin(),i),
                                std::next(b.begin(),j), 1.,
                                std::multiplies<value_type>())*
                        m_alphas[i]/m_alphas[j]*Tinv(j,j);
                }
                else if (i>j)
                {
                    Tinv(i, j) =
                        sign(i-j)*std::accumulate(std::next(c.begin(),j+1),
                                std::next(c.begin(),i+1), 1.,
                                std::multiplies<value_type>())*
                        m_betas[i+1]/m_betas[j+1]*Tinv(j,j);
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
    thrust::host_vector<real_type> m_alphas, m_betas;
    unsigned m_size;
};


/**
* @brief USE THIS ONE Compute the inverse of a general tridiagonal matrix. The algorithm does not rely on the determinant.
* @note For some special cases division by zero occurs (can be fixed if necessary cf. 2nd paper)
*  This is the algorihm of "On the inverses of general tridiagonal matrices" by Hou-Biao Li, Ting-Zhu Huang, Xing-Ping Liu, Hong Li
*  Appears to be the same as the algorithm in "ON AN INVERSE FORMULA OF A TRIDIAGONAL MATRIX" by Tomoyuki Sugimoto
*
* @tparam real_type float or double
*/
template< class real_type>
class TridiagInvDF
{
  public:
    using value_type = real_type; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    TridiagInvDF(){}
    /**
     * @brief Construct from vector
     *
     * @param copyable vector
     */
    TridiagInvDF(const thrust::host_vector<real_type>& copyable)
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
    void operator()(const dg::TriDiagonal<thrust::host_vector<real_type>>& T, dg::SquareMatrix<real_type>& Tinv)
    {
        this->operator()(
                T.O, // 0 diagonal
                T.P, // +1 diagonal
                T.M, // -1 diagonal
                Tinv);
    }
    /**
     * @brief Compute the inverse of a tridiagonal matrix T
     *
     * @param T tridiagonal matrix
     * @return the inverse of the tridiagonal matrix (coordinate format)
     **/
    dg::SquareMatrix<real_type> operator()(const dg::TriDiagonal<thrust::host_vector<real_type>>& T)
    {
        dg::SquareMatrix<real_type> Tinv;
        this->operator()( T, Tinv);
        return Tinv;
    }
     /**
     * @brief Compute the inverse of a tridiagonal matrix with diagonal vectors a,b,c
     *
     * The diagonal vectors are given as in the \c dg::TriDiagonal matrix format
     * @param a  "0" diagonal vector (index 0 is on row 0, \c O in \c dg::TriDiagonal)
     * @param b "+1" diagonal vector (index 0 is on row 0, last index outside, \c P in \c dg::TriDiagonal)
     * @param c "-1" diagonal vector (index 0 is on row 0, outside of matrix, \c M in \c dg::TriDiagonal)
     * @param Tinv the inverse of the tridiagonal matrix (coordinate format)
     *  gets resized if necessary
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void operator()(const ContainerType0& a, const ContainerType1& b,
            const ContainerType2& c, dg::SquareMatrix<real_type>& Tinv)
    {
        Tinv.resize(m_size);
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
        helper = a[0] + b[0]* m_theta[1];
        if (helper==0) throw dg::Error( dg::Message(_ping_)<< "Failure: No inverse exists\n");
        else Tinv(0,0) = 1.0/helper;

        if (m_size == 1) helper = a[m_size-1];
        else helper = a[m_size-1] + c[m_size-1]*m_phi[m_size-2];

        if (helper==0) throw dg::Error( dg::Message(_ping_)<< "Failure: No inverse exists\n");
        else Tinv( m_size -1 , m_size - 1) = 1.0/helper;

        for( unsigned i=1; i<m_size-1; i++)
        {
            helper = a[i] + c[i]*m_phi[i-1] + b[i]* m_theta[i+1];
            if (helper==0) throw dg::Error( dg::Message(_ping_)<< "Failure: No inverse exists\n");
            else Tinv(i,i) = 1.0/helper;
        }
        //Off-diagonal entries
        for( unsigned j=0; j<m_size-1; j++) //row index
        {
            for (unsigned i=j+1; i<m_size; i++)
            {
                Tinv(i,j) = m_theta[i]*Tinv(i-1, j);
            }
        }
        for( unsigned j=1; j<m_size; j++) //row index
        {
            for (int i=j-1; i>=0; i--)
            {
                Tinv(i,j) = m_phi[i]*Tinv(i+1,j);
            }
        }
    }
  private:
    thrust::host_vector<real_type> m_phi, m_theta;
    unsigned m_size;
};

/**
* @brief Compute the inverse of a general tridiagonal matrix.
*
* @attention If roughly the matrix size m>150 the algorithm is unstable. However,
*  it performs extremely fast if it stays below this value.  This is the
*  algorihm of "Inversion of a Tridiagonal Jacobi Matrix" by Riaz A. Usmani
*
* @tparam real_type float or double
*/
template< class real_type>
class TridiagInvD
{
  public:
    using value_type = real_type; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    TridiagInvD(){}
    /**
     * @brief Construct from vector
     *
     * @param copyable vector
     */
    TridiagInvD(const thrust::host_vector<real_type>& copyable)
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
    void operator()(const dg::TriDiagonal<thrust::host_vector<real_type>>& T, dg::SquareMatrix<real_type>& Tinv)
    {
        this->operator()(
                T.O, // 0 diagonal
                T.P, // +1 diagonal
                T.M, // -1 diagonal
                Tinv);
    }
    /**
     * @brief Compute the inverse of a tridiagonal matrix T
     *
     * @param T tridiagonal matrix
     * @return the inverse of the tridiagonal matrix (coordinate format)
     **/
    dg::SquareMatrix<real_type> operator()(const dg::TriDiagonal<thrust::host_vector<real_type>>& T)
    {
        dg::SquareMatrix<real_type> Tinv;
        this->operator()( T, Tinv);
        return Tinv;
    }
     /**
     * @brief Compute the inverse of a tridiagonal matrix with diagonal vectors a,b,c
     *
     * The diagonal vectors are given as in the \c dg::TriDiagonal matrix format
     * @param a  "0" diagonal vector (index 0 is on row 0, \c O in \c dg::TriDiagonal)
     * @param b "+1" diagonal vector (index 0 is on row 0, last index outside, \c P in \c dg::TriDiagonal)
     * @param c "-1" diagonal vector (index 0 is on row 0, outside of matrix, \c M in \c dg::TriDiagonal)
     * @param Tinv the inverse of the tridiagonal matrix (coordinate format)
     *  gets resized if necessary
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void operator()(const ContainerType0& a, const ContainerType1& b,
            const ContainerType2& c, dg::SquareMatrix<real_type>& Tinv)
    {
        Tinv.resize( m_size);
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
                if (i<j) {
                    Tinv(i,j) =
                        std::accumulate(std::next(b.begin(),i),
                                std::next(b.begin(),j), 1.,
                                std::multiplies<value_type>())*sign(i+j) *
                        m_theta[i] * m_phi[j+1]/m_theta[m_size];
                }
                else if (i==j)
                {
                    Tinv(i,j) =  m_theta[i] * m_phi[i+1]/m_theta[m_size];
                }
                else // if (i>j)
                {
                    Tinv(i,j) =
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
    thrust::host_vector<real_type> m_phi, m_theta;
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
void invert(const dg::TriDiagonal<thrust::host_vector<value_type>>& T,
        dg::SquareMatrix<value_type>& Tinv)
{
    TridiagInvDF<value_type>( T.O.size())(T, Tinv);
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
dg::SquareMatrix<value_type> invert(
        const dg::TriDiagonal<thrust::host_vector<value_type>>& T)
{
    return TridiagInvDF<value_type>( T.O.size())(T);
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
std::array<value_type, 2> compute_extreme_EV( const dg::TriDiagonal<thrust::host_vector<value_type>>& T)
{
    dg::SquareMatrix<value_type> evecs;
    // We use P as "subdiagonal" because it is symmetric and the first element must be on 0 index
    thrust::host_vector<value_type> evals( T.O), subdiagonal( T.P), Z, work;
    lapack::stev('N', evals,  subdiagonal, Z, work);
    value_type EVmax = dg::blas1::reduce( evals, 0., dg::AbsMax<value_type>());
    value_type EVmin = dg::blas1::reduce( evals, EVmax, dg::AbsMin<value_type>());
    return std::array<value_type, 2>{EVmin, EVmax};
}


///@}

} // namespace mat
} // namespace dg
