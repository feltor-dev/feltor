#ifndef _DG_CHEB_
#define _DG_CHEB_

#include <cmath>

#include "blas.h"

/*!@file
 * Polynomial Preconditioners and solvers
 */

namespace dg
{

/**
* @brief Preconditioned Chebyshev iteration for solving
* \f$ PAx=Pb\f$
*
* Chebyshev iteration is not well-suited for solving matrix equations
* on its own. Rather, it is suited as a smoother for a multigrid algorithm
* and also as a Preconditioner for the Conjugate Gradient method.
* It does not contain scalar products,
* which makes it appaeling for both small and highly parallelized systems.
*
* Given the minimum and maximum Eigenvalue of the matrix A we define
* \f[ \theta = (\lambda_\min+\lambda_\max)/2 \quad \delta = (\lambda_\max - \lambda_\min)/2 \\
*     \rho_0 := \frac{\delta}{\theta},\ x_0 := x, \ x_{1} = x_0+\frac{1}{\theta} P(b-Ax_0) \\
*     \rho_{k}:=\left(\frac{2\theta}{\delta}-\rho_{k-1}\right)^{-1} \\
*     x_{k+1} := x_k + \rho_k\left( \rho_{k-1}(x_k - x_{k-1})
*     + \frac{2}{\delta} P( b - Ax_k) \right)
* \f]
* The preconditioned version is obtained by applying the regular version to
 * \f$ \bar A\bar x = \bar b\f$ with \f$ \bar A
 * := {E^{-1}}^\mathrm{T} A E^{-1} \f$, \f$ \bar x := Ex\f$ and \f$ \bar b :=
 * {E^{-1}}^\mathrm{T}\f$, where \f$ P = {E^{-1}}^\mathrm{T} E^{-1}\f$ is
 * the preconditioner. The bounds on the spectrum then need to be on the \f$PA\f$ matrix.
* @note The maximum Eigenvalue of \f$ A\f$ and \f$ P A\f$ can be estimated
* using the \c EVE class.
* \sa For more information see the book
* <a href="https://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf">"Iteratvie Methods for Sparse Linear Systems" 2nd edition by Yousef Saad </a>
* @note If the initial vector is zero Chebyshev iteration will produce the
* Chebyshev polynomial \f$ C_k( A) b\f$ applied to the right hand side
* and the preconditioned version produces
* \f$ C_{k-1}(PA)Pb = E^{-1} C_{k-1}(
* {E^{-1}}^\mathrm{T} A E^{-1}){E^{-1}}^\mathrm{T}\f$
*
* @attention Chebyshev iteration may diverge if the elliptical bound of the
* Eigenvalues is not accurate (in particular if \f$\lambda_\max\f$ is underestimated) or if an ellipsis is not a good fit for the
* spectrum of the matrix
*
* @ingroup invert
*
* @copydoc hide_ContainerType
*/
template< class ContainerType>
class ChebyshevIteration
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    ChebyshevIteration() = default;
    ///@copydoc construct()
    ChebyshevIteration( const ContainerType& copyable):
        m_ax(copyable), m_z( m_ax), m_xm1(m_ax){}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_ax;}

    /**
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     */
    void construct( const ContainerType& copyable) {
        m_xm1 = m_z = m_ax = copyable;
    }
    /**
     * @brief Solve the system \f$ Ax = b\f$ using \c num_iter Chebyshev iteration
     *
     * The iteration stops when the maximum number of iterations is reached
     * @param A A symmetric, positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param min_ev an estimate of the minimum Eigenvalue
     * @param max_ev an estimate of the maximum Eigenvalue of \f$ A\f$ (must be larger than \c min_ev)
     * Use \c EVE to get this value
     * @param num_iter the number of iterations \c k (equals the number of times A is applied)
     * If 0 the function returns immediately
     * @param x_is_zero If true, the first matrix-vector multiplication is avoided
     * by assuming x is zero. (This works even if x is not actually 0)
     * This is in particular in the case when Chebyshev Iteration is used as a Preconditioner
     * @note In the \c x_is_zero mode \c k iterations  will produce the \c k-1
     * Chebyshev polynomial applied to
     * the right hand side \f$ x = C_{k-1}(A)b\f$
     *
     * @copydoc hide_matrix
     * @copydoc hide_ContainerType
     */
    template< class MatrixType, class ContainerType0, class ContainerType1>
    void solve( MatrixType&& A, ContainerType0& x, const ContainerType1& b,
        value_type min_ev, value_type max_ev, unsigned num_iter, bool x_is_zero = false)
    {
        if( num_iter == 0)
            return;
        assert ( min_ev < max_ev);
        value_type theta = (min_ev+max_ev)/2., delta = (max_ev-min_ev)/2.;
        value_type rhokm1 = delta/theta, rhok=0;
        if( !x_is_zero)
        {
            dg::blas1::copy( x, m_xm1); //x_{k-1}
            dg::blas2::symv( std::forward<MatrixType>(A), x, m_ax);
            dg::blas1::axpbypgz( 1./theta, b, -1./theta, m_ax, 1., x); //x_1
        }
        else
        {
            dg::blas1::copy( 0., m_xm1); //x_{k-1}
            dg::blas1::axpby( 1./theta, b, 0., x); //x_1
        }
        for ( unsigned k=1; k<num_iter; k++)
        {
            rhok = 1./(2.*theta/delta - rhokm1);
            dg::blas2::symv( std::forward<MatrixType>(A), x, m_ax);
            dg::blas1::evaluate( m_xm1, dg::equals(), PairSum(),
                             1.+rhok*rhokm1, x,
                            -rhok*rhokm1,    m_xm1,
                             2.*rhok/delta,  b,
                            -2.*rhok/delta,  m_ax
                            );
            using std::swap;
            swap( x, m_xm1);
            rhokm1 = rhok;
        }
    }
    /**
     * @brief Solve the system \f$ PAx = Pb\f$ using \c num_iter Preconditioned Chebyshev iteration
     *
     * The iteration stops when the maximum number of iterations is reached
     * @param A A symmetric, positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P the Preconditioner (\f$ M^{-1}\f$ in the above notation
     * @param min_ev an estimate of the minimum Eigenvalue
     * @param max_ev an estimate of the maximum Eigenvalue of \f$ P A\f$ (must be larger than \c min_ev)
     * Use \c EVE to get this value
     * @param num_iter the number of iterations k (equals the number of times \c A is applied)
     * If 0 the function returns immediately
     * @param x_is_zero If true, the first matrix-vector multiplication is avoided
     * by assuming x is zero. (This works even if x is not actually 0)
     * This is in particular in the case when Chebyshev Iteration is used as a Preconditioner
     * @note In the \c x_is_zero mode \c k iterations  will produce the \c k-1 Chebyshev polynomial applied to
     * the right hand side \f$ x = C_{k-1}(PA)Pb = E^{-1} C_{k-1}(
     * {E^{-1}}^\mathrm{T} A E^{-1}){E^{-1}}^\mathrm{T}\f$
     *
     * @copydoc hide_matrix
     * @copydoc hide_ContainerType
     */
    template< class MatrixType0, class MatrixType1, class ContainerType0, class ContainerType1>
    void solve( MatrixType0&& A, ContainerType0& x, const ContainerType1& b,
            MatrixType1&& P, value_type min_ev, value_type max_ev, unsigned num_iter,
            bool x_is_zero = false)
    {
        if( num_iter == 0)
            return;
        assert ( min_ev < max_ev);
        value_type theta = (min_ev+max_ev)/2., delta = (max_ev-min_ev)/2.;
        value_type rhokm1 = delta/theta, rhok=0;
        if( !x_is_zero)
        {
            dg::blas1::copy( x, m_xm1); //x_{k-1}
            dg::blas2::symv( std::forward<MatrixType0>(A), x, m_ax);
            dg::blas1::axpby( 1., b, -1., m_ax); //r_0
            dg::blas2::symv( std::forward<MatrixType1>(P), m_ax, m_z);
            dg::blas1::axpby( 1./theta, m_z, 1., x); //x_{k-1}
        }
        else
        {
            dg::blas2::symv( std::forward<MatrixType1>(P), b, x);
            if( num_iter == 1) return;
            dg::blas1::scal( m_xm1, 0.);
            dg::blas1::scal( x, 1./theta);
        }
        for ( unsigned k=1; k<num_iter; k++)
        {
            rhok = 1./(2.*theta/delta - rhokm1);
            dg::blas2::symv( std::forward<MatrixType0>(A), x, m_ax);
            dg::blas1::axpby( 1., b, -1., m_ax); //r_k
            dg::blas2::symv( P, m_ax, m_z);
            dg::blas1::axpbypgz(
                             1.+rhok*rhokm1, x,
                             2.*rhok/delta,  m_z,
                            -rhok*rhokm1,    m_xm1
                            );
            using std::swap;
            swap( x, m_xm1);
            rhokm1 = rhok;
        }
    }
  private:
    ContainerType m_ax, m_z, m_xm1;
};

 /** @class hide_polynomial
 *
 * @note This class can be used as a Preconditioner in the CG algorithm. The CG
 * algorithm forms an approximation to the solution in the form \f$ x_{k+1} =
 * x_0 + P_k(A) r_0\f$ where \f$ P_k\f$ is a polynomial of degree \c k, which
 * is optimal in minimizing the A-norm. Thus a polynomial preconditioner cannot
 * decrease the number of matrix-vector multiplications needed to achieve a
 * certain accuracy.  However, since polynomial preconditioners do not use scalar products they may
 * offset the increased overhead if the dot product becomes a bottleneck for performance or scalability.
  */

/**
 * @brief Chebyshev Polynomial Preconditioner \f$ C( A)\f$
 *
 * @copydoc hide_polynomial
 * @sa ChebyshevIteration
 * @tparam Matrix Preferably a reference type
 * @tparam ContainerType
 * @ingroup invert
 */
template<class Matrix, class ContainerType>
struct ChebyshevPreconditioner
{
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    /**
     * @brief  Construct the k-th Chebyshev Polynomial
     *
     * @param op The Matrix (copied, so maybe choose a reference type for shallow copying) will be called as \c dg::blas2::symv( op, x, y)
     * @param copyable A ContainerType must be copy-constructible from this
     * @param ev_min an estimate of the minimum Eigenvalue (It is important to get a good value here. Unfortunately, we currently have no perfect way of getting this value, as a suggestion use \c 0.01*max_ev)
     * @param ev_max an estimate of the maximum Eigenvalue of \f$ A\f$ (must be larger than \c min_ev)
     * Use \c EVE to get this value
     * @param degree degree k of the Polynomial (5 should be a good number)
     */
    ChebyshevPreconditioner( Matrix op, const ContainerType& copyable, value_type ev_min,
            value_type ev_max, unsigned degree):
        m_op(op), m_ch( copyable),
        m_ev_min(ev_min), m_ev_max(ev_max), m_degree(degree){}

    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y)
    {
        //m_ch.solve( m_op, y, x, m_op.precond(), m_ev_min, m_ev_max, m_degree+1, true);
        m_ch.solve( m_op, y, x, m_ev_min, m_ev_max, m_degree+1, true);
    }
    private:
    Matrix m_op;
    ChebyshevIteration<ContainerType> m_ch;
    value_type m_ev_min, m_ev_max;
    unsigned m_degree;
};

/**
 * @brief Approximate inverse Chebyshev Polynomial Preconditioner \f$ A^{-1} = \frac{c_0}{2} I + \sum_{k=1}^{r}c_kT_k( Z)\f$
 *
 * This is the polynomial preconditioner as proposed by <a href="https://ieeexplore.ieee.org/document/1245544">Dag and Semlyen, A New Preconditioned Conjugate Gradient Power Flow, IEEE Transactions on power Systems, 18, (2003)</a>
 * We have \f$ c_k = \sqrt{\lambda_\min\lambda_\max}^{-1} (\sqrt{\lambda_\min/\lambda_\max}-1)^k / (\sqrt{\lambda_\min/\lambda_\max }+ 1)^k\f$ and \f$ Z = 2 ( A - (\lambda_\max + \lambda_\min)I/2)/(\lambda_\max-\lambda_\min)\f$
 *
 * They propose to use \f$ \lambda_\min = \lambda_\max / (5r)\f$ where r is the degree
 * of the polynomial
 * @copydoc hide_polynomial
 * @tparam Matrix Preferably a reference type
 * @tparam ContainerType
 * @ingroup invert
 */
template<class Matrix, class ContainerType>
struct ModifiedChebyshevPreconditioner
{
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    /**
     * @brief  Construct the k-th Chebyshev Polynomial approximate
     *
     * @param op The Matrix (copied, so maybe choose a reference type for shallow copying) will be called as \c dg::blas2::symv( op, x, y)
     * @param copyable A ContainerType must be copy-constructible from this
     * @param ev_min an estimate of the minimum Eigenvalue (It is important to
     * get a good value here. The authors propose to use
     * \f$ \lambda_\min = \lambda_\max / (5r)\f$ where \c r is the \c degree
     * @param ev_max an estimate of the maximum Eigenvalue of \f$ A\f$ (must be larger than \c min_ev)
     * Use \c EVE to get this value
     * @param degree degree k of the Polynomial (5 should be a good number)
     */
    ModifiedChebyshevPreconditioner( Matrix op, const ContainerType& copyable, value_type ev_min,
            value_type ev_max, unsigned degree):
        m_op(op), m_ax(copyable), m_z1(m_ax), m_z2(m_ax),
        m_ev_min(ev_min), m_ev_max(ev_max), m_degree(degree){}

    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y)
    {
        value_type theta = (m_ev_min+m_ev_max)/2., delta = (m_ev_max-m_ev_min)/2.;
        value_type c_k = 1./sqrt(m_ev_min*m_ev_max);
        dg::blas1::axpby( c_k/2., x, 0., y);
        if( m_degree == 0) return;
        dg::blas2::symv( m_op, x, m_ax);
        dg::blas1::axpby( 1./delta, m_ax, -theta/delta, x, m_z1); //T_{k-1} x
        c_k *= (sqrt( m_ev_min/m_ev_max) - 1.)/(sqrt(m_ev_min/m_ev_max)+1);
        dg::blas1::axpby( c_k, m_z1, 1., y);
        if( m_degree == 1) return;
        dg::blas1::copy( x, m_z2); //T_{k-2} x
        for( unsigned i=1; i<m_degree; i++)
        {
            dg::blas2::symv( m_op, m_z1, m_ax);
            dg::blas1::axpby( 1./delta, m_ax, -theta/delta, m_z1, m_ax); //Z T_{k-1}
            dg::blas1::axpby( 2., m_ax, -1., m_z2, m_z2); //T_k
            c_k *= (sqrt( m_ev_min/m_ev_max) - 1.)/(sqrt(m_ev_min/m_ev_max)+1);
            dg::blas1::axpby( c_k, m_z2, 1., y);
            using std::swap;
            swap(m_z1,m_z2);
        }
    }
    private:
    Matrix m_op;
    ContainerType m_ax, m_z1, m_z2;
    value_type m_ev_min, m_ev_max;
    unsigned m_degree;
};

/**
 * @brief Least Squares Polynomial Preconditioner \f$ M^{-1} s( AM^{-1})\f$
 *
 * Implements the least squares polynomial preconditioner as suggested by
 * <a href= "https://doi.org/10.1137/0906059"> Yousef Saad, Practical Use of Polynomial Preconditionings for the Conjugate Gradient Method,SIAM J. Sci. and Stat. Comput., 6(4), 865–881 (1985) </a>
 * @note The least squares polynomial might (or might not) perform
 * better than Chebyshev Polynomials and does not need an estimate of the
 * lowest Eigenvalue
 * @copydoc hide_polynomial
 *
 * \sa For more information see the book
 * <a href="https://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf">"Iteratvie Methods for Sparse Linear Systems" 2nd edition by Yousef Saad </a>
 * @tparam Matrix Preferably a reference type
 * @tparam InnerPreconditioner Preferably a reference type
 * @copydoc hide_ContainerType
 * @ingroup invert
 */
template<class Matrix, class InnerPreconditioner, class ContainerType>
struct LeastSquaresPreconditioner
{
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    /**
     * @brief  Construct k-th Least Squares Polynomial
     *
     * @param op The Matrix (copied, so maybe choose a reference type for shallow copying) will be called as \c dg::blas2::symv( op, x, y)
     * @param P The inner Preconditioner (copied, so maybe choose a reference type for shallow copying) will be called as \c dg::blas2::symv( op, x, y)
     * @param copyable A ContainerType must be copy-constructible from this
     * @param ev_max An estimate of the largest Eigenvalue of \f$ M^{-1} A\f$. Use \c EVE to get this value
     * @param degree degree k of the Polynomial (5 should be a good number - only up to degree 10 polynomials are implemented at the moment)
     */
    LeastSquaresPreconditioner( Matrix op, InnerPreconditioner P, const ContainerType& copyable, value_type ev_max, unsigned degree):
        m_op(op), m_p(P), m_z(copyable),
        m_ev_max( ev_max), m_degree(degree){
            m_c = coeffs(degree);
    }
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_z;}

    template<class ContainerType0, class ContainerType1>
    void symv( const ContainerType0& x, ContainerType1& y)
    {
        //Horner scheme
        dg::blas1::axpby( m_c[m_degree],x, 0., m_z);
        for( int i=m_degree-1; i>=0; i--)
        {
            //dg::blas1::copy( m_z, y);
            dg::blas2::symv( m_p, m_z, y);
            dg::blas2::symv( m_op, y, m_z);
            dg::blas1::axpby( m_c[i], x, +4./m_ev_max, m_z);
        }
        //dg::blas1::copy( m_z, y);
        dg::blas2::symv( m_p, m_z, y);
    }
    private:
    std::vector<value_type> coeffs( unsigned degree){
        switch( degree){
            case 0: return {1.};
            case 1: return {5., -1.};
            case 2: return { 14., -7., 1.};
            case 3: return {30., -27., 9., -1.};
            case 4: return {55., -77., 44., -11., 1.};
            case 5: return {91., -182., 156., -65., 13., -1. };
            case 6: return {140., -378., 450., -275., 90., -15., 1. };
            case 7: return {204., -714.,1122., -935., 442., -119., 17., -1.};
            case 8: return {285.,-1254., 2508., -2717., 1729., -665., 152., -19., 1.};
            case 9: return {385., -2079., 5148.,-7007.,5733.,-2940.,952.,-189.,21.,-1.};
            default:
                if (degree > 10)
                    std::cerr << "WARNING: LeastSquares Case "<<degree<<" not implemented. Taking 10 instead!\n";
                return {506., -3289., 9867.,-16445.,16744.,-10948.,4692.,-1311.,230.,-23.,1. };
        };
    }
    std::vector<value_type> m_c;
    Matrix m_op;
    InnerPreconditioner m_p;
    ContainerType m_z;
    value_type m_ev_max;
    unsigned m_degree;
};

///@cond
template<class M, class V>
struct TensorTraits<ChebyshevPreconditioner<M,V>>
{
    using value_type      = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
template<class M, class V>
struct TensorTraits<ModifiedChebyshevPreconditioner<M,V>>
{
    using value_type      = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};
template<class M, class P, class V>
struct TensorTraits<LeastSquaresPreconditioner<M,P,V>>
{
    using value_type      = get_value_type<V>;
    using tensor_category = SelfMadeMatrixTag;
};

///@endcond

//template<class Matrix, class Container>
//struct WrapperSpectralShift
//{
//    WrapperSpectralShift( Matrix& op, value_type ev_max):
//        m_op(op), m_ev_max(ev_max){}
//    template<class ContainerType0, class ContainerType1>
//    void symv( const ContainerType0& x, ContainerType1& y)
//    {
//        dg::blas1::axpby( m_ev_max, x, 0., y);
//        dg::blas2::symv( -1., m_op, x, 1., y);
//    }
//
//    private:
//    Matrix& m_op;
//    value_type m_ev_max;
//
//};
//template<class M, class V>
//struct TensorTraits<detail::WrapperSpectralShift<M,V>>
//{
//    using value_type      = get_value_type<V>;
//    using tensor_category = SelfMadeMatrixTag;
//};

} //namespace dg

#endif // _DG_CHEB_
