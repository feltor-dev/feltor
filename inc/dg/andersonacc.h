#pragma once

#include <functional>
#include "blas.h"
#include "topology/operator.h"

namespace dg{
///@cond
//const double m_EPS = 2.2204460492503131e-16;
namespace detail{

// delete 1st column and shift everything to the left once
template<class ContainerType, class value_type>
void QRdelete1( std::vector<ContainerType>& Q, dg::SquareMatrix<value_type>& R, unsigned mMax)
{
    for(unsigned i = 0; i<mMax-1;i++){
        value_type temp = sqrt(R(i,i+1)*R(i,i+1)+R(i+1,i+1)*R(i+1,i+1));
        value_type c = R(i,i+1)/temp;
        value_type s = R(i+1,i+1)/temp;
        R(i,i+1) = temp;
        R(i+1,i+1) = 0;
        if (i < mMax-2) {
            for (unsigned j = i+2; j < mMax; j++){
                temp = c * R(i,j) + s * R(i+1,j);
                R(i+1,j) = - s * R(i,j) + c * R(i+1,j);
                R(i,j) = temp;
            }
        }
        dg::blas1::subroutine( [c,s]DG_DEVICE( double& qi, double& qip) {
                double tmp = c*qi + s*qip;
                qip = - s*qi + c*qip;
                qi = tmp;
            }, Q[i], Q[i+1]);
        // Q(i + 1) = s ∗ Q(ℓ, i) + c ∗ Q(ℓ, i + 1).
        // Q(i) = c ∗ Q(ℓ, i) + s ∗ Q(ℓ, i + 1).
    } //Check for error in keeping the last row.!!!
    for(unsigned i = 0; i<mMax-1;i++)
        for(unsigned j = 0; j < mMax-1; j++)
            R(i,j) = R(i,j+1);
    return;
}

}//namespace detail
///@endcond



/*!@brief Anderson Acceleration of Fixed Point/Richardson Iteration for the nonlinear equation \f$ f(x) = b\f$
 *
 * This class implements the Anderson acceleration of the fixed point iteration algorithm
 *  described by https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf
with implementation details in //https://users.wpi.edu/~walker/Papers/anderson_accn_algs_imps.pdf
 As recommended by  https://arxiv.org/pdf/1803.06673.pdf we periodically restart the acceleration to
 improve convergence behaviour.
 *  @ingroup invert
 * @snippet bicgstabl_t.cpp andersonacc
 * @copydoc hide_ContainerType
 */
template<class ContainerType>
struct AndersonAcceleration
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@brief Allocate nothing, Call \c construct method before usage
    AndersonAcceleration() = default;
    /*! @brief Allocate memory for Fixed point iteration
     *
     * This version sets mMax to zero reducing the solve method to Fixed Point
     * (or Richardson if the damping parameter is != 1 in the \c solve()
     * method) iteration
     * @param copyable A ContainerType must be copy-constructible from this
     */
    AndersonAcceleration(const ContainerType& copyable):
        AndersonAcceleration( copyable, 0){}
    /*! @brief Allocate memory for the object
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param mMax The maximum number of vectors to include in the optimization
     * procedure. \c mMax+1 is the number of solutions involved in computing
     * the new solution.  Something between 3 and 10 are good values but higher
     * values mean more storage space that needs to be reserved.  If \c mMax==0
     * then the algorithm is equivalent to Fixed Point (or Richardson if the
     * damping parameter is != 1 in the \c solve() method) iteration i.e. no
     * optimization and only 1 solution needed to compute a new solution.
     */
    AndersonAcceleration(const ContainerType& copyable, unsigned mMax ):
        m_g_old( copyable), m_fval( copyable), m_f_old(copyable),
        m_DG( mMax, copyable), m_Q( m_DG),
        m_gamma( mMax, 0.),
        m_R( mMax), m_mMax( mMax)
    {
    }

    /**
    * @brief Perfect forward parameters to one of the constructors
    *
    * @tparam Params deduced by the compiler
    * @param ps parameters forwarded to constructors
    */
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = AndersonAcceleration( std::forward<Params>( ps)...);
    }

    ///@copydoc dg::PCG::copyable()
    const ContainerType& copyable() const{ return m_fval;}
    ///@copydoc dg::PCG::set_throw_on_fail(bool)
    void set_throw_on_fail( bool throw_on_fail){
        m_throw_on_fail = throw_on_fail;
    }

    /*!@brief Solve the system \f$ f(x) = b \f$ in the given norm
     *
     * Iterates until \f$ ||f(x)-b|| < a_{\mathrm{tol}} + r_{\mathrm{tol}} ||b||\f$
     *
     * @param f The function \c y=f(x) in the form \c f(x,y). The first argument is the input and the second the output.
     * @param x Contains an initial guess on input and the solution on output.
     * @param b The right hand side vector.
     * @param weights The weights define the norm for the stopping condition of the solver and the scalar product in which the least square problem is computed
     * @param rtol Relative error condition with respect to \c b
     * @param atol Absolute error condition
     * @param max_iter Maxmimum number of iterations
     * @param damping Paramter to prevent too large jumps around the actual
     * solution. Hard to determine in general but values between 1e-2 and 1e-4
     * are good values to begin with. This is the parameter that appears in
     * Richardson iteration. It is beter to have it too small than too large
     * (where it can lead to divergence of the solver)
     * @param restart Number >= mMax that indicates after how many iterations
     * to restart the acceleration. Periodic restarts are important for this
     * method.  Per default it should be the same value as \c mMax but \c mMax+1
     * or higher could be valuable to consider (but usually are worse
     * than \c mMax). Lower values \c restart<mMax are equivalent to setting
     * \c mMax=restart.
     * @param verbose If true writes intermediate errors to \c std::cout
     * @return Number of iterations used to achieve desired precision
     * @note The method will throw \c dg::Fail if the desired accuracy is not reached within \c max_iterations
     * You can unset this behaviour with the \c set_throw_on_fail member
     * @copydoc hide_matrix
     * @copydoc hide_ContainerType
     */
    template<class MatrixType, class ContainerType0, class ContainerType1, class ContainerType2>
    unsigned solve( MatrixType&& f, ContainerType0& x, const ContainerType1& b,
            const ContainerType2& weights,
        value_type rtol, value_type atol, unsigned max_iter,
        value_type damping, unsigned restart, bool verbose);

    private:
    ContainerType m_g_old, m_fval, m_f_old;
    std::vector<ContainerType> m_DG, m_Q;
    std::vector<value_type> m_gamma;
    dg::SquareMatrix<value_type> m_R;

    unsigned m_mMax;
    bool m_throw_on_fail = true;
};
///@cond

template<class ContainerType>
template<class MatrixType, class ContainerType0, class ContainerType1, class ContainerType2>
unsigned AndersonAcceleration<ContainerType>::solve(
    MatrixType&& func, ContainerType0& x, const ContainerType1& b, const ContainerType2& weights,
    value_type rtol, value_type atol, unsigned max_iter,
    value_type damping, unsigned restart,  bool verbose )
{
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
    if (m_mMax == 0){
        if(verbose)DG_RANK0 std::cout<< "No acceleration will occur" << std::endl;
    }

    unsigned mAA = 0;
    value_type nrmb = sqrt( dg::blas2::dot( b, weights, b));
    value_type tol = atol+rtol*nrmb;
    if(verbose)DG_RANK0 std::cout << "Solve with mMax = "<<m_mMax<<" rtol = "
        <<rtol<<" atol = "<<atol<< " tol = " << tol <<" max_iter =  "<<max_iter
        <<" damping = "<<damping<<" restart = "<<restart<<std::endl;

    ContainerType0& m_gval = x;
    // - use weights for orthogonalization (works because minimization is also true if V_m is unitary in the W scalar product
    for(unsigned iter=0;iter < max_iter; iter++)
    {
        if ( restart != 0 && iter % (restart) == 0) {
            mAA = 0;
            if(verbose)DG_RANK0 std::cout << "Iter = " << iter << std::endl;
        }

        dg::apply( std::forward<MatrixType>(func), x, m_fval);
        dg::blas1::axpby( -1., b, 1., m_fval); //f(x) = func - b (residual)
        value_type res_norm = sqrt(dg::blas2::dot(m_fval,weights,m_fval));  //l2norm(m_fval)

        if(verbose)DG_RANK0 std::cout << "res_norm = " << res_norm << " Against tol = " << tol << std::endl;
        // Test for stopping
        if (res_norm <= tol){
            if(verbose)DG_RANK0 std::cout << "Terminate with residual norm = " << res_norm << std::endl;
            return iter+1;
        }

        dg::blas1::axpby(1.,x,-damping,m_fval,m_gval);                      // m_gval = x - damping*m_fval
        // Without acceleration, x =  g(x) is the next approximate solution.
        if( m_mMax == 0) continue;

        if( iter == 0)
        {
            std::swap(m_fval,m_f_old);
            dg::blas1::copy(m_gval,m_g_old);
            continue;
        }

        // Apply Anderson acceleration.

        if (mAA < m_mMax) {

            dg::blas1::axpby(1.,m_fval,-1.,m_f_old, m_Q[mAA]);                 //Q = m_fval-m_f_old;
            dg::blas1::axpby(1.,m_gval,-1.,m_g_old,m_DG[mAA]);        //Update m_DG = [m_DG   m_gval-m_g_old];

        } else {

            std::rotate(m_DG.begin(), m_DG.begin() + 1, m_DG.end());  //Rotate to the left hopefully this works... otherwise for i = 0 .. mMax-2 m_DG[i] = m_DG[i+1], and finally m_DG[mMax-1] = update...
            dg::blas1::axpby(1.,m_gval,-1.,m_g_old,m_DG[m_mMax-1]);     //Update last m_DG entry

            detail::QRdelete1(m_Q,m_R,m_mMax);                      // If the column dimension of Q is mMax, delete the first column (the oldest vector) and shift composition to the left
            dg::blas1::axpby(1.,m_fval,-1.,m_f_old, m_Q[m_mMax-1]);                 //Q = m_fval-m_f_old;
            mAA = m_mMax-1; // mAA = m_mMax-1

        }


        // update the QR decomposition to incorporate the new column.
        // MW: This is modified Gram-Schmidt which delivers a reduced QR-factorization
        for (unsigned j = 0; j < mAA; j++) {
            m_R(j,mAA) = dg::blas2::dot(m_Q[j],weights,m_Q[mAA]);      //Q(:,j)’*Q(mAA); //Changed mAA -> mAA-1

            dg::blas1::axpby(-m_R(j,mAA),m_Q[j],1.,m_Q[mAA]);  //m_Q[mAA] = Q(mAA) - R(j,mAA)*Q(:,j);
        }
        m_R(mAA,mAA) = sqrt(dg::blas2::dot(m_Q[mAA],weights,m_Q[mAA]));
        dg::blas1::scal(m_Q[mAA], 1./m_R(mAA,mAA));

        //Calculate condition number of R to figure whether to keep going or call QR delete to reduce Q and R.
        //value_type condDF = cond(R,mAA+1);
        //Here should be the check for whether to proceed.

        //Solve least squares problem.
        for(int i = (int)mAA; i>=0; i--){
            m_gamma[i] = dg::blas2::dot(m_Q[i],weights,m_fval);
            for(int j = i + 1; j < (int)mAA+1; j++){
                m_gamma[i] = DG_FMA( -m_R(i,j), m_gamma[j], m_gamma[i]) ;
            }
            m_gamma[i] /= m_R(i,i);
        }

        std::swap(m_fval,m_f_old);
        dg::blas1::copy(m_gval,m_g_old);

        //Update new approximate solution x = m_gval - m_DG*gamma
        //for (unsigned i = 0; i < mAA+1; i++) {
        //    dg::blas1::axpby(-m_gamma[i],m_DG[i],1.,x);
        //}
        // ATTENTION: x is an alias for gval
        dg::blas2::gemv( -1., dg::asDenseMatrix( dg::asPointers(m_DG), mAA+1),
            std::vector<value_type>{m_gamma.begin(), m_gamma.begin()+mAA+1},
            1., x);

        mAA++;
    }
    if( m_throw_on_fail)
    {
        throw dg::Fail( tol, Message(_ping_)
            <<"After "<<max_iter<<" Anderson iterations with rtol "<<rtol<<" atol "<<atol<<" damping "<<damping<<" restart "<<restart);
    }
    return max_iter;

}
///@endcond
//
/*!
 * @brief If you are looking for fixed point iteration: it is a special case of Anderson Acceleration
 * @ingroup invert
 */
template<class ContainerType>
using FixedPointIteration = AndersonAcceleration<ContainerType>;

}//namespace dg
