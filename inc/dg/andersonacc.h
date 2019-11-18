#pragma once

#include <functional>
#include "blas.h"

namespace dg{
///@cond
//const double m_EPS = 2.2204460492503131e-16;
namespace detail{

template<class ContainerType, class Mat>
void QRdelete1(std::vector<ContainerType>& Q,Mat& R, unsigned m)
{
    using value_type = dg::get_value_type<ContainerType>;
    for(unsigned i = 0; i<m-1;i++){
        value_type temp = sqrt(R[i][i+1]*R[i][i+1]+R[i+1][i+1]*R[i+1][i+1]);
        value_type c = R[i][i+1]/temp;
        value_type s = R[i+1][i+1]/temp;
        R[i][i+1] = temp;
        R[i+1][i+1] = 0;
        if (i < m-2) {
            for (unsigned j = i+2; j < m; j++){
                temp = c * R[i][j] + s * R[i+1][j];
                R[i+1][j] = - s * R[i][j] + c * R[i+1][j];
                R[i][j] = temp;
            }
        }
        //Collapse look into blas1 routines
        dg::blas1::axpby(-s,Q[i],c,Q[i+1]); // Q(i + 1) = s ∗ Q(ℓ, i) + c ∗ Q(ℓ, i + 1).
        dg::blas1::axpbypgz(c,Q[i],s,Q[i+1],0.,Q[i]); //Q(i) = c ∗ Q(ℓ, i) + s ∗ Q(ℓ, i + 1).
    } //Check for error in keeping the last row.!!!
    for(int i = 0; i<(int)m-1;i++)
        for(int j = 0; j < (int)m-1; j++)
            R[i][j] = R[i][j+1];
    return;
}
}//namespace detail
///@endcond



/*!@brief Anderson Acceleration of Fixed Point/Richardson Iteration for the nonlinear equation \f[ f(x) = b\f]
 *
 * This class implements the Anderson acceleration of the fixed point iteration algorithm
 *  described by https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf
 As recommended by  https://arxiv.org/pdf/1803.06673.pdf we periodically restart the acceleration to
 improve convergence behaviour.
 *  @ingroup invert
 * @snippet andersonacc_t.cu doxygen
 * @copydoc hide_ContainerType
 */
template<class ContainerType>
struct AndersonAcceleration
{
    using value_type = get_value_type<ContainerType>;//!< the value type of the time variable (float or double)
    using container_type = ContainerType; //!< the type of the vector class in use
    ///@brief Allocate nothing, Call \c construct method before usage
    AndersonAcceleration(){}
    ///@copydoc construct()
    AndersonAcceleration(const ContainerType& copyable, unsigned mMax ):
        m_gval( copyable), m_g_old( m_gval), m_fval( m_gval), m_f_old(m_gval),
        m_df( m_gval), m_DG( mMax, copyable), m_Q( m_DG),
        m_gamma( mMax, 0.), m_Ry( m_gamma),
        m_R( mMax, m_gamma), m_mMax( mMax)
    {}

    /*! @brief Allocate memory for the object
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param mMax The maximum number of vectors to include in the optimization procedure. \c mMax+1 is the number
     * of solutions involved in computing the new solution.
     *  Something between 3 and 10 are good values but higher values mean more storage space that needs to be reserved.
     *  If \c mMax==0 then the algorithm is equivalent to Fixed Point (or Richardson if the damping parameter is used in the \c solve() method) iteration i.e. no optimization and only 1 solution needed to compute a new solution.
     */
    void construct( const ContainerType& copyable, unsigned mMax)
    {
        *this = AndersonAcceleration(copyable, mMax);
    }
    const ContainerType& copyable() const{ return m_gval;}

    /*!@brief Solve the system \f$ f(x) = b \f$ in the given norm
     *
     * Iterates until \f$ ||f(x)-b|| < a_{\mathrm{tol}} + r_{\mathrm{tol}} ||b||\f$
     *
     * @param f The function \c y=f(x) in the form \c f(x,y). The first argument is the input and the second the output.
     * @param x Contains an initial guess on input and the solution on output.
     * @param b The right hand side vector.
     * @param weights The weights define the norm for the stopping condition of the solver
     * @param rtol Relative error condition with respect to \c b
     * @param atol Absolute error condition
     * @param max_iter Maxmimum number of iterations
     * @param damping Paramter to prevent too large jumps around the actual solution. Hard to determine in general but values between 0.1 and 1e-3 are good values to begin with. This is the parameter that appears in Richardson iteration.
     * @param restart Number >= 1 that indicates after how many iterations to restart the acceleration. Periodic restarts are important for this method.  Per default it should be the same value as \c mMax but \c mMax+1 or higher could also be valuable to consider.
     * @param verbose If true writes intermediate errors to \c std::cout . Avoid in MPI code.
     * @return Number of iterations used to achieve desired precision
     */
    template<class BinarySubroutine, class ContainerType2>
    unsigned solve( BinarySubroutine& f, ContainerType& x, const ContainerType& b, const ContainerType2& weights,
        value_type rtol, value_type atol, unsigned max_iter,
        value_type damping, unsigned restart, bool verbose);

    private:
    ContainerType m_gval, m_g_old, m_fval, m_f_old, m_df;
    std::vector<ContainerType> m_DG, m_Q;
    std::vector<value_type> m_gamma, m_Ry;
    std::vector<std::vector<value_type>> m_R;

    unsigned m_mMax;
};
///@cond

template<class ContainerType>
template<class BinarySubroutine, class ContainerType2>
unsigned AndersonAcceleration<ContainerType>::solve(
    BinarySubroutine& func, ContainerType& x, const ContainerType& b, const ContainerType2& weights,
    value_type rtol, value_type atol, unsigned max_iter,
    value_type damping, unsigned restart,  bool verbose )
{
    if (m_mMax == 0){
        if(verbose)std::cout<< "No acceleration will occur" << std::endl;
    }

    unsigned mAA = 0;
    value_type nrmb = sqrt( dg::blas2::dot( b, weights, b));
    value_type tol = atol+rtol*nrmb;
    if(verbose)std::cout << "tol = " << tol << std::endl;

    for(unsigned iter=0;iter < max_iter; iter++)
    {

        // Restart from mAA=1 (note that it's incremented further down) if a certain number of iterations are reached.
        if (iter % (restart) == 0) {
            mAA = 0;
            if(verbose)std::cout << "Iter = " << iter << std::endl;
        }

        func( x, m_fval);
        dg::blas1::axpby( -1., b, 1., m_fval); //f(x) = func - b (residual)
        value_type res_norm = sqrt(dg::blas2::dot(m_fval,weights,m_fval));  //l2norm(m_fval)

        if(verbose)std::cout << "res_norm = " << res_norm << " Against tol = " << tol << std::endl;
        // Test for stopping
        if (res_norm <= tol){
            if(verbose)std::cout << "Terminate with residual norm = " << res_norm << std::endl;
            return iter+1;
        }

        dg::blas1::axpby(1.,x,-damping,m_fval,m_gval);                      // m_gval = x - damping*m_fval

        if (m_mMax == 0){
            // Without acceleration, update x <- g(x) to obtain the next approximate solution.

            dg::blas1::copy(m_gval,x);                                    //x = m_gval;

        } else {
            // Apply Anderson acceleration.

            if(iter > 0){                                         // Update the m_df vector and the m_DG array.t,

                dg::blas1::axpby(1.,m_fval,-1.,m_f_old, m_df);                 //m_df = m_fval-m_f_old;

                if (mAA < m_mMax) {

                    dg::blas1::axpby(1.,m_gval,-1.,m_g_old,m_DG[mAA]);        //Update m_DG = [m_DG   m_gval-m_g_old];

                } else {

                    std::rotate(m_DG.begin(), m_DG.begin() + 1, m_DG.end());  //Rotate to the left hopefully this works... otherwise for i = 0 .. mMax-2 m_DG[i] = m_DG[i+1], and finally m_DG[mMax-1] = update...
                    dg::blas1::axpby(1.,m_gval,-1.,m_g_old,m_DG[m_mMax-1]);     //Update last m_DG entry

                }
                mAA = mAA + 1;
            }

            dg::blas1::copy(m_fval,m_f_old);                                //m_f_old = m_fval;

            dg::blas1::copy(m_gval,m_g_old);                                //m_g_old = m_gval;

            if(mAA==0){ //only the very first iteration

                dg::blas1::copy(m_gval,x);                                // If mAA == 0, update x <- g(x) to obtain the next approximate solution.

            } else {                                                    // If mAA > 0, solve the least-squares problem and update the solution.

                if (mAA == 1) {                                         // If mAA == 1, form the initial QR decomposition.

                    m_R[0][0] = sqrt(dg::blas1::dot(m_df, m_df));
                    dg::blas1::axpby(1./m_R[0][0],m_df,0.,m_Q[0]);

                } else {                                                // If mAA > 1, update the QR decomposition.

                    if ((mAA > m_mMax)) {                                 // If the column dimension of Q is mMax, delete the first column and update the decomposition.


                        mAA = mAA - 1;
                        detail::QRdelete1(m_Q,m_R,mAA);

                    }
                    // Now update the QR decomposition to incorporate the new column.
                    for (unsigned j = 1; j < mAA; j++) {
                        m_R[j-1][mAA-1] = dg::blas1::dot(m_Q[j-1],m_df);      //Q(:,j)’*m_df; //Changed mAA -> mAA-1

                        dg::blas1::axpby(-m_R[j-1][mAA-1],m_Q[j-1],1.,m_df);  //m_df = m_df - R(j,mAA)*Q(:,j);
                    }
                    m_R[mAA-1][mAA-1] = sqrt(dg::blas1::dot(m_df,m_df));
                    dg::blas1::axpby(1./m_R[mAA-1][mAA-1],m_df,0.,m_Q[mAA-1]);
                }

                //Calculate condition number of R to figure whether to keep going or call QR delete to reduce Q and R.
                //value_type condDF = cond(R,mAA);
                //Here should be the check for whether to proceed.

                //Solve least squares problem.
                for(int i = (int)mAA-1; i>=0; i--){
                    m_gamma[i] = dg::blas1::dot(m_Q[i],m_fval);
                    for(int j = i + 1; j < (int)mAA; j++){
                        m_gamma[i] -= m_R[i][j]*m_gamma[j];
                    }
                    m_gamma[i] /= m_R[i][i];
                }

                //Update new approximate solution x = m_gval - m_DG*gamma
                dg::blas1::copy(m_gval,x);
                for (unsigned i = 0; i < mAA; i++) {
                    dg::blas1::axpby(-m_gamma[i],m_DG[i],1.,x);
                }

            }//Should all mAA
        }

    }
    return max_iter;

}
///@endcond

}//namespace dg
