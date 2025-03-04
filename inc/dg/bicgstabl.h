#ifndef _DG_BICGSTABl_
#define _DG_BICGSTABl_

#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "blas.h"
#include "functors.h"
#include "backend/typedefs.h"

/*!@file
 * BICGSTABl class
 *
 * @author Aslak Poulsen, Matthias Wiesenberger
 */

namespace dg{

/**
* @brief Preconditioned BICGSTAB(l) method to solve
* \f$ Ax=b\f$
*
* @ingroup invert
* @snippet bicgstabl_t.cpp bicgstabl
*
* @note BICGSTAB(l) is a method for solving non-symmetrical linear systems.
* BICGSTAB(l) is a modification of BICGSTAB that aims to improve convergence.
* See a paper here
* https://pdfs.semanticscholar.org/c185/7ceab3c9ab4dbcb6a52fb62916f5757c0b38.pdf
*
*/
template< class ContainerType>
class BICGSTABl
{
  public:
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    BICGSTABl() = default;
    /**
     * @brief Allocate memory for the preconditioned BICGSTABl method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations (there is 2 matrix-vector products plus 2 Preconditioner-vector products per iteration)
     * @param l_input Size of polynomial used for stabilisation.
     * Usually 2 or 4 is a good number (makes \c l_input Bi-CG iterations before computing the minimal residual)
     * @note \c l_input=1 computes exactly the same as Bi-CGstab does
     */
    BICGSTABl( const ContainerType& copyable, unsigned max_iterations,
            unsigned l_input):
        max_iter(max_iterations),
        m_l(l_input),
        m_tmp(copyable)
    {
        rhat.assign(m_l+1,copyable);
        uhat.assign(m_l+1,copyable);
        sigma.assign(m_l+1,0);
        gamma.assign(m_l+1,0);
        gammap.assign(m_l+1,0);
        gammapp.assign(m_l+1,0);
        tau.assign( m_l+1, std::vector<value_type>( m_l+1, 0));
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
        *this = BICGSTABl( std::forward<Params>( ps)...);
    }
    ///@brief Set the maximum number of iterations
    ///@param new_max New maximum number
    void set_max( unsigned new_max) {max_iter = new_max;}
    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_max() const {return max_iter;}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_tmp;}

    ///@brief Set or unset debugging output during iterations
    ///@param verbose If true, additional output will be written to \c std::cout during solution
    void set_verbose( bool verbose){ m_verbose = verbose;}

    ///@copydoc dg::PCG::set_throw_on_fail(bool)
    void set_throw_on_fail( bool throw_on_fail){
        m_throw_on_fail = throw_on_fail;
    }

    /**
     * @brief Solve \f$ Ax = b\f$ using a preconditioned BICGSTABl method
     *
     * The iteration stops if \f$ ||P(Ax-b)||_W < \epsilon( ||Pb||_S + C) \f$ where \f$C\f$ is
     * the absolute error in units of \f$ \epsilon\f$ and \f$ W \f$ defines a square norm
     * @attention The stopping criterion differs from that of \c CG or \c LGMRES by the preconditioner. It is unfortunately cumbersome to obtain the real residual in this algorithm. If \c P is diagonal there is the opportunity to use \c W to offset its effect.
     * @param A A matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may not be the same vector.
     * @param P The preconditioner to be used
     * @param W Weights used to define the scalar product and the norm for the error condition
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     *
     * @return Number of iterations used to achieve desired precision (in each iteration the matrix has to be applied twice)
     * @note The method will throw \c dg::Fail if the desired accuracy is not reached within \c max_iterations
     * You can unset this behaviour with the \c set_throw_on_fail member
     * @copydoc hide_matrix
     * @copydoc hide_ContainerType
     */
    template< class MatrixType0, class ContainerType0, class ContainerType1, class MatrixType1, class ContainerType2 >
    unsigned solve( MatrixType0&& A, ContainerType0& x, const ContainerType1& b, MatrixType1&& P, const ContainerType2& W, value_type eps = 1e-12, value_type nrmb_correction = 1);

  private:
    unsigned max_iter, m_l;
    ContainerType m_tmp;
    std::vector<ContainerType> rhat;
    std::vector<ContainerType> uhat;
    std::vector<value_type> sigma, gamma, gammap, gammapp;
    std::vector<std::vector<value_type>> tau;
    bool m_verbose = false, m_throw_on_fail = true;

};
///@cond

template< class ContainerType>
template< class Matrix, class ContainerType0, class ContainerType1, class Preconditioner, class ContainerType2>
unsigned BICGSTABl< ContainerType>::solve( Matrix&& A, ContainerType0& x, const ContainerType1& b, Preconditioner&& P, const ContainerType2& W, value_type eps, value_type nrmb_correction)
{
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
    dg::blas2::symv(std::forward<Preconditioner>(P),b,m_tmp);
    value_type nrmb = sqrt(dg::blas2::dot(W,m_tmp));
    value_type tol = eps*(nrmb + nrmb_correction);
    if( nrmb == 0)
    {
        blas1::copy( 0., x);
        return 0;
    }
    dg::blas2::symv(std::forward<Matrix>(A),x,m_tmp);
    dg::blas1::axpby(1.,b,-1.,m_tmp);
    if( sqrt( blas2::dot(W,m_tmp) ) < tol) //if x happens to be the solution
        return 0;
    dg::blas2::symv(std::forward<Preconditioner>(P),m_tmp,rhat[0]);

    dg::blas1::copy(0., uhat[0]);

    value_type rho_0 = 1;
    value_type alpha = 0;
    value_type omega = 1;
    ContainerType0& xhat=x; // alias x for ease of notation

    for (unsigned k = 0; k < max_iter; k+= m_l){

        rho_0 = -omega*rho_0;

        /// Bi-CG part ///
        for(unsigned j = 0; j<m_l;j++)
        {
            value_type rho_1 = dg::blas2::dot(rhat[j],W,b);
            value_type beta = alpha*rho_1/rho_0;
            rho_0 = rho_1;
            for(unsigned i = 0; i<=j;i++)
            {
                dg::blas1::axpby(1.,rhat[i],-1.0*beta,uhat[i]);
            }
            dg::blas2::symv(std::forward<Matrix>(A),uhat[j],m_tmp);
            dg::blas2::symv(std::forward<Preconditioner>(P),m_tmp,uhat[j+1]);
            if( rho_0 == 0)
                alpha = 0;
            else
                alpha = rho_0/dg::blas2::dot(uhat[j+1],W,b);
            for(unsigned i = 0; i<=j; i++)
            {
                dg::blas1::axpby(-1.0*alpha,uhat[i+1],1.,rhat[i]);
            }
            dg::blas2::symv(std::forward<Matrix>(A),rhat[j],m_tmp);
            dg::blas2::symv(std::forward<Preconditioner>(P),m_tmp,rhat[j+1]);
            dg::blas1::axpby(alpha,uhat[0],1.,xhat);
        }

        /// Minimal Residual part: modified Gram-Schmidt ///
        for(unsigned j = 1; j<=m_l; j++){
            for(unsigned i = 1; i<j;i++){
                tau[i][j] = 1.0/sigma[i]*dg::blas2::dot(rhat[j],W,rhat[i]);
                dg::blas1::axpby(-tau[i][j],rhat[i],1.,rhat[j]);
            }
            sigma[j] = dg::blas2::dot(rhat[j],W,rhat[j]);
            gammap[j] = 1.0/sigma[j]*dg::blas2::dot(rhat[0],W,rhat[j]);
        }

        gamma[m_l] = gammap[m_l];
        omega = gamma[m_l];

        for(unsigned j=m_l-1;j>=1;j--){
            value_type tmp = 0;
            for(unsigned i=j+1;i<=m_l;i++){
                tmp += tau[j][i]*gamma[i];
            }
            gamma[j] = gammap[j]-tmp;
        }
        for(unsigned j=1;j<=m_l-1;j++){
            value_type tmp = 0.;
            for(unsigned i=j+1;i<=m_l-1;i++){
                tmp += tau[j][i]*gamma[i+1];
            }
            gammapp[j] = gamma[j+1]+tmp;
        }
        dg::blas1::axpby(gamma[1],rhat[0],1.,xhat);
        dg::blas1::axpby(-gammap[m_l],rhat[m_l],1.,rhat[0]);
        dg::blas1::axpby(-gamma[m_l],uhat[m_l],1.,uhat[0]);
        for(unsigned j = 1; j<=m_l-1; j++){
            dg::blas1::axpby(gammapp[j],rhat[j],1.,xhat);
            dg::blas1::axpby(-gamma[j],uhat[j],1.,uhat[0]);
            dg::blas1::axpby(-gammap[j],rhat[j],1.,rhat[0]);
        }

        // rhat[0] is P dot the actual residual
        value_type err = sqrt(dg::blas2::dot(W,rhat[0]));
        if( m_verbose)
            DG_RANK0 std::cout << "# Error is now : " << err << " Against " << tol << std::endl;
        if( err < tol){
            if( m_verbose)
                DG_RANK0 std::cout << "# Exited with error : " << err << " After " << k+m_l << " Iterations." << std::endl;
            return k+m_l;
        }
    }
    if( m_verbose)
        DG_RANK0 std::cout << "# Failed to converge within max_iter" << std::endl;
    if( m_throw_on_fail)
    {
        throw dg::Fail( tol, Message(_ping_)
                <<"After "<<max_iter<<" BICGSTABL iterations with rtol "<<eps<<" and atol "<<eps*nrmb_correction );
    }
    return max_iter;
}
///@endcond

}//namespace dg
#endif
