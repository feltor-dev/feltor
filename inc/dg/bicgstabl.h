#ifndef _DG_BICGSTABl_
#define _DG_BICGSTABl_

#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "blas.h"
#include "functors.h"

/*!@file
 * BICGSTABl class
 *
 * @author Aslak Poulsen
 */

namespace dg{

/**
* @brief Preconditioned BICGSTAB(l) method to solve
* \f[ Ax=b\f]
*
* @ingroup invert
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
    BICGSTABl(){}
    ///@copydoc construct()
    BICGSTABl( const ContainerType& copyable, unsigned max_iterations, unsigned l_input){
        construct(copyable, max_iterations, l_input);
    }
    ///@brief Set the maximum number of iterations
    ///@param new_max New maximum number
    void set_max( unsigned new_max) {max_iter = new_max;}
    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_max() const {return max_iter;}
    /**
     * @brief Allocate memory for the preconditioned BICGSTABl method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations
     * @param l_input Size of polynomial used for stabilisation. Usually 2 or 4 is a good number.
     */
    void construct(const ContainerType& copyable, unsigned max_iterations, unsigned l_input){
        max_iter = max_iterations;
        l = l_input;
        m_tmp=copyable;
        rhat.assign(l+1,copyable);
        uhat.assign(l+1,copyable);
        sigma.assign(l+1,0);
        gamma.assign(l+1,0);
        gammap.assign(l+1,0);
        gammapp.assign(l+1,0);
        for(unsigned i = 0; i < l; i++){
            tau.push_back(std::vector<value_type>());
            for(unsigned j = 0; j < l; j++){
                tau[i].push_back(0);
            }
        }
    }
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_tmp;}

    /**
     * @brief Solve \f$ Ax = b\f$ using a preconditioned BICGSTABl method
     *
     * The iteration stops if \f$ ||Ax-b||_S < \epsilon( ||b||_S + C) \f$ where \f$C\f$ is
     * the absolute error in units of \f$ \epsilon\f$ and \f$ S \f$ defines a square norm
     * @param A A matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param S (Inverse) Weights used to compute the norm for the error condition
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     *
     * @return Number of iterations used to achieve desired precision
     * @copydoc hide_matrix
     * @tparam ContainerTypes must be usable with \c MatrixType and \c ContainerType in \ref dispatch
     * @tparam Preconditioner A type for which the blas2::symv(Preconditioner&, ContainerType&, ContainerType&) function is callable.
     * @tparam SquareNorm A type for which the blas2::dot( const SquareNorm&, const ContainerType&) function is callable. This can e.g. be one of the ContainerType types.
     */
    template< class MatrixType, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm >
    unsigned solve( MatrixType& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps = 1e-12, value_type nrmb_correction = 1);

  private:
    unsigned max_iter, l;
    ContainerType m_tmp;
    std::vector<ContainerType> rhat;
    std::vector<ContainerType> uhat;
    std::vector<value_type> sigma, gamma, gammap, gammapp;
    std::vector<std::vector<value_type>> tau;

};
///@cond

template< class ContainerType>
template< class Matrix, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm>
unsigned BICGSTABl< ContainerType>::solve( Matrix& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps, value_type nrmb_correction)
{
    value_type nrmb = sqrt(dg::blas2::dot(S,b));
    value_type tol = eps*(nrmb + nrmb_correction);
    if( nrmb == 0)
    {
        blas1::copy( 0., x);
        return 0;
    }

    dg::blas1::copy(0., uhat[0]);
    dg::blas2::symv(A,x,m_tmp);
    dg::blas1::axpby(1.,b,-1.,m_tmp);
    if( sqrt( blas2::dot(S,m_tmp) ) < tol) //if x happens to be the solution
        return 0;
    dg::blas2::symv(P,m_tmp,rhat[0]); // MW: Technically this is not allowed symv must store
    // output in a separate vector (also check lgmres.h)

    value_type rho_0 = 1;
    value_type alpha = 0;
    value_type omega = 1;
    ContainerType0& xhat=x; // alias x for ease of notation

    for (unsigned k = 0; k < max_iter; k++){

        rho_0 = -omega*rho_0;

        /// Bi-CG part ///
        for(unsigned j = 0; j<l;j++)
        {
            value_type rho_1 = dg::blas1::dot(rhat[j],b);
            value_type beta = alpha*rho_1/rho_0;
            rho_0 = rho_1;
            for(unsigned i = 0; i<=j;i++)
            {
                dg::blas1::axpby(1.,rhat[i],-1.0*beta,uhat[i]);
            }
            dg::blas2::symv(A,uhat[j],m_tmp);
            dg::blas2::symv(P,m_tmp,uhat[j+1]);
            if( rho_0 == 0)
                alpha = 0;
            else
                alpha = rho_0/dg::blas1::dot(uhat[j+1],b);
            for(unsigned i = 0; i<=j; i++)
            {
                dg::blas1::axpby(-1.0*alpha,uhat[i+1],1.,rhat[i]);
            }
            dg::blas2::symv(A,rhat[j],m_tmp);
            dg::blas2::symv(P,m_tmp,rhat[j+1]);
            dg::blas1::axpby(alpha,uhat[0],1.,xhat);
        }

        /// MR part ///
        for(unsigned j = 1; j<=l; j++){
            for(unsigned i = 1; i<j;i++){
                tau[i][j] = 1.0/sigma[i]*dg::blas1::dot(rhat[j],rhat[i]);
                dg::blas1::axpby(-tau[i][j],rhat[i],1.,rhat[j]);
            }
            sigma[j] = dg::blas1::dot(rhat[j],rhat[j]);
            gammap[j] = 1.0/sigma[j]*dg::blas1::dot(rhat[0],rhat[j]);
        }

        gamma[l] = gammap[l];
        omega = gamma[l];

        for(unsigned j=l-1;j>=1;j--){
            value_type tmp = 0;
            for(unsigned i=j+1;i<=l;i++){
                tmp += tau[j][i]*gamma[i];
            }
            gamma[j] = gammap[j]-tmp;
        }
        for(unsigned j=1;j<=l-1;j++){
            value_type tmp = 0.;
            for(unsigned i=j+1;i<=l-1;i++){
                tmp += tau[j][i]*gamma[i+1];
            }
            gammapp[j] = gamma[j+1]+tmp;
        }
        dg::blas1::axpby(gamma[1],rhat[0],1.,xhat);
        dg::blas1::axpby(-gammap[l],rhat[l],1.,rhat[0]);
        dg::blas1::axpby(-gamma[l],uhat[l],1.,uhat[0]);
        for(unsigned j = 1; j<=l-1; j++){
            dg::blas1::axpby(gammapp[j],rhat[j],1.,xhat);
            dg::blas1::axpby(-gamma[j],uhat[j],1.,uhat[0]);
            dg::blas1::axpby(-gammap[j],rhat[j],1.,rhat[0]);
        }

        value_type err = sqrt(dg::blas2::dot(S,rhat[0]));
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif //MPI
        std::cout << "# Error is now : " << err << " Against " << tol << std::endl;
#endif //DG_DEBUG
        if( err < tol){
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    if(rank==0)
#endif //MPI
            std::cout << "# Exited with error : " << err << " After " << k << " Iterations." << std::endl;
#endif //DG_DEBUG
            return k;
        }
    }
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    if(rank==0)
#endif //MPI
    std::cout << "# Failed to converge within max_iter" << std::endl;
#endif //DG_DEBUG
    return max_iter;
}
///@endcond

}//namespace dg
#endif
