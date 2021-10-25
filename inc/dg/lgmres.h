#ifndef _DG_LGMRES_
#define _DG_LGMRES_

#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "blas.h"
#include "functors.h"

/*!@file
 * LGMRES class
 *
 * @author Aslak Poulsen
 */

namespace dg{

/**
* @brief Functor class for the preconditioned LGMRES method to solve
* \f[ Ax=b\f]
*
* @ingroup invert
*
* @note GMRES is a method for solving non-symmetrical linear systems.
* LGMRES is a modification of restarted GMRES that aims to improve convergence.
* This implementation is adapted from:
* https://github.com/KellyBlack/GMRES/blob/master/GMRES.h
* with LGMRES elements from
* https://github.com/haranjackson/NewtonKrylov/blob/master/lgmres.cpp
* A paper can be found at
* https://www.cs.colorado.edu/~jessup/SUBPAGES/PS/lgmres.pdf
*
* Basically the Krylov subspace is augmented by \c k approximations to the error.
* The storage requirement is m+3k vectors
*
* @note the first cycle of LGMRES(m,k) is equivalent to the first cycle of GMRES(m+k)
* @note Only \c m matrix-vector multiplications need to be computed in LGMRES(m,k) per restart cycle irrespective of the value of \c k
* @attention LGMRES can stagnate if the matrix A is not positive definite. The use of a preconditioner is
* paramount in such a situation
*/
template< class ContainerType>
class LGMRES
{
  public:
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    LGMRES(){}
    /**
     * @brief Allocate memory for the preconditioned LGMRES method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_inner Maximum number of vectors to be saved in gmres. Usually 30 seems to be a decent number.
     * @param max_outer Maximum number of solutions (actually approximations to the error) saved for restart. Usually 1...3 is a good number. The Krylov Dimension is augmented to \c max_inner+max_outer. \c max_outer=0 corresponds to standard GMRES.
     * @param Restarts Maximum number of restarts. This can be set high just in case. Like e.g. gridsize/max_outer.
     */
    LGMRES( const ContainerType& copyable, unsigned max_inner, unsigned max_outer, unsigned Restarts):
        m_tmp(copyable),
        m_dx(copyable),
        m_residual( copyable),
        m_maxRestarts( Restarts),
        m_inner_m( max_inner),
        m_outer_k( max_outer),
        m_krylovDimension( max_inner+max_outer)
    {
        //Declare Hessenberg matrix
        m_H.assign( m_krylovDimension+1, std::vector<value_type>( m_krylovDimension, 0));
        //Declare givens rotation matrix
        m_givens.assign( m_krylovDimension+1, {0,0});
        //Declare s that minimizes the residual:
        m_s.assign(m_krylovDimension+1,0);
        // m+k+1 orthogonal basis vectors:
        m_V.assign(m_krylovDimension+1,copyable);
        m_W.assign(m_krylovDimension,copyable);
        // k augmented pairs
        m_outer_w.assign(m_outer_k+1,copyable);
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
        *this = LGMRES( std::forward<Params>( ps)...);
    }
    ///@brief Set the number of restarts
    ///@param new_Restarts New maximum number of restarts
    void set_max( unsigned new_Restarts) {m_maxRestarts = new_Restarts;}
    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_maxRestarts() const {return m_maxRestarts;}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_tmp;}

    /**
     * @brief Solve \f$ Ax = b\f$ using a preconditioned LGMRES method
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
    template <class Preconditioner, class ContainerType0>
    void Update(Preconditioner& P, ContainerType &dx, ContainerType0 &x,
            unsigned dimension, const std::vector<std::vector<value_type>> &H,
            std::vector<value_type> &s, const std::vector<ContainerType> &W);
    std::vector<std::vector<value_type>> m_H, m_givens;
    ContainerType m_tmp, m_dx, m_residual;
    std::vector<ContainerType> m_V, m_W, m_outer_w;
    std::vector<value_type> m_s;
    unsigned m_maxRestarts, m_inner_m, m_outer_k, m_krylovDimension;
};
///@cond

template< class ContainerType>
template < class Preconditioner, class ContainerType0>
void LGMRES<ContainerType>::Update(Preconditioner& P, ContainerType &dx,
        ContainerType0 &x,
        unsigned dimension, const std::vector<std::vector<value_type>> &H,
        std::vector<value_type> &s, const std::vector<ContainerType> &W)
{
    // Solve for the coefficients, i.e. solve for c in
    // H*c=s, but we do it in place.
    for (int lupe = dimension; lupe >= 0; --lupe)
    {
        s[lupe] = s[lupe]/H[lupe][lupe];
        if(lupe > 0){
            for (int innerLupe = lupe - 1; innerLupe >= 0; --innerLupe)
            {
                // Subtract off the parts from the upper diagonal of the matrix.
                s[innerLupe] -=  s[lupe]*H[innerLupe][lupe];
            }
        }
	}
    //std::cout << "HessenbergA\n";
    //for( unsigned i=0; i<dimension; i++)
    //{
    //    for( unsigned k=0; k<i; k++)
    //        std::cout << "X ";
    //    for( unsigned k=i; k<dimension; k++)
    //        std::cout << H[i][k]<<" ";
    //    std::cout << std::endl;
    //}
    //std::cout << "HessenbergB\n";

    // Finally update the approximation. V_m*s
    dg::blas1::axpby(s[0],W[0],0.,dx);
    for (unsigned lupe = 1; lupe <= dimension; lupe++)
        dg::blas1::axpby(s[lupe],W[lupe],1.,dx);
    // right preconditioner
    dg::blas2::symv( P, dx, m_tmp);
    dg::blas1::axpby(1.,m_tmp,1.,x);
}

template< class ContainerType>
template< class Matrix, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm>
unsigned LGMRES< ContainerType>::solve( Matrix& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps, value_type nrmb_correction)
{
    // suggested Improvements:
    // - Use right preconditioned system such that residual norm is available in minimization
    // - do not compute Az explicitly but save on iterations
    // - too many vectors stored ( reduce storage requirements)
    // - first cycle equivalent to GMRES(m+k)
    // - use SquareNorm for orthogonalization (works because 6.29 and 6.30 are also true if V_m is unitary in the S scalar product, the Hessenberg matrix is still formed in the regular 2-norm, just define J(y) with S-norm in 6.26 and form V_m with a Gram-Schmidt process in the S-norm
    value_type nrmb = sqrt( blas2::dot( S, b));
    value_type tol = eps*(nrmb + nrmb_correction);
    if( nrmb == 0)
    {
        blas1::copy( 0., x);
        return 0;
    }

    unsigned restartCycle = 0;
    value_type rho = 1.;
    do
	{
        dg::blas2::symv(A,x,m_residual);
        dg::blas1::axpby(1.,b,-1.,m_residual);
        rho = sqrt(dg::blas2::dot(S,m_residual));
        if( rho < tol) //if x happens to be the solution
            return 0;
        // left Preconditioning
        //dg::blas2::symv(P,m_tmp,m_residual);
        //value_type rho = sqrt(dg::blas2::dot(m_residual,S,m_residual));
        //if( rho < tol) //if P happens to produce the solution
        //    return 0;
        //
        // The first vector in the Krylov subspace is the normalized residual.
        dg::blas1::axpby(1.0/rho,m_residual,0.,m_V[0]);

		m_s[0] = rho;

		// Go through and generate the pre-determined number of vectors for the Krylov subspace.
		for( unsigned iteration=0;iteration<m_krylovDimension;++iteration)
		{
            unsigned outer_w_count = std::min(restartCycle,m_outer_k);
            if(iteration < m_inner_m){
                dg::blas2::symv(P,m_V[iteration],m_tmp);
                dg::blas2::symv(A,m_tmp,m_V[iteration+1]);
                dg::blas1::copy(m_V[iteration],m_W[iteration]);
            } else if( iteration < m_inner_m + outer_w_count){ // size of W
                // MW: I don't think we need that multiplication
                dg::blas2::symv(P,m_outer_w[iteration-m_inner_m],m_tmp);
                dg::blas2::symv(A,m_tmp,m_V[iteration+1]);
                dg::blas1::copy(m_outer_w[iteration-m_inner_m],m_W[iteration]);
            }

			// Get the next entry in the vectors that form the basis for the Krylov subspace.
            // Arnoldi modified Gram-Schmidt orthogonalization
            for(unsigned row=0;row<=iteration;++row)
			{
                m_H[row][iteration] = dg::blas2::dot(m_V[iteration+1],S,m_V[row]);
                dg::blas1::axpby(-m_H[row][iteration],m_V[row],1.,m_V[iteration+1]);
			}
            m_H[iteration+1][iteration]
                = sqrt(dg::blas2::dot(m_V[iteration+1],S,m_V[iteration+1]));
            dg::blas1::scal(m_V[iteration+1],1.0/m_H[iteration+1][iteration]);

            // Now solve the least squares problem
            // using Givens Rotations transforming H into
            // an upper triangular matrix (see Saad Chapter 6.5.3)

            // First apply previous rotations to the current matrix.
            value_type tmp = 0;
			for (unsigned row = 0; row < iteration; row++)
			{
				tmp = m_givens[row][0]*m_H[row][iteration] + // c_row
					m_givens[row][1]*m_H[row+1][iteration];  // s_row
				m_H[row+1][iteration] = -m_givens[row][1]*m_H[row][iteration]
					+ m_givens[row][0]*m_H[row+1][iteration];
				m_H[row][iteration]  = tmp;
			}

			// Figure out the next Givens rotation.
			if(m_H[iteration+1][iteration] == 0.0)
			{
				// It is already upper triangular. Just leave it be....
				m_givens[iteration][0] = 1.0; // c_i
				m_givens[iteration][1] = 0.0; // s_i
			}
			else if (fabs(m_H[iteration+1][iteration]) > fabs(m_H[iteration][iteration]))
			{
				// The off diagonal entry has a larger
				// magnitude. Use the ratio of the
				// diagonal entry over the off diagonal.
				tmp = m_H[iteration][iteration]/m_H[iteration+1][iteration];
				m_givens[iteration][1] = 1.0/sqrt(1.0+tmp*tmp);
				m_givens[iteration][0] = tmp*m_givens[iteration][1];
			}
			else
			{
				// The off diagonal entry has a smaller
				// magnitude. Use the ratio of the off
				// diagonal entry to the diagonal entry.
				tmp = m_H[iteration+1][iteration]/m_H[iteration][iteration];
				m_givens[iteration][0] = 1.0/sqrt(1.0+tmp*tmp);
				m_givens[iteration][1] = tmp*m_givens[iteration][0];
			}
            // Apply the new Givens rotation on the new entry in the upper Hessenberg matrix.
			tmp = m_givens[iteration][0]*m_H[iteration][iteration] +
				  m_givens[iteration][1]*m_H[iteration+1][iteration];
			m_H[iteration+1][iteration] = -m_givens[iteration][1]*m_H[iteration][iteration] +
				  m_givens[iteration][0]*m_H[iteration+1][iteration]; // zero
			m_H[iteration][iteration] = tmp;
			// Finally apply the new Givens rotation on the s vector
			tmp = m_givens[iteration][0]*m_s[iteration]; // + m_givens[iteration][1]*m_s[iteration+1];
			m_s[iteration+1] = -m_givens[iteration][1]*m_s[iteration];// + m_givens[iteration][1]*m_s[iteration+1];
			m_s[iteration] = tmp;

            rho = fabs(m_s[iteration+1]);
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif //MPI
            std::cout << "# rho = " << rho << std::endl;
#endif //DG_DEBUG
            if( rho < tol)
			{
                Update(P,m_dx,x,iteration,m_H,m_s,m_W);
                return(iteration+restartCycle*m_krylovDimension);
            }
        }
        Update(P,m_dx,x,m_krylovDimension-1,m_H,m_s,m_W);
        // do not(?) normalize new z vector
        //value_type nx = sqrt(dg::blas2::dot(m_dx,S,m_dx));
        //if(nx>0.){
        //    if (restartCycle<m_outer_k){
        //        dg::blas1::axpby(1.0/nx,m_dx,0.,m_outer_w[restartCycle]); //new outer entry = dx/nx
        //    } else {
        //        std::rotate(m_outer_w.begin(),m_outer_w.begin()+1,m_outer_w.end()); //rotate one to the left.
        //        dg::blas1::axpby(1.0/nx,m_dx,0.,m_outer_w[m_outer_k]);
        //    }
        //}
        if( m_outer_k > 1)
            std::rotate(m_outer_w.rbegin(),m_outer_w.rbegin()+1,m_outer_w.rend());
        dg::blas1::copy(m_dx,m_outer_w[0]);

        restartCycle ++;
    // Go through the requisite number of restarts.
    } while( (restartCycle < m_maxRestarts) && (rho > tol));
    return restartCycle*m_krylovDimension;
}
///@endcond
}//namespace dg
#endif
