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
 * @author Aslak Poulsen, Matthias Wiesenberger
 */

namespace dg{

/**
* @brief Functor class for the right preconditioned LGMRES method to solve
* \f$ Ax=b\f$
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
* \sa For more information see the book
* <a href="https://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf">Iteratvie Methods for Sparse Linear Systems" 2nd edition by Yousef Saad </a>
*
* Basically the Krylov subspace is augmented by \c k approximations to the error.
* The storage requirement is m+3k vectors
*
* @snippet bicgstabl_t.cpp lgmres
*
* @note We use **right preconditioning** because this makes the residual norm automatically available in each iteration
* @note the orthogonalization is done with respect to a user-provided inner product \c W.
* @note the first cycle of LGMRES(m,k) is equivalent to the first cycle of GMRES(m+k)
* @note Only \c m matrix-vector multiplications need to be computed in LGMRES(m,k) per restart cycle irrespective of the value of \c k
*
* @attention There are a lot of calls to \c dot in LGMRES such that GPUs may struggle for small vector sizes
* @attention LGMRES can stagnate if the matrix A is not positive definite. The
* use of a preconditioner is paramount in such a situation
*/
template< class ContainerType>
class LGMRES
{
  public:
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    LGMRES() = default;
    /**
     * @brief Allocate memory for the preconditioned LGMRES method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_inner Maximum number inner gmres iterations per restart.
     * Usually 20-30 seems to be a decent number. Per iteration a matrix-vector product and a preconditioner-vector product needs to be computed.
     * @param max_outer Maximum number of solutions (actually approximations to the error) saved for restart. Usually 1...3 is a good number. The Krylov Dimension is thus augmented to \c max_inner+max_outer. No new matrix-vector products need to be computed for the additional solutions. \c max_outer=0 corresponds to standard GMRES.
     * @param max_restarts Maximum number of restarts. The total maximum number of iterations/ matrix-vector products is thus \c max_restarts*max_inner
     */
    LGMRES( const ContainerType& copyable, unsigned max_inner, unsigned max_outer, unsigned max_restarts):
        m_tmp(copyable),
        m_dx(copyable),
        m_residual( copyable),
        m_maxRestarts( max_restarts),
        m_inner_m( max_inner),
        m_outer_k( max_outer),
        m_krylovDimension( max_inner+max_outer)
    {
        if( m_inner_m < m_outer_k)
            std::cerr << "WARNING (LGMRES): max_inner is smaller than the restart dimension max_outer. Did you swap the constructor parameters?\n";
        //Declare Hessenberg matrix
        m_H.assign( m_krylovDimension+1, std::vector<value_type>( m_krylovDimension, 0));
        m_HH = m_H; //copy of H to be stored unaltered
        //Declare givens rotation matrix
        m_givens.assign( m_krylovDimension+1, {0,0});
        //Declare s that minimizes the residual:
        m_s.assign(m_krylovDimension+1,0);
        // m+k+1 orthogonal basis vectors:
        // k augmented pairs
        m_outer_w.assign(m_outer_k,copyable);
        m_outer_Az.assign(m_outer_k,copyable);
        m_V.assign(m_krylovDimension+1,copyable);
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
    ///@brief Get the current maximum number of restarts
    ///@return the current maximum of restarts
    unsigned get_max() const {return m_maxRestarts;}
    ///@copydoc dg::PCG::set_throw_on_fail(bool)
    void set_throw_on_fail( bool throw_on_fail){
        m_throw_on_fail = throw_on_fail;
    }
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_tmp;}

    /**
     * @brief Solve \f$ Ax = b\f$ using a right preconditioned LGMRES method
     *
     * The iteration stops if \f$ ||Ax-b||_W < \epsilon( ||b||_W + C) \f$ where \f$C\f$ is
     * the absolute error in units of \f$ \epsilon\f$ and \f$ W \f$ defines a square norm
     * @param A A matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param W A diagonal matrix (a vector) that is used to define the scalar
     * product in which the orthogonalization in LGMRES is computed and that
     * defines the norm in which the stopping criterion is computed
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     *
     * @return Number of times the matrix A and the preconditioner P were
     * multiplied to achieve the desired precision
     * @note The method will throw \c dg::Fail if the desired accuracy is not reached within \c max_restarts
     * You can unset this behaviour with the \c set_throw_on_fail member
     * @copydoc hide_matrix
     * @copydoc hide_ContainerType
     */
    template< class MatrixType0, class ContainerType0, class ContainerType1, class MatrixType1, class ContainerType2 >
    unsigned solve( MatrixType0&& A, ContainerType0& x, const ContainerType1& b, MatrixType1&& P, const ContainerType2& W, value_type eps = 1e-12, value_type nrmb_correction = 1);

    /**
     * @brief If last call to solve converged or not
     *
     * @return true if convergence was reached, false else
     */
    bool converged() const{
        return m_converged;
    }

  private:
    template <class Preconditioner, class ContainerType0>
    void Update(Preconditioner&& P, ContainerType &dx, ContainerType0 &x,
            unsigned dimension, const std::vector<std::vector<value_type>> &H,
            std::vector<value_type> &s, const std::vector<const ContainerType*> &W);
    std::vector<std::array<value_type,2>> m_givens;
    std::vector<std::vector<value_type>> m_H, m_HH;
    ContainerType m_tmp, m_dx, m_residual;
    std::vector<ContainerType> m_V, m_outer_w, m_outer_Az;
    std::vector<value_type> m_s;
    unsigned m_maxRestarts, m_inner_m, m_outer_k, m_krylovDimension;
    bool m_converged = true, m_throw_on_fail = true;
};
///@cond

template< class ContainerType>
template < class Preconditioner, class ContainerType0>
void LGMRES<ContainerType>::Update(Preconditioner&& P, ContainerType &dx,
        ContainerType0 &x,
        unsigned dimension, const std::vector<std::vector<value_type>> &H,
        std::vector<value_type> &s, const std::vector<const ContainerType*> &W)
{
    // Solve for the coefficients, i.e. solve for c in
    // H*c=s, but we do it in place.
    for (int lupe = dimension; lupe >= 0; --lupe)
    {
        s[lupe] = s[lupe]/H[lupe][lupe];
        for (int innerLupe = lupe - 1; innerLupe >= 0; --innerLupe)
        {
            // Subtract off the parts from the upper diagonal of the matrix.
            s[innerLupe] =  DG_FMA( -s[lupe],H[innerLupe][lupe], s[innerLupe]);
        }
	}

    // Finally update the approximation. W_m*s
    dg::blas2::gemv( dg::asDenseMatrix( W, dimension+1), std::vector<value_type>( s.begin(), s.begin()+dimension+1), dx);
    // right preconditioner
    dg::blas2::gemv( std::forward<Preconditioner>(P), dx, m_tmp);
    dg::blas1::axpby(1.,m_tmp,1.,x);
}

template< class ContainerType>
template< class Matrix, class ContainerType0, class ContainerType1, class Preconditioner, class ContainerType2>
unsigned LGMRES< ContainerType>::solve( Matrix&& A, ContainerType0& x, const ContainerType1& b, Preconditioner&& P, const ContainerType2& S, value_type eps, value_type nrmb_correction)
{
    // Improvements over old implementation:
    // - Use right preconditioned system such that residual norm is available in minimization
    // - do not compute Az explicitly but save on iterations
    // - first cycle equivalent to GMRES(m+k)
    // - use weights for orthogonalization (works because in Saad book 6.29 and 6.30 are also true if V_m is unitary in the S scalar product, the Hessenberg matrix is still formed in the regular 2-norm, just define J(y) with S-norm in 6.26 and form V_m with a Gram-Schmidt process in the W-norm)
    value_type nrmb = sqrt( blas2::dot( S, b));
    value_type tol = eps*(nrmb + nrmb_correction);
    m_converged = true;
    if( nrmb == 0)
    {
        blas1::copy( 0., x);
        return 0;
    }

    unsigned restartCycle = 0;
    unsigned counter = 0;
    value_type rho = 1.;
    // DO NOT HOLD THESE AS PRIVATE!! MAKES BUG IN COPY!!
    std::vector<ContainerType const*> m_W, m_Vptr;
    m_W.assign(m_krylovDimension,nullptr);
    m_Vptr.assign(m_krylovDimension+1,nullptr);
    for( unsigned i=0; i<m_krylovDimension+1; i++)
        m_Vptr[i] = &m_V[i];
    do
	{
        dg::blas2::gemv(std::forward<Matrix>(A),x,m_residual);
        dg::blas1::axpby(1.,b,-1.,m_residual);
        rho = sqrt(dg::blas2::dot(S,m_residual));
        counter ++;
        if( rho < tol) //if x happens to be the solution
            return counter;
        // The first vector in the Krylov subspace is the normalized residual.
        dg::blas1::axpby(1.0/rho,m_residual,0.,m_V[0]);

		m_s[0] = rho;
        for(unsigned lupe=1;lupe<=m_krylovDimension;++lupe)
			m_s[lupe] = 0.0;

		// Go through and generate the pre-determined number of vectors for the Krylov subspace.
		for( unsigned iteration=0;iteration<m_krylovDimension;++iteration)
		{
            unsigned outer_w_count = std::min(restartCycle,m_outer_k);
            if(iteration < m_krylovDimension-outer_w_count){
                m_W[iteration] = &m_V[iteration];
                dg::blas2::gemv(std::forward<Preconditioner>(P),*m_W[iteration],m_tmp);
                dg::blas2::gemv(std::forward<Matrix>(A),m_tmp,m_V[iteration+1]);
                counter++;
            } else if( iteration < m_krylovDimension){ // size of W
                unsigned w_idx = iteration - (m_krylovDimension - outer_w_count);
                m_W[iteration] = &m_outer_w[w_idx];
                dg::blas1::copy( m_outer_Az[w_idx], m_V[iteration+1]);
            }

			// Get the next entry in the vectors that form the basis for the Krylov subspace.
            // Arnoldi modified Gram-Schmidt orthogonalization
            for(unsigned row=0;row<=iteration;++row)
			{
                m_HH[row][iteration] = m_H[row][iteration]
                    = dg::blas2::dot(m_V[iteration+1],S,m_V[row]);
                dg::blas1::axpby(-m_H[row][iteration],m_V[row],1.,m_V[iteration+1]);

			}
            m_HH[iteration+1][iteration] = m_H[iteration+1][iteration]
                = sqrt(dg::blas2::dot(m_V[iteration+1],S,m_V[iteration+1]));
            dg::blas1::scal(m_V[iteration+1],1.0/m_H[iteration+1][iteration]);

            // Now solve the least squares problem
            // using Givens Rotations transforming H into
            // an upper triangular matrix (see Saad Chapter 6.5.3)
            // corresponding to QR-decomposition of H

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
			tmp = m_givens[iteration][0]*m_s[iteration] + m_givens[iteration][1]*m_s[iteration+1];
			m_s[iteration+1] = -m_givens[iteration][1]*m_s[iteration] + m_givens[iteration][1]*m_s[iteration+1];
			m_s[iteration] = tmp;

            rho = fabs(m_s[iteration+1]);
            if( rho < tol)
			{
                Update(std::forward<Preconditioner>(P),m_dx,x,iteration,m_H,m_s,m_W);
                return counter;
            }
        }
        Update(std::forward<Preconditioner>(P),m_dx,x,m_krylovDimension-1,m_H,m_s,m_W);
        if( m_outer_k > 1)
        {
            std::rotate(m_outer_w.rbegin(),m_outer_w.rbegin()+1,m_outer_w.rend());
            std::rotate(m_outer_Az.rbegin(),m_outer_Az.rbegin()+1,m_outer_Az.rend());
        }
        if( m_outer_k > 0)
        {
            dg::blas1::copy(m_dx,m_outer_w[0]);
            // compute A P dx
            std::vector<value_type> coeffs( m_krylovDimension+1, 0.);
            for( unsigned i=0; i<m_krylovDimension+1; i++)
            {
                coeffs[i] = 0.;
                for( unsigned k=0; k<m_krylovDimension; k++)
                    coeffs[i] = DG_FMA( m_HH[i][k],m_s[k], coeffs[i]);
            }
            dg::blas2::gemv( dg::asDenseMatrix( m_Vptr), coeffs, m_outer_Az[0]);
        }

        restartCycle ++;
    // Go through the requisite number of restarts.
    } while( (restartCycle < m_maxRestarts) && (rho > tol));
    if( rho > tol)
    {
        if( m_throw_on_fail)
        {
            throw dg::Fail( eps, Message(_ping_)
                <<"After "<<counter<<" LGMRES iterations");
        }
        m_converged = false;
    }
    return counter;
}
///@endcond
}//namespace dg
#endif
