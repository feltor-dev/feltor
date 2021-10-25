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
*/
template< class ContainerType>
class LGMRES
{
  public:
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    LGMRES(){}
    ///@copydoc construct()
    LGMRES( const ContainerType& copyable, unsigned max_outer, unsigned max_inner, unsigned Restarts){
        construct(copyable, max_outer, max_inner, Restarts);
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
     * @brief Allocate memory for the preconditioned LGMRES method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_inner Maximum number of vectors to be saved in gmres. Usually 30 seems to be a decent number.
     * @param max_outer Maximum number of (additional) solutions saved for restart. Usually 3-10 seems to be a good number. The Krylov Dimension is \c max_inner+max_outer
     * @param Restarts Maximum number of restarts. This can be set high just in case. Like e.g. gridsize/max_outer.
     */
    void construct(const ContainerType& copyable, unsigned max_outer, unsigned max_inner, unsigned Restarts){
        m_outer_k = max_outer;
        m_inner_m = max_inner;
        m_maxRestarts = Restarts;
        m_krylovDimension = m_inner_m + m_outer_k;
        //Declare Hessenberg matrix
        for(unsigned i = 0; i < m_krylovDimension+1; i++){
            m_H.push_back(std::vector<value_type>());
            for(unsigned j = 0; j < m_krylovDimension; j++){
                m_H[i].push_back(0);
            }
        }
        //Declare givens rotation matrix
        for(unsigned i = 0; i < m_krylovDimension+1; i++){
            givens.push_back(std::vector<value_type>());
            for(unsigned j = 0; j < 2; j++){
                givens[i].push_back(0);
            }
        }
        //Declare s that minimizes the residual...
        m_s.assign(m_krylovDimension+1,0);

        //The residual which will be used to calculate the solution.
        m_V.assign(m_krylovDimension+1,copyable);
        m_W.assign(m_krylovDimension,copyable);
        //In principle we don't need this many... but just to be on board with the algorithm
        m_outer_w.assign(m_outer_k+1,copyable);
        m_tmp = copyable;
        m_dx = copyable;
        m_residual = copyable;
    }

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
    template < class ContainerType0>
    void Update(ContainerType &dx, ContainerType0 &x, unsigned dimension,
            const std::vector<std::vector<value_type>> &H,
            std::vector<value_type> &s, const std::vector<ContainerType> &W);
    std::vector<std::vector<value_type>> m_H, givens;
    ContainerType m_tmp, m_dx, m_residual;
    std::vector<ContainerType> m_V, m_W, m_outer_w;
    std::vector<value_type> m_s;
    unsigned m_maxRestarts, m_inner_m, m_outer_k, m_krylovDimension;
};
///@cond

template< class ContainerType>
template < class ContainerType0>
void LGMRES<ContainerType>::Update(ContainerType &dx, ContainerType0 &x,
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

    // Finally update the approximation.
    dg::blas1::scal(dx,0.);
    for (unsigned lupe = 0; lupe <= dimension; lupe++)
        dg::blas1::axpby(s[lupe],W[lupe],1.,dx);
    dg::blas1::axpby(1.,dx,1.,x);
}

template< class ContainerType>
template< class Matrix, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm>
unsigned LGMRES< ContainerType>::solve( Matrix& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps, value_type nrmb_correction)
{
    value_type nrmb = sqrt( blas2::dot( S, b));
    value_type tol = eps*(nrmb + nrmb_correction);
    if( nrmb == 0)
    {
        blas1::copy( 0., x);
        return 0;
    }
    dg::blas2::symv(A,x,m_tmp);
    dg::blas1::axpby(1.,b,-1.,m_tmp);
    value_type normres = sqrt(dg::blas2::dot(S,m_tmp));
    if( normres < tol) //if x happens to be the solution
        return 0;
    dg::blas2::symv(P,m_tmp,m_residual);
    value_type rho = sqrt(dg::blas1::dot(m_residual,m_residual));

    unsigned restartCycle = 0;

    // Go through the requisite number of restarts.

    while( (restartCycle < m_maxRestarts) && (normres > tol))
	{
        // The first vector in the Krylov subspace is the normalized residual.
        dg::blas1::axpby(1.0/rho,m_residual,0.,m_V[0]);

        m_s.assign(m_krylovDimension+1,0);
		m_s[0] = rho;

		// Go through and generate the pre-determined number of vectors for the Krylov subspace.
		for( unsigned iteration=0;iteration<m_krylovDimension;++iteration)
		{
            unsigned outer_v_count = std::min(restartCycle,m_outer_k);
            if(iteration < outer_v_count){
                // MW: I don't think we need that multiplication
                dg::blas2::symv(A,m_outer_w[iteration],m_tmp);
                dg::blas1::copy(m_outer_w[iteration],m_W[iteration]);
            } else if (iteration == outer_v_count) {
                dg::blas2::symv(A,m_V[0],m_tmp);
                dg::blas1::copy(m_V[0],m_W[iteration]);
            } else {
                dg::blas2::symv(A,m_V[iteration],m_tmp);
                dg::blas1::copy(m_V[iteration],m_W[iteration]);
            }

			// Get the next entry in the vectors that form the basis for the Krylov subspace.
            dg::blas2::symv(P,m_tmp,m_V[iteration+1]);

            for(unsigned row=0;row<=iteration;++row)
			{
                m_H[row][iteration] = dg::blas1::dot(m_V[iteration+1],m_V[row]);
                dg::blas1::axpby(-m_H[row][iteration],m_V[row],1.,m_V[iteration+1]);
			}
            m_H[iteration+1][iteration] = sqrt(dg::blas1::dot(m_V[iteration+1],m_V[iteration+1]));
            dg::blas1::scal(m_V[iteration+1],1.0/m_H[iteration+1][iteration]);

			// Apply the Givens Rotations to insure that H is
			// an upper diagonal matrix. First apply previous
			// rotations to the current matrix.
			value_type tmp;
			for (unsigned row = 0; row < iteration; row++)
			{
				tmp = givens[row][0]*m_H[row][iteration] +
					givens[row][1]*m_H[row+1][iteration];
				m_H[row+1][iteration] = -givens[row][1]*m_H[row][iteration]
					+ givens[row][0]*m_H[row+1][iteration];
				m_H[row][iteration]  = tmp;
			}

			// Figure out the next Givens rotation.
			if(m_H[iteration+1][iteration] == 0.0)
			{
				// It is already lower diagonal. Just leave it be....
				givens[iteration][0] = 1.0;
				givens[iteration][1] = 0.0;
			}
			else if (fabs(m_H[iteration+1][iteration]) > fabs(m_H[iteration][iteration]))
			{
				// The off diagonal entry has a larger
				// magnitude. Use the ratio of the
				// diagonal entry over the off diagonal.
				tmp = m_H[iteration][iteration]/m_H[iteration+1][iteration];
				givens[iteration][1] = 1.0/sqrt(1.0+tmp*tmp);
				givens[iteration][0] = tmp*givens[iteration][1];
			}
			else
			{
				// The off diagonal entry has a smaller
				// magnitude. Use the ratio of the off
				// diagonal entry to the diagonal entry.
				tmp = m_H[iteration+1][iteration]/m_H[iteration][iteration];
				givens[iteration][0] = 1.0/sqrt(1.0+tmp*tmp);
				givens[iteration][1] = tmp*givens[iteration][0];
			}
            // Apply the new Givens rotation on the new entry in the upper Hessenberg matrix.
			tmp = givens[iteration][0]*m_H[iteration][iteration] +
				  givens[iteration][1]*m_H[iteration+1][iteration];
			m_H[iteration+1][iteration] = -givens[iteration][1]*m_H[iteration][iteration] +
				  givens[iteration][0]*m_H[iteration+1][iteration];
			m_H[iteration][iteration] = tmp;
			// Finally apply the new Givens rotation on the s vector
			tmp = givens[iteration][0]*m_s[iteration] + givens[iteration][1]*m_s[iteration+1];
			m_s[iteration+1] = -givens[iteration][1]*m_s[iteration] + givens[iteration][1]*m_s[iteration+1];
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
                Update(m_dx,x,iteration,m_H,m_s,m_W);
                //dg::blas2::symv(A,x,m_residual);
                //dg::blas1::axpby(1.,b,-1.,m_residual);
                //std::cout << sqrt(dg::blas2::dot(S,m_residual) )<< std::endl;
                return(iteration+restartCycle*m_krylovDimension);
            }
        }
        Update(m_dx,x,m_krylovDimension-1,m_H,m_s,m_W);
        value_type nx = sqrt(dg::blas1::dot(m_dx,m_dx));
        if(nx>0.){
            if (restartCycle<m_outer_k){
                dg::blas1::axpby(1.0/nx,m_dx,0.,m_outer_w[restartCycle]); //new outer entry = dx/nx
            } else {
                std::rotate(m_outer_w.begin(),m_outer_w.begin()+1,m_outer_w.end()); //rotate one to the left.
                dg::blas1::axpby(1.0/nx,m_dx,0.,m_outer_w[m_outer_k]);
            }
        }
        dg::blas2::symv(A,x,m_tmp);
        dg::blas1::axpby(1.,b,-1.,m_tmp);
        normres = sqrt(dg::blas2::dot(S,m_tmp));
        dg::blas2::symv(P,m_tmp,m_residual);
        //value_type rho = sqrt(dg::blas1::dot(m_residual,m_residual));
        restartCycle ++;
    }
    return restartCycle*m_krylovDimension;
}
///@endcond
}//namespace dg
#endif
