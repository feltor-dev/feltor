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
    void set_max( unsigned new_Restarts) {numberRestarts = new_Restarts;}
    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_numberRestarts() const {return numberRestarts;}
    /**
     * @brief Allocate memory for the preconditioned LGMRES method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_inner Maximum number of vectors to be saved in gmres. Usually 30 seems to be a decent number.
     * @param max_outer Maximum number of solutions saved for restart. Usually 3-10 seems to be a good number.
     * @param Restarts Maximum number of restarts. This can be set high just in case. Like e.g. gridsize/max_outer.
     */
    void construct(const ContainerType& copyable, unsigned max_outer, unsigned max_inner, unsigned Restarts){
        outer_k = max_outer;
        inner_m = max_inner;
        numberRestarts = Restarts;
        krylovDimension = inner_m + outer_k;
        //Declare Hessenberg matrix
        for(unsigned i = 0; i < krylovDimension+1; i++){
            H.push_back(std::vector<value_type>());
            for(unsigned j = 0; j < krylovDimension; j++){
                H[i].push_back(0);
            }
        }
        //Declare givens rotation matrix
        for(unsigned i = 0; i < krylovDimension+1; i++){
            givens.push_back(std::vector<value_type>());
            for(unsigned j = 0; j < 2; j++){
                givens[i].push_back(0);
            }
        }
        //Declare s that minimizes the residual... something like that.
        //s(krylovDimension+1);
        s.assign(krylovDimension,0);

        //The residual which will be used to calculate the solution.
        V.assign(krylovDimension+1,copyable);
        W.assign(krylovDimension,copyable);
        //In principle we don't need this many... but just to be on board with the algorithm
        outer_v.assign(outer_k,copyable);
        z = copyable;
        dx = copyable;
        residual = copyable;
    }

    /**
     * @brief Solve \f$ Ax = b\f$ using a preconditioned LGMRES method
     *
     * The iteration stops if \f$ ||Ax||_S < \epsilon( ||b||_S + C) \f$ where \f$C\f$ is
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
    template < class Hess, class HessContainerType1, class HessContainerType2, class HessContainerType3  >
    void Update(HessContainerType1 &dx, HessContainerType1 &x, unsigned dimension, Hess &H, HessContainerType2 &s, HessContainerType3 &V);
    value_type tolerance;
    std::vector<std::vector<value_type>> H, givens;
    ContainerType z, dx, residual;
    std::vector<ContainerType> V, W, outer_v;
    std::vector<value_type> s;
    unsigned numberRestarts, inner_m, outer_k, krylovDimension;
};
///@cond

template< class ContainerType>
template < class Hess, class HessContainerType1, class HessContainerType2, class HessContainerType3  >
void LGMRES< ContainerType>::Update(HessContainerType1 &dx, HessContainerType1 &x, unsigned dimension, Hess &H, HessContainerType2 &s, HessContainerType3 &V)
{
    // Solve for the coefficients, i.e. solve for c in
    // H*c=s, but we do it in place.
    int lupe;
    for (lupe = dimension; lupe >= 0; --lupe)
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

    // Finally update the approximation.
    dg::blas1::scal(dx,0.);
    for (lupe = 0; lupe <= (int)dimension; lupe++)
        dg::blas1::axpby(s[lupe],V[lupe],1.,dx);
    dg::blas1::axpby(1.,dx,1.,x);
}

template< class ContainerType>
template< class Matrix, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm>
unsigned LGMRES< ContainerType>::solve( Matrix& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps, value_type nrmb_correction)
{
    dg::blas2::symv(A,x,residual);
    dg::blas1::axpby(1.,b,-1.,residual);
    dg::blas2::symv(P,residual,residual);
    value_type rho = sqrt(dg::blas1::dot(residual,residual));
    value_type normres = sqrt(dg::blas2::dot(S,residual));
    value_type normRHS = sqrt(dg::blas1::dot(b,b));
    value_type normedRHS = sqrt(dg::blas2::dot(S,b));

    tolerance = eps;

    unsigned totalRestarts = 0;

    if(normRHS < 1.0E-5)
        normRHS = 1.0;

    // Go through the requisite number of restarts.
	unsigned iteration = 0;

    while( (totalRestarts < numberRestarts) && (normres > tolerance*normedRHS))
	{
        // The first vector in the Krylov subspace is the normalized residual.
        dg::blas1::axpby(1.0/rho,residual,0.,V[0]);

        for(unsigned lupe=0;lupe<=krylovDimension;++lupe)
			s[lupe] = 0.0;
		s[0] = rho;

		// Go through and generate the pre-determined number of vectors for the Krylov subspace.
		for( iteration=0;iteration<krylovDimension;++iteration)
		{
            unsigned outer_v_count = std::min(totalRestarts,outer_k);
            if(iteration < outer_v_count){
                //dg::blas1::copy(outer_v[totalRestarts-outer_v_count+iteration],z);
                dg::blas1::copy(outer_v[iteration],z);
            } else if (iteration == outer_v_count) {
                dg::blas1::copy(V[0],z);
            } else {
                dg::blas1::copy(V[iteration],z);
            }

			// Get the next entry in the vectors that form the basis for the Krylov subspace.
            dg::blas2::symv(A,z,V[iteration+1]);
            dg::blas2::symv(P,V[iteration+1],V[iteration+1]);
            unsigned row;

            for(row=0;row<=iteration;++row)
			{
                H[row][iteration] = dg::blas1::dot(V[iteration+1],V[row]);
                dg::blas1::axpby(-H[row][iteration],V[row],1.,V[iteration+1]);
			}
            H[iteration+1][iteration] = sqrt(dg::blas1::dot(V[iteration+1],V[iteration+1]));
            dg::blas1::scal(V[iteration+1],1.0/H[iteration+1][iteration]);
            dg::blas1::copy(z,W[iteration]);

			// Apply the Givens Rotations to insure that H is
			// an upper diagonal matrix. First apply previous
			// rotations to the current matrix.
			value_type tmp;
			for (row = 0; row < iteration; row++)
			{
				tmp = givens[row][0]*H[row][iteration] +
					givens[row][1]*H[row+1][iteration];
				H[row+1][iteration] = -givens[row][1]*H[row][iteration]
					+ givens[row][0]*H[row+1][iteration];
				H[row][iteration]  = tmp;
			}

			// Figure out the next Givens rotation.
			if(H[iteration+1][iteration] == 0.0)
			{
				// It is already lower diagonal. Just leave it be....
				givens[iteration][0] = 1.0;
				givens[iteration][1] = 0.0;
			}
			else if (fabs(H[iteration+1][iteration]) > fabs(H[iteration][iteration]))
			{
				// The off diagonal entry has a larger
				// magnitude. Use the ratio of the
				// diagonal entry over the off diagonal.
				tmp = H[iteration][iteration]/H[iteration+1][iteration];
				givens[iteration][1] = 1.0/sqrt(1.0+tmp*tmp);
				givens[iteration][0] = tmp*givens[iteration][1];
			}
			else
			{
				// The off diagonal entry has a smaller
				// magnitude. Use the ratio of the off
				// diagonal entry to the diagonal entry.
				tmp = H[iteration+1][iteration]/H[iteration][iteration];
				givens[iteration][0] = 1.0/sqrt(1.0+tmp*tmp);
				givens[iteration][1] = tmp*givens[iteration][0];
			}
            // Apply the new Givens rotation on the new entry in the upper Hessenberg matrix.
			tmp = givens[iteration][0]*H[iteration][iteration] +
				  givens[iteration][1]*H[iteration+1][iteration];
			H[iteration+1][iteration] = -givens[iteration][1]*H[iteration][iteration] +
				  givens[iteration][0]*H[iteration+1][iteration];
			H[iteration][iteration] = tmp;
			// Finally apply the new Givens rotation on the s vector
			tmp = givens[iteration][0]*s[iteration] + givens[iteration][1]*s[iteration+1];
			s[iteration+1] = -givens[iteration][1]*s[iteration] + givens[iteration][1]*s[iteration+1];
			s[iteration] = tmp;

            rho = fabs(s[iteration+1]);
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif //MPI
            std::cout << "# rho = " << rho << std::endl;
#endif //DG_DEBUG
			if(rho < tolerance*normRHS)
			{

                Update(dx,x,iteration,H,s,W);
                dg::blas2::symv(A,x,residual);
                dg::blas1::axpby(1.,b,-1.,residual);
                std::cout << sqrt(dg::blas2::dot(S,residual) )<< std::endl;
                return(iteration+totalRestarts*krylovDimension);
            }
        }
        Update(dx,x,iteration-1,H,s,W);
        value_type nx = sqrt(dg::blas1::dot(dx,dx));
        if(nx>0.){
            if (totalRestarts<outer_k){
                dg::blas1::axpby(1.0/nx,dx,0.,outer_v[totalRestarts]); //new outer entry = dx/nx
            } else {
                std::rotate(outer_v.begin(),outer_v.begin()+1,outer_v.end()); //rotate one to the left.
                dg::blas1::axpby(1.0/nx,dx,0.,outer_v[outer_k]);
            }
        }
        dg::blas2::symv(A,x,residual);
        dg::blas1::axpby(1.,b,-1.,residual);
        normres = sqrt(dg::blas2::dot(S,residual));
        dg::blas2::symv(P,residual,residual);
        value_type rho = sqrt(dg::blas1::dot(residual,residual));
        totalRestarts += 1;
    }
    return totalRestarts*krylovDimension;
}
///@endcond
}//namespace dg
#endif
