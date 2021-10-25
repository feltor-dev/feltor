#ifndef _DG_CG_
#define _DG_CG_

#include <cmath>

#include "blas.h"
#include "functors.h"
#include "extrapolation.h"

#ifdef DG_BENCHMARK
#include "backend/timer.h"
#endif //DG_BENCHMARK

/*!@file
 * Conjugate gradient class and functions
 */

namespace dg{

//// TO DO: check for better stopping criteria using condition number estimates?

/**
* @brief Preconditioned conjugate gradient method to solve
* \f[ M^{-1}Ax=M^{-1}b\f]
*
* @ingroup invert
*
* @sa This implements the PCG algorithm as given in https://en.wikipedia.org/wiki/Conjugate_gradient_method
or the book
* <a href="https://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf">Iteratvie Methods for Sparse Linear Systems" 2nd edition by Yousef Saad </a>
* @note Conjugate gradients might become unstable for positive semidefinite
* matrices arising e.g. in the discretization of the periodic laplacian
* @attention beware the sign: a negative definite matrix does @b not work in Conjugate gradient
*
* @snippet cg2d_t.cu doxygen
* @copydoc hide_ContainerType
*/
template< class ContainerType>
class CG
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    CG(){}
    /**
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    CG( const ContainerType& copyable, unsigned max_iterations):
        r(copyable), p(r), ap(r), max_iter(max_iterations){}
    ///@brief Set the maximum number of iterations
    ///@param new_max New maximum number
    void set_max( unsigned new_max) {max_iter = new_max;}
    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_max() const {return max_iter;}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return r;}

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
        *this = CG( std::forward<Params>( ps)...);
    }
    ///@brief DEPRECATED: use solve method instead
    ///@copydetails solve(MatrixType&,ContainerType0&,const ContainerType1&,Preconditioner&,value_type,value_type)
    template< class MatrixType, class ContainerType0, class ContainerType1, class Preconditioner >
    unsigned operator()( MatrixType& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P , value_type eps = 1e-12, value_type nrmb_correction = 1)
    {
        return solve(A,x,b,P,eps,nrmb_correction);
    }
    /**
     * @brief Solve the system A*x = b using a preconditioned conjugate gradient method
     *
     * The iteration stops if \f$ ||b - Ax|| < \epsilon( ||b|| + C) \f$ where \f$C\f$ is
     * the absolute error in units of \f$ \epsilon\f$
     * @param A A symmetric, positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     * @attention This version uses the Preconditioner to compute the norm for the error condition (this safes one scalar product)
     *
     * @return Number of iterations used to achieve desired precision
     * @note Required memops per iteration (\c P is assumed vector):
             - 11  reads + 3 writes
             - plus the number of memops for \c A;
     * @copydoc hide_matrix
     * @tparam ContainerTypes must be usable with \c MatrixType and \c ContainerType in \ref dispatch
     * @tparam Preconditioner A class for which the <tt> blas2::symv(value_type, const Preconditioner&, const ContainerType&, value_type, ContainerType&) and
     blas2::dot( const Preconditioner&, const ContainerType&) </tt> functions are callable.
     */
    template< class MatrixType, class ContainerType0, class ContainerType1, class Preconditioner >
    unsigned solve( MatrixType& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P , value_type eps = 1e-12, value_type nrmb_correction = 1);
    ///@brief DEPRECATED: use solve method instead
    ///@copydetails solve(MatrixType&,ContainerType0&,const ContainerType1&,Preconditioner&,SquareNorm&,value_type,value_type,int)
    template< class MatrixType, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm >
    unsigned operator()( MatrixType& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps = 1e-12, value_type nrmb_correction = 1, int test_frequency = 1)
    {
        return solve(A,x,b,P,S,eps,nrmb_correction,test_frequency);
    }
    //version of CG where Preconditioner is not trivial
    /**
     * @brief Solve \f$ Ax = b\f$ using a preconditioned conjugate gradient method
     *
     * The iteration stops if \f$ ||Ax-b||_S < \epsilon( ||b||_S + C) \f$ where \f$C\f$ is
     * the absolute error in units of \f$ \epsilon\f$ and \f$ S \f$ defines a square norm
     * @param A A symmetric positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param S (Inverse) Weights used to compute the norm for the error condition
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     * @param test_frequency if set to 1 then the norm of the error is computed in every iteration to test if the loop can be terminated. Sometimes, especially for small sizes the dot product is expensive to compute, then it is beneficial to set this parameter to e.g. 10, which means that the errror condition is only evaluated every 10th iteration.
     *
     * @return Number of iterations used to achieve desired precision
     * @note Required memops per iteration (\c P and \c S are assumed vectors):
             - 15  reads + 4 writes
             - plus the number of memops for \c A;
     * @copydoc hide_matrix
     * @tparam ContainerTypes must be usable with \c MatrixType and \c ContainerType in \ref dispatch
     * @tparam Preconditioner A type for which the blas2::symv(Preconditioner&, ContainerType&, ContainerType&) function is callable.
     * @tparam SquareNorm A type for which the blas2::dot( const SquareNorm&, const ContainerType&) function is callable. This can e.g. be one of the ContainerType types.
     */
    template< class MatrixType, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm >
    unsigned solve( MatrixType& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps = 1e-12, value_type nrmb_correction = 1, int test_frequency = 1);
  private:
    ContainerType r, p, ap;
    unsigned max_iter;
};

/*
    compared to unpreconditioned compare
    dot(r,r), axpby()
    to
    dot( r,P,r), symv(P)
    i.e. it will be slower, if P needs to be stored
    (but in our case P_{ii} can be computed directly
    compared to normal preconditioned compare
    ddot(r,P,r), dsymv(P)
    to
    ddot(r,z), dsymv(P), axpby(), (storage for z)
    i.e. it's surely faster if P contains no more elements than z
    (which is the case for diagonal scaling)
    NOTE: the same comparison hold for A with the result that A contains
    significantly more elements than z whence ddot(r,A,r) is far slower than ddot(r,z)
*/
///@cond
template< class ContainerType>
template< class Matrix, class ContainerType0, class ContainerType1, class Preconditioner>
unsigned CG< ContainerType>::solve( Matrix& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, value_type eps, value_type nrmb_correction)
{
    value_type nrmb = sqrt( blas2::dot( P, b));
    value_type tol = eps*(nrmb + nrmb_correction);
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif //MPI
    {
    std::cout << "# Norm of b "<<nrmb <<"\n";
    std::cout << "# Residual errors: \n";
    }
#endif //DG_DEBUG
    if( nrmb == 0)
    {
        blas1::copy( 0., x);
        return 0;
    }
    blas2::symv( A,x,r);
    blas1::axpby( 1., b, -1., r);
    blas2::symv( P, r, p );//<-- compute p_0
    //note that dot does automatically synchronize
    value_type nrm2r_old = blas2::dot( P,r); //and store the norm of it
    if( sqrt( nrm2r_old ) < tol) //if x happens to be the solution
        return 0;
    value_type alpha, nrm2r_new;
    for( unsigned i=1; i<max_iter; i++)
    {
        blas2::symv( A, p, ap);
        alpha = nrm2r_old /blas1::dot( p, ap);
        blas1::axpby( alpha, p, 1.,x);
	        //here one could add a ifstatement to remove accumulated floating point error
//             if (i % 100==0) {
//                   blas2::symv( A,x,r);
//                   blas1::axpby( 1., b, -1., r);
//             }
//             else {
//                   blas1::axpby( -alpha, ap, 1., r);
//             }
        blas1::axpby( -alpha, ap, 1., r);
        nrm2r_new = blas2::dot( P, r);
#ifdef DG_DEBUG
#ifdef MPI_VERSION
        if(rank==0)
#endif //MPI
        {
            std::cout << "# Absolute "<<sqrt( nrm2r_new) <<"\t ";
            std::cout << "#  < Critical "<<tol <<"\t ";
            std::cout << "# (Relative "<<sqrt( nrm2r_new)/nrmb << ")\n";
        }
#endif //DG_DEBUG
        if( sqrt( nrm2r_new) < tol)
            return i;
        blas2::symv(1.,P, r, nrm2r_new/nrm2r_old, p );
        nrm2r_old=nrm2r_new;

    }
    return max_iter;
}

template< class ContainerType>
template< class Matrix, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm>
unsigned CG< ContainerType>::solve( Matrix& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps, value_type nrmb_correction, int save_on_dots )
{
    value_type nrmb = sqrt( blas2::dot( S, b));
    value_type tol = eps*(nrmb + nrmb_correction);
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif //MPI
    {
    std::cout << "# Norm of S b "<<nrmb <<"\n";
    std::cout << "# Residual errors: \n";
    }
#endif //DG_DEBUG
    if( nrmb == 0)
    {
        blas1::copy( 0., x);
        return 0;
    }
    blas2::symv( A,x,r);
    blas1::axpby( 1., b, -1., r);
    //note that dot does automatically synchronize
    if( sqrt( blas2::dot(S,r) ) < tol) //if x happens to be the solution
        return 0;
    blas2::symv( P, r, p );//<-- compute p_0
    value_type nrmzr_old = blas1::dot( p,r); //and store the scalar product
    value_type alpha, nrmzr_new;
    for( unsigned i=1; i<max_iter; i++)
    {
        blas2::symv( A, p, ap);
        alpha =  nrmzr_old/blas1::dot( p, ap);
        blas1::axpby( alpha, p, 1.,x);
        blas1::axpby( -alpha, ap, 1., r);
        if( 0 == i%save_on_dots )
        {
#ifdef DG_DEBUG
#ifdef MPI_VERSION
            if(rank==0)
#endif //MPI
            {
                std::cout << "# Absolute r*S*r "<<sqrt( blas2::dot(S,r)) <<"\t ";
                std::cout << "#  < Critical "<<tol <<"\t ";
                std::cout << "# (Relative "<<sqrt( blas2::dot(S,r) )/nrmb << ")\n";
            }
#endif //DG_DEBUG
                if( sqrt( blas2::dot(S,r)) < tol)
                    return i;
        }
        blas2::symv(P,r,ap);
        nrmzr_new = blas1::dot( ap, r);
        blas1::axpby(1.,ap, nrmzr_new/nrmzr_old, p );
        nrmzr_old=nrmzr_new;
    }
    return max_iter;
}
///@endcond


/**
 * @brief Wrapper around CG and Extrapolation to solve the Equation \f[ Ax = W  b \f]
 *
 * where \f$A\f$ was made symmetric
 * by appropriate weights \f$W\f$ (s. comment below).
 * Uses solutions from the last calls to
 * extrapolate a solution for the current call.
 *
 * @ingroup invert
 * @snippet elliptic2d_b.cu invert
 * @note A note on weights, inverse weights and preconditioning.
 * A normalized DG-discretized derivative or operator is normally not symmetric.
 * The diagonal coefficient matrix that is used to make the operator
 * symmetric is called weights W, i.e. \f$ \hat O = W\cdot O\f$ is symmetric.
 * Now, to compute the correct scalar product of the right hand side the
 * inverse weights have to be used i.e. \f$ W\rho\cdot W \rho /W\f$.
 * Independent from this, a preconditioner should be used to solve the
 * symmetric matrix equation. The inverse of \f$W\f$ is
 * a good general purpose preconditioner.
 * @attention beware the sign: a negative definite matrix does @b not work in Conjugate gradient
 * @sa Extrapolation MultigridCG2d
 * @copydoc hide_ContainerType
 */
template<class ContainerType>
struct Invert
{
    typedef typename TensorTraits<ContainerType>::value_type value_type;

    ///@brief Allocate nothing
    Invert() { multiplyWeights_ = true; nrmb_correction_ = 1.; }

    ///@copydoc construct()
    Invert(const ContainerType& copyable, unsigned max_iter, value_type eps, int extrapolationType = 2, bool multiplyWeights = true, value_type nrmb_correction = 1)
    {
        construct( copyable, max_iter, eps, extrapolationType, multiplyWeights, nrmb_correction);
    }
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return cg.copyable();}

    /**
     * @brief Allocate memory
     *
     * @param copyable Needed to construct the two previous solutions
     * @param max_iter maximum iteration in conjugate gradient
     * @param eps relative error in conjugate gradient
     * @param extrapolationType number of last values to use for extrapolation of the current guess
     * @param multiplyWeights if true the rhs shall be multiplied by the weights before cg is applied
     * @param nrmb_correction the absolute error \c C in units of \c eps in conjugate gradient
     */
    void construct( const ContainerType& copyable, unsigned max_iter, value_type eps, int extrapolationType = 2, bool multiplyWeights = true, value_type nrmb_correction = 1.)
    {
        m_ex.set_max( extrapolationType, copyable);
        m_rhs = copyable;
        set_size( copyable, max_iter);
        set_accuracy( eps, nrmb_correction);
        multiplyWeights_=multiplyWeights;
    }


    /**
     * @brief Set vector storage and maximum number of iterations
     *
     * @param assignable
     * @param max_iterations
     */
    void set_size( const ContainerType& assignable, unsigned max_iterations) {
        cg.construct( assignable, max_iterations);
        m_ex.set_max( m_ex.get_max(), assignable);
    }

    /**
     * @brief Set accuracy parameters for following inversions
     *
     * @param eps the relative error in conjugate gradient
     * @param nrmb_correction the absolute error \c C in units of \c eps in conjugate gradient
     */
    void set_accuracy( value_type eps, value_type nrmb_correction = 1.) {
        eps_ = eps;
        nrmb_correction_ = nrmb_correction;
    }

    /**
     * @brief Set the maximum number of iterations
     * @param new_max New maximum number
     */
    void set_max( unsigned new_max) {cg.set_max( new_max);}
    /**
     * @brief Get the current maximum number of iterations
     * @return the current maximum
     */
    unsigned get_max() const {return cg.get_max();}

    ///@brief DEPRECATED: use solve method instead
    ///@copydetails solve(SymmetricOp&,ContainerType0&,const ContainerType1&)
    template< class SymmetricOp, class ContainerType0, class ContainerType1 >
    unsigned operator()( SymmetricOp& op, ContainerType0& phi, const ContainerType1& rho)
    {
        return solve(op, phi, rho, op.weights(), op.inv_weights(), op.precond());
    }
    /**
     * @brief Solve linear problem
     *
     * Solves the Equation \f[ \hat O \phi = W\rho \f] using a preconditioned
     * conjugate gradient method. The initial guess comes from an extrapolation
     * of the last solutions
     * @copydoc hide_symmetric_op
     * @param op selfmade symmetric Matrix operator class
     * @param phi solution (write only)
     * @param rho right-hand-side (will be multiplied by \c weights)
     * @note computes inverse weights from the weights
     * @note If the Macro \c DG_BENCHMARK is defined this function will write timings to \c std::cout
     *
     * @return number of iterations used
     */
    template< class SymmetricOp, class ContainerType0, class ContainerType1 >
    unsigned solve( SymmetricOp& op, ContainerType0& phi, const ContainerType1& rho)
    {
        return solve(op, phi, rho, op.weights(), op.inv_weights(), op.precond());
    }

    ///@brief DEPRECATED: use solve method instead
    ///@copydetails solve(MatrixType&,ContainerType0&,const ContainerType1&,const SquareNorm0&,const SquareNorm1&,Preconditioner&)
    template< class MatrixType, class ContainerType0, class ContainerType1, class SquareNorm0, class SquareNorm1, class Preconditioner >
    unsigned operator()( MatrixType& op, ContainerType0& phi, const ContainerType1& rho, const SquareNorm0& weights, const SquareNorm1& inv_weights, Preconditioner& p)
    {
        return solve( op, phi, rho, weights, inv_weights, p);
    }

    /**
     * @brief Solve linear problem
     *
     * Solves the Equation \f[ \hat O \phi = W\rho \f] using a preconditioned
     * conjugate gradient method. The initial guess comes from an extrapolation
     * of the last solutions.
     * @copydoc hide_matrix
     * @tparam ContainerTypes must be usable with \c ContainerType in \ref dispatch
     * @tparam SquareNorm A type for which the blas2::dot( const Matrix&, const Vector&) function is callable. This can e.g. be one of the container types.
     * @tparam Preconditioner A type for which the <tt> blas2::symv(Matrix&, Vector1&, Vector2&) </tt> function is callable.
     * @param op symmetric Matrix operator class
     * @param phi solution (write only)
     * @param rho right-hand-side (will be multiplied by \c weights)
     * @param weights The weights that normalize the symmetric operator
     * @param inv_weights The inverse of the weights that normalize the symmetric operator
     * @param p The preconditioner
     * @note (15+N)memops per iteration where N is the memops contained in \c op.
     *   If the Macro \c DG_BENCHMARK is defined this function will write timings to \c std::cout
     *
     * @return number of iterations used
     */
    template< class MatrixType, class ContainerType0, class ContainerType1, class SquareNorm0, class SquareNorm1, class Preconditioner >
    unsigned solve( MatrixType& op, ContainerType0& phi, const ContainerType1& rho, const SquareNorm0& weights, const SquareNorm1& inv_weights, Preconditioner& p)
    {
        assert( phi.size() != 0);
        assert( &rho != &phi);
#ifdef DG_BENCHMARK
        Timer t;
        t.tic();
#endif //DG_BENCHMARK
        m_ex.extrapolate( phi);

        unsigned number;
        if( multiplyWeights_ )
        {
            dg::blas2::symv( weights, rho, m_rhs);
            number = cg( op, phi, m_rhs, p, inv_weights, eps_, nrmb_correction_);
        }
        else
            number = cg( op, phi, rho, p, inv_weights, eps_, nrmb_correction_);

        m_ex.update(phi);
#ifdef DG_BENCHMARK
        t.toc();
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif //MPI
        {
            std::cout << "# of cg iterations \t"<< number << "\t";
            std::cout << "# took \t"<<t.diff()<<"s\n";
        }
#endif //DG_BENCHMARK
        return number;
    }

  private:
    value_type eps_, nrmb_correction_;
    dg::CG< ContainerType > cg;
    Extrapolation<ContainerType> m_ex;
    ContainerType m_rhs;
    bool multiplyWeights_;
};

} //namespace dg



#endif //_DG_CG_
