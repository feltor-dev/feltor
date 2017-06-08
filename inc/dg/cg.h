#ifndef _DG_CG_
#define _DG_CG_

#include <cmath>

#include "blas.h"
#include "functors.h"

#ifdef DG_BENCHMARK
#include "backend/timer.cuh"
#endif //DG_BENCHMARK

/*!@file
 * Conjugate gradient class and functions
 */

namespace dg{

//// TO DO: check for better stopping criteria using condition number estimates

/**
* @brief Functor class for the preconditioned conjugate gradient method to solve
* \f[ Ax=b\f]
*
 @ingroup invert
 @tparam Vector The Vector class: needs to model Assignable 

 The following 3 pseudo - BLAS routines need to be callable 
 \li value_type dot = dg::blas1::dot( const Vector&, const Vector&); 
 \li dg::blas1::axpby();  with the Vector type
 \li dg::blas2::symv(Matrix& m, Vector1& x, Vector2& y ); with the Matrix type
 \li value_type dot = dg::blas2::dot( );  with the Preconditioner type
 \li dg::blas2::symv( ); with the Preconditioner type

 @note Conjugate gradients might become unstable for positive semidefinite
 matrices arising e.g. in the discretization of the periodic laplacian
*/
template< class Vector>
class CG
{
  public:
    typedef typename VectorTraits<Vector>::value_type value_type;//!< value type of the Vector class
    /**
     * @brief Allocate nothing, 
     */
    CG(){}
      /**
       * @brief Reserve memory for the pcg method
       *
       * @param copyable A Vector must be copy-constructible from this
       * @param max_iter Maximum number of iterations to be used
       */
    CG( const Vector& copyable, unsigned max_iter):r(copyable), p(r), ap(r), max_iter(max_iter){}
    /**
     * @brief Set the maximum number of iterations 
     *
     * @param new_max New maximum number
     */
    void set_max( unsigned new_max) {max_iter = new_max;}
    /**
     * @brief Get the current maximum number of iterations
     *
     * @return the current maximum
     */
    unsigned get_max() const {return max_iter;}

    /**
     * @brief Set internal storage and maximum number of iterations
     *
     * @param copyable
     * @param max_iterations
     */
    void construct( const Vector& copyable, unsigned max_iterations) { 
        ap = p = r = copyable;
        max_iter = max_iterations;
    }
    /**
     * @brief Solve the system A*x = b using a preconditioned conjugate gradient method
     *
     * The iteration stops if \f$ ||Ax|| < \epsilon( ||b|| + C) \f$ where \f$C\f$ is 
     * a correction factor to the absolute error
     @tparam Matrix The matrix class: no requirements except for the 
            BLAS routines
     @tparam Preconditioner no requirements except for the blas routines. Thus far the dg library
        provides only diagonal preconditioners, which should be enough if the result is extrapolated from
        previous timesteps.
     * In every iteration the following BLAS functions are called: \n
       symv 1x, dot 1x, axpby 2x, Prec. dot 1x, Prec. symv 1x
     * @param A A symmetric positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param eps The relative error to be respected
     * @param nrmb_correction Correction factor C for norm of b
     * @attention This versions uses the Preconditioner to compute the norm for the error condition (this safes one scalar product)
     *
     * @return Number of iterations used to achieve desired precision
     */
    template< class Matrix, class Preconditioner >
    unsigned operator()( Matrix& A, Vector& x, const Vector& b, Preconditioner& P , value_type eps = 1e-12, value_type nrmb_correction = 1);
    //version of CG where Preconditioner is not trivial
    /**
     * @brief Solve the system A*x = b using a preconditioned conjugate gradient method
     *
     * The iteration stops if \f$ ||Ax||_S < \epsilon( ||b||_S + C) \f$ where \f$C\f$ is 
     * a correction factor to the absolute error and \f$ S \f$ defines a square norm
     @tparam Matrix The matrix class: no requirements except for the 
            BLAS routines
     @tparam Preconditioner no requirements except for the blas routines. Thus far the dg library
        provides only diagonal preconditioners, which should be enough if the result is extrapolated from
        previous timesteps.
     @tparam SquareNorm  (usually is the same as the container class)

     * In every iteration the following BLAS functions are called: \n
       symv 1x, dot 1x, axpby 2x, Prec. dot 1x, Prec. symv 1x
     * @param A A symmetric positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param S Weights used to compute the norm for the error condition
     * @param eps The relative error to be respected
     * @param nrmb_correction Correction factor C for norm of b
     *
     * @return Number of iterations used to achieve desired precision
     */
    template< class Matrix, class Preconditioner, class SquareNorm >
    unsigned operator()( Matrix& A, Vector& x, const Vector& b, Preconditioner& P, SquareNorm& S, value_type eps = 1e-12, value_type nrmb_correction = 1);
  private:
    Vector r, p, ap; 
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
template< class Vector>
template< class Matrix, class Preconditioner>
unsigned CG< Vector>::operator()( Matrix& A, Vector& x, const Vector& b, Preconditioner& P, value_type eps, value_type nrmb_correction)
{
    value_type nrmb = sqrt( blas2::dot( P, b));
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif //MPI
    {
    std::cout << "Norm of b "<<nrmb <<"\n";
    std::cout << "Residual errors: \n";
    }
#endif //DG_DEBUG
    if( nrmb == 0)
    {
        blas1::axpby( 1., b, 0., x);
        return 0;
    }
    blas2::symv( A,x,r);
    blas1::axpby( 1., b, -1., r);
    blas2::symv( P, r, p );//<-- compute p_0
    //note that dot does automatically synchronize
    value_type nrm2r_old = blas2::dot( P,r); //and store the norm of it
    if( sqrt( nrm2r_old ) < eps*(nrmb + nrmb_correction)) //if x happens to be the solution
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
        std::cout << "Absolute "<<sqrt( nrm2r_new) <<"\t ";
        std::cout << " < Critical "<<eps*nrmb + eps <<"\t ";
        std::cout << "(Relative "<<sqrt( nrm2r_new)/nrmb << ")\n";
    }
#endif //DG_DEBUG
        if( sqrt( nrm2r_new) < eps*(nrmb + nrmb_correction)) 
            return i;
        blas2::symv(1.,P, r, nrm2r_new/nrm2r_old, p );
        nrm2r_old=nrm2r_new;

    }
    return max_iter;
}

template< class Vector>
template< class Matrix, class Preconditioner, class SquareNorm>
unsigned CG< Vector>::operator()( Matrix& A, Vector& x, const Vector& b, Preconditioner& P, SquareNorm& S, value_type eps, value_type nrmb_correction)
{
    value_type nrmb = sqrt( blas2::dot( S, b));
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif //MPI
    {
    std::cout << "Norm of S b "<<nrmb <<"\n";
    std::cout << "Residual errors: \n";
    }
#endif //DG_DEBUG
    if( nrmb == 0)
    {
        blas1::copy( b, x);
        return 0;
    }
    blas2::symv( A,x,r);
    blas1::axpby( 1., b, -1., r);
    //note that dot does automatically synchronize
    if( sqrt( blas2::dot(S,r) ) < eps*(nrmb + nrmb_correction)) //if x happens to be the solution
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
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    if(rank==0)
#endif //MPI
    {
        std::cout << "Absolute r*S*r "<<sqrt( blas2::dot(S,r)) <<"\t ";
        std::cout << " < Critical "<<eps*nrmb + eps <<"\t ";
        std::cout << "(Relative "<<sqrt( blas2::dot(S,r) )/nrmb << ")\n";
    }
#endif //DG_DEBUG
        if( sqrt( blas2::dot(S,r)) < eps*(nrmb + nrmb_correction)) 
            return i;
        blas2::symv(P,r,ap);
        nrmzr_new = blas1::dot( ap, r); 
        blas1::axpby(1.,ap, nrmzr_new/nrmzr_old, p );
        nrmzr_old=nrmzr_new;
    }
    return max_iter;
}


/**
 * @brief Smart conjugate gradient solver. 
 
 * Solve a symmetric linear inversion problem using a conjugate gradient method and 
 * the last two solutions.
 *
 * @ingroup invert
 * Solves the Equation \f[ \hat O \phi = W \cdot \rho \f]
 * for any operator \f$\hat O\f$ that was made symmetric 
 * by appropriate weights \f$W\f$ (s. comment below). 
 * It uses solutions from the last two calls to 
 * extrapolate a solution for the current call.
 * @tparam container The Vector class to be used
 * @note A note on weights, inverse weights and preconditioning. 
 * A normalized DG-discretized derivative or operator is normally not symmetric. 
 * The diagonal coefficient matrix that is used to make the operator 
 * symmetric is called weights W, i.e. \f$ \hat O = W\cdot O\f$ is symmetric. 
 * Now, to compute the correct scalar product of the right hand side the
 * inverse weights have to be used i.e. \f$ W\rho\cdot W \rho /W\f$.
 * Independent from this, a preconditioner should be used to solve the
 * symmetric matrix equation. The inverse of \f$W\f$ is 
 * a good general purpose preconditioner. 
 */
template<class container>
struct Invert
{
    typedef typename VectorTraits<container>::value_type value_type;

    /**
     * @brief Allocate nothing
     *
     */
    Invert() { multiplyWeights_ = true; set_extrapolationType(2); nrmb_correction_ = 1.; }

    /**
     * @brief Constructor
     *
     * @param copyable Needed to construct the two previous solutions
     * @param max_iter maximum iteration in conjugate gradient
     * @param eps relative error in conjugate gradient
     * @param extrapolationType number of last values to use for extrapolation of the current guess
     * @param multiplyWeights if true the rhs shall be multiplied by the weights before cg is applied
     * @param nrmb_correction Correction factor for norm of b (cf. CG)
     */
    Invert(const container& copyable, unsigned max_iter, value_type eps, int extrapolationType = 2, bool multiplyWeights = true, value_type nrmb_correction = 1)
    {
        construct( copyable, max_iter, eps, extrapolationType, multiplyWeights, nrmb_correction);
    }

    /**
     * @brief to be called after default constructor
     *
     * @param copyable Needed to construct the two previous solutions
     * @param max_iter maximum iteration in conjugate gradient
     * @param eps relative error in conjugate gradient
     * @param extrapolationType number of last values to use for extrapolation of the current guess
     * @param multiplyWeights if true the rhs shall be multiplied by the weights before cg is applied
     * @param nrmb_correction Correction factor for norm of b (cf. CG)
     */
    void construct( const container& copyable, unsigned max_iter, value_type eps, int extrapolationType = 2, bool multiplyWeights = true, value_type nrmb_correction = 1.) 
    {
        set_size( copyable, max_iter);
        set_accuracy( eps, nrmb_correction);
        multiplyWeights_=multiplyWeights;
        set_extrapolationType( extrapolationType);
    }


    /**
     * @brief Set vector storage and maximum number of iterations
     *
     * @param assignable
     * @param max_iterations
     */
    void set_size( const container& assignable, unsigned max_iterations) {
        cg.construct(assignable, max_iterations);
        phi0 = phi1 = phi2 = assignable;
    }

    /**
     * @brief Set accuracy parameters for following inversions
     *
     * @param eps
     * @param nrmb_correction
     */
    void set_accuracy( value_type eps, value_type nrmb_correction = 1.) { 
        eps_ = eps; 
        nrmb_correction_ = nrmb_correction;
    }

    /**
     * @brief Set the extrapolation Type for following inversions
     *
     * @param extrapolationType number of last values to use for next extrapolation of initial guess
     */
    void set_extrapolationType( int extrapolationType)
    {
        assert( extrapolationType <= 3 && extrapolationType >= 0);
        switch(extrapolationType)
        {
            case(0): alpha[0] = 0, alpha[1] = 0, alpha[2] = 0;
                     break;
            case(1): alpha[0] = 1, alpha[1] = 0, alpha[2] = 0;
                     break;
            case(2): alpha[0] = 2, alpha[1] = -1, alpha[2] = 0;
                     break;
            case(3): alpha[0] = 3, alpha[1] = -3, alpha[2] = 1;
                     break;
            default: alpha[0] = 2, alpha[1] = -1, alpha[2] = 0;
        }
    }
    /**
     * @brief Set the maximum number of iterations 
     *
     * @param new_max New maximum number
     */
    void set_max( unsigned new_max) {cg.set_max( new_max);}
    /**
     * @brief Get the current maximum number of iterations
     *
     * @return the current maximum
     */
    unsigned get_max() const {return cg.get_max();}

    /**
    * @brief Return last solution
    */
    const container& get_last() const { return phi0;}

    /**
     * @brief Solve linear problem
     *
     * Solves the Equation \f[ \hat O \phi = W\rho \f] using a preconditioned 
     * conjugate gradient method. The initial guess comes from an extrapolation 
     * of the last solutions
     * @tparam SymmetricOp Symmetric operator with the SelfMadeMatrixTag
        The functions weights() and precond() need to be callable and return
        weights and the preconditioner for the conjugate gradient method.
        The Operator is assumed to be symmetric!
     * @param op selfmade symmetric Matrix operator class
     * @param phi solution (write only)
     * @param rho right-hand-side
     * @note computes inverse weights from the weights 
     *
     * @return number of iterations used 
     */
    template< class SymmetricOp >
    unsigned operator()( SymmetricOp& op, container& phi, const container& rho)
    {
        container inv_weights( op.weights());
        dg::blas1::transform( inv_weights, inv_weights, dg::INVERT<double>());
        return this->operator()(op, phi, rho, op.weights(), op.precond(), inv_weights);
    }

    /**
     * @brief Solve linear problem
     *
     * Solves the Equation \f[ \hat O \phi = W\rho \f] using a preconditioned 
     * conjugate gradient method. The initial guess comes from an extrapolation 
     * of the last solutions.
     * @tparam SymmetricOp Symmetric matrix or operator (with the selfmade tag)
     * @tparam Preconditioner class of the Preconditioner
     * @param op selfmade symmetric Matrix operator class
     * @param phi solution (write only)
     * @param rho right-hand-side
     * @param w The weights that made the operator symmetric
     * @param p The preconditioner  
     * @note computes inverse weights from the weights 
     *
     * @return number of iterations used 
     */
    template< class SymmetricOp, class Preconditioner >
    unsigned operator()( SymmetricOp& op, container& phi, const container& rho, const container& w, Preconditioner& p)
    {
        container inv_weights( w);
        dg::blas1::transform( inv_weights, inv_weights, dg::INVERT<double>());
        return this->operator()(op, phi, rho, w, p, inv_weights);
    }

    /**
     * @brief Solve linear problem
     *
     * Solves the Equation \f[ \hat O \phi = W\rho \f] using a preconditioned 
     * conjugate gradient method. The initial guess comes from an extrapolation 
     * of the last solutions.
     * @tparam SymmetricOp Symmetric matrix or operator (with the selfmade tag)
     * @tparam Preconditioner class of the Preconditioner
     * @param op selfmade symmetric Matrix operator class
     * @param phi solution (write only)
     * @param rho right-hand-side
     * @param w The weights that made the operator symmetric
     * @param p The preconditioner  
     * @param inv_weights The inverse weights used to compute the scalar product in the CG solver
     * @note Calls the most general form of the CG solver with SquareNorm being the container class
     *
     * @return number of iterations used 
     */
    template< class SymmetricOp, class Preconditioner >
    unsigned operator()( SymmetricOp& op, container& phi, const container& rho, const container& w, Preconditioner& p, const container& inv_weights)
    {
        assert( phi0.size() != 0);
        assert( &rho != &phi);
        blas1::axpby( alpha[0], phi0, alpha[1], phi1, phi); // 1. phi0 + 0.*phi1 = phi
        blas1::axpby( alpha[2], phi2, 1., phi); // 0. phi2 + 1. phi0 + 0.*phi1 = phi

        unsigned number;
#ifdef DG_BENCHMARK
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        Timer t;
        t.tic();
#endif //DG_BENCHMARK
        if( multiplyWeights_ ) 
        {
            dg::blas2::symv( w, rho, phi2);
            number = cg( op, phi, phi2, p, inv_weights, eps_, nrmb_correction_);
        }
        else
            number = cg( op, phi, rho, p, inv_weights, eps_, nrmb_correction_);
#ifdef DG_BENCHMARK
#ifdef MPI_VERSION
        if(rank==0)
#endif //MPI
        std::cout << "# of cg iterations \t"<< number << "\t";
        t.toc();
#ifdef MPI_VERSION
        if(rank==0)
#endif //MPI
        std::cout<< "took \t"<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
        phi1.swap( phi2);
        phi0.swap( phi1);
        
        blas1::axpby( 1., phi, 0, phi0);
        return number;
    }

  private:
    value_type eps_, nrmb_correction_;
    container phi0, phi1, phi2;
    dg::CG< container > cg;
    value_type alpha[3];
    bool multiplyWeights_; 
};

/**
 * @brief This struct holds a matrix and applies its inverse to vectors 
 *
 * The inverse is computed with a conjugate gradient method
 * @ingroup invert
 * @tparam SymmetricOp A symmetric Matrix type
 * @tparam container The container type to use
 */
template<class SymmetricOp, class container>
struct Inverse
{
    typedef typename VectorTraits<container>::value_type value_type;
    Inverse( SymmetricOp& op, container& copyable, unsigned max_iter, value_type eps, int extrapolationType=0): 
        x_(copyable), b_(copyable), op_( op), invert_( copyable, max_iter, eps, extrapolationType, false, 1.){}
    /**
     * @brief Computes Op^{-1} b = x
     *
     * @param b
     * @param x
     */
    template<class OtherContainer>
    void symv( const OtherContainer& b, OtherContainer& x)
    {
        //std::cout << "Number in inverse "<<invert( op, x, b, op.weights(), op.precond())<<std::endl;
        dg::blas1::transfer(b,b_);
        invert_( op_, x_, b_); 
        dg::blas1::transfer(x_,x);
    }
    private:
    container x_,b_;
    SymmetricOp op_;
    Invert<container> invert_;
};

///@cond
template< class M, class V>
struct MatrixTraits< Inverse< M, V > >
{
    typedef typename Inverse<M, V>::value_type value_type;
    typedef SelfMadeMatrixTag matrix_category;
};

///@endcond


/**
 * @brief Function version of CG class
 *
 * @ingroup invert 
 * @tparam Matrix Matrix type
 * @tparam Vector Vector type (must be constructible from given size)
 * @tparam Preconditioner Preconditioner type
 * @param A Matrix 
 * @param x contains initial guess on input and solution on output
 * @param b right hand side
 * @param P Preconditioner
 * @param eps relative error
 * @param max_iter maximum iterations allowed
 *
 * @return number of iterations
 */
/*
template< class Matrix, class Vector, class Preconditioner>
unsigned cg( Matrix& A, Vector& x, const Vector& b, const Preconditioner& P, typename VectorTraits<Vector>::value_type eps, unsigned max_iter)
{
    typedef typename VectorTraits<Vector>::value_type value_type;
    value_type nrmb = sqrt( blas2::dot( P, b)); //norm of b
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif //MPI
    {
    std::cout << "Norm of b "<<nrmb <<"\n";
    std::cout << "Residual errors: \n";
    }
#endif //DG_DEBUG
    if( nrmb == 0)
    {
        blas1::axpby( 1., b, 0., x); //x=b
        return 0;
    }
    Vector r(x.size()), p(x.size()), ap(x.size()); //1% time at 20 iterations
    //r = b; blas2::symv( -1., A, x, 1.,r); //compute r_0 
    blas2::symv( A,x,r); //r=A x
    blas1::axpby( 1., b, -1., r); //r=b-Ax
    blas2::symv( P, r, p );//<-- compute p_0  //p=P(b-Ax)
    //note that dot does automatically synchronize
    value_type nrm2r_old = blas2::dot( P,r); //and store the norm of it // norm of r^2 = ||r||^2 = r^T P r 
    if( sqrt( nrm2r_old ) < eps*nrmb + eps)
        return 0;
    value_type alpha, nrm2r_new;
    for( unsigned i=1; i<max_iter; i++)
    {
        blas2::symv( A, p, ap); // ap = A ( P(b-Ax))
        alpha = nrm2r_old /blas1::dot( p, ap); // alpha = ||r||^2 / ( ((b-Ax)^T P^T) A (P (b-Ax))  )
        blas1::axpby( alpha, p, 1.,x);
        //here one could add a ifstatement to remove accumulated floating point error
        //(if i modulo sqrt(n)) r=b-Ax else ...
        blas1::axpby( -alpha, ap, 1., r); // r = r-alpha*A ( P(b-Ax))
        nrm2r_new = blas2::dot( P, r);  //||r_new||^2 =  r^T P r 
#ifdef DG_DEBUG
#ifdef MPI_VERSION
        if(rank==0)
#endif //MPI
        {
        std::cout << "Absolute "<<sqrt( nrm2r_new) <<"\t ";
        std::cout << " < Critical "<<eps*nrmb + eps <<"\t ";
        std::cout << "(Relative "<<sqrt( nrm2r_new)/nrmb << ")\n";
        }
#endif //DG_DEBUG
        if( sqrt( nrm2r_new) < eps*nrmb + eps) 
            return i;
        blas2::symv(1.,P, r, nrm2r_new/nrm2r_old, p ); //p= 1*P*r + ||r_new||^2 /||r_old||^2 *p
        nrm2r_old=nrm2r_new;
    }
    return max_iter;
}
*/

} //namespace dg



#endif //_DG_CG_
