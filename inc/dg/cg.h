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

//// TO DO: check for better stopping criteria using condition number estimates?

/**
* @brief Functor class for the preconditioned conjugate gradient method to solve
* \f[ Ax=b\f]
*
 @ingroup invert
 @copydoc hide_container

 @note Conjugate gradients might become unstable for positive semidefinite
 matrices arising e.g. in the discretization of the periodic laplacian
*/
template< class container>
class CG
{
  public:
    typedef typename VectorTraits<container>::value_type value_type;//!< value type of the container class
    ///@brief Allocate nothing, 
    CG(){}
      /**
       * @brief Reserve memory for the pcg method
       *
       * @param copyable A container must be copy-constructible from this
       * @param max_iter Maximum number of iterations to be used
       */
    CG( const container& copyable, unsigned max_iter):r(copyable), p(r), ap(r), max_iter(max_iter){}
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
    void construct( const container& copyable, unsigned max_iterations) { 
        ap = p = r = copyable;
        max_iter = max_iterations;
    }
    /**
     * @brief Solve the system A*x = b using a preconditioned conjugate gradient method
     *
     * The iteration stops if \f$ ||b - Ax|| < \epsilon( ||b|| + C) \f$ where \f$C\f$ is 
     * a correction factor to the absolute error
     * @copydoc hide_matrix
     * @tparam Preconditioner A class for which the blas2::symv() and 
     blas2::dot( const Matrix&, const Vector&) functions are callable. Currently Preconditioner must be the same as container (diagonal preconditioner) except when container is std::vector<container_type> then Preconditioner can be container_type
     * @param A A symmetric positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param eps The relative error to be respected
     * @param nrmb_correction Correction factor C for norm of b
     * @attention This version uses the Preconditioner to compute the norm for the error condition (this safes one scalar product)
     *
     * @return Number of iterations used to achieve desired precision
     */
    template< class Matrix, class Preconditioner >
    unsigned operator()( Matrix& A, container& x, const container& b, Preconditioner& P , value_type eps = 1e-12, value_type nrmb_correction = 1);
    //version of CG where Preconditioner is not trivial
    /**
     * @brief Solve the system A*x = b using a preconditioned conjugate gradient method
     *
     * The iteration stops if \f$ ||Ax||_S < \epsilon( ||b||_S + C) \f$ where \f$C\f$ is 
     * a correction factor to the absolute error and \f$ S \f$ defines a square norm
     * @copydoc hide_matrix
     * @tparam Preconditioner A type for which the blas2::symv(Matrix&, Vector1&, Vector2&) function is callable. 
     * @tparam SquareNorm A type for which the blas2::dot( const Matrix&, const Vector&) function is callable. This can e.g. be one of the container types.
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
    unsigned operator()( Matrix& A, container& x, const container& b, Preconditioner& P, SquareNorm& S, value_type eps = 1e-12, value_type nrmb_correction = 1);
  private:
    container r, p, ap; 
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
template< class container>
template< class Matrix, class Preconditioner>
unsigned CG< container>::operator()( Matrix& A, container& x, const container& b, Preconditioner& P, value_type eps, value_type nrmb_correction)
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

template< class container>
template< class Matrix, class Preconditioner, class SquareNorm>
unsigned CG< container>::operator()( Matrix& A, container& x, const container& b, Preconditioner& P, SquareNorm& S, value_type eps, value_type nrmb_correction)
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
///@endcond


/**
* @brief Class that stores up to three solutions of iterative methods and
can be used to get initial guesses based on past solutions

 \f[ x_{init} = \alpha_0 x_0 + \alpha_{-1}x_{-1} + \alpha_{-2} x_{-2}\f]
 where the indices indicate the current (0) and past (negative) solutions.
*
* @copydoc hide_container
* @ingroup misc
*/
template<class container>
struct Extrapolation
{
    /*! @brief Set extrapolation type without initializing values
     * @param type number of vectors to use for extrapolation ( 0<=type<=3)
     * @attention the update function must be used at least type times before the extrapolate function can be called
     */
    Extrapolation( unsigned type = 2){ set_type(type); }
    /*! @brief Set extrapolation type and initialize values
     * @param type number of vectors to use for extrapolation ( 0<=type<=3)
     * @param init the vectors are initialized with this value
     */
    Extrapolation( unsigned type, const container& init) { 
        set_type(type, init); 
    }
    ///@copydoc Extrapolation(unsigned)
    void set_type( unsigned type)
    {
        m_type = type;
        m_x.resize( type);
        assert( m_type <= 3 );
    }
    ///@copydoc Extrapolation(unsigned,const container&)
    void set_type( unsigned type, const container& init)
    {
        m_x.assign( type, init);
        m_type = type;
        assert( m_type <= 3 );
    }
    ///read the current extrapolation type
    unsigned get_type( ) const{return m_type;}

    /**
    * @brief Extrapolate values 
    *
    * @param new_x (write only) contains extrapolated value on output ( may alias the tail)
    */
    void extrapolate( container& new_x) const{
        switch(m_type)
        {
            case(0): 
                     break;
            case(1): dg::blas1::copy( m_x[0], new_x);
                     break;
            case(2): dg::blas1::axpby( 2., m_x[0], -1., m_x[1], new_x);
                     break;
            case(3): dg::blas1::axpby( 1., m_x[2], -3., m_x[1], new_x);
                     dg::blas1::axpby( 3., m_x[0], 1., new_x);
                     break;
            default: dg::blas1::axpby( 2., m_x[0], -1., m_x[1], new_x);
        }
    }

    
    /**
    * @brief move the all values one step back and copy the given vector as current head
    *
    * @param new_head the new head ( may alias the tail)
    */
    void update( const container& new_head){
        if( m_type == 0) return;
        //push out last value
        for (unsigned u=m_type-1; u>0; u--)
            m_x[u].swap( m_x[u-1]);
        blas1::copy( new_head, m_x[0]);
    }

    /**
     * @brief return the current head 
     * @return current head (undefined if m_type==0)
     */
    const container& head()const{return m_x[0];}
    ///write access to tail value ( the one that will be deleted in the next update
    container& tail(){return m_x[m_type-1];}
    ///read access to tail value ( the one that will be deleted in the next update
    const container& tail()const{return m_x[m_type-1];}

    private:
    unsigned m_type;
    std::vector<container> m_x;
};


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
 * @copydoc hide_container
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

    ///@brief Allocate nothing
    Invert() { multiplyWeights_ = true; nrmb_correction_ = 1.; }

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
        m_ex.set_type( extrapolationType);
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
    void set_size( const container& assignable, unsigned max_iterations) {
        cg.construct(assignable, max_iterations);
        m_ex.set_type( m_ex.get_type(), assignable);
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
    void set_extrapolationType( int extrapolationType) {
        m_ex.set_type( extrapolationType);
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

    /// @brief Return last solution
    const container& get_last() const { return m_ex.head();}

    /**
     * @brief Solve linear problem
     *
     * Solves the Equation \f[ \hat O \phi = W\rho \f] using a preconditioned 
     * conjugate gradient method. The initial guess comes from an extrapolation 
     * of the last solutions
     * @copydoc hide_symmetric_op
     * @param op selfmade symmetric Matrix operator class
     * @param phi solution (write only)
     * @param rho right-hand-side
     * @note computes inverse weights from the weights 
     * @note If the Macro DG_BENCHMARK is defined this function will write timings to std::cout
     *
     * @return number of iterations used 
     */
    template< class SymmetricOp >
    unsigned operator()( SymmetricOp& op, container& phi, const container& rho)
    {
        return this->operator()(op, phi, rho, op.weights(), op.inv_weights(), op.precond());
    }

    /**
     * @brief Solve linear problem
     *
     * Solves the Equation \f[ \hat O \phi = W\rho \f] using a preconditioned 
     * conjugate gradient method. The initial guess comes from an extrapolation 
     * of the last solutions.
     * @copydoc hide_matrix
     * @tparam Preconditioner A type for which the blas2::symv(Matrix&, Vector1&, Vector2&) function is callable. 
     * @param op symmetric Matrix operator class
     * @param phi solution (write only)
     * @param rho right-hand-side
     * @param weights The weights that normalize the symmetric operator
     * @param inv_weights The inverse of the weights that normalize the symmetric operator
     * @param p The preconditioner  
     * @note If the Macro DG_BENCHMARK is defined this function will write timings to std::cout
     *
     * @return number of iterations used 
     */
    template< class Matrix, class Preconditioner >
    unsigned operator()( Matrix& op, container& phi, const container& rho, const container& weights, const container& inv_weights, Preconditioner& p)
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
            dg::blas2::symv( rho, weights, m_ex.tail());
            number = cg( op, phi, m_ex.tail(), p, inv_weights, eps_, nrmb_correction_);
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
            std::cout<< "took \t"<<t.diff()<<"s\n";
        }
#endif //DG_BENCHMARK
        return number;
    }

  private:
    value_type eps_, nrmb_correction_;
    dg::CG< container > cg;
    Extrapolation<container> m_ex;
    bool multiplyWeights_; 
};

} //namespace dg



#endif //_DG_CG_
