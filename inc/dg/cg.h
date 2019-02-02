#ifndef _DG_CG_
#define _DG_CG_

#include <cmath>

#include "blas.h"
#include "functors.h"

#ifdef DG_BENCHMARK
#include "backend/timer.h"
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
* @ingroup invert
*
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
    ///@copydoc construct()
    CG( const ContainerType& copyable, unsigned max_iterations):r(copyable), p(r), ap(r), max_iter(max_iterations){}
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
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    void construct( const ContainerType& copyable, unsigned max_iterations) {
        ap = p = r = copyable;
        max_iter = max_iterations;
    }
    /**
     * @brief Solve the system A*x = b using a preconditioned conjugate gradient method
     *
     * The iteration stops if \f$ ||b - Ax|| < \epsilon( ||b|| + C) \f$ where \f$C\f$ is
     * a correction factor to the absolute error
     * @param A A symmetric, positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param eps The relative error to be respected
     * @param nrmb_correction Correction factor C for norm of b
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
    unsigned operator()( MatrixType& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P , value_type eps = 1e-12, value_type nrmb_correction = 1);
    //version of CG where Preconditioner is not trivial
    /**
     * @brief Solve \f$ Ax = b\f$ using a preconditioned conjugate gradient method
     *
     * The iteration stops if \f$ ||Ax||_S < \epsilon( ||b||_S + C) \f$ where \f$C\f$ is
     * a correction factor to the absolute error and \f$ S \f$ defines a square norm
     * @param A A symmetric positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param S Weights used to compute the norm for the error condition
     * @param eps The relative error to be respected
     * @param nrmb_correction Correction factor C for norm of b
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
    unsigned operator()( MatrixType& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps = 1e-12, value_type nrmb_correction = 1);
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
unsigned CG< ContainerType>::operator()( Matrix& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, value_type eps, value_type nrmb_correction)
{
    value_type nrmb = sqrt( blas2::dot( P, b));
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
        std::cout << "# Absolute "<<sqrt( nrm2r_new) <<"\t ";
        std::cout << "#  < Critical "<<eps*nrmb + eps <<"\t ";
        std::cout << "# (Relative "<<sqrt( nrm2r_new)/nrmb << ")\n";
    }
#endif //DG_DEBUG
        if( sqrt( nrm2r_new) < eps*(nrmb + nrmb_correction))
            return i;
        blas2::symv(1.,P, r, nrm2r_new/nrm2r_old, p );
        nrm2r_old=nrm2r_new;

    }
    return max_iter;
}

template< class ContainerType>
template< class Matrix, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm>
unsigned CG< ContainerType>::operator()( Matrix& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps, value_type nrmb_correction)
{
    value_type nrmb = sqrt( blas2::dot( S, b));
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
        std::cout << "# Absolute r*S*r "<<sqrt( blas2::dot(S,r)) <<"\t ";
        std::cout << "#  < Critical "<<eps*nrmb + eps <<"\t ";
        std::cout << "# (Relative "<<sqrt( blas2::dot(S,r) )/nrmb << ")\n";
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
//
//implement the classical 3-term recursion with explicit residual
template< class ContainerType>
class ChebyshevIteration
{
  public:
    using container_type = ContainerType;
    using value_type = get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    ChebyshevIteration(){}
    ///@copydoc construct()
    ChebyshevIteration( const ContainerType& copyable):
        m_r(copyable), m_x1(m_r){}
    ///@brief Return an object of same size as the object used for construction
    ///@return A copyable object; what it contains is undefined, its size is important
    const ContainerType& copyable()const{ return m_r;}

    /**
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    void construct( const ContainerType& copyable) {
        m_x1 = m_r = copyable;
    }
    /**
     * @brief Solve the system A*x = b using Chebyshev iteration
     *
     * The iteration stops when the maximum number of iterations is reached
     * @param A A symmetric, positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param lmin the minimum Eigenvalue
     * @param lmax the minimum Eigenvalue
     * @param num_iter the number of iterations
     *
     * @copydoc hide_matrix
     * @tparam ContainerTypes must be usable with \c MatrixType and \c ContainerType in \ref dispatch
     */
    template< class MatrixType, class ContainerType0, class ContainerType1>
    void solve( MatrixType& A, ContainerType0& x, const ContainerType1& b, double lmin, double lmax, unsigned num_iter)
    {
        assert ( lmin < lmax);
        double a = (lmin+lmax)/2., c = (lmin-lmax)/2.;
        double betan = -c*c/2./a;
        double gamman = -a;
        dg::blas1::copy( x, m_x1);
        dg::blas2::symv( A, x, m_r);
        dg::blas1::axpby( 1., b, -1., m_r);
        //-(r0+a*x0)/gamma0, m_x1=0
        dg::blas1::axpby( -1./gamman, m_r, -a/gamman, x);
        dg::blas1::copy( 0., m_x1);
        for ( unsigned u=1; u<num_iter; u++)
        {
            betan = (c/2.)*(c/2.)/gamman;
            gamman = -(a+betan);

            dg::blas2::symv( A, x, m_r);
            dg::blas1::axpby( 1., b, -1., m_r);
            dg::blas1::evaluate( m_x1, dg::equals(), PairSum(),
                            -1./gamman, m_r,
                            -a/gamman, x,
                            -betan/gamman,m_x1);
            x.swap(m_x1);
        }
    }
    //version of CG where Preconditioner is not trivial
  private:
    ContainerType m_r, m_x1;
};


/**
* @brief Extrapolate based on up to three past solutions
*
* The intention of this class is to provide an initial guess for iterative solvers
* based on past solutions:
 \f[ x_{init} = \alpha_0 x_0 + \alpha_{-1}x_{-1} + \alpha_{-2} x_{-2}\f]
 where the indices indicate the current (0) and past (negative) solutions.
 Choose between 1 (constant), 2 (linear) or 3 (parabola) extrapolation.
 The user can choose to provide a time value \c t_i associated with the \c x_i, which
 are then used to compute the coefficients \c alpha_i (using Lagrange interpolation).
 Otherwise an equidistant distribution is assumed.
*
* @note Since extrapolation with higher order polynomials is so prone to oscillations
* anything higher than linear rarely leads to anything useful. So best stick to
* constant or linear extrapolation
* @copydoc hide_ContainerType
* @ingroup invert
* @sa https://en.wikipedia.org/wiki/Extrapolation
*/
template<class ContainerType>
struct Extrapolation
{
    using value_type = get_value_type<ContainerType>;
    using container_type = ContainerType;
    /*! @brief Set extrapolation order without initializing values
     * @param number number of vectors to use for extrapolation.
         Choose between 1 (constant), 2 (linear) or 3 (parabola) extrapolation.
     * @attention the update function must be used at least \c number times before the extrapolate function can be called
     */
    Extrapolation( unsigned number = 2){
        set_number(number);
    }
    /*! @brief Set extrapolation order and initialize values
     * @param number number of vectors to use for extrapolation.
         Choose between 1 (constant), 2 (linear) or 3 (parabola) extrapolation.
     * @param t_init the times are initialized with the values <tt> t_init, t_init-1 , t_init-2 </tt>
     * @param init the vectors are initialized with this value
     */
    Extrapolation( unsigned number, value_type t_init, const ContainerType& init) {
        set_number(number, t_init, init);
    }
    /*! @brief Set extrapolation order and initialize values (equidistant)
     * @param number number of vectors to use for extrapolation.
         Choose between 1 (constant), 2 (linear) or 3 (parabola) extrapolation.
     * @param init the vectors are initialized with this value
     * @note the times are initialized with the values <tt> 0, -1 , -2 </tt>
     */
    Extrapolation( unsigned number, const ContainerType& init) {
        set_number(number, 0, init);
    }
    ///@copydoc Extrapolation(unsigned)
    void set_number( unsigned number)
    {
        assert( number <= 3 );
        m_number = number;
        m_t.resize( number);
        m_x.resize( number);
        for(unsigned i=0; i<m_t.size(); i++)
            m_t[i] = -(value_type)i;
    }
    ///@copydoc Extrapolation(unsigned,value_type,const ContainerType&)
    void set_number( unsigned number, value_type t_init, const ContainerType& init)
    {
        //init times 0, -1, -2
        assert( number <= 3 );
        m_x.assign( number, init);
        m_t.assign( number, t_init);
        m_number = number;
        for(unsigned i=0; i<m_t.size(); i++)
            m_t[i] = t_init - (value_type)i;
    }
    ///@copydoc Extrapolation(unsigned,const ContainerType&)
    void set_number( unsigned number, const ContainerType& init)
    {
        //init times 0, -1, -2
        set_number( number, 1, init);
        for(unsigned i=0; i<m_t.size(); i++)
            m_t[i] = -(value_type)i;
    }
    ///return the current extrapolation number
    unsigned get_number( ) const{
        return m_number;
    }

    /**
    * @brief Extrapolate value to given time
    * @param t time to which to extrapolate
    * @param new_x (write only) contains extrapolated value on output ( may alias the tail)
    * @tparam ContainerType0 must be usable with \c ContainerType in \ref dispatch
    */
    template<class ContainerType0>
    void extrapolate( value_type t, ContainerType0& new_x) const{
        switch(m_number)
        {
            case(0):
                     break;
            case(1): dg::blas1::copy( m_x[0], new_x);
                     break;
            case(3): {
                value_type f0 = (t-m_t[1])*(t-m_t[2])/(m_t[0]-m_t[1])/(m_t[0]-m_t[2]);
                value_type f1 = (t-m_t[0])*(t-m_t[2])/(m_t[1]-m_t[0])/(m_t[1]-m_t[2]);
                value_type f2 = (t-m_t[0])*(t-m_t[1])/(m_t[2]-m_t[0])/(m_t[2]-m_t[1]);
                dg::blas1::evaluate( new_x, dg::equals(), dg::PairSum(),
                        f0, m_x[0], f1, m_x[1], f2, m_x[2]);
                 break;
            }
            default: {
                value_type f0 = (t-m_t[1])/(m_t[0]-m_t[1]);
                value_type f1 = (t-m_t[0])/(m_t[1]-m_t[0]);
                dg::blas1::axpby( f0, m_x[0], f1, m_x[1], new_x);
            }
        }
    }

    /**
    * @brief Extrapolate value
    * @param new_x (write only) contains extrapolated value on output ( may alias the tail)
    * @note Assumes that extrapolation time equals last inserted time+1
    * @tparam ContainerType0 must be usable with \c ContainerType in \ref dispatch
    */
    template<class ContainerType0>
    void extrapolate( ContainerType0& new_x) const{
        value_type t = m_t[0] +1.;
        extrapolate( t, new_x);
    }


    /**
    * @brief insert a new entry, deleting the oldest entry or update existing entry
    * @param t_new the time for the new entry
    * @param new_entry the new entry ( may alias the tail), replaces value of existing entry if \c t_new already exists
    * @tparam ContainerType0 must be usable with \c ContainerType in \ref dispatch
    */
    template<class ContainerType0>
    void update( value_type t_new, const ContainerType0& new_entry){
        if( m_number == 0) return;
        //check if entry is already there to avoid division by zero errors
        for( unsigned i=0; i<m_number; i++)
            if( fabs(t_new - m_t[i]) <1e-14)
            {
                blas1::copy( new_entry, m_x[i]);
                return;
            }
        //push out last value (keep track of what is oldest value
        for (unsigned u=m_number-1; u>0; u--)
        {
            std::swap( m_t[u], m_t[u-1]);
            m_x[u].swap( m_x[u-1]);
        }
        m_t[0] = t_new;
        blas1::copy( new_entry, m_x[0]);
    }
    /**
    * @brief insert a new entry
    * @param new_entry the new entry ( may alias the tail)
    * @note Assumes new time equals last inserted time+1
    * @tparam ContainerType0 must be usable with \c ContainerType in \ref dispatch
    */
    template<class ContainerType0>
    void update( const ContainerType0& new_entry){
        value_type t_new = m_t[0] + 1;
        update( t_new, new_entry);
    }

    /**
     * @brief return the current head (the one most recently inserted)
     * @return current head (undefined if number==0)
     */
    const ContainerType& head()const{
        return m_x[0];
    }
    ///write access to tail value ( the one that will be deleted in the next update
    ContainerType& tail(){
        return m_x[m_number-1];
    }
    ///read access to tail value ( the one that will be deleted in the next update
    const ContainerType& tail()const{
        return m_x[m_number-1];
    }

    private:
    unsigned m_number;
    std::vector<value_type> m_t;
    std::vector<ContainerType> m_x;
};


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
     * @param nrmb_correction Correction factor for norm of b (cf. CG)
     */
    void construct( const ContainerType& copyable, unsigned max_iter, value_type eps, int extrapolationType = 2, bool multiplyWeights = true, value_type nrmb_correction = 1.)
    {
        m_ex.set_number( extrapolationType);
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
        m_ex.set_number( m_ex.get_number(), assignable);
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
        m_ex.set_number( extrapolationType);
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
    const ContainerType& get_last() const { return m_ex.head();}

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
    unsigned operator()( SymmetricOp& op, ContainerType0& phi, const ContainerType1& rho)
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
    unsigned operator()( MatrixType& op, ContainerType0& phi, const ContainerType1& rho, const SquareNorm0& weights, const SquareNorm1& inv_weights, Preconditioner& p)
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
            dg::blas2::symv( weights, rho, m_ex.tail());
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
            std::cout << "# took \t"<<t.diff()<<"s\n";
        }
#endif //DG_BENCHMARK
        return number;
    }

  private:
    value_type eps_, nrmb_correction_;
    dg::CG< ContainerType > cg;
    Extrapolation<ContainerType> m_ex;
    bool multiplyWeights_;
};

} //namespace dg



#endif //_DG_CG_
