#pragma once

// #include <random>
#include "blas.h"
#include "topology/operator.h"

namespace dg
{

/**
 * @brief Compute \f$ a = (B^T W B)^{-1} WB^T b\f$ for given \f$ B\f$, weights \f$ W\f$ and right hand side \f$ b\f$
 *
 * This is the normal form of a least squares problem: given vectors \f$ b_i\f$
 * find coefficients \f$ a_i\f$ such that \f$ a_i b_i\f$ is as close as possible
 * to a target vector \f$ b\f$, i.e. \f$  \min_a || B a - b||_W \f$ where the
 * \f$ b_i\f$ constitute the columns of the matrix \f$ B\f$ and \f$ W\f$ are
 * weights.
 * This can be transformed into the solution of the *normal equation*
 * \f[ B^T W B a = B^T W b\f]
 * @note With a little trick this function can be used to compute the least squares
 * fit through a given list of points
 *
 * @snippet{trimleft} extrapolation_t.cpp least_squares
 *
 * @note the algorithm used directly computes the components of \f$ B^T W B\f$
 * followed by an LU decomposition.
 * @ingroup extrapolation
 * @param bs a number of input vectors all with the same size
 * @param b must have the same size as the bs
 * @param weights define the norm that is minimized
 * @return the minimal coefficients a
 * @copydoc hide_ContainerType
 */
template<class ContainerType0, class ContainerType1, class ContainerType2>
std::vector<double> least_squares( const std::vector<ContainerType0>& bs, const ContainerType1 & b, const ContainerType2& weights)
{
    // it would be interesting to see how this algorithm fares against
    // Gram-Schmidt/QR-factorization
    // This implementation should have as many scalar dots as Gram-Schmidt
    // namely (size^2+size)/2
    // Solve B^T B a = B^T b
    unsigned size = bs.size();
    // B^T B is the "Gram matrix"
    dg::SquareMatrix<double> op( size, 0.); // B^T B
    std::vector<double> rhs( size, 0.);
    std::vector<double> a(size,0.);
    for( unsigned i=0; i<size; i++)
    {
        for( unsigned j=i; j<size; j++)
            op(i,j) = dg::blas2::dot( bs[i], weights, bs[j]);
        for( unsigned j=0; j<i; j++)
            op(i,j) = op(j,i);
        rhs[i] = dg::blas2::dot( bs[i], weights, b);
    }
    // possibly replace with Cholesky factorization?
    std::vector<unsigned> p;
    dg::create::lu_pivot( op, p);
    dg::lu_solve<double>( op, p, rhs);
    return rhs;
}

/**
 * @brief An alias for <tt> least_squares( bs, b, 1.) </tt>
 * @ingroup extrapolation
 */
template<class ContainerType0, class ContainerType1>
std::vector<double> least_squares( const std::vector<ContainerType0>& bs, const ContainerType1 & b)
{
    return least_squares( bs, b, 1.);
}
/**
 * @brief %Evaluate a least squares fit
 *
 * This class gathers pairs of (features, labels) vectors \f$ (\vec x_i, \vec y_i)\f$
 * and then constructs a guess for \f$ y\f$ for given unkown \f$ \vec x\f$
 * by constructing the least squares coefficients
 * \f[ \min ||a_i \vec x_i - \vec x||\f]
 * to get
 * \f[ \vec y = a_i \vec y_i\f]
 * @note In the context of generating initial guesses for a matrix equation from
 * previous solutions this method is equivalent to the "rolling QR" algorithm described in https://arxiv.org/pdf/2009.10863.pdf
 * <a href="https://arxiv.org/pdf/2009.10863.pdf">Austin, A.P. and Chalmers N. and Warburton, T. INITIAL GUESSES FOR SEQUENCES OF LINEAR SYSTEMS IN A
GPU-ACCELERATED INCOMPRESSIBLE FLOW SOLVER (2021)</a>. This means it works for matrix equations that are constant in time.
 * @note This works best if the unkown function \f$ \vec y = f(\vec x) \f$ is linear and
 * if the \f$ x_i\f$ are orthogonal
 *
 * @snippet{trimleft} extrapolation_t.cpp LeastSquaresExtrapolation
 * @ingroup extrapolation
 * @copydoc hide_ContainerType
 */
template<class ContainerType0, class ContainerType1>
struct LeastSquaresExtrapolation
{
    using value_type = get_value_type<ContainerType0>;
    using container_type = ContainerType0;
    LeastSquaresExtrapolation( ){ m_counter = 0; }
    /*! @brief Set maximum number of vectors and allocate memory
     * @param max maximum of vectors to use for fit
     * @param copyable0 the memory for the x is allocated based on this vector
     * @param copyable1 the memory for the y is allocated based on this vector
     */
    LeastSquaresExtrapolation( unsigned max, const ContainerType0& copyable0, const ContainerType1& copyable1) {
        set_max(max, copyable0, copyable1);
    }
    ///@copydoc LeastSquaresExtrapolation(unsigned,const ContainerType0&,const ContainerType1&)
    void set_max( unsigned max, const ContainerType0& copyable0,
                    const ContainerType1& copyable1)
    {
        m_counter = 0;
        m_x.assign( max, copyable0);
        m_y.assign( max, copyable1);
        m_max = max;
    }
    ///return the current extrapolation max
    ///This may not coincide with the max set in the constructor if values have not been updated yet
    unsigned get_max( ) const{
        return m_counter;
    }

    /**
    * @brief extrapolate value at a new unkown value \f$ y = \alpha f(x) + \beta y \f$
    * @param alpha Quality of life parameter
    * @param x (read only) value to extrapolate for
    * @param beta Quality of life parameter
    * @param y (write only) contains extrapolated value on output
    */
    void extrapolate( double alpha, const ContainerType0& x, double beta,
            ContainerType1& y) const{
        unsigned size = m_counter;
        thrust::host_vector<double> rhs( size, 0.), a(rhs), opIi(rhs); // B^T b
        for( unsigned i=0; i<size; i++)
            rhs[i] = dg::blas1::dot( m_x[i], x);
        // a = op_inv * rhs
        dg::blas1::scal( y, beta);
        for( unsigned i=0; i<size; i++)
        {
            for( unsigned j=0; j<size; j++)
                opIi[j] = m_op_inv(i,j);
            a[i] = dg::blas1::dot( rhs, opIi) ;
            dg::blas1::axpby( alpha*a[i], m_y[i], 1., y);
        }
    }
    /**
    * @brief extrapolate value at a new unkown value \f$ y = f(x) \f$
    * @param x (read only) value to extrapolate for
    * @param y (write only) contains extrapolated value on output
    */
    void extrapolate( const ContainerType0& x, ContainerType1& y) const{
        extrapolate( 1., x, 0., y);
    }

    /**
    * @brief insert a new entry / train the machine learning algorithm
    * @param x_new the input
    * @param y_new the corresponding output
    * @attention if x_new is in the span of the existing x (i.e. x_new is a
    * linear combination of the x_i, then the new value pair is rejected
    * and the function returns)
    * @note It is very good at recognizing linear dependence because it
    * computes the linear algebra in extended accuracy using exblas
    */
    void update( const ContainerType0& x_new, const ContainerType1& y_new){
        if( m_max == 0) return;
        unsigned size = m_counter < m_max ? m_counter + 1 : m_max;
        dg::SquareMatrix<double> op( size, 0.), op_inv( size, 0.); // B^T B
        //i = 0
        op(0,0) = dg::blas1::dot( x_new, x_new);
        for( unsigned j=1; j<size; j++)
            op(0,j) = op( j, 0) = dg::blas1::dot( x_new, m_x[j-1]);
        // recursively fill in previous values
        for( unsigned i=1; i<size; i++)
            for( unsigned j=1; j<size; j++)
                op(i,j) = m_op(i-1,j-1);
        // test if new value is linearly independent or zero
        // maybe one can get a better control (with a tolerance value) over
        // this test
        try{
            op_inv = dg::create::inverse( op);
        }
        catch ( std::runtime_error & e){
            return;
        }
        m_op_inv = op_inv, m_op = op;
        if( m_counter < m_max)
            m_counter++;
        //push out last value (keep track of what is oldest value
        std::rotate( m_x.rbegin(), m_x.rbegin()+1, m_x.rend());
        std::rotate( m_y.rbegin(), m_y.rbegin()+1, m_y.rend());
        blas1::copy( x_new, m_x[0]);
        blas1::copy( y_new, m_y[0]);
    }

    private:
    unsigned m_max, m_counter;
    std::vector<ContainerType0> m_x;
    std::vector<ContainerType1> m_y;
    dg::SquareMatrix<double> m_op, m_op_inv;
};

/**
* @brief Extrapolate a polynomial passing through up to three points
*
* This class constructs an interpolating polynomial through up to three given points
* and evaluates its value or its derivative at a new point. The points can be updated to get a new polynomial.
*
* The intention of this class is to provide an initial guess for iterative solvers
* based on past solutions:
 \f[ x_{init} = \alpha_0 x_0 + \alpha_{-1}x_{-1} + \alpha_{-2} x_{-2}\f]
 where the indices indicate the current (0) and past (negative) solutions.
 Choose between 1 (constant), 2 (linear) or 3 (parabola) extrapolation.  The user can choose to provide a time value \c t_i associated with the \c x_i, which
 are then used to compute the coefficients \c alpha_i (using Lagrange interpolation).
 Otherwise an equidistant distribution is assumed.
*
 * @snippet{trimleft} extrapolation_t.cpp Extrapolation
 *
* @note Since extrapolation with higher order polynomials is so prone to oscillations
* anything higher than linear rarely leads to anything useful. So best stick to
* constant or linear extrapolation
* @note The derivative of the interpolating polynomial at a new point reduces to familiar finite difference formulas
* @copydoc hide_ContainerType
* @ingroup extrapolation
* @sa https://en.wikipedia.org/wiki/Extrapolation
*/
template<class ContainerType>
struct Extrapolation
{
    using value_type = get_value_type<ContainerType>;
    using container_type = ContainerType;
    /*! @brief Leave values uninitialized
     */
    Extrapolation( ){ m_counter = 0; }
    /*! @brief Set maximum extrapolation order and allocate memory
     * @param max maximum of vectors to use for extrapolation.
         Choose between 0 (no extrapolation) 1 (constant), 2 (linear) or 3
         (parabola) extrapolation.  Higher values currently default back to a
         linear extrapolation
     * @param copyable the memory is allocated based on this vector
     */
    Extrapolation( unsigned max, const ContainerType& copyable) {
        set_max(max, copyable);
    }
    ///@copydoc Extrapolation(unsigned,const ContainerType&)
    void set_max( unsigned max, const ContainerType& copyable)
    {
        m_counter = 0;
        m_x.assign( max, copyable);
        m_t.assign( max, 0);
        m_max = max;
    }
    /**
     * @brief Current extrapolation count
     *
     * @note This may not coincide with the max set in the constructor if
     * values have not been updated yet
     * @return The current size of the extrapolation
     */
    unsigned get_max( ) const{
        return m_counter;
    }


    /**
     * @brief Check if time exists in current points
     *
     * @note The check is numerical and returns true if \f$|t - t_i| < 10^{-14}\f$
     * for all i
     * @note If extrapolate() is called on an exisiting time,
     * the corresponding vector is returned while update() will overwrite
     * the existing vector
     * @param t The time to check for
     *
     * @return true if time exists in current points, false else
     */
    bool exists( value_type t)const{
        if( m_max == 0) return false;
        for( unsigned i=0; i<m_counter; i++)
            if( fabs(t - m_t[i]) <1e-14)
                return true;
        return false;
    }

    /**
    * @brief Extrapolate value to given time
    *
    * Construt and evaluate the interpolating polynomial at a given point
    * @param t time to which to extrapolate (or at which interpolating
    * polynomial is evaluated)
    * @param new_x (write only) contains extrapolated value on output
    * @tparam ContainerType0 must be usable with \c ContainerType in \ref dispatch
    * @attention If the update function has not been called enough times to fill all values the result depends: (i) never called => new_x is zero (ii) called at least once => the interpolating polynomial is constructed with all available values
    */
    template<class ContainerType0>
    void extrapolate( value_type t, ContainerType0& new_x) const{
        switch(m_counter)
        {
            case(0): dg::blas1::copy( 0, new_x);
                     break;
            case(1): dg::blas1::copy( m_x[0], new_x);
                     break;
            case(3): {
                value_type f0 = (t-m_t[1])*(t-m_t[2])/(m_t[0]-m_t[1])/(m_t[0]-m_t[2]);
                value_type f1 =-(t-m_t[0])*(t-m_t[2])/(m_t[0]-m_t[1])/(m_t[1]-m_t[2]);
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
    * @brief %Evaluate first derivative of interpolating polynomial
    *
    * Equivalent to constructing the interpolating polynomial, deriving it once
    * and then evaluating it at the required point
    * @param t time at which derivative of interpolating polynomial is evaluated
    * @param dot_x (write only) contains derived value on output
    * @note If t is chosen as the latest time of update t0, then the result coincides
    * with the backward difference formula of order  \c max
    * @attention If max==1, the result is 0 (derivative of a constant)
    * @attention If the update function has not been called enough times to fill all values the result depends: (i) never called => dot_x is zero (ii) called at least once => the interpolating polynomial is constructed with all available values
    * @tparam ContainerType0 must be usable with \c ContainerType in \ref dispatch
    */
    template<class ContainerType0>
    void derive( value_type t, ContainerType0& dot_x) const{
        switch(m_counter)
        {
            case(0): dg::blas1::copy( 0, dot_x);
                     break;
            case(1): dg::blas1::copy( 0, dot_x);
                     break;
            case(3): {
                value_type f0 =-(-2.*t+m_t[1]+m_t[2])/(m_t[0]-m_t[1])/(m_t[0]-m_t[2]);
                value_type f1 = (-2.*t+m_t[0]+m_t[2])/(m_t[0]-m_t[1])/(m_t[1]-m_t[2]);
                value_type f2 =-(-2.*t+m_t[0]+m_t[1])/(m_t[2]-m_t[0])/(m_t[2]-m_t[1]);
                dg::blas1::evaluate( dot_x, dg::equals(), dg::PairSum(),
                        f0, m_x[0], f1, m_x[1], f2, m_x[2]);
                break;
            }
            default: {
                value_type f0 = 1./(m_t[0]-m_t[1]);
                value_type f1 = 1./(m_t[1]-m_t[0]);
                dg::blas1::axpby( f0, m_x[0], f1, m_x[1], dot_x);
            }
        }
    }

    /**
    * @brief Extrapolate value (equidistant version)
    * @param new_x (write only) contains extrapolated value on output
    * @note Assumes that extrapolation time equals last inserted time+1
    * @tparam ContainerType0 must be usable with \c ContainerType in \ref dispatch
    */
    template<class ContainerType0>
    void extrapolate( ContainerType0& new_x) const{
        value_type t = m_t[0] +1.;
        extrapolate( t, new_x);
    }


    /**
     * @brief EXPERIMENTAL Perform a least squares extrapolation
     *
     * This algorithm computes \f$ b_i = A x_i\f$, then solves the least squares
     * problem \f$ \min_a || B a - b||_W \f$ with \f$ b_i\f$ the columns of
     * \f$ B\f$ to compute \f$ x_{new} = \sum_i a_i x_i\f$.
     * @note This is different from \c LeastSquaresExtrapolation if the matrix
     * \f$ A\f$ is time-dependent.
     * @note So far this was only tested for toefl simulations with default
     * parameters where we get mixed results. Depending on the \c max
     * parameter one can see a slow down or a speed-up. Around 10 we observed a
     * positive effect of about 10\% acceleration
     * @param A the matrix
     * @param b the right hand side
     * @param new_x Contains initial guess based on a least squares
     * extrapolation on output
     * @param weights The weights/volume in which to minimize
     */
    template<class MatrixType0, class ContainerType0, class ContainerType1>
    void extrapolate_least_squares( MatrixType0&& A, const ContainerType0& b,
            ContainerType0& new_x, const ContainerType1& weights)
    {
        if( m_counter < m_max)
        {
            extrapolate( new_x);
            return;
        }
        // allocate m_b if not yet done
        if( m_b.empty())
            m_b.assign( m_max, b);
        std::vector<const ContainerType*> x_ptrs = dg::asPointers( m_x);
        // An attempt at the algorithm in https://arxiv.org/abs/2309.02156
        //if( m_b.empty())
        //    m_b.assign( mm, b);
        // if( m_zx.empty())
        //     m_zx.assign( mm, b);
        // // compress the subspace via random linear combinations
        // if( mm < m_max)
        // {
        //     std::random_device rd{};
        //     std::mt19937 gen{rd()};
        //     std::normal_distribution<double> dist{0.0, 1.0};
        //     std::vector<double> z(m_max);
        //     for( unsigned i=0; i<mm; i++)
        //     {
        //         for( unsigned k=0; k<m_max; k++)
        //             z[k] = dist(gen);
        //         dg::blas2::gemv( 1., dg::asDenseMatrix(x_ptrs), z, 0., m_zx[i]);
        //     }
        //     x_ptrs = dg::asPointers( m_zx);
        // }
        // First compute bs
        for( unsigned u=0; u<m_max; u++)
            dg::apply( A, *x_ptrs[u], m_b[u]);

        try{
            std::vector<double> a = least_squares( m_b, b, weights);
            dg::blas2::gemv( 1., dg::asDenseMatrix(x_ptrs), a, 0., new_x);
        }
        catch( std::runtime_error& err)
        {
            return extrapolate( new_x);
        }

    }


    /**
    * @brief %Evaluate first derivative of interpolating polynomial (equidistant version)
    * @param dot_x (write only) contains derived value on output
    * @note Assumes that time equals t0 such that a backward difference formula will be evaluated
    * @tparam ContainerType0 must be usable with \c ContainerType in \ref dispatch
    */
    template<class ContainerType0>
    void derive( ContainerType0& dot_x) const{
        derive( m_t[0], dot_x);
    }

    /**
    * @brief insert a new entry, deleting the oldest entry or update existing entry
    * @param t_new the time for the new entry
    * @param new_entry the new entry ( replaces value of existing entry if \c t_new already exists)
    * @tparam ContainerType0 must be usable with \c ContainerType in \ref dispatch
    */
    template<class ContainerType0>
    void update( value_type t_new, const ContainerType0& new_entry){
        if( m_max == 0) return;
        //check if entry is already there to avoid division by zero errors
        for( unsigned i=0; i<m_counter; i++)
            if( fabs(t_new - m_t[i]) <1e-14)
            {
                blas1::copy( new_entry, m_x[i]);
                return;
            }
        if( m_counter < m_max) //don't update counter if Time entry was rejected
            m_counter++;
        //push out last value (keep track of what is oldest value
        std::rotate( m_x.rbegin(), m_x.rbegin()+1, m_x.rend());
        std::rotate( m_t.rbegin(), m_t.rbegin()+1, m_t.rend());
        m_t[0] = t_new;
        blas1::copy( new_entry, m_x[0]);
    }
    /**
    * @brief insert a new entry
    * @param new_entry the new entry
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
     * @return current head (undefined if max==0)
     */
    const ContainerType& head()const{
        return m_x[0];
    }
    ///DEPRECATED write access to tail value ( the one that will be deleted in the next update, undefined if max==0)
    ContainerType& tail(){
        return m_x[m_max-1];
    }
    ///DEPRECATED read access to tail value ( the one that will be deleted in the next update, undefined if max==0)
    const ContainerType& tail()const{
        return m_x[m_max-1];
    }

    private:
    unsigned m_max, m_counter;
    std::vector<value_type> m_t;
    std::vector<ContainerType> m_x;
    //only allocated if least squares extrapolate is used
    std::vector<ContainerType> m_b;
};


}//namespace dg
