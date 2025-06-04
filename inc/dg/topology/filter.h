#pragma once
#include "dg/functors.h"
#include "fast_interpolation.h"

/**@file
* @brief Modal Filtering
*/

namespace dg
{

namespace create
{

/**
 * @brief Create a modal filter block \f$ V D V^{-1}\f$
 *
 * where \f$ V\f$ is the Vandermonde matrix (the backward transformation matrix)
 * and \f$ D \f$ is a diagonal matrix with \f$ D_{ii} = \sigma(i)\f$
 * @sa A discussion of the effects of the modal filter on advection schemes can be found here https://mwiesenberger.github.io/advection
 * @note basically the result is that it is usually not advantageous to use a modal filter
 * @tparam UnaryOp Model of Unary Function \c real_type \c sigma(unsigned) The input will be the modal number \c i where \f$ i=0,...,n-1\f$ and \c n is the number of polynomial coefficients in use. The output is the filter strength for the given mode number
 * @param op the unary function
 * @param n number of polynomial coefficients for forward and backward transformation
 * @return The product \f$ V D V^{-1}\f$

 * @note The idea is to use the result in connection with \c dg::create::fast_transform() to create a matrix that applies the filter to vectors. For example
 * to create a modal filter that acts in two dimensions:
 * @code{.cpp}
 * // create filter:
 * auto filter = dg::create::fast_transform(
 *      dg::create::modal_filter( op, grid.nx()),
 *      dg::create::modal_filter( op, grid.ny()), grid);
 * //apply filter:
 * dg::blas2::symv( filter, x, y);
 * @endcode
 * @ingroup misc
 */
template<class UnaryOp>
dg::SquareMatrix<std::invoke_result_t<UnaryOp, unsigned>> modal_filter( UnaryOp op, unsigned n )
{
    using real_type = std::invoke_result_t<UnaryOp, unsigned>;
    SquareMatrix<real_type> backward=dg::DLT<real_type>::backward(n);
    SquareMatrix<real_type> forward=dg::DLT<real_type>::forward(n);
    SquareMatrix<real_type> filter( n, 0);
    for( unsigned i=0; i<n; i++)
        filter(i,i) = op( i);
    filter = backward*filter*forward;
    return filter;
}

} //namespace create

///@cond
namespace detail{

template<class real_type>
DG_DEVICE void pix_sort( real_type& a, real_type& b)
{
    if( a > b) // swap
    {
        real_type tmp = a;
        a = b;
        b = tmp;
    }
}

template<class real_type>
DG_DEVICE real_type median3( real_type* p)
{
    pix_sort(p[0],p[1]) ; pix_sort(p[1],p[2]) ; pix_sort(p[0],p[1]) ;
    return (p[1]) ;
}

template<class real_type>
DG_DEVICE real_type median5( real_type* p)
{
    pix_sort(p[0],p[1]) ; pix_sort(p[3],p[4]) ; pix_sort(p[0],p[3]) ;
    pix_sort(p[1],p[4]) ; pix_sort(p[1],p[2]) ; pix_sort(p[2],p[3]) ;
    pix_sort(p[1],p[2]) ; return (p[2]) ;
}

template<class real_type>
DG_DEVICE real_type median9( real_type* p)
{
    pix_sort(p[1], p[2]) ; pix_sort(p[4], p[5]) ; pix_sort(p[7], p[8]) ;
    pix_sort(p[0], p[1]) ; pix_sort(p[3], p[4]) ; pix_sort(p[6], p[7]) ;
    pix_sort(p[1], p[2]) ; pix_sort(p[4], p[5]) ; pix_sort(p[7], p[8]) ;
    pix_sort(p[0], p[3]) ; pix_sort(p[5], p[8]) ; pix_sort(p[4], p[7]) ;
    pix_sort(p[3], p[6]) ; pix_sort(p[1], p[4]) ; pix_sort(p[2], p[5]) ;
    pix_sort(p[4], p[7]) ; pix_sort(p[4], p[2]) ; pix_sort(p[6], p[4]) ;
    pix_sort(p[4], p[2]) ; return (p[4]) ;
}

template<class real_type, class Functor>
DG_DEVICE real_type median( unsigned i, const int* row_offsets,
            const int* column_indices, Functor f, const real_type* x )
{
    int n = row_offsets[i+1]-row_offsets[i];
    if( n == 3)
    {
        real_type p[3];
        int k = row_offsets[i];
        for( int l = 0; l<3; l++)
            p[l] =  f(x[column_indices[k+l]]);
        return detail::median3( p);
    }
    if ( n == 5)
    {
        real_type p[5];
        int k = row_offsets[i];
        for( int l = 0; l<5; l++)
            p[l] =  f(x[column_indices[k+l]]);
        return detail::median5(p);
    }
    if( n == 9)
    {
        real_type p[9];
        int k = row_offsets[i];
        for( int l = 0; l<9; l++)
            p[l] =  f(x[column_indices[k+l]]);
        return detail::median9( p);

    }
    int less, greater, equal;
    real_type  min, max, guess, maxltguess, mingtguess;

    min = max = f(x[column_indices[row_offsets[i]]]) ;
    for (int k=row_offsets[i]+1 ; k<row_offsets[i+1] ; k++) {
        if (f(x[column_indices[k]])<min) min=f(x[column_indices[k]]);
        if (f(x[column_indices[k]])>max) max=f(x[column_indices[k]]);
    }

    while (1) {
        guess = (min+max)/2;
        less = 0; greater = 0; equal = 0;
        maxltguess = min ;
        mingtguess = max ;
        for (int k=row_offsets[i]; k<row_offsets[i+1]; k++) {
            if (f(x[column_indices[k]])<guess) {
                less++;
                if (f(x[column_indices[k]])>maxltguess)
                    maxltguess = f(x[column_indices[k]]) ;
            } else if (f(x[column_indices[k]])>guess) {
                greater++;
                if (f(x[column_indices[k]])<mingtguess)
                    mingtguess = f(x[column_indices[k]]) ;
            } else equal++;
        }
        if (less <= (n+1)/2 && greater <= (n+1)/2) break ;
        else if (less>greater) max = maxltguess ;
        else min = mingtguess;
    }
    if (less >= (n+1)/2) return maxltguess;
    else if (less+equal >= (n+1)/2) return guess;
    else return mingtguess;
}

}//namespace detail
///@endcond

///@addtogroup filters
///@{
/**
 * @brief Compute (lower) Median of input numbers
 *
 * The (lower) median of N numbers is the N/2 (rounded up) largest number.
 * Another definition is implicit
 * \f$ \text{Median}(x) := \{ m : \sum_i \text{sgn}(x_i - m) = 0\} \f$
 * The Median is taken over all points contained
 * in the stencil given by the row and column indices. The matrix values are ignored.
 * @sa dg::blas2::stencil dg::create::window_stencil
 */
struct CSRMedianFilter
{
    template<class real_type>
    DG_DEVICE
    void operator()( unsigned i, const int* row_offsets,
            const int* column_indices, const real_type* /*values*/,
            const real_type* x, real_type* y)
    {
        // http://ndevilla.free.fr/median/median/index.html
        // ignore the values array ...
        y[i] = detail::median( i, row_offsets, column_indices, []DG_DEVICE(double x){return x;}, x);
    }
};


/**
 * @brief Switching median filter
 *
 \f[ y_i = \begin{cases}
      \text{Median}( x) \text{ if } |x_i - \text{Median}(x)| > \alpha \sigma\\
      x_i \text{ else}
      \end{cases}
 \f]
 with
 \f[
 \sigma = \text{Median}(|x-\text{Median}(x)|)
 \f]
 the median absolute deviation and \f$ \alpha\f$ a constant.
 The Median is taken over all points contained
 in the stencil given by the row and column indices. The matrix values are ignored.
 @note Adaptive Switching Median Filter from Akkoul "A New Adaptive Switching Median Filter" IEEE Signal processing letters (2010)
 * @sa dg::blas2::stencil dg::create::window_stencil
 */
template<class real_type>
struct CSRSWMFilter
{
    CSRSWMFilter( real_type alpha) : m_alpha( alpha) {}
    DG_DEVICE
    void operator()( unsigned i, const int* row_offsets,
            const int* column_indices, const real_type* /*values*/,
            const real_type* x, real_type* y)
    {
        real_type median = detail::median( i, row_offsets, column_indices,
            []DG_DEVICE(double x){return x;}, x);
        real_type amd = detail::median( i, row_offsets, column_indices,
            [median]DG_DEVICE(double x){return fabs(x-median);}, x);

        if( fabs( x[i] - median) > m_alpha*amd)
        {
            y[i] = median;
        }
        else
            y[i] = x[i];
    }
    private:
    real_type m_alpha ;
};

/**
 * @brief %Average filter that computes the average of all points in the stencil
 * @sa dg::blas2::stencil dg::create::window_stencil
 */
struct CSRAverageFilter
{
    template<class real_type>
    DG_DEVICE
    void operator()( unsigned i, const int* row_offsets,
            const int* column_indices, const real_type* /*values*/,
            const real_type* x, real_type* y)
    {
        y[i] = 0;
        int n = row_offsets[i+1]-row_offsets[i];
        for( int k=row_offsets[i]; k<row_offsets[i+1]; k++)
            y[i] += x[column_indices[k]]/(real_type)n;
    }
};
/**
 * @brief Test filter that computes the symv csr matrix-vector product if used
 * @sa dg::blas2::stencil dg::create::window_stencil
 */
struct CSRSymvFilter
{
    template<class real_type>
    DG_DEVICE
    void operator()( unsigned i, const int* row_offsets,
            const int* column_indices, const real_type* values,
            const real_type* x, real_type* y)
    {
        y[i] = 0;
        for( int k=row_offsets[i]; k<row_offsets[i+1]; k++)
            y[i] += x[column_indices[k]]*values[k];
    }
};

/**
 * @brief Generalized slope limiter for dG methods
 *
 * Consider the one-dimensional case. The first step is to transform the given
 * values to compute modal coefficients.
 * The linear part is given by
 * \f$ u_h^1(x) = u_{n0}p_{n0}(x) + u_{n1}p_{n1}(x)\f$ with \f$p_{n0}(x) = 1\f$
 * and \f$ p_{n1}(x) = 2(x-x_n)/h\f$.
 * Then the limiter is defined via
 * \f[
 * \Lambda\Pi ( u_h^1)|_n = u_{n0} + \textrm{minmod}\left( u_{n1}, ( u_{(n+1)0} - u_{n0}), (u_{(n)0} - u_{(n-1)0})\right)p_{n1}(x)
 * \f]
 * If the result of the minmod function is \f$ u_{n1}\f$, then \f$ \Lambda\Pi( u_h)|_n = u_h|_n\f$, else \f$ \Lambda\Pi(u_h)|_n = \Lambda\Pi(u_h^1)|_n\f$
 * Must be applied in combination with \c limiter_stencil
 * @note This limiter in a dG advection scheme has mixed success, generally
 * maybe because we use it as a Kronecker product of two 1d filters?
 *
 * @sa dg::blas2::stencil dg::create::limiter_stencil
 */
template<class real_type>
struct CSRSlopeLimiter
{
    CSRSlopeLimiter( real_type mod = (real_type)0) :
        m_mod(mod) {}
    DG_DEVICE
    void operator()( unsigned i, const int* row_offsets,
            const int* column_indices, const real_type* values,
            const real_type* x, real_type* y)
    {
        int k = row_offsets[i];
        int n = (row_offsets[i+1] - row_offsets[i])/3;
        if( n == 0) //only every n-th thread does something
            return;
        for( int u=0; u<n; u++)
            y[column_indices[k+1*n+u]] = x[column_indices[k+1*n + u]]; // copy input
        // Transform
        real_type uM = 0, u0 = 0, uP = 0, u1 = 0;
        for( int u=0; u<n; u++)
        {
            uM += x[column_indices[k+0*n + u]]*fabs(values[k+u]);
            u0 += x[column_indices[k+1*n + u]]*fabs(values[k+u]);
            u1 += x[column_indices[k+1*n + u]]*values[k+n+u];
            uP += x[column_indices[k+2*n + u]]*fabs(values[k+u]);
        }
        if( values[k]<0) //DIR boundary condition
            uM *= -1;
        if( values[k+2*n]>0) //DIR boundary condition
            uP *= -1;

        dg::MinMod minmod;
        if( fabs( u1) <= m_mod)
            return;
        real_type m = minmod( u1, uP - u0, u0 - uM);
        if( m == u1)
            return;
        // Else transform back
        for( int u=0; u<n; u++)
            y[column_indices[k+1*n+u]] =
             values[k+2*n]>0 ? u0 - m*values[k+2*n+u] : u0 + m*values[k+2*n+u];
    }
    private:
    real_type m_mod;
};


///@}
}//namespace dg
