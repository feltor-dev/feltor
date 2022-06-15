#pragma once
#include "dg/topology/functions.h"
#include "dg/backend/config.h"

namespace dg{
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
 * @sa dg::blas2::filtered_symv
 */
struct CSRMedianFilter
{
    template<class real_type>
    DG_DEVICE
    void operator()( unsigned i, const int* row_offsets,
            const int* column_indices, const real_type* values,
            const real_type* x, real_type* y)
    {
        // http://ndevilla.free.fr/median/median/index.html
        // ignore the values array ...
        y[i] = detail::median( i, row_offsets, column_indices, []DG_DEVICE(double x){return x;}, x);
    }
};


// Adaptive Switching Median Filter from Akkoul " A New Adaptive Switching Median Filter" IEEE Signal processing letters (2010)
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
 * @sa dg::blas2::filtered_symv
 */
template<class real_type>
struct CSRSWMFilter
{
    CSRSWMFilter( real_type alpha) : m_alpha( alpha) {}
    DG_DEVICE
    void operator()( unsigned i, const int* row_offsets,
            const int* column_indices, const real_type* values,
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
 * @sa dg::blas2::filtered_symv
 */
struct CSRAverageFilter
{
    template<class real_type>
    DG_DEVICE
    void operator()( unsigned i, const int* row_offsets,
            const int* column_indices, const real_type* values,
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
 * @sa dg::blas2::filtered_symv
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
///@}

}//namespace dg
