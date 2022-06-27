#pragma once
#include "dg/functors.h"
#include "fast_interpolation.h"

/**@file
* @brief Modal Filtering
*/

namespace dg
{

///@cond
namespace create
{

template<class UnaryOp, class real_type>
dg::Operator<real_type> modal_op( UnaryOp op, const DLT<real_type>& dlt )
{
    Operator<real_type> backward=dlt.backward();
    Operator<real_type> forward=dlt.forward();
    Operator<real_type> filter( dlt.n(), 0);
    for( unsigned i=0; i<dlt.n(); i++)
        filter(i,i) = op( i);
    filter = backward*filter*forward;
    return filter;
}

} //namespace create
///@endcond

/**
 * @brief Struct that applies a given modal filter to a vector
 *
 * \f[ y = V D V^{-1}\f]
 * where \f$ V\f$ is the Vandermonde matrix (the backward transformation matrix)
 * and \f$ D \f$ is a diagonal matrix with \f$ D_{ii} = \sigma(i)\f$
 * @sa A discussion of the effects of the modal filter on advection schemes can be found here https://mwiesenberger.github.io/advection
 * @note basically the result is that it is usually not advantageous to use a modal filter
 * @copydoc hide_matrix
 * @copydoc hide_ContainerType
 * @ingroup misc
 */
template<class MatrixType, class ContainerType>
struct ModalFilter
{
    using real_type = get_value_type<ContainerType>;
    ModalFilter() = default;
    /**
     * @brief Create arbitrary filter
     *
     * @tparam Topology Any grid
     * @tparam UnaryOp Model of Unary Function \c real_type \c sigma(unsigned) The input will be the modal number \c i where \f$ i=0,...,n-1\f$ and \c n is the number of polynomial coefficients in use. The output is the filter strength for the given mode number
     * @param sigma The filter to evaluate on the normalized modal coefficients
     * @param t The topology to apply the modal filter on
     * @param ps parameters that are forwarded to the creation of a ContainerType (e.g. when a std::vector is to be created it is the vector size)
     */
    template<class UnaryOp, class Topology, class ...Params>
    ModalFilter( UnaryOp sigma, const Topology& t, Params&& ...ps) :
        m_tmp( dg::construct<ContainerType>(dg::evaluate( dg::zero, t),
        std::forward<Params>(ps)...)), m_filter ( dg::create::fast_transform(
        create::modal_op(sigma, t.dltx()), create::modal_op(sigma, t.dlty()),
        t), std::forward<Params>(ps)...)
            { }

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
        *this = ModalFilter( std::forward<Params>( ps)...);
    }

    void operator()( ContainerType& y) {
        operator()( 1., y, 0., m_tmp);
        using std::swap;
        swap( y, m_tmp);
    }
    void operator()( const ContainerType& x, ContainerType& y) const{ operator()( 1., x,0,y);}
    void operator()(real_type alpha, const ContainerType& x, real_type beta, ContainerType& y) const
    {
        m_filter.symv( alpha, x, beta, y);
    }
    private:
    ContainerType m_tmp;
    MultiMatrix<MatrixType, ContainerType> m_filter;
};

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
            const int* column_indices, const real_type* values,
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
 * @sa dg::blas2::stencil dg::create::window_stencil
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
 * Consider the one-dimensional case and further assume that the
 * modal coefficients are given. (Else use \c dg::create::fast_transform
 * to convert to L-space). The linear part is given by \f$ u_h^1(x) = u_{n0}p_{n0}(x) + u_{n1}p_{n1}(x)\f$ with \f$p_{n0}(x) = 1\f$ and \f$ p_{n1}(x) = 2(x-x_n)/h\f$. Then the limiter is defined via
 * \f[
 * \Lambda\Pi ( u_h^1)|_n = u_{n0} + \textrm{minmod}\left( u_{n1}, ( u_{(n+1)0} - u_{n0}), (u_{(n)0} - u_{(n-1)0})\right)p_{n1}(x)
 * \f]
 * If the result of the minmod function is \f$ u_{n1}\f$, then \f$ \Lambda\Pi( u_h)|_n = u_h|_n\f$, else \f$ \Lambda\Pi(u_h)|_n = \Lambda\Pi(u_h^1)|_n\f$
 * Must be applied to coefficients transformed to L-space in combination with \c limiter_stencil
 *
 * @sa dg::blas2::stencil dg::create::fast_transform dg::create::limiter_stencil
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
        if( (row_offsets[i+1] - row_offsets[i]) == 0) //only every n-th thread does something
            return;
        int n = abs((int)values[0]);
        for( int u=0; u<n; u++)
            y[i+u] = x[i+u]; // copy input
        dg::MinMod minmod;
        real_type dx = x[column_indices[k+2]];
        if( fabs( dx) <= m_mod)
            return;
        real_type m = minmod( dx,
            (x[column_indices[k+1]] - x[column_indices[k]]),
            (x[column_indices[k+3]] - x[column_indices[k+1]]));
        if( m == dx)
            return;
        y[i+1] = m;
        for( int u=2; u<n; u++)
            y[i+u] = (real_type)0;
    }

    private:
    real_type m_mod;
};

///@}
}//namespace dg
