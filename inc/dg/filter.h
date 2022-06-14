#pragma once
#include "dg/topology/functions.h"
#include "dg/backend/config.h"

namespace dg{

    // Idea: maybe use if( | m - f_0| < rel*|f_0| + abs)
    // to use filter only at places where it is needed
///@addtogroup filters
///@{
/**
 * @brief Compute (lower) Median of input numbers
 *
 * The (lower) median of N numbers is the N/2 (rounded up) largest number
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
        int n = row_offsets[i+1]-row_offsets[i];
        if( n == 3)
        {
            real_type p[3];
            int k = row_offsets[i];
            for( int l = 0; l<3; l++)
                p[l] =  x[column_indices[k+l]];
            pix_sort(p[0],p[1]) ; pix_sort(p[1],p[2]) ; pix_sort(p[0],p[1]) ;
            y[i] = (p[1]) ;
        }
        else if ( n == 5)
        {
            real_type p[5];
            int k = row_offsets[i];
            for( int l = 0; l<5; l++)
                p[l] =  x[column_indices[k+l]];
            pix_sort(p[0],p[1]) ; pix_sort(p[3],p[4]) ; pix_sort(p[0],p[3]) ;
            pix_sort(p[1],p[4]) ; pix_sort(p[1],p[2]) ; pix_sort(p[2],p[3]) ;
            pix_sort(p[1],p[2]) ; y[i] = (p[2]) ;
        }
        else if( n == 9)
        {
            real_type p[9];
            int k = row_offsets[i];
            for( int l = 0; l<9; l++)
                p[l] =  x[column_indices[k+l]];

            pix_sort(p[1], p[2]) ; pix_sort(p[4], p[5]) ; pix_sort(p[7], p[8]) ;
            pix_sort(p[0], p[1]) ; pix_sort(p[3], p[4]) ; pix_sort(p[6], p[7]) ;
            pix_sort(p[1], p[2]) ; pix_sort(p[4], p[5]) ; pix_sort(p[7], p[8]) ;
            pix_sort(p[0], p[3]) ; pix_sort(p[5], p[8]) ; pix_sort(p[4], p[7]) ;
            pix_sort(p[3], p[6]) ; pix_sort(p[1], p[4]) ; pix_sort(p[2], p[5]) ;
            pix_sort(p[4], p[7]) ; pix_sort(p[4], p[2]) ; pix_sort(p[6], p[4]) ;
            pix_sort(p[4], p[2]) ; y[i] = (p[4]) ;
        }
        else
        {
            int less, greater, equal;
            real_type  min, max, guess, maxltguess, mingtguess;

            min = max = x[column_indices[row_offsets[i]]] ;
            for (int k=row_offsets[i]+1 ; k<row_offsets[i+1] ; k++) {
                if (x[column_indices[k]]<min) min=x[column_indices[k]];
                if (x[column_indices[k]]>max) max=x[column_indices[k]];
            }

            while (1) {
                guess = (min+max)/2;
                less = 0; greater = 0; equal = 0;
                maxltguess = min ;
                mingtguess = max ;
                for (int k=row_offsets[i]; k<row_offsets[i+1]; k++) {
                    if (x[column_indices[k]]<guess) {
                        less++;
                        if (x[column_indices[k]]>maxltguess)
                            maxltguess = x[column_indices[k]] ;
                    } else if (x[column_indices[k]]>guess) {
                        greater++;
                        if (x[column_indices[k]]<mingtguess)
                            mingtguess = x[column_indices[k]] ;
                    } else equal++;
                }
                if (less <= (n+1)/2 && greater <= (n+1)/2) break ;
                else if (less>greater) max = maxltguess ;
                else min = mingtguess;
            }
            if (less >= (n+1)/2) y[i] = maxltguess;
            else if (less+equal >= (n+1)/2) y[i] = guess;
            else y[i] =  mingtguess;
        }
    }
    private:
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
};


// Adaptive Switching Median Filter from Akkoul " A New Adaptive Switching Median Filter" IEEE Signal processing letters (2010)
template<class real_type>
struct CSRASWMFilter
{
    CSRASWMFilter( real_type alpha, real_type eps = 0.01,
        real_type delta = 0.1) :
        m_eps(eps), m_delta( delta), m_alpha( alpha) {}
    DG_DEVICE
    void operator()( unsigned i, const int* row_offsets,
            const int* column_indices, const real_type* values,
            const real_type* x, real_type* y)
    {
        real_type m_old = 1e10, m_new = 0;

        m_new = 0.;
        for( int k=row_offsets[i]; k<row_offsets[i+1]; k++)
            m_new += x[column_indices[k]];
        m_new /= (real_type)(row_offsets[i+1]-row_offsets[i]);
        // compute weighted mean until converged
        while ( fabs(m_old -  m_new) > m_eps*fabs(x[i])+m_eps)
        {
            m_old = m_new;
            m_new = 0.;
            real_type sum = 0.;
            for( int k=row_offsets[i]; k<row_offsets[i+1]; k++)
            {
                sum += 1./(fabs( x[column_indices[k]] - m_old) + m_delta);
                m_new += x[column_indices[k]]/
                    (fabs( x[column_indices[k]] - m_old) + m_delta);
            }
            m_new /= sum;
        }

        // compute weighted standard deviation
        real_type sigma = 0;
        real_type sum = 0.;
        for( int k=row_offsets[i]; k<row_offsets[i+1]; k++)
        {
            sum += 1./(fabs( x[column_indices[k]] - m_new) + m_delta);
            sigma += ( x[column_indices[k]] - m_new)*(x[column_indices[k]] - m_new)/
                (fabs( x[column_indices[k]] - m_old) + m_delta);
        }
        sigma = sqrt( sigma/sum);
        if( fabs( x[i] - m_new) > m_alpha*sigma)
        {
            dg::CSRMedianFilter()( i, row_offsets, column_indices, values, x, y);
        }
        else
            y[i] =x[i];
    }
    private:
    real_type m_eps, m_delta, m_alpha ;
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
