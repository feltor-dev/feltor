#pragma once
#include "dg/topology/functions.h"
#include "dg/backend/config.h"

namespace dg{

    // Idea: maybe use if( | m - f_0| < rel*f_0 + abs)
    // to use filter only at places where it is needed
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
/**
 * @brief Average filter that computes the average of all points in the stencil
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

}//namespace dg
