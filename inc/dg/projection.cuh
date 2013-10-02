#include <vector>
#include <cusp/coo_matrix.h>
#include "grid.cuh"
#include "matrix_traits_thrust.h"


namespace dg{

namespace create{
namespace detail{
double LegendreP( unsigned n, double x)
{
    if( n==0) return 1;
    if( n==1) return x;
    return ((2*n+1)*x*LegendreP( n-1, x) - n*LegendreP( n-2, x))/(double)(n+1);
}
}//namespace detail
/**
 * @brief Create a projection matrix 
 *
 * Size is n_new*N_factor x n_old. It's intention is to project a given 
 * polynomial (x-space) in a cell to N subcells. 
 * @param n_old degree of given polynomial
 * @param n_new degree of polynomial in subcells (must be smaller than n_old)
 * @param N number of subcells 
 *
 * @return projection matrix in vector
 */
std::vector<double> projection( unsigned n_old, unsigned n_new, unsigned N)
{
    assert( n_old > 0);
    assert( n_new <= n_old && n_new > 0);
    assert( N > 0);
    dg::Grid1d<double> g_( -1, 1, n_new, N);
    thrust::host_vector<double> x = dg::create::abscissas( g_);
    unsigned rows = n_new*N, cols_i = n_new, cols = n_old;
    std::vector<double> project( rows*cols_i); 
    for( unsigned k=0; k<rows; k++)
        for( unsigned j=0; j<cols_i; j++)
        {
            project[ k*cols + j] = detail::LegendreP( j, x[k]) ;
        }
            
    std::vector<double> total( rows*cols, 0);
    DLT<double> dlt = g_.dlt();
    //multiply p_ki*f_ij
    for( unsigned k=0; k<rows; k++)
        for( unsigned j=0; j<cols; j++)
            for( unsigned i=0; i<cols_i; i++)
                total[ k*cols+j] += project[ k*cols_i+i]*dlt.forward()[i*cols+j];
    return total;

}


cusp::coo_matrix< int, double, cusp::host_memory> diagonal_matrix( unsigned N, std::vector<double>& v, unsigned v_rows, unsigned v_cols)
{
    cusp::coo_matrix<int, double, cusp::host_memory> A(N*v_rows, N*v_cols, N*v_rows*v_cols);
    unsigned number = 0;
    for( unsigned k=0; k<N; k++)
        for( unsigned i=0; i<v_rows; i++)
            for( unsigned j=0; j<v_cols; j++)
            {
                A.row_indices[number]      = k*v_rows+i;
                A.column_indices[number]   = k*v_cols+j;
                A.values[number]           = v[i*v_cols+j];
                number++;
            }
    return A;
}


}//namespace create
}//namespace dg
