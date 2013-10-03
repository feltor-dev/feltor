#include <vector>
#include <cusp/coo_matrix.h>
#include "grid.cuh"
#include "matrix_traits_thrust.h"


namespace dg{

namespace create{
namespace detail{
double LegendreP( unsigned n, double x)
{
    if( n==0 ) return 1;
    if( n==1 ) return x;
    return ((double)(2*n-1)*x*LegendreP( n-1, x) - (double)(n-1)*LegendreP( n-2, x))/(double)(n);
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


cusp::coo_matrix< int, double, cusp::host_memory> diagonal_matrix( unsigned N, const std::vector<double>& v, unsigned v_rows, unsigned v_cols)
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

cusp::coo_matrix< int, double, cusp::host_memory> projection1d( const Grid1d<double>& g1, const Grid1d<double>& g2)
{
    assert( g1.x0() == g2.x0()); assert( g1.x1() == g2.x1());
    assert( g2.N() % g1.N() == 0);
    unsigned Nf = g2.N()/g1.N();
    std::vector<double> p = dg::create::projection( g1.n(), g2.n(), Nf);
    return dg::create::diagonal_matrix( g1.N(), p, g2.n()*Nf, g1.n());
}
//2D Version
std::vector<double> tensor( const std::vector<double>& v1, const std::vector<double>& v2, unsigned n_old, unsigned n_new, unsigned N1f, unsigned N2f)
{
    std::vector<double> prod( n_old*n_old*n_new*n_new*N1f*N2f);
    for( unsigned i=0; i<n_new*N1f; i++)
        for( unsigned j=0; j<n_old; j++)
            for( unsigned k=0; k<n_new*N2f; k++)
                for( unsigned l=0; l<n_old; l++)
                    prod[ i*n_old*n_old*N2f+k*n_old*n_old+ j*n_old+l] = v1[i*n_old+j]*v2[k*n_old+l];
    return prod;

}
cusp::coo_matrix< int, double, cusp::host_memory> projection2d( const Grid<double>& g1, const Grid<double>& g2)
{
    assert( g1.x0() == g2.x0()); assert( g1.x1() == g2.x1());
    assert( g1.y0() == g2.y0()); assert( g1.y1() == g2.y1());
    assert( g2.Nx() % g1.Nx() == 0);
    assert( g2.Ny() % g1.Ny() == 0);
    unsigned Nfx = g2.Nx()/g1.Nx();
    unsigned Nfy = g2.Ny()/g1.Ny();
    Grid1d<double> g1x( g1.x0(), g1.x1(), g1.n(), g1.Nx()); 
    Grid1d<double> g1y( g1.y0(), g1.y1(), g1.n(), g1.Ny());
    Grid1d<double> g2x( g2.x0(), g2.x1(), g2.n(), g2.Nx()); 
    Grid1d<double> g2y( g2.y0(), g2.y1(), g2.n(), g2.Ny());
    std::vector<double> px = dg::create::projection( g1.n(), g2.n(), Nfx);
    std::vector<double> py = dg::create::projection( g1.n(), g2.n(), Nfy);
    std::vector<double> p = tensor( py, px, g1.n(), g2.n(), Nfy, Nfx);
    return dg::create::diagonal_matrix( g1.Nx()*g1.Ny(), p, g2.n()*g2.n()*Nfx*Nfy, g1.n()*g1.n() );
}


}//namespace create
}//namespace dg
