#pragma once

#include "grid.cuh"
#include "creation.cuh"
#include "operator_matrix.cuh"

namespace dg{

namespace create{
namespace detail{

std::vector<double> coefficients( double xn, unsigned n)
{
    assert( xn <= 1. && xn >= -1.);
    std::vector<double> px(n);
    if( xn == -1)
    {
        for( unsigned i=0; i<n; i++)
            px[i] = (double)pow( -1, i);
    }
    else if( xn == 1)
    {
        for( unsigned i=0; i<n; i++)
            px[i] = 1.;
    }
    else
    {
        px[0] = 1.;
        if( n > 1)
        {
            px[1] = xn;
            for( unsigned i=1; i<n-1; i++)
                px[i+1] = ((double)(2*i+1)*xn*px[i]-(double)i*px[i-1])/(double)(i+1);
        }
    }
    return px;
}

}//namespace detail
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( std::vector<double>& x, std::vector<double>& y, const Grid2d<double>& g)
{
    assert( x.size() == g.size());
    assert( y.size() == g.size());
    cusp::coo_matrix<int, double, cusp::host_memory> A( g.size(), g.size(), g.size()*g.n()*g.n());

    int number = 0;
    for( unsigned i=0; i<g.size(); i++)
    {
        assert( x[i] >= g.x0() && x[i] <= g.x1());
        assert( y[i] >= g.y0() && y[i] <= g.y1());

        //determine which cell (x,y) lies in 
        unsigned n = (unsigned)floor(x[i]/g.hx());
        unsigned m = (unsigned)floor(y[i]/g.hy());

        //determine normalized coordinates
        double xn = 2.*x[i]/g.hx() - (double)(2*n+1); 
        double yn = 2.*y[i]/g.hy() - (double)(2*m+1); 

        std::vector<double> px = detail::coefficients(xn, g.n()), py = detail::coefficients( yn, g.n());
        std::vector<double> pxy( g.n()*g.n());
        for(unsigned k=0; k<py.size(); k++)
            for( unsigned l=0; l<px.size(); l++)
                pxy[k*px.size()+l]= py[k]*px[l];
        unsigned col_begin = m*g.Nx()*g.n()*g.n() + n*g.n()*g.n();
        detail::add_line( A, number, i,  col_begin, pxy);
    }
    dg::Operator<double> forward( g.dlt().forward());
    dg::Operator<double> forward2d = dg::tensor( forward, forward);
    Matrix ward = dg::tensor( g.Nx()*g.Ny(), forward2d), B;
    cusp::multiply( A, ward, B);
    B.sort_by_row_and_column();
    return B;

}

}//namespace create
} //namespace dg
