#pragma once

#include "grid.h"
#include "creation.cuh"
#include "operator_tensor.cuh"

namespace dg{

namespace create{
    ///@cond
namespace detail{

/**
 * @brief Evaluate n Legendre poloynomial on given abscissa
 *
 * @param xn normalized x-value on which to evaluate the polynomials: -1<=xn<=1
 * @param n  maximum order of the polynomial
 *
 * @return array of coefficients beginning with p_0(x_n) until p_{n-1}(x_n)
 */
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
///@endcond

/**
 * @brief Create interpolation matrix
 *
 * Transforms from a vector given in XSPACE to the points in XSPACE
 * @param x X-coordinates of interpolation points
 * @param y Y-coordinates of interpolation points
 * @param g The Grid on which to operate
 *
 * @return interpolation matrix
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const thrust::host_vector<double>& x, const thrust::host_vector<double>& y, const Grid2d<double>& g)
{
    assert( x.size() == g.size());
    assert( y.size() == g.size());
    cusp::coo_matrix<int, double, cusp::host_memory> A( g.size(), g.size(), g.size()*g.n()*g.n());

    int number = 0;
    for( unsigned i=0; i<g.size(); i++)
    {
        if (!(x[i] >= g.x0() && x[i] <= g.x1())) {
            std::cerr << "xi = " << x[i] <<std::endl;
        }
        assert(x[i] >= g.x0() && x[i] <= g.x1());
        
        if (!(y[i] >= g.y0() && y[i] <= g.y1())) {
            std::cerr << "yi = " << y[i] <<std::endl;
        }
        assert( y[i] >= g.y0() && y[i] <= g.y1());

        //determine which cell (x,y) lies in 
        unsigned n = (unsigned)floor((x[i]-g.x0())/g.hx());
        unsigned m = (unsigned)floor((y[i]-g.y0())/g.hy());

        //determine normalized coordinates
        double xn = 2.*(x[i]-g.x0())/g.hx() - (double)(2*n+1); 
        double yn = 2.*(y[i]-g.y0())/g.hy() - (double)(2*m+1); 

        //evaluate 2d Legendre polynomials at (xn, yn)...
        std::vector<double> px = detail::coefficients( xn, g.n()), 
                            py = detail::coefficients( yn, g.n());
        std::vector<double> pxy( g.n()*g.n());
        for(unsigned k=0; k<py.size(); k++)
            for( unsigned l=0; l<px.size(); l++)
                pxy[k*px.size()+l]= py[k]*px[l];

        //...these are the matrix coefficients with which to multiply 
        //unsigned col_begin = m*g.Nx()*g.n()*g.n() + n*g.n()*g.n();
        //detail::add_line( A, number, i,  col_begin, pxy);
        unsigned col_begin = m*g.Nx()*g.n()*g.n() + n*g.n();
        detail::add_line( A, number, i,  col_begin, g.n(), g.Nx(), pxy);
        //choose layout from comments
    }
    typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;
    
    dg::Operator<double> forward( g.dlt().forward());
    Matrix transformX = dg::tensor( g.Nx(), forward);
    Matrix transformY = dg::tensor( g.Ny(), forward);
    Matrix ward = dg::dgtensor( g.n(), transformY, transformX);

    Matrix B;
    cusp::multiply( A, ward, B);
    B.sort_by_row_and_column();
    return B;

}

}//namespace create
} //namespace dg
