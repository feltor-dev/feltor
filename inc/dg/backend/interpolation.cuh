#pragma once

#include "grid.h"
#include "evaluation.cuh"
#include "functions.h"
#include "creation.cuh"
#include "tensor.cuh"
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
 * @param g The Grid on which to operate
 *
 * @return interpolation matrix
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const thrust::host_vector<double>& x, const Grid1d<double>& g)
{
    cusp::coo_matrix<int, double, cusp::host_memory> A( x.size(), g.size(), x.size()*g.n()*g.n());

    int number = 0;
    for( unsigned i=0; i<x.size(); i++)
    {
        if (!(x[i] >= g.x0() && x[i] <= g.x1())) {
            std::cerr << "xi = " << x[i] <<std::endl;
        }
        assert(x[i] >= g.x0() && x[i] <= g.x1());

        //determine which cell (x) lies in 
        unsigned n = (unsigned)floor((x[i]-g.x0())/g.h());
        //determine normalized coordinates
        double xn = 2.*(x[i]-g.x0())/g.h() - (double)(2*n+1); 
        //evaluate 2d Legendre polynomials at (xn, yn)...
        std::vector<double> px = detail::coefficients( xn, g.n());
        //...these are the matrix coefficients with which to multiply 
        unsigned col_begin = n*g.n();
        detail::add_line( A, number, i,  col_begin, px);
        //choose layout from comments
    }
    typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;
    
    dg::Operator<double> forward( g.dlt().forward());
    Matrix transformX = dg::tensor( g.N(), forward);

    Matrix B;
    cusp::multiply( A, transformX, B);
    B.sort_by_row_and_column();
    return B;
}

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
    assert( x.size() == y.size());
    cusp::coo_matrix<int, double, cusp::host_memory> A( x.size(), g.size(), x.size()*g.n()*g.n());

    int number = 0;
    for( unsigned i=0; i<x.size(); i++)
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
        unsigned n =(unsigned)floor((x[i]-g.x0())/g.hx());
        unsigned m = (unsigned)floor((y[i]-g.y0())/g.hy());

        //determine normalized coordinates
        double xn = 2.*(x[i]-g.x0())/g.hx() - (double)(2*(n)+1); 
        double yn = 2.*(y[i]-g.y0())/g.hy() - (double)(2*(m)+1); 

        //evaluate 2d Legendre polynomials at (xn, yn)...
        std::vector<double> px = detail::coefficients( xn, g.n()), 
                            py = detail::coefficients( yn, g.n());
        std::vector<double> pxy( g.n()*g.n());
        for(unsigned k=0; k<py.size(); k++)
            for( unsigned l=0; l<px.size(); l++)
                pxy[k*px.size()+l]= py[k]*px[l];
        //these are the matrix coefficients with which to multiply 
//         unsigned col_begin = m*g.Nx()*g.n()*g.n() + n*g.n()*g.n();
        //detail::add_line( A, number, i,  col_begin, pxy);
        if ( x[i]==g.x0() || x[i]==g.x1()  || y[i]==g.y0()  || y[i]==g.y1())
        {
           for(unsigned k=0; k<py.size(); k++)
            for( unsigned l=0; l<px.size(); l++)
                pxy[k*px.size()+l]= 0; 
            n=(n==(unsigned)floor((g.x1()-g.x0())/g.hx())) ? n-1 :n;
            m=(m==(unsigned)floor((g.y1()-g.y0())/g.hy())) ? m-1 :m;
            
        }
        unsigned col_begin = (m)*g.Nx()*g.n()*g.n() + (n)*g.n();

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

/**
 * @brief Create interpolation matrix
 *
 * Transforms from a vector defined on the grid to the given points 
 * @param x X-coordinates of interpolation points
 * @param y Y-coordinates of interpolation points
 * @param y Z-coordinates of interpolation points
 * @param g The Grid on which to operate
 *
 * @return interpolation matrix
 * @note The values of x, y and z must lie within the boundaries of g
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const thrust::host_vector<double>& x, const thrust::host_vector<double>& y, const thrust::host_vector<double>& z, const Grid3d<double>& g)
{
    assert( x.size() == y.size());
    assert( y.size() == z.size());
    //assert( z.size() == g.size());
    cusp::coo_matrix<int, double, cusp::host_memory> A( x.size(), g.size(), x.size()*g.n()*g.n());

    int number = 0;
    for( unsigned i=0; i<x.size(); i++)
    {
        if (!(x[i] >= g.x0() && x[i] <= g.x1())) {
            std::cerr << g.x0()<<"< xi = " << x[i] <<" < "<<g.x1()<<std::endl;
        }
        assert(x[i] >= g.x0() && x[i] <= g.x1());
        
        if (!(y[i] >= g.y0() && y[i] <= g.y1())) {
            std::cerr << g.y0()<<"< yi = " << y[i] <<" < "<<g.y1()<<std::endl;
        }
        assert( y[i] >= g.y0() && y[i] <= g.y1());
        if (!(z[i] >= g.z0() && z[i] <= g.z1())) {
            std::cerr << g.z0()<<"< zi = " << z[i] <<" < "<<g.z1()<<std::endl;
        }
        assert( z[i] >= g.z0() && z[i] <= g.z1());

        //determine which cell (x,y) lies in 
        unsigned n = (unsigned)floor((x[i]-g.x0())/g.hx());
        unsigned m = (unsigned)floor((y[i]-g.y0())/g.hy());
        unsigned l = (unsigned)floor((z[i]-g.z0())/g.hz());

        //determine normalized coordinates
        double xn = 2.*(x[i]-g.x0())/g.hx() - (double)(2*n+1); 
        double yn = 2.*(y[i]-g.y0())/g.hy() - (double)(2*m+1); 
        double zn = 2.*(z[i]-g.z0())/g.hz() - (double)(2*l+1); 

        //evaluate 2d Legendre polynomials at (xn, yn)...
        std::vector<double> px = detail::coefficients( xn, g.n()), 
                            py = detail::coefficients( yn, g.n()),
                            pz = detail::coefficients( zn, 1 );
        std::vector<double> pxyz( g.n()*g.n());
        for(unsigned k=0; k<g.n(); k++)
            for( unsigned j=0; j<g.n(); j++)
                pxyz[k*g.n()+j]= pz[0]*py[k]*px[j];

        //...these are the matrix coefficients with which to multiply 
        unsigned col_begin = ((l*g.Ny()+ m)*g.Nx()*g.n() + n)*g.n();
        detail::add_line( A, number, i,  col_begin, g.n(), g.Nx(), pxyz);
        //choose layout from comments
    }
    typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;
    
    dg::Operator<double> forward( g.dlt().forward());
    Matrix transformX = dg::tensor( g.Nx(), forward);
    Matrix transformY = dg::tensor( g.Ny(), forward);
    Matrix transformZ = dg::tensor( g.Nz(), delta(1));
    Matrix ward = dg::dgtensor( g.n(), transformY, transformX);
    Matrix ward2 = dg::dgtensor( 1, transformZ, ward);

    Matrix B;
    cusp::multiply( A, ward2, B);
    B.sort_by_row_and_column();
    return B;
}
/**
 * @brief Create interpolation between two grids
 *
 * This matrix can be applied to vectors defined on the old grid to obtain
 * its values on the new grid.
 * 
 * @param g_new The new points 
 * @param g_old The old grid
 *
 * @return Interpolation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const Grid1d<double>& g_new, const Grid1d<double>& g_old)
{
    //assert both grids are on the same box
    assert( g_new.x0() >= g_old.x0());
    assert( g_new.x1() <= g_old.x1());
    thrust::host_vector<double> pointsX = dg::evaluate( dg::coo1, g_new);
    return interpolation( pointsX, g_old);

}
/**
 * @brief Create interpolation between two grids
 *
 * This matrix can be applied to vectors defined on the old grid to obtain
 * its values on the new grid.
 * 
 * @param g_new The new points 
 * @param g_old The old grid
 *
 * @return Interpolation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const Grid2d<double>& g_new, const Grid2d<double>& g_old)
{
    //assert both grids are on the same box
    assert( g_new.x0() >= g_old.x0());
    assert( g_new.x1() <= g_old.x1());
    assert( g_new.y0() >= g_old.y0());
    assert( g_new.y1() <= g_old.y1());
    thrust::host_vector<double> pointsX = dg::evaluate( dg::coo1, g_new);
    thrust::host_vector<double> pointsY = dg::evaluate( dg::coo2, g_new);
    return interpolation( pointsX, pointsY, g_old);

}

/**
 * @brief Create interpolation between two grids
 *
 * This matrix can be applied to vectors defined on the old grid to obtain
 * its values on the new grid.
 * 
 * @param g_new The new points 
 * @param g_old The old grid
 *
 * @return Interpolation matrix
 * @note The boundaries of the old brid must lie within the boundaries of the new grid
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const Grid3d<double>& g_new, const Grid3d<double>& g_old)
{
    //assert both grids are on the same box
    assert( g_new.x0() >= g_old.x0());
    assert( g_new.x1() <= g_old.x1());
    assert( g_new.y0() >= g_old.y0());
    assert( g_new.y1() <= g_old.y1());
    assert( g_new.z0() >= g_old.z0());
    assert( g_new.z1() <= g_old.z1());
    thrust::host_vector<double> pointsX = dg::evaluate( dg::coo1, g_new);
    thrust::host_vector<double> pointsY = dg::evaluate( dg::coo2, g_new);
    thrust::host_vector<double> pointsZ = dg::evaluate( dg::coo3, g_new);
    return interpolation( pointsX, pointsY, pointsZ, g_old);

}
}//namespace create
} //namespace dg
