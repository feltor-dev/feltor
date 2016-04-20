#pragma once
//#include <iomanip>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include "grid.h"
#include "evaluation.cuh"
#include "functions.h"
#include "creation.cuh"
#include "tensor.cuh"
#include "operator_tensor.cuh"

/*! @file

  Contains 1D, 2D and 3D matrix creation functions
  */

namespace dg{
///@addtogroup typedefs
///@{
//interpolation matrices
typedef cusp::csr_matrix<int, double, cusp::host_memory> IHMatrix; //!< CSR host Matrix
typedef cusp::csr_matrix<int, double, cusp::device_memory> IDMatrix; //!< CSR device Matrix
///@}

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
    
/**
 * @brief The n-th Legendre Polynomial on [-1;1]
 */
struct Legendre
{
    Legendre( unsigned n): n_(n+1), m_(0){}
    Legendre( unsigned n, unsigned m): n_(n+1), m_(m+1){}
    double operator()( double x)
    {
        //compute p_i(xn) and return the last value
        std::vector<double> px = coefficients(x, n_);
        return px[n_-1];
    }
    double operator()( double x, double y)
    {
        std::vector<double> px = coefficients(x, n_);
        std::vector<double> py = coefficients(y, m_);
        return px[n_-1]*py[m_-1];
    }
    private:
    unsigned n_, m_;
};


}//namespace detail
///@endcond
///@addtogroup utilities
///@{
/**
 * @brief Create interpolation matrix
 *
 * The matrix, when applied to a vector, interpolates its values to the given coordinates
 * @param x X-coordinates of interpolation points
 * @param g The Grid on which to operate
 *
 * @return interpolation matrix
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const thrust::host_vector<double>& x, const Grid1d<double>& g)
{
    cusp::coo_matrix<int, double, cusp::host_memory> A( x.size(), g.size(), x.size()*g.n());

    int number = 0;
    dg::Operator<double> forward( g.dlt().forward());
    for( unsigned i=0; i<x.size(); i++)
    {
        if (!(x[i] >= g.x0() && x[i] <= g.x1())) {
            std::cerr << "xi = " << x[i] <<std::endl;
        }
        assert(x[i] >= g.x0() && x[i] <= g.x1());

        //determine which cell (x) lies in 
        double xnn = (x[i]-g.x0())/g.h();
        unsigned n = (unsigned)floor(xnn);
        //determine normalized coordinates
        double xn = 2.*xnn - (double)(2*n+1); 
        //intervall correction
        if (n==g.N()) {
            n-=1;
            xn = 1.;
        }
        //evaluate 2d Legendre polynomials at (xn, yn)...
        std::vector<double> px = detail::coefficients( xn, g.n());
        //...these are the matrix coefficients with which to multiply 
        std::vector<double> pxF(px.size(),0);
        for( unsigned l=0; l<g.n(); l++)
            for( unsigned k=0; k<g.n(); k++)
                pxF[l]+= px[k]*forward(k,l);
        unsigned col_begin = n*g.n();
        detail::add_line( A, number, i,  col_begin, pxF);
    }
    return A;
}

/**
 * @brief Create interpolation matrix
 *
 * The matrix, when applied to a vector, interpolates its values to the given coordinates
 * @param x X-coordinates of interpolation points
 * @param y Y-coordinates of interpolation points
 * @param g The Grid on which to operate
 * @param globalbcz NEU for common interpolation. DIR for zeros at Box
 *
 * @return interpolation matrix
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const thrust::host_vector<double>& x, const thrust::host_vector<double>& y, const Grid2d<double>& g , dg::bc globalbcz = dg::NEU)
{
    assert( x.size() == y.size());
    cusp::coo_matrix<int, double, cusp::host_memory> A( x.size(), g.size(), x.size()*g.n()*g.n());

    dg::Operator<double> forward( g.dlt().forward());
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

        //determine which cell (x,y) lies in 

        double xnn = (x[i]-g.x0())/g.hx();
        double ynn = (y[i]-g.y0())/g.hy();
        unsigned n = (unsigned)floor(xnn);
        unsigned m = (unsigned)floor(ynn);
        //determine normalized coordinates

        double xn =  2.*xnn - (double)(2*n+1); 
        double yn =  2.*ynn - (double)(2*m+1); 
        //interval correction
        if (n==g.Nx()) {
            n-=1;
            xn = 1.;
        }
        if (m==g.Ny()) {
            m-=1;
            yn =1.;
        }


        //evaluate 2d Legendre polynomials at (xn, yn)...
        std::vector<double> px = detail::coefficients( xn, g.n()), 
                            py = detail::coefficients( yn, g.n());
        std::vector<double> pxF(g.n(),0), pyF(g.n(), 0);
        for( unsigned l=0; l<g.n(); l++)
            for( unsigned k=0; k<g.n(); k++)
            {
                pxF[l]+= px[k]*forward(k,l);
                pyF[l]+= py[k]*forward(k,l);
            }
        std::vector<double> pxy( g.n()*g.n());
        //these are the matrix coefficients with which to multiply 
        for(unsigned k=0; k<pyF.size(); k++)
            for( unsigned l=0; l<pxF.size(); l++)
                pxy[k*px.size()+l]= pyF[k]*pxF[l];
        if (globalbcz == dg::DIR)
        {
            if ( x[i]==g.x0() || x[i]==g.x1()  || y[i]==g.y0()  || y[i]==g.y1())
//             if ( fabs(x[i]-g.x0())<1e-10 || fabs(x[i]-g.x1())<1e-10  || fabs(y[i]-g.y0())<1e-10  || fabs(y[i]-g.y1())<1e-10)
            {
                //zeroe boundary values 
                for(unsigned k=0; k<py.size(); k++)
                for( unsigned l=0; l<px.size(); l++)
                    pxy[k*px.size()+l]= 0; 
            }
        }
        unsigned col_begin = (m)*g.Nx()*g.n()*g.n() + (n)*g.n();
        detail::add_line( A, number, i,  col_begin, g.n(), g.Nx(), pxy); 
    }
    if (globalbcz == DIR_NEU ) std::cerr << "DIR_NEU NOT IMPLEMENTED "<<std::endl;
    if (globalbcz == NEU_DIR ) std::cerr << "NEU_DIR NOT IMPLEMENTED "<<std::endl;
    if (globalbcz == dg::PER ) std::cerr << "PER NOT IMPLEMENTED "<<std::endl;
    return A;
}



/**
 * @brief Create interpolation matrix
 *
 * The matrix, when applied to a vector, interpolates its values to the given coordinates
 * @param x X-coordinates of interpolation points
 * @param y Y-coordinates of interpolation points
 * @param z Z-coordinates of interpolation points
 * @param g The Grid on which to operate
 * @param globalbcz determines what to do if values lie exactly on the boundary
 *
 * @return interpolation matrix
 * @note The values of x, y and z must lie within the boundaries of g
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const thrust::host_vector<double>& x, const thrust::host_vector<double>& y, const thrust::host_vector<double>& z, const Grid3d<double>& g, dg::bc globalbcz= dg::NEU)
{
    assert( x.size() == y.size());
    assert( y.size() == z.size());
    //assert( z.size() == g.size());
    cusp::coo_matrix<int, double, cusp::host_memory> A( x.size(), g.size(), x.size()*g.n()*g.n());

    dg::Operator<double> forward( g.dlt().forward());
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
        double xnn = (x[i]-g.x0())/g.hx();
        double ynn = (y[i]-g.y0())/g.hy();
        double znn = (z[i]-g.z0())/g.hz();
        unsigned n = (unsigned)floor(xnn);
        unsigned m = (unsigned)floor(ynn);
        unsigned l = (unsigned)floor(znn);
        //n=(n==g.Nx()) ? n-1 :n;
        //m=(m==g.Ny()) ? m-1 :m;
        //l=(l==g.Nz()) ? l-1 :l;

        //determine normalized coordinates
        double xn = 2.*xnn - (double)(2*n+1); 
        double yn = 2.*ynn - (double)(2*m+1); 
        double zn = 2.*znn - (double)(2*l+1); 
        if (n==g.Nx()) {
            n-=1;
            xn = 1.;
        }
        if (m==g.Ny()) {
            m-=1;
            yn =1.;
        }
         if (l==g.Nz()) {
            l-=1;
            zn =1.;
        }
        //evaluate 2d Legendre polynomials at (xn, yn)...
        std::vector<double> px = detail::coefficients( xn, g.n()), 
                            py = detail::coefficients( yn, g.n()),
                            pz = detail::coefficients( zn, 1 );
        std::vector<double> pxF(g.n(),0), pyF(g.n(), 0);
        for( unsigned l=0; l<g.n(); l++)
            for( unsigned k=0; k<g.n(); k++)
            {
                pxF[l]+= px[k]*forward(k,l);
                pyF[l]+= py[k]*forward(k,l);
            }
        std::vector<double> pxyz( g.n()*g.n());
        for(unsigned k=0; k<g.n(); k++)
            for( unsigned j=0; j<g.n(); j++)
                pxyz[k*g.n()+j]= pz[0]*pyF[k]*pxF[j];

        //...these are the matrix coefficients with which to multiply 
        if (globalbcz == dg::DIR)
        {
            if ( x[i]==g.x0() || x[i]==g.x1()  || y[i]==g.y0()  || y[i]==g.y1())
            {
                //zeroe boundary values 
                for(unsigned k=0; k<g.n(); k++)
                    for( unsigned j=0; j<g.n(); j++)
                        pxyz[k*g.n()+j]= 0;
            }
        }
        unsigned col_begin = ((l*g.Ny()+ m)*g.Nx()*g.n() + n)*g.n();
        detail::add_line( A, number, i,  col_begin, g.n(), g.Nx(), pxyz);
        //choose layout from comments
    }
    return A;
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
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
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

thrust::host_vector<double> transform( const Operator<double>& op, const thrust::host_vector<double>& in, const Grid2d<double>& g)
{
    assert( op.size() == g.n());
    thrust::host_vector<double> out(in.size(), 0);
    for( unsigned i=0; i<g.Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
    for( unsigned o=0; o<g.n(); o++)
    for( unsigned m=0; m<g.n(); m++)
        out[((i*g.n() + k)*g.Nx() + j)*g.n() + l] += op(k,o)*op( l, m)*in[((i*g.n() + o)*g.Nx() + j)*g.n() + m];
    return out;
}
thrust::host_vector<double> forward_transform( const thrust::host_vector<double>& in, const Grid2d<double>& g)
{
    dg::Operator<double> forward( g.dlt().forward());
    return transform( forward, in, g);
}

cusp::coo_matrix<int, double, cusp::host_memory> projection( const Grid2d<double>& g_coarse, const Grid2d<double>& g_fine)
{

    assert( g_coarse.x0() >= g_fine.x0());
    assert( g_coarse.x1() <= g_fine.x1());
    assert( g_coarse.y0() >= g_fine.y0());
    assert( g_coarse.y1() <= g_fine.y1());
    assert( g_fine.Nx() % g_coarse.Nx() == 0);
    assert( g_fine.Ny() % g_coarse.Ny() == 0);

    unsigned num_cellsX = g_fine.Nx() / g_coarse.Nx();
    unsigned num_cellsY = g_fine.Ny() / g_coarse.Ny();

    //construct elemental grid with fine number of cells and polynomials
    Grid2d<double> g_elemental( -1., 1., -1., 1., g_fine.n(), num_cellsX, num_cellsY);
    //now evaluate the coarse Legendre polynomials on the fine grid
    thrust::host_vector<double> coeffsX[g_coarse.n()*g_coarse.n()];
    thrust::host_vector<double> coeffsL[g_coarse.n()*g_coarse.n()];
    Operator<double> sisj = dg::create::pipj( g_elemental.n());
    Operator<double> forward( g_elemental.dlt().forward());
    for( unsigned p=0; p<g_coarse.n(); p++) //y
        for( unsigned q=0; q<g_coarse.n(); q++)//x
        {
            detail::Legendre legendre( q, p); 
            coeffsX[p*g_coarse.n()+q] = dg::evaluate( legendre, g_elemental);
            //forward transform coefficients
            coeffsL[p*g_coarse.n()+q] = transform( forward, coeffsX[p*g_coarse.n() + q], g_elemental);

            //multiply by S matrix 
            coeffsX[p*g_coarse.n()+q] = transform( sisj, coeffsL[p*g_coarse.n() + q], g_elemental);
            //std::cout << "p "<<p<<" q "<<q<<"\n";
            //for( unsigned i=0; i<g_fine.n(); i++)
            //{
            //    for( unsigned j=0; j<g_fine.n(); j++)
            //        std::cout << coeffsX[p*g_coarse.n()+q][i*g_fine.n()*num_cellsX+j] <<" ";
            //    std::cout << "\n";
            //}
            //std::cout <<std::endl;
            //multiply by forward transpose
            coeffsL[p*g_coarse.n()+q] = transform( forward.transpose(), coeffsX[p*g_coarse.n() + q], g_elemental);
            
        }
    Grid2d<double> gc_elemental( -1., 1., -1., 1., g_coarse.n(), 1, 1);
    Operator<double> backward( gc_elemental.dlt().backward());
    Operator<double> sisj_inv = dg::create::pipj_inv( gc_elemental.n());
    Operator<double> left = backward*sisj_inv;    
    //multiply left over all coarse polynomials
    for( unsigned k=0; k<g_coarse.n(); k++)
    for( unsigned q=0; q<g_coarse.n(); q++)
    {
        for( unsigned i=0; i<g_elemental.size(); i++)
        {
                coeffsX[k*g_coarse.n()+q][i] = 0.;
                for( unsigned m=0; m<g_coarse.n(); m++)
                for( unsigned l=0; l<g_coarse.n(); l++)
                {
                    coeffsX[k*g_coarse.n()+q][i] += left(k,m)*left(q,l)*coeffsL[m*g_coarse.n()+l][i];
                }
                coeffsX[k*g_coarse.n()+q][i] *= g_fine.hx()*g_fine.hy()/g_coarse.hx()/g_coarse.hy();
        }
    }
    cusp::coo_matrix<int, double, cusp::host_memory> A( g_coarse.size(), g_fine.size(), g_coarse.size()*num_cellsX*num_cellsY*g_fine.n()*g_fine.n());
    int number = 0;
    for( unsigned i=0; i<g_coarse.Ny(); i++)
    for( unsigned k=0; k<g_coarse.n(); k++)
    for( unsigned j=0; j<g_coarse.Nx(); j++)
    for( unsigned q=0; q<g_coarse.n(); q++)
    {
        unsigned line = ((i*g_coarse.n()+k)*g_coarse.Nx()+j)*g_coarse.n()+q;
        //add correct line to A
        for( unsigned m=0; m<num_cellsY; m++)
        for( unsigned n=0; n<num_cellsX; n++)
        {
            //column for indices (i,j,m,n) the (0,0) element
            unsigned col_begin = (((i*num_cellsY+m)*g_fine.n()*g_coarse.Nx()+j)*num_cellsX+n)*g_fine.n();
            std::vector<double> temp(g_fine.n()*g_fine.n());
            for( unsigned e=0; e<g_fine.n(); e++)
                for( unsigned h=0; h<g_fine.n(); h++)
                    temp[e*g_fine.n()+h] = coeffsX[k*g_coarse.n()+q]
                        [((m*g_fine.n() + e)*num_cellsX + n)*g_fine.n()+h];
            detail::add_line( A, number, line,  col_begin, g_fine.n(), g_fine.Nx(), temp); 
        }
    }

    A.sort_by_row_and_column();
    return A;
}

cusp::coo_matrix<int, double, cusp::host_memory> projection( const Grid3d<double>& g_coarse, const Grid3d<double>& g_fine)
{

    assert( g_coarse.z0() >= g_fine.z0());
    assert( g_coarse.z1() <= g_fine.z1());
    assert( g_coarse.Nz() == g_fine.Nz());
    const unsigned Nz = g_coarse.Nz();

    Grid2d<double> g2d_coarse( g_coarse.x0(), g_coarse.x1(), g_coarse.y0(), g_coarse.y1(), g_coarse.n(), g_coarse.Nx(), g_coarse.Ny());
    Grid2d<double> g2d_fine( g_fine.x0(), g_fine.x1(), g_fine.y0(), g_fine.y1(), g_fine.n(), g_fine.Nx(), g_fine.Ny());
    cusp::coo_matrix<int, double, cusp::host_memory> A2d = projection( g2d_coarse, g2d_fine);

    cusp::coo_matrix<int, double, cusp::host_memory> A( A2d.num_rows*Nz, A2d.num_cols*Nz, A2d.num_entries*Nz);
    for( unsigned i=0; i<Nz; i++)
        for( unsigned j=0; j<A2d.num_entries; j++)
        {
            A.column_indices[i*A2d.num_entries+j] = i*A2d.num_cols + A2d.column_indices[j];
            A.row_indices[i*A2d.num_entries+j] = i*A2d.num_rows + A2d.row_indices[j];
            A.values[i*A2d.num_entries+j] = A2d.values[j];
        }
    return A;
}
///@}


}//namespace create

/**
 * @brief Interpolate a single point
 *
 * The matrix, when applied to a vector, interpolates its values to the given coordinates
 * @param x X-coordinate of interpolation point
 * @param y Y-coordinate of interpolation point
 * @param v The vector to interpolate in LSPACE
 * @param g The Grid on which to operate
 *
 * @return interpolated point
 */
double interpolate( double x, double y,  const thrust::host_vector<double>& v, const Grid2d<double>& g )
{
    assert( v.size() == g.size());

    dg::Operator<double> forward( g.dlt().forward());
    if (!(x >= g.x0() && x <= g.x1())) {
        std::cerr << g.x0()<<"< xi = " << x <<" < "<<g.x1()<<std::endl;
    }
    
    assert(x >= g.x0() && x <= g.x1());
    
    if (!(y >= g.y0() && y <= g.y1())) {
        std::cerr << g.y0()<<"< yi = " << y <<" < "<<g.y1()<<std::endl;
    }
    assert( y >= g.y0() && y <= g.y1());

    //determine which cell (x,y) lies in 

    double xnn = (x-g.x0())/g.hx();
    double ynn = (y-g.y0())/g.hy();
    unsigned n = (unsigned)floor(xnn);
    unsigned m = (unsigned)floor(ynn);
    //determine normalized coordinates

    double xn =  2.*xnn - (double)(2*n+1); 
    double yn =  2.*ynn - (double)(2*m+1); 
    //interval correction
    if (n==g.Nx()) {
        n-=1;
        xn = 1.;
    }
    if (m==g.Ny()) {
        m-=1;
        yn =1.;
    }
    //evaluate 2d Legendre polynomials at (xn, yn)...
    std::vector<double> px = create::detail::coefficients( xn, g.n()), 
                        py = create::detail::coefficients( yn, g.n());
    //std::vector<double> pxF(g.n(),0), pyF(g.n(), 0);
    //for( unsigned l=0; l<g.n(); l++)
    //    for( unsigned k=0; k<g.n(); k++)
    //    {
    //        pxF[l]+= px[k]*forward(k,l);
    //        pyF[l]+= py[k]*forward(k,l);
    //    }
    //these are the matrix coefficients with which to multiply 
    unsigned col_begin = (m)*g.Nx()*g.n()*g.n() + (n)*g.n();
    //multiply x 
    double value = 0;
    for( unsigned i=0; i<g.n(); i++)
        for( unsigned j=0; j<g.n(); j++)
            value += v[col_begin + i*g.Nx()*g.n() + j]*px[j]*py[i];
            //value += v[col_begin + i*g.Nx()*g.n() + j]*pxF[j]*pyF[i];
    return value;
}

} //namespace dg
