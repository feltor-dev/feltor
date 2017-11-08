#pragma once
//#include <iomanip>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include "grid.h"
#include "evaluation.cuh"
#include "functions.h"
#include "creation.cuh"
#include "dgtensor.cuh"
#include "operator_tensor.cuh"

/*! @file

  @brief contains 1D, 2D and 3D interpolation matrix creation functions
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

}//namespace detail
///@endcond
///@addtogroup interpolation
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
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const thrust::host_vector<double>& x, const Grid1d& g)
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
 * @param bcx determines what to do when a point lies exactly on the boundary in x:  DIR generates zeroes in the interpolation matrix, 
 NEU and PER interpolate the inner side polynomial. (DIR_NEU and NEU_DIR apply NEU / DIR to the respective left or right boundary )
 * @param bcy determines what to do when a point lies exactly on the boundary in y. Behaviour correponds to bcx.
 * @attention all points (x,y) must lie within or on the boundaries of g.
 *
 * @return interpolation matrix
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const thrust::host_vector<double>& x, const thrust::host_vector<double>& y, const aTopology2d& g , dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU)
{
    assert( x.size() == y.size());
    std::vector<double> gauss_nodes = g.dlt().abscissas(); 
    dg::Operator<double> forward( g.dlt().forward());
    cusp::array1d<double, cusp::host_memory> values;
    cusp::array1d<int, cusp::host_memory> row_indices;
    cusp::array1d<int, cusp::host_memory> column_indices;

    for( int i=0; i<(int)x.size(); i++)
    {
        //assert that point is inside the grid boundaries
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
        unsigned nn = (unsigned)floor(xnn);
        unsigned mm = (unsigned)floor(ynn);
        //determine normalized coordinates
        double xn =  2.*xnn - (double)(2*nn+1); 
        double yn =  2.*ynn - (double)(2*mm+1); 
        //interval correction
        if (nn==g.Nx()) {
            nn-=1;
            xn = 1.;
        }
        if (mm==g.Ny()) {
            mm-=1;
            yn =1.;
        }
        //Test if the point is a Gauss point since then no interpolation is needed
        int idxX =-1, idxY = -1;
        for( unsigned k=0; k<g.n(); k++)
        {
            if( fabs( xn - gauss_nodes[k]) < 1e-14)
                idxX = nn*g.n() + k; //determine which grid column it is
            if( fabs( yn - gauss_nodes[k]) < 1e-14)
                idxY = mm*g.n() + k;  //determine grid line
        }
        if( idxX < 0 && idxY < 0 ) //there is no corresponding point
        {
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
            if (  (x[i] == g.x0() && (bcx==dg::DIR || bcx==dg::DIR_NEU) )
                ||(x[i] == g.x1() && (bcx==dg::DIR || bcx==dg::NEU_DIR) )
                ||(y[i] == g.y0() && (bcy==dg::DIR || bcy==dg::DIR_NEU) )
                ||(y[i] == g.y1() && (bcy==dg::DIR || bcy==dg::NEU_DIR) ))
            {
                //zeroe boundary values 
                for(unsigned k=0; k<py.size(); k++)
                for( unsigned l=0; l<px.size(); l++)
                    pxy[k*px.size()+l]= 0; 
            }
            for( unsigned k=0; k<g.n(); k++)
                for( unsigned l=0; l<g.n(); l++)
                {
                    row_indices.push_back( i);
                    column_indices.push_back( (mm*g.n()+k)*g.n()*g.Nx()+nn*g.n() + l);
                    values.push_back( pxy[k*g.n()+l]);
                }
        }
        else if ( idxX < 0 && idxY >=0) //there is a corresponding line
        {
            std::vector<double> px = detail::coefficients( xn, g.n());
            std::vector<double> pxF(g.n(),0);
            for( unsigned l=0; l<g.n(); l++)
                for( unsigned k=0; k<g.n(); k++)
                    pxF[l]+= px[k]*forward(k,l);
            for( unsigned l=0; l<g.n(); l++)
            {
                row_indices.push_back( i);
                column_indices.push_back( (idxY)*g.Nx()*g.n() + nn*g.n() + l);
                values.push_back( pxF[l]);
            }
        }
        else if ( idxX >= 0 && idxY < 0) //there is a corresponding column
        {
            std::vector<double> py = detail::coefficients( yn, g.n());
            std::vector<double> pyF(g.n(),0);
            for( unsigned l=0; l<g.n(); l++)
                for( unsigned k=0; k<g.n(); k++)
                    pyF[l]+= py[k]*forward(k,l);
            for( unsigned k=0; k<g.n(); k++)
            {
                row_indices.push_back(i);
                column_indices.push_back((mm*g.n()+k)*g.Nx()*g.n() + idxX);
                values.push_back(pyF[k]);
            }
        }
        else //the point already exists
        {
            row_indices.push_back(i);
            column_indices.push_back(idxY*g.Nx()*g.n() + idxX); 
            values.push_back(1.);
        }

    }
    cusp::coo_matrix<int, double, cusp::host_memory> A( x.size(), g.size(), values.size());
    A.row_indices = row_indices; A.column_indices = column_indices; A.values = values;

    return A;
}



/**
 * @brief Create interpolation matrix
 *
 * The matrix, when applied to a vector, interpolates its values to the given coordinates. In z-direction only a nearest neighbor interpolation is used
 * @param x X-coordinates of interpolation points
 * @param y Y-coordinates of interpolation points
 * @param z Z-coordinates of interpolation points
 * @param g The Grid on which to operate
 * @param bcx determines what to do when a point lies exactly on the boundary in x:  DIR generates zeroes in the interpolation matrix, 
 NEU and PER interpolate the inner side polynomial. (DIR_NEU and NEU_DIR apply NEU / DIR to the respective left or right boundary )
 * @param bcy determines what to do when a point lies exactly on the boundary in y. Behaviour correponds to bcx.
 *
 * @return interpolation matrix
 * @attention all points (x, y, z) must lie within or on the boundaries of g
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const thrust::host_vector<double>& x, const thrust::host_vector<double>& y, const thrust::host_vector<double>& z, const aTopology3d& g, dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU)
{
    assert( x.size() == y.size());
    assert( y.size() == z.size());
    std::vector<double> gauss_nodes = g.dlt().abscissas(); 
    dg::Operator<double> forward( g.dlt().forward());
    cusp::array1d<double, cusp::host_memory> values;
    cusp::array1d<int, cusp::host_memory> row_indices;
    cusp::array1d<int, cusp::host_memory> column_indices;

    for( int i=0; i<(int)x.size(); i++)
    {
        //assert that point is inside the grid boundaries
        if (!(x[i] >= g.x0() && x[i] <= g.x1())) {
            std::cerr << g.x0()<<"< xi = " << x[i] <<" < "<<g.x1()<<std::endl;
        } assert(x[i] >= g.x0() && x[i] <= g.x1());
        if (!(y[i] >= g.y0() && y[i] <= g.y1())) {
            std::cerr << g.y0()<<"< yi = " << y[i] <<" < "<<g.y1()<<std::endl;
        } assert( y[i] >= g.y0() && y[i] <= g.y1());
        if (!(z[i] >= g.z0() && z[i] <= g.z1())) {
            std::cerr << g.z0()<<"< zi = " << z[i] <<" < "<<g.z1()<<std::endl;
        } assert( z[i] >= g.z0() && z[i] <= g.z1());

        //determine which cell (x,y) lies in 
        double xnn = (x[i]-g.x0())/g.hx();
        double ynn = (y[i]-g.y0())/g.hy();
        double znn = (z[i]-g.z0())/g.hz();
        unsigned nn = (unsigned)floor(xnn);
        unsigned mm = (unsigned)floor(ynn);
        unsigned ll = (unsigned)floor(znn);
        //determine normalized coordinates
        double xn = 2.*xnn - (double)(2*nn+1); 
        double yn = 2.*ynn - (double)(2*mm+1); 
        //interval correction
        if (nn==g.Nx()) {
            nn-=1;
            xn = 1.;
        }
        if (mm==g.Ny()) {
            mm-=1;
            yn =1.;
        }
        if (ll==g.Nz()) {
            ll-=1;
        }
        //Test if the point is a Gauss point since then no interpolation is needed
        int idxX =-1, idxY = -1;
        for( unsigned k=0; k<g.n(); k++)
        {
            if( fabs( xn - gauss_nodes[k]) < 1e-14)
                idxX = nn*g.n() + k; //determine which grid column it is
            if( fabs( yn - gauss_nodes[k]) < 1e-14)
                idxY = mm*g.n() + k;  //determine grid line
        } //in z-direction we don't interpolate
        if( idxX < 0 && idxY < 0 ) //there is no corresponding point
        {
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
            std::vector<double> pxyz( g.n()*g.n());
            //these are the matrix coefficients with which to multiply 
            for(unsigned k=0; k<pyF.size(); k++)
                for( unsigned l=0; l<pxF.size(); l++)
                    pxyz[k*g.n()+l]= 1.*pyF[k]*pxF[l];
            if (  (x[i] == g.x0() && (bcx==dg::DIR || bcx==dg::DIR_NEU) )
                ||(x[i] == g.x1() && (bcx==dg::DIR || bcx==dg::NEU_DIR) )
                ||(y[i] == g.y0() && (bcy==dg::DIR || bcy==dg::DIR_NEU) )
                ||(y[i] == g.y1() && (bcy==dg::DIR || bcy==dg::NEU_DIR) ))
            {
                //zeroe boundary values 
                for(unsigned k=0; k<g.n(); k++)
                for(unsigned l=0; l<g.n(); l++)
                    pxyz[k*g.n()+l]= 0; 
            }
            for( unsigned k=0; k<g.n(); k++)
                for( unsigned l=0; l<g.n(); l++)
                {
                    row_indices.push_back( i);
                    column_indices.push_back( ((ll*g.Ny()+mm)*g.n()+k)*g.n()*g.Nx()+nn*g.n() + l);
                    values.push_back( pxyz[k*g.n()+l]);
                }
        }
        else if ( idxX < 0 && idxY >=0) //there is a corresponding line
        {
            std::vector<double> px = detail::coefficients( xn, g.n());
            std::vector<double> pxF(g.n(),0);
            for( unsigned l=0; l<g.n(); l++)
                for( unsigned k=0; k<g.n(); k++)
                    pxF[l]+= px[k]*forward(k,l);
            for( unsigned l=0; l<g.n(); l++)
            {
                row_indices.push_back( i);
                column_indices.push_back( (ll*g.Ny()*g.n() + idxY)*g.Nx()*g.n() + nn*g.n() + l);
                values.push_back( pxF[l]);
            }
        }
        else if ( idxX >= 0 && idxY < 0) //there is a corresponding column
        {
            std::vector<double> py = detail::coefficients( yn, g.n());
            std::vector<double> pyF(g.n(),0);
            for( unsigned l=0; l<g.n(); l++)
                for( unsigned k=0; k<g.n(); k++)
                    pyF[l]+= py[k]*forward(k,l);
            for( unsigned k=0; k<g.n(); k++)
            {
                row_indices.push_back(i);
                column_indices.push_back(((ll*g.Ny()+mm)*g.n()+k)*g.Nx()*g.n() + idxX);
                values.push_back(pyF[k]);
            }
        }
        else //the point already exists
        {
            row_indices.push_back(i);
            column_indices.push_back((ll*g.Ny()*g.n()+idxY)*g.Nx()*g.n() + idxX); 
            values.push_back(1.);
        }

    }
    cusp::coo_matrix<int, double, cusp::host_memory> A( x.size(), g.size(), values.size());
    A.row_indices = row_indices; A.column_indices = column_indices; A.values = values;

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
 * @note also check the transformation matrix, which is the more general solution
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const Grid1d& g_new, const Grid1d& g_old)
{
    //assert both grids are on the same box
    assert( g_new.x0() >= g_old.x0());
    assert( g_new.x1() <= g_old.x1());
    thrust::host_vector<double> pointsX = dg::evaluate( dg::cooX1d, g_new);
    return interpolation( pointsX, g_old);

}
///@copydoc interpolation(const Grid1d&,const Grid1d&)
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const aTopology2d& g_new, const aTopology2d& g_old)
{
    //assert both grids are on the same box
    assert( g_new.x0() >= g_old.x0());
    assert( g_new.x1() <= g_old.x1());
    assert( g_new.y0() >= g_old.y0());
    assert( g_new.y1() <= g_old.y1());
    thrust::host_vector<double> pointsX = dg::evaluate( dg::cooX2d, g_new);

    thrust::host_vector<double> pointsY = dg::evaluate( dg::cooY2d, g_new);
    return interpolation( pointsX, pointsY, g_old);

}

///@copydoc interpolation(const Grid1d&,const Grid1d&)
cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const aTopology3d& g_new, const aTopology3d& g_old)
{
    //assert both grids are on the same box
    assert( g_new.x0() >= g_old.x0());
    assert( g_new.x1() <= g_old.x1());
    assert( g_new.y0() >= g_old.y0());
    assert( g_new.y1() <= g_old.y1());
    assert( g_new.z0() >= g_old.z0());
    assert( g_new.z1() <= g_old.z1());
    thrust::host_vector<double> pointsX = dg::evaluate( dg::cooX3d, g_new);
    thrust::host_vector<double> pointsY = dg::evaluate( dg::cooY3d, g_new);
    thrust::host_vector<double> pointsZ = dg::evaluate( dg::cooZ3d, g_new);
    return interpolation( pointsX, pointsY, pointsZ, g_old);

}
///@}


/**
 * @brief Transform a vector from XSPACE to LSPACE
 *
 * @param in input
 * @param g grid
 *
 * @ingroup misc
 * @return the vector in LSPACE
 */
thrust::host_vector<double> forward_transform( const thrust::host_vector<double>& in, const aTopology2d& g)
{
    thrust::host_vector<double> out(in.size(), 0);
    dg::Operator<double> forward( g.dlt().forward());
    for( unsigned i=0; i<g.Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
    for( unsigned m=0; m<g.n(); m++)
    for( unsigned o=0; o<g.n(); o++)
        out[((i*g.n() + k)*g.Nx() + j)*g.n() + l] += forward(k,o)*forward( l, m)*in[((i*g.n() + o)*g.Nx() + j)*g.n() + m];
    return out;
}

}//namespace create

/**
 * @brief Interpolate a single point
 *
 * @param x X-coordinate of interpolation point
 * @param y Y-coordinate of interpolation point
 * @param v The vector to interpolate in LSPACE, s.a. dg::forward_transform( )
 * @param g The Grid on which to operate
 *
 * @ingroup interpolation
 * @return interpolated point
 * @note g.contains(x,y) must return true
 */
double interpolate( double x, double y,  const thrust::host_vector<double>& v, const aTopology2d& g )
{
    assert( v.size() == g.size());

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
    //dg::Operator<double> forward( g.dlt().forward());
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
