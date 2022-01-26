#pragma once
//#include <iomanip>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include "grid.h"
#include "evaluation.h"
#include "functions.h"
#include "operator_tensor.h"

/*! @file

  @brief 1D, 2D and 3D interpolation matrix creation functions
  */

namespace dg{
///@addtogroup typedefs
///@{
template<class real_type>
using IHMatrix_t = cusp::csr_matrix<int, real_type, cusp::host_memory>;
template<class real_type>
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
//Ell matrix can be almost 3x faster than csr for GPU
//However, sometimes matrices contain outlier rows that do not fit in ell
using IDMatrix_t = cusp::csr_matrix<int, real_type, cusp::device_memory>;
#else
// csr matrix can be much faster than ell for CPU (we have our own symv implementation!)
using IDMatrix_t = cusp::csr_matrix<int, real_type, cusp::device_memory>;
#endif
using IHMatrix = IHMatrix_t<double>;
using IDMatrix = IDMatrix_t<double>;
#ifndef MPI_VERSION
namespace x{
//introduce into namespace x
using IHMatrix = IHMatrix;
using IDMatrix = IDMatrix;
} //namespace x
#endif //MPI_VERSION

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
template<class real_type>
std::vector<real_type> coefficients( real_type xn, unsigned n)
{
    assert( xn <= 1. && xn >= -1.);
    std::vector<real_type> px(n);
    if( xn == -1)
    {
        for( unsigned u=0; u<n; u++)
            px[u] = (u%2 == 0) ? +1. : -1.;
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
                px[i+1] = ((real_type)(2*i+1)*xn*px[i]-(real_type)i*px[i-1])/(real_type)(i+1);
        }
    }
    return px;
}

// evaluate n base polynomials for n given abscissas
template<class real_type>
std::vector<real_type> lagrange( real_type x, const std::vector<real_type>& xi)
{
    unsigned n = xi.size();
    std::vector<real_type> l( n , 1.);
    for( unsigned i=0; i<n; i++)
        for( unsigned k=0; k<n; k++)
        {
            if ( k != i)
                l[i] *= (x-xi[k])/(xi[i]-xi[k]);
        }
    return l;
}

//THERE IS A BUG FOR PERIODIC BC !!
template<class real_type>
std::vector<real_type> choose_1d_abscissas( real_type X,
        unsigned points_per_line, const RealGrid1d<real_type>& g,
        const thrust::host_vector<real_type>& abs,
        thrust::host_vector<unsigned>& cols)
{
    assert( abs.size() >= points_per_line && "There must be more points to interpolate\n");
    dg::bc bcx = g.bcx();
    //determine which cell (X) lies in
    real_type xnn = (X-g.x0())/g.h();
    unsigned n = (unsigned)floor(xnn);
    //intervall correction
    if (n==g.N() && bcx != dg::PER) {
        n-=1;
    }
    // look for closest abscissa
    std::vector<real_type> xs( points_per_line, 0);
    // X <= *it
    auto it = std::lower_bound( abs.begin()+n*g.n(), abs.begin() + (n+1)*g.n(),
            X);
    cols.resize( points_per_line, 0);
    switch( points_per_line)
    {
        case 1: xs[0] = 1.;
                if( it == abs.begin())
                    cols[0] = 0;
                else if( it == abs.end())
                    cols[0] = it - abs.begin() - 1;
                else
                {
                    if ( fabs(X - *it) < fabs( X - *(it-1)))
                        cols[0] = it - abs.begin();
                    else
                        cols[0] = it - abs.begin() -1;
                }
                break;
        case 2: if( it == abs.begin())
                {
                    if( bcx == dg::PER)
                    {
                        xs[0] = *it;
                        xs[1] = *(abs.end() -1)-g.lx();;
                        cols[0] = 0, cols[1] = abs.end()-abs.begin()-1;
                    }
                    else
                    {
                        //xs[0] = *it;
                        //xs[1] = *(it+1);
                        //cols[0] = 0, cols[1] = 1;
                        // This makes it consistent with fem_t
                        xs.resize(1);
                        xs[0] = *it;
                        cols[0] = 0;
                    }
                }
                else if( it == abs.end())
                {
                    if( bcx == dg::PER)
                    {
                        xs[0] = *(abs.begin())+g.lx();
                        xs[1] = *(it-1);
                        cols[0] = 0, cols[1] = it-abs.begin()-1;
                    }
                    else
                    {
                        //xs[0] = *(it-2);
                        //xs[0] = *(it-1);
                        //cols[0] = it - abs.begin() - 2;
                        //cols[1] = it - abs.begin() - 1;
                        // This makes it consistent with fem_t
                        xs.resize(1);
                        xs[0] = *(it-1);
                        cols[0] = it-abs.begin()-1;
                    }
                }
                else
                {
                    xs[0] = *(it-1);
                    xs[1] = *it;
                    cols[0] = it - abs.begin() - 1;
                    cols[1] = cols[0]+1;
                }
                break;
        case 4: if( it <= abs.begin() +1)
                {
                    if( bcx == dg::PER)
                    {
                        xs[0] = *abs.begin(), cols[0] = 0;
                        xs[1] = *(abs.begin()+1), cols[1] = 1;
                        xs[2] = it == abs.begin() ? *(abs.end() -2) : *(abs.begin()+2);
                        cols[2] = it == abs.begin() ? abs.end()-abs.begin() -2 : 2;
                        xs[3] = *(abs.end() -1);
                        cols[3] = abs.end()-abs.begin() -1;
                    }
                    else
                    {
                        it = abs.begin();
                        xs[0] = *it,     xs[1] = *(it+1);
                        xs[2] = *(it+2), xs[3] = *(it+3);
                        cols[0] = 0, cols[1] = 1;
                        cols[2] = 2, cols[3] = 3;
                    }
                }
                else if( it >= abs.end() -1)
                {
                    if( bcx == dg::PER)
                    {
                        xs[0] = *abs.begin(), cols[0] = 0;
                        xs[1] = it == abs.end() ? *(abs.begin()+1) : *(abs.end() -3) ;
                        cols[1] = it == abs.end() ? 1 :  abs.end()-abs.begin()-3 ;
                        xs[2] = *(abs.end() - 2), cols[2] = abs.end()-abs.begin()-2;
                        xs[3] = *(abs.end() - 1), cols[3] = abs.end()-abs.begin()-1;
                    }
                    else
                    {
                        it = abs.end();
                        xs[0] = *(it-4), xs[1] = *(it-3);
                        xs[2] = *(it-2), xs[3] = *(it-1);
                        cols[0] = it - abs.begin() - 4;
                        cols[1] = cols[0]+1;
                        cols[2] = cols[1]+1;
                        cols[3] = cols[2]+1;
                    }
                }
                else
                {
                    xs[0] = *(it-2), xs[1] = *(it-1);
                    xs[2] = *(it  ), xs[3] = *(it+1);
                    cols[0] = it - abs.begin() - 2;
                    cols[1] = cols[0]+1;
                    cols[2] = cols[1]+1;
                    cols[3] = cols[2]+1;
                }
                break;
    }
    return xs;
}

}//namespace detail
///@endcond
///@addtogroup interpolation
///@{
/*!@class hide_bcx_doc
 * @param bcx determines what to do when a point lies outside the boundary in x. If \c dg::PER, the point will be shifted topologically back onto the domain. Else the
 * point will be mirrored at the boundary: \c dg::NEU will then simply interpolate at the resulting point, \c dg::DIR will take the negative of the interpolation.
 (\c dg::DIR_NEU and \c dg::NEU_DIR apply \c dg::NEU / \c dg::DIR to the respective left or right boundary )
 * This means the result of the interpolation is as if the interpolated function were Fourier transformed with the correct boundary condition and thus extended beyond the grid boundaries.
 * Note that if a point lies directly on the boundary between two grid cells, the value of the polynomial to the right is taken.
*/
/*!@class hide_method
 * @param method Several interpolation methods are available: **dg** uses the native
 * dG interpolation scheme given by the grid, **nearest** searches for the
 * nearest point and copies its value, **linear** searches for the two (in 2d
 * four, etc.) closest points and linearly interpolates their values, **cubic**
 * searches for the four (in 2d 16, etc) closest points and interpolates a
 * cubic polynomial
 */

/**
 * @brief Create interpolation matrix
 *
 * The created matrix has \c g.size() columns and \c x.size() rows. Per default
 * it uses polynomial interpolation given by the dG polynomials, i.e. the
 * interpolation has order \c g.n() .
 * When applied to a vector the result contains the interpolated values at the
 * given interpolation points.  The given boundary conditions determine how
 * interpolation points outside the grid domain are treated.
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @param x X-coordinates of interpolation points
 * @param g The Grid on which to operate
 * @copydoc hide_bcx_doc
 * @copydoc hide_method
 *
 * @return interpolation matrix
 * @attention does **not** remove explicit zeros in the interpolation matrix
 */
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation(
        const thrust::host_vector<real_type>& x,
        const RealGrid1d<real_type>& g,
        dg::bc bcx = dg::NEU,
        std::string method = "dg")
{
    cusp::array1d<real_type, cusp::host_memory> values;
    cusp::array1d<int, cusp::host_memory> row_indices;
    cusp::array1d<int, cusp::host_memory> column_indices;
    if( method == "dg")
    {
        dg::Operator<real_type> forward( g.dlt().forward());
        for( unsigned i=0; i<x.size(); i++)
        {
            real_type X = x[i];
            bool negative = false;
            g.shift( negative, X, bcx);

            //determine which cell (x) lies in
            real_type xnn = (X-g.x0())/g.h();
            unsigned n = (unsigned)floor(xnn);
            //determine normalized coordinates
            real_type xn = 2.*xnn - (real_type)(2*n+1);
            //intervall correction
            if (n==g.N()) {
                n-=1;
                xn = 1.;
            }
            //evaluate 2d Legendre polynomials at (xn, yn)...
            std::vector<real_type> px = detail::coefficients( xn, g.n());
            //...these are the matrix coefficients with which to multiply
            std::vector<real_type> pxF(px.size(),0);
            for( unsigned l=0; l<g.n(); l++)
                for( unsigned k=0; k<g.n(); k++)
                    pxF[l]+= px[k]*forward(k,l);
            unsigned cols = n*g.n();
            for ( unsigned l=0; l<g.n(); l++)
            {
                row_indices.push_back(i);
                column_indices.push_back( cols + l);
                values.push_back(negative ? -pxF[l] : pxF[l]);
            }
        }
    }
    else
    {
        unsigned points_per_line = 1;
        if( method == "nearest")
            points_per_line = 1;
        else if( method == "linear")
            points_per_line = 2;
        else if( method == "cubic")
            points_per_line = 4;
        else
            throw std::runtime_error( "Interpolation method "+method+" not recognized!\n");
        thrust::host_vector<real_type> abs = dg::create::abscissas( g);
        dg::RealGrid1d<real_type> gx( g.x0(), g.x1(), g.n(), g.N(), bcx);
        for( unsigned i=0; i<x.size(); i++)
        {
            real_type X = x[i];
            bool negative = false;
            g.shift( negative, X, bcx);

            thrust::host_vector<unsigned> cols;
            std::vector<real_type> xs  = detail::choose_1d_abscissas( X,
                    points_per_line, gx, abs, cols);

            std::vector<real_type> px = detail::lagrange( X, xs);
            // px may have size != points_per_line (at boundary)
            for ( unsigned l=0; l<px.size(); l++)
            {
                row_indices.push_back(i);
                column_indices.push_back( cols[l]);
                values.push_back(negative ? -px[l] : px[l]);
            }
        }
    }
    cusp::coo_matrix<int, real_type, cusp::host_memory> A(
            x.size(), g.size(), values.size());
    A.row_indices = row_indices;
    A.column_indices = column_indices;
    A.values = values;
    return A;
}

/**
 * @brief Create interpolation matrix
 *
 * The created matrix has \c g.size() columns and \c x.size() rows. Per default
 * it uses polynomial interpolation given by the dG polynomials, i.e. the
 * interpolation has order \c g.n() .
 * When applied to a vector the result contains the interpolated values at the
 * given interpolation points.  The given boundary conditions determine how
 * interpolation points outside the grid domain are treated.
 * @snippet topology/interpolation_t.cu doxygen
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @param x X-coordinates of interpolation points
 * @param y Y-coordinates of interpolation points (\c y.size() must equal \c x.size())
 * @param g The Grid on which to operate
 * @copydoc hide_bcx_doc
 * @param bcy analogous to \c bcx, applies to y direction
 * @copydoc hide_method
 *
 * @return interpolation matrix
 * @attention removes explicit zeros in the interpolation matrix
 */
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation(
        const thrust::host_vector<real_type>& x,
        const thrust::host_vector<real_type>& y,
        const aRealTopology2d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU,
        std::string method = "dg")
{
    assert( x.size() == y.size());
    cusp::array1d<real_type, cusp::host_memory> values;
    cusp::array1d<int, cusp::host_memory> row_indices;
    cusp::array1d<int, cusp::host_memory> column_indices;
    if( method == "dg")
    {
        std::vector<real_type> gauss_nodesx = g.dltx().abscissas();
        std::vector<real_type> gauss_nodesy = g.dlty().abscissas();
        dg::Operator<real_type> forwardx( g.dltx().forward());
        dg::Operator<real_type> forwardy( g.dlty().forward());


        for( int i=0; i<(int)x.size(); i++)
        {
            real_type X = x[i], Y = y[i];
            bool negative=false;
            g.shift( negative,X,Y, bcx, bcy);

            //determine which cell (x,y) lies in
            real_type xnn = (X-g.x0())/g.hx();
            real_type ynn = (Y-g.y0())/g.hy();
            unsigned nn = (unsigned)floor(xnn);
            unsigned mm = (unsigned)floor(ynn);
            //determine normalized coordinates
            real_type xn =  2.*xnn - (real_type)(2*nn+1);
            real_type yn =  2.*ynn - (real_type)(2*mm+1);
            //interval correction
            if (nn==g.Nx()) {
                nn-=1;
                xn =1.;
            }
            if (mm==g.Ny()) {
                mm-=1;
                yn =1.;
            }
            //Test if the point is a Gauss point since then no interpolation is needed
            int idxX =-1, idxY = -1;
            for( unsigned k=0; k<g.nx(); k++)
            {
                if( fabs( xn - gauss_nodesx[k]) < 1e-14)
                    idxX = nn*g.nx() + k; //determine which grid column it is
            }
            for( unsigned k=0; k<g.ny(); k++)
            {
                if( fabs( yn - gauss_nodesy[k]) < 1e-14)
                    idxY = mm*g.ny() + k;  //determine grid line
            }
            if( idxX < 0 && idxY < 0 ) //there is no corresponding point
            {
                //evaluate 2d Legendre polynomials at (xn, yn)...
                std::vector<real_type> px = detail::coefficients( xn, g.nx()),
                                       py = detail::coefficients( yn, g.ny());
                std::vector<real_type> pxF(g.nx(),0), pyF(g.ny(), 0);
                for( unsigned l=0; l<g.nx(); l++)
                    for( unsigned k=0; k<g.nx(); k++)
                        pxF[l]+= px[k]*forwardx(k,l);
                for( unsigned l=0; l<g.ny(); l++)
                    for( unsigned k=0; k<g.ny(); k++)
                        pyF[l]+= py[k]*forwardy(k,l);
                //these are the matrix coefficients with which to multiply
                for(unsigned k=0; k<g.ny(); k++)
                    for( unsigned l=0; l<g.nx(); l++)
                    {
                        row_indices.push_back( i);
                        column_indices.push_back( ((mm*g.ny()+k)*g.Nx()+nn)*g.nx() + l);
                        real_type pxy = pyF[k]*pxF[l];
                        if( !negative)
                            values.push_back(  pxy);
                        else
                            values.push_back( -pxy);
                    }
            }
            else if ( idxX < 0 && idxY >=0) //there is a corresponding line
            {
                std::vector<real_type> px = detail::coefficients( xn, g.nx());
                std::vector<real_type> pxF(g.nx(),0);
                for( unsigned l=0; l<g.nx(); l++)
                    for( unsigned k=0; k<g.nx(); k++)
                        pxF[l]+= px[k]*forwardx(k,l);
                for( unsigned l=0; l<g.nx(); l++)
                {
                    row_indices.push_back( i);
                    column_indices.push_back( ((idxY)*g.Nx() + nn)*g.nx() + l);
                    if( !negative)
                        values.push_back( pxF[l]);
                    else
                        values.push_back(-pxF[l]);

                }
            }
            else if ( idxX >= 0 && idxY < 0) //there is a corresponding column
            {
                std::vector<real_type> py = detail::coefficients( yn, g.ny());
                std::vector<real_type> pyF(g.ny(),0);
                for( unsigned l=0; l<g.ny(); l++)
                    for( unsigned k=0; k<g.ny(); k++)
                        pyF[l]+= py[k]*forwardy(k,l);
                for( unsigned k=0; k<g.ny(); k++)
                {
                    row_indices.push_back(i);
                    column_indices.push_back((mm*g.ny()+k)*g.Nx()*g.nx() + idxX);
                    if( !negative)
                        values.push_back( pyF[k]);
                    else
                        values.push_back(-pyF[k]);

                }
            }
            else //the point already exists
            {
                row_indices.push_back(i);
                column_indices.push_back(idxY*g.Nx()*g.nx() + idxX);
                if( !negative)
                    values.push_back( 1.);
                else
                    values.push_back(-1.);
            }

        }
    }
    else
    {
        unsigned points_per_line = 1;
        if( method == "nearest")
            points_per_line = 1;
        else if( method == "linear")
            points_per_line = 2;
        else if( method == "cubic")
            points_per_line = 4;
        else
            throw std::runtime_error( "Interpolation method "+method+" not recognized!\n");
        RealGrid1d<real_type> gx(g.x0(), g.x1(), g.nx(), g.Nx(), bcx);
        RealGrid1d<real_type> gy(g.y0(), g.y1(), g.ny(), g.Ny(), bcy);
        thrust::host_vector<real_type> absX = dg::create::abscissas( gx);
        thrust::host_vector<real_type> absY = dg::create::abscissas( gy);

        for( unsigned i=0; i<x.size(); i++)
        {
            real_type X = x[i], Y = y[i];
            bool negative = false;
            g.shift( negative, X, Y, bcx, bcy);

            thrust::host_vector<unsigned> colsX, colsY;
            std::vector<real_type> xs  = detail::choose_1d_abscissas( X,
                    points_per_line, gx, absX, colsX);
            std::vector<real_type> ys  = detail::choose_1d_abscissas( Y,
                    points_per_line, gy, absY, colsY);

            //evaluate 2d Legendre polynomials at (xn, yn)...
            std::vector<real_type> pxy( points_per_line*points_per_line);
            std::vector<real_type> px = detail::lagrange( X, xs),
                                   py = detail::lagrange( Y, ys);
            // note: px , py may have size != points_per_line at boundary
            for(unsigned k=0; k<py.size(); k++)
                for( unsigned l=0; l<px.size(); l++)
                    pxy[k*px.size()+l]= py[k]*px[l];
            for( unsigned k=0; k<py.size(); k++)
                for( unsigned l=0; l<px.size(); l++)
                {
                    if( fabs(pxy[k*px.size() +l]) > 1e-14)
                    {
                        row_indices.push_back( i);
                        column_indices.push_back( (colsY[k])*g.nx()*g.Nx() +
                            colsX[l]);
                        values.push_back( negative ? - pxy[k*px.size()+l]
                                :  pxy[k*px.size()+l]);
                    }
                }
        }
    }
    cusp::coo_matrix<int, real_type, cusp::host_memory> A( x.size(),
            g.size(), values.size());
    A.row_indices = row_indices;
    A.column_indices = column_indices;
    A.values = values;

    return A;
}



/**
 * @brief Create interpolation matrix
 *
 * The created matrix has \c g.size() columns and \c x.size() rows. Per default
 * it uses polynomial interpolation given by the dG polynomials, i.e. the
 * interpolation has order \c g.n() .
 * When applied to a vector the result contains the interpolated values at the
 * given interpolation points.
 * @snippet topology/interpolation_t.cu doxygen3d
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @param x X-coordinates of interpolation points
 * @param y Y-coordinates of interpolation points (\c y.size() must equal \c x.size())
 * @param z Z-coordinates of interpolation points (\c z.size() must equal \c x.size())
 * @param g The Grid on which to operate
 * @copydoc hide_bcx_doc
 * @param bcy analogous to \c bcx, applies to y direction
 * @param bcz analogous to \c bcx, applies to z direction
 * @copydoc hide_method
 *
 * @return interpolation matrix
 * @attention removes explicit zeros from the interpolation matrix
 * @attention all points (x, y, z) must lie within or on the boundaries of g
 */
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation(
        const thrust::host_vector<real_type>& x,
        const thrust::host_vector<real_type>& y,
        const thrust::host_vector<real_type>& z,
        const aRealTopology3d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU, dg::bc bcz = dg::PER,
        std::string method = "dg")
{
    assert( x.size() == y.size());
    assert( y.size() == z.size());
    cusp::array1d<real_type, cusp::host_memory> values;
    cusp::array1d<int, cusp::host_memory> row_indices;
    cusp::array1d<int, cusp::host_memory> column_indices;

    if( method == "dg")
    {
        std::vector<real_type> gauss_nodesx = g.dltx().abscissas();
        std::vector<real_type> gauss_nodesy = g.dlty().abscissas();
        std::vector<real_type> gauss_nodesz = g.dltz().abscissas();
        dg::Operator<real_type> forwardx( g.dltx().forward());
        dg::Operator<real_type> forwardy( g.dlty().forward());
        dg::Operator<real_type> forwardz( g.dltz().forward());
        for( int i=0; i<(int)x.size(); i++)
        {
            real_type X = x[i], Y = y[i], Z = z[i];
            bool negative = false;
            g.shift( negative,X,Y,Z, bcx, bcy, bcz);

            //determine which cell (x,y,z) lies in
            real_type xnn = (X-g.x0())/g.hx();
            real_type ynn = (Y-g.y0())/g.hy();
            real_type znn = (Z-g.z0())/g.hz();
            unsigned nn = (unsigned)floor(xnn);
            unsigned mm = (unsigned)floor(ynn);
            unsigned ll = (unsigned)floor(znn);
            //determine normalized coordinates
            real_type xn = 2.*xnn - (real_type)(2*nn+1);
            real_type yn = 2.*ynn - (real_type)(2*mm+1);
            real_type zn = 2.*znn - (real_type)(2*ll+1);
            //interval correction
            if (nn==g.Nx()) {
                nn-=1;
                xn =1.;
            }
            if (mm==g.Ny()) {
                mm-=1;
                yn =1.;
            }
            if (ll==g.Nz()) {
                ll-=1;
                zn =1.;
            }
            //Test if the point is a Gauss point since then no interpolation is needed
            int idxX =-1, idxY = -1, idxZ = -1;
            for( unsigned k=0; k<g.nx(); k++)
            {
                if( fabs( xn - gauss_nodesx[k]) < 1e-14)
                    idxX = nn*g.nx() + k; //determine which grid column it is
            }
            for( unsigned k=0; k<g.ny(); k++)
            {
                if( fabs( yn - gauss_nodesy[k]) < 1e-14)
                    idxY = mm*g.ny() + k;  //determine grid line
            }
            for( unsigned k=0; k<g.nz(); k++)
            {
                if( fabs( zn - gauss_nodesz[k]) < 1e-14)
                    idxZ = ll*g.nz() + k;  //determine grid line
            }
            if( idxX >= 0 && idxY >= 0 && idxZ >= 0) //the point already exists
            {
                row_indices.push_back(i);
                column_indices.push_back((idxZ*g.Ny()*g.ny()+idxY)*g.Nx()*g.nx() + idxX);
                if( !negative)
                    values.push_back( 1.);
                else
                    values.push_back(-1.);
            }
            else if ( idxX < 0 && idxY >=0 && idxZ >= 0)
            {
                std::vector<real_type> px = detail::coefficients( xn, g.nx());
                std::vector<real_type> pxF(g.nx(),0);
                for( unsigned l=0; l<g.nx(); l++)
                    for( unsigned k=0; k<g.nx(); k++)
                        pxF[l]+= px[k]*forwardx(k,l);
                for( unsigned l=0; l<g.nx(); l++)
                {
                    row_indices.push_back( i);
                    column_indices.push_back( (idxZ*g.Ny()*g.ny() +
                                idxY)*g.Nx()*g.nx() + nn*g.nx() + l);
                    if( !negative)
                        values.push_back( pxF[l]);
                    else
                        values.push_back(-pxF[l]);
                }
            }
            else if ( idxX >= 0 && idxY < 0 && idxZ >= 0)
            {
                std::vector<real_type> py = detail::coefficients( yn, g.ny());
                std::vector<real_type> pyF(g.ny(),0);
                for( unsigned l=0; l<g.ny(); l++)
                    for( unsigned k=0; k<g.ny(); k++)
                        pyF[l]+= py[k]*forwardy(k,l);
                for( unsigned k=0; k<g.ny(); k++)
                {
                    row_indices.push_back(i);
                    column_indices.push_back(((idxZ*g.Ny()+mm)*g.ny()+k)*g.Nx()*g.nx() + idxX);
                    if(!negative)
                        values.push_back( pyF[k]);
                    else
                        values.push_back(-pyF[k]);
                }
            }
            else
            {
                //evaluate 3d Legendre polynomials at (xn, yn, zn)...
                std::vector<real_type> px = detail::coefficients( xn, g.nx()),
                                       py = detail::coefficients( yn, g.ny()),
                                       pz = detail::coefficients( zn, g.nz());
                std::vector<real_type> pxF(g.nx(),0), pyF(g.ny(), 0), pzF( g.nz(), 0);
                for( unsigned l=0; l<g.nx(); l++)
                    for( unsigned k=0; k<g.nx(); k++)
                        pxF[l]+= px[k]*forwardx(k,l);
                for( unsigned l=0; l<g.ny(); l++)
                    for( unsigned k=0; k<g.ny(); k++)
                        pyF[l]+= py[k]*forwardy(k,l);
                for( unsigned l=0; l<g.nz(); l++)
                    for( unsigned k=0; k<g.nz(); k++)
                        pzF[l]+= pz[k]*forwardz(k,l);
                //these are the matrix coefficients with which to multiply
                for( unsigned s=0; s<g.nz(); s++)
                for( unsigned k=0; k<g.ny(); k++)
                for( unsigned l=0; l<g.nx(); l++)
                {
                    row_indices.push_back( i);
                    column_indices.push_back(
                        ((((ll*g.nz()+s)*g.Ny()+mm)*g.ny()+k)*g.Nx()+nn)*g.nx()+l);
                    real_type pxyz = pzF[s]*pyF[k]*pxF[l];
                    if( !negative)
                        values.push_back( pxyz);
                    else
                        values.push_back(-pxyz);
                }
            }
        }
    }
    else
    {
        unsigned points_per_line = 1;
        if( method == "nearest")
            points_per_line = 1;
        else if( method == "linear")
            points_per_line = 2;
        else if( method == "cubic")
            points_per_line = 4;
        else
            throw std::runtime_error( "Interpolation method "+method+" not recognized!\n");
        RealGrid1d<real_type> gx(g.x0(), g.x1(), g.nx(), g.Nx(), bcx);
        RealGrid1d<real_type> gy(g.y0(), g.y1(), g.ny(), g.Ny(), bcy);
        RealGrid1d<real_type> gz(g.z0(), g.z1(), g.nz(), g.Nz(), bcz);
        thrust::host_vector<real_type> absX = dg::create::abscissas( gx);
        thrust::host_vector<real_type> absY = dg::create::abscissas( gy);
        thrust::host_vector<real_type> absZ = dg::create::abscissas( gz);
        for( unsigned i=0; i<x.size(); i++)
        {
            real_type X = x[i], Y = y[i], Z = z[i];
            bool negative = false;
            g.shift( negative, X, Y, Z, bcx, bcy, bcz);

            thrust::host_vector<unsigned> colsX, colsY, colsZ;
            std::vector<real_type> xs  = detail::choose_1d_abscissas( X,
                    points_per_line, gx, absX, colsX);
            std::vector<real_type> ys  = detail::choose_1d_abscissas( Y,
                    points_per_line, gy, absY, colsY);
            std::vector<real_type> zs  = detail::choose_1d_abscissas( Z,
                    points_per_line, gz, absZ, colsZ);

            //evaluate 3d Legendre polynomials at (xn, yn, zn)...
            std::vector<real_type> pxyz( points_per_line*points_per_line
                    *points_per_line);
            std::vector<real_type> px = detail::lagrange( X, xs),
                                   py = detail::lagrange( Y, ys),
                                   pz = detail::lagrange( Z, zs);
            // note: px, py, pz may have size != points_per_line at boundary
            for( unsigned m=0; m<pz.size(); m++)
            for( unsigned k=0; k<py.size(); k++)
            for( unsigned l=0; l<px.size(); l++)
                pxyz[(m*py.size()+k)*px.size()+l]= pz[m]*py[k]*px[l];
            for( unsigned m=0; m<pz.size(); m++)
            for( unsigned k=0; k<py.size(); k++)
            for( unsigned l=0; l<px.size(); l++)
            {
                if( fabs(pxyz[(m*py.size()+k)*px.size() +l]) > 1e-14)
                {
                    row_indices.push_back( i);
                    column_indices.push_back( ((colsZ[m])*g.ny()*g.Ny() +
                                colsY[k])*g.nx()*g.Nx() + colsX[l]);
                    values.push_back( negative ?
                            -pxyz[(m*py.size()+k)*px.size()+l]
                          :  pxyz[(m*py.size()+k)*px.size()+l] );
                }
            }
        }
    }
    cusp::coo_matrix<int, real_type, cusp::host_memory> A( x.size(), g.size(),
            values.size());
    A.row_indices = row_indices;
    A.column_indices = column_indices;
    A.values = values;

    return A;
}
/**
 * @brief Create interpolation between two grids
 *
 * This matrix interpolates vectors on the old grid \c g_old to the %Gaussian nodes of the new grid \c g_new. The interpolation is of the order \c g_old.n()
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @sa for integer multiples between old and new %grid you may want to consider the dg::create::fast_interpolation %functions
 *
 * @param g_new The new grid
 * @param g_old The old grid
 *
 * @return Interpolation matrix with \c g_old.size() columns and \c g_new.size() rows
 * @attention The 1d version does **not** remove explicit zeros from the
 * interpolation matrix, but the 2d and 3d versions do
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 * @note When interpolating a 2d grid to a 3d grid the third coordinate is simply ignored, i.e. the 2d vector will be trivially copied Nz times into the 3d vector
 * @note also check the transformation matrix, which is the more general solution
 */
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const RealGrid1d<real_type>& g_new, const RealGrid1d<real_type>& g_old)
{
    //assert both grids are on the same box
    assert( g_new.x0() >= g_old.x0());
    assert( g_new.x1() <= g_old.x1());
    thrust::host_vector<real_type> pointsX = dg::evaluate( dg::cooX1d, g_new);
    return interpolation( pointsX, g_old);

}
///@copydoc interpolation(const RealGrid1d<real_type>&,const RealGrid1d<real_type>&)
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const aRealTopology2d<real_type>& g_new, const aRealTopology2d<real_type>& g_old)
{
    //assert both grids are on the same box
    assert( g_new.x0() >= g_old.x0());
    assert( g_new.x1() <= g_old.x1());
    assert( g_new.y0() >= g_old.y0());
    assert( g_new.y1() <= g_old.y1());
    thrust::host_vector<real_type> pointsX = dg::evaluate( dg::cooX2d, g_new);

    thrust::host_vector<real_type> pointsY = dg::evaluate( dg::cooY2d, g_new);
    return interpolation( pointsX, pointsY, g_old);

}

///@copydoc interpolation(const RealGrid1d<real_type>&,const RealGrid1d<real_type>&)
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const aRealTopology3d<real_type>& g_new, const aRealTopology3d<real_type>& g_old)
{
    //assert both grids are on the same box
    assert( g_new.x0() >= g_old.x0());
    assert( g_new.x1() <= g_old.x1());
    assert( g_new.y0() >= g_old.y0());
    assert( g_new.y1() <= g_old.y1());
    assert( g_new.z0() >= g_old.z0());
    assert( g_new.z1() <= g_old.z1());
    thrust::host_vector<real_type> pointsX = dg::evaluate( dg::cooX3d, g_new);
    thrust::host_vector<real_type> pointsY = dg::evaluate( dg::cooY3d, g_new);
    thrust::host_vector<real_type> pointsZ = dg::evaluate( dg::cooZ3d, g_new);
    return interpolation( pointsX, pointsY, pointsZ, g_old);

}
///@copydoc interpolation(const RealGrid1d<real_type>&,const RealGrid1d<real_type>&)
template<class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation( const aRealTopology3d<real_type>& g_new, const aRealTopology2d<real_type>& g_old)
{
    //assert both grids are on the same box
    assert( g_new.x0() >= g_old.x0());
    assert( g_new.x1() <= g_old.x1());
    assert( g_new.y0() >= g_old.y0());
    assert( g_new.y1() <= g_old.y1());
    thrust::host_vector<real_type> pointsX = dg::evaluate( dg::cooX3d, g_new);
    thrust::host_vector<real_type> pointsY = dg::evaluate( dg::cooY3d, g_new);
    return interpolation( pointsX, pointsY, g_old);

}
///@}


}//namespace create

/**
 * @brief Transform a vector from dg::xspace (nodal values) to dg::lspace (modal values)
 *
 * @param in input
 * @param g grid
 *
 * @ingroup misc
 * @return the vector in LSPACE
 */
template<class real_type>
thrust::host_vector<real_type> forward_transform( const thrust::host_vector<real_type>& in, const aRealTopology2d<real_type>& g)
{
    thrust::host_vector<real_type> out(in.size(), 0);
    dg::Operator<real_type> forwardx( g.dltx().forward());
    dg::Operator<real_type> forwardy( g.dlty().forward());
    for( unsigned i=0; i<g.Ny(); i++)
    for( unsigned k=0; k<g.ny(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.nx(); l++)
    for( unsigned o=0; o<g.ny(); o++)
    for( unsigned m=0; m<g.nx(); m++)
        out[((i*g.ny() + k)*g.Nx() + j)*g.nx() + l] +=
            forwardy(k,o)*forwardx( l, m)*in[((i*g.ny() + o)*g.Nx() + j)*g.nx() + m];
    return out;
}


/**
 * @brief Interpolate a vector on a single point on a 1d Grid
 *
 * @param sp Indicate whether the elements of the vector
 * v are in xspace (nodal values) or lspace (modal values)
 *  (choose dg::xspace if you don't know what is going on here,
 *      It is faster to interpolate in dg::lspace so consider
 *      transforming v using dg::forward_transform( )
 *      if you do it very many times)
 * @param v The vector to interpolate
 * @param x X-coordinate of interpolation point
 * @param g The Grid on which to operate
 * @copydoc hide_bcx_doc
 *
 * @ingroup interpolation
 * @return interpolated point
 */
template<class real_type>
real_type interpolate(
    dg::space sp,
    const thrust::host_vector<real_type>& v,
    real_type x,
    const RealGrid1d<real_type>& g,
    dg::bc bcx = dg::NEU)
{
    assert( v.size() == g.size());
    bool negative = false;
    g.shift( negative, x, bcx);

    //determine which cell (x) lies in

    real_type xnn = (x-g.x0())/g.h();
    unsigned n = (unsigned)floor(xnn);
    //determine normalized coordinates

    real_type xn =  2.*xnn - (real_type)(2*n+1);
    //interval correction
    if (n==g.N()) {
        n-=1;
        xn = 1.;
    }
    //evaluate 1d Legendre polynomials at (xn)...
    std::vector<real_type> px = create::detail::coefficients( xn, g.n());
    if( sp == dg::xspace)
    {
        dg::Operator<real_type> forward( g.dlt().forward());
        std::vector<real_type> pxF(g.n(),0);
        for( unsigned l=0; l<g.n(); l++)
            for( unsigned k=0; k<g.n(); k++)
                pxF[l]+= px[k]*forward(k,l);
        for( unsigned k=0; k<g.n(); k++)
            px[k] = pxF[k];
    }
    //these are the matrix coefficients with which to multiply
    unsigned cols = (n)*g.n();
    //multiply x
    real_type value = 0;
    for( unsigned j=0; j<g.n(); j++)
    {
        if(negative)
            value -= v[cols + j]*px[j];
        else
            value += v[cols + j]*px[j];
    }
    return value;
}

/**
 * @brief Interpolate a vector on a single point on a 2d Grid
 *
 * @param sp Indicate whether the elements of the vector
 * v are in xspace (nodal values) or lspace  (modal values)
 *  (choose dg::xspace if you don't know what is going on here,
 *      It is faster to interpolate in dg::lspace so consider
 *      transforming v using dg::forward_transform( )
 *      if you do it very many times)
 * @param v The vector to interpolate in dg::xspace, or dg::lspace s.a. dg::forward_transform( )
 * @param x X-coordinate of interpolation point
 * @param y Y-coordinate of interpolation point
 * @param g The Grid on which to operate
 * @copydoc hide_bcx_doc
 * @param bcy analogous to \c bcx, applies to y direction
 *
 * @ingroup interpolation
 * @return interpolated point
 */
template<class real_type>
real_type interpolate(
    dg::space sp,
    const thrust::host_vector<real_type>& v,
    real_type x, real_type y,
    const aRealTopology2d<real_type>& g,
    dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU )
{
    assert( v.size() == g.size());
    bool negative = false;
    g.shift( negative, x,y, bcx, bcy);

    //determine which cell (x,y) lies in

    real_type xnn = (x-g.x0())/g.hx();
    real_type ynn = (y-g.y0())/g.hy();
    unsigned n = (unsigned)floor(xnn);
    unsigned m = (unsigned)floor(ynn);
    //determine normalized coordinates

    real_type xn =  2.*xnn - (real_type)(2*n+1);
    real_type yn =  2.*ynn - (real_type)(2*m+1);
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
    std::vector<real_type> px = create::detail::coefficients( xn, g.nx()),
                           py = create::detail::coefficients( yn, g.ny());
    if( sp == dg::xspace)
    {
        dg::Operator<real_type> forwardx( g.dltx().forward());
        dg::Operator<real_type> forwardy( g.dlty().forward());
        std::vector<real_type> pxF(g.nx(),0), pyF(g.ny(), 0);
        for( unsigned l=0; l<g.nx(); l++)
            for( unsigned k=0; k<g.nx(); k++)
                pxF[l]+= px[k]*forwardx(k,l);
        for( unsigned l=0; l<g.ny(); l++)
            for( unsigned k=0; k<g.ny(); k++)
                pyF[l]+= py[k]*forwardy(k,l);
        px = pxF, py = pyF;
    }
    //these are the matrix coefficients with which to multiply
    unsigned cols = (m)*g.Nx()*g.ny()*g.nx() + (n)*g.nx();
    //multiply x
    real_type value = 0;
    for( unsigned i=0; i<g.ny(); i++)
        for( unsigned j=0; j<g.nx(); j++)
        {
            if(negative)
                value -= v[cols + i*g.Nx()*g.nx() + j]*px[j]*py[i];
            else
                value += v[cols + i*g.Nx()*g.nx() + j]*px[j]*py[i];
        }
    return value;
}

} //namespace dg
