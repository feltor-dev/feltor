#pragma once
//#include <iomanip>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include "dg/backend/typedefs.h"
#include "dg/backend/view.h"
#include "grid.h"
#include "evaluation.h"
#include "functions.h"
#include "operator_tensor.h"
#include "xspacelib.h"

/*! @file

  @brief Interpolation matrix creation functions
  */

namespace dg{

namespace create{
///@cond
namespace detail{
/*!@class hide_shift_doc
 * @brief Shift any point coordinate to a corresponding grid coordinate according to the boundary condition
 *
 * If the given point is already inside the grid, the function does nothing, else along each dimension the following happens: check the boundary condition.
 *If \c dg::PER, the point will be shifted topologically back onto the domain (modulo operation). Else the
 * point will be mirrored at the closest boundary. If the boundary is a Dirichlet boundary (happens for \c dg::DIR, \c dg::DIR_NEU and \c dg::NEU_DIR; the latter two apply \c dg::DIR to the respective left or right boundary )
 * an additional sign flag is swapped. This process is repeated until the result lies inside the grid. This function forms the basis for extending/periodifying a
 * function discretized on the grid beyond the grid boundaries.
 * @sa interpolate
 * @note For periodic boundaries the right boundary point is considered outside the grid and is shifted to the left boundary point.
 * @param negative swap value if there was a sign swap (happens when a point is mirrored along a Dirichlet boundary)
 * @param x point to shift (inout) the result is guaranteed to lie inside the grid
 */
template<class real_type>
void shift( bool& negative, real_type & x, dg::bc bc, real_type x0, real_type x1)
{
    // This is now a free function because not all Topologies may have it
    if( bc == dg::PER)
    {
        real_type N0 = floor((x-x0)/(x1-x0)); // ... -2[ -1[ 0[ 1[ 2[ ...
        x = x - N0*(x1-x0); //shift
    }
    //mirror along boundary as often as necessary
    while( (x<x0) || (x>x1) )
    {
        if( x < x0){
            x = 2.*x0 - x;
            //every mirror swaps the sign if Dirichlet
            if( bc == dg::DIR || bc == dg::DIR_NEU)
                negative = !negative;//swap sign
        }
        if( x > x1){
            x = 2.*x1 - x;
            if( bc == dg::DIR || bc == dg::NEU_DIR) //notice the different boundary NEU_DIR to the above DIR_NEU !
                negative = !negative; //swap sign
        }
    }
}

/**
 * @brief Evaluate n Legendre poloynomial on given abscissa
 *
 * @param xn normalized x-value on which to evaluate the polynomials: -1<=xn<=1
 * @param n  maximum order of the polynomial
 *
 * @return array of coefficients beginning with p_0(x_n) until p_{n-1}(x_n)
 */
template<class real_type>
std::vector<real_type> legendre( real_type xn, unsigned n)
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
        unsigned points_per_line, double lx,// const RealGrid1d<real_type>& g,
        const thrust::host_vector<real_type>& abs, dg::bc bcx,
        thrust::host_vector<unsigned>& cols)
{
    // Select points for nearest, linear or cubic interpolation
    // lx is needed for PER bondary conditions
    assert( abs.size() >= points_per_line && "There must be more points to interpolate\n");
    //determine which cell (X) lies in
    // abs must be sorted for std::lower_bound to work
    auto it = std::lower_bound( abs.begin(), abs.end(), X);

    std::vector<real_type> xs( points_per_line, 0);
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
                        xs[1] = *(abs.end() -1)-lx;;
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
                        xs[0] = *(abs.begin())+lx;
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

template<class real_type>
void interpolation_row( dg::space sp, real_type X,
    const RealGrid1d<real_type>& g, dg::bc bcx,
    cusp::array1d<int, cusp::host_memory>& cols,
    cusp::array1d<real_type, cusp::host_memory>& vals)
{
    bool negative = false;
    detail::shift( negative, X, bcx, g.x0(), g.x1());

    //determine which cell (x) lies in
    real_type xnn = (X-g.x0())/g.h();
    unsigned nn = (unsigned)floor(xnn);
    //determine normalized coordinates
    real_type xn = 2.*xnn - (real_type)(2*nn+1);
    //intervall correction
    if (nn==g.N()) {
        nn-=1;
        xn = 1.;
    }
    //Test if the point is a Gauss point since then no interpolation is needed
    int idxX =-1;
    std::vector<real_type> gauss_nodesx = dg::DLT<real_type>::abscissas(g.n());
    for( unsigned k=0; k<g.nx(); k++)
    {
        // Due to rounding errors xn will not perfectly be gauss_nodex[k] ( errors ~ 1e-14 possible)
        // even if x are the abscissas of the grid
        if( fabs( xn - gauss_nodesx[k]) < 1e-13)
            idxX = nn*g.nx() + k; //determine which grid column it is
    }
    if( idxX < 0 or sp == dg::lspace) //there is no corresponding point
    {
        //evaluate 1d Legendre polynomials at (xn, yn)...
        std::vector<real_type> px = detail::legendre( xn, g.n());
        //...these are the matrix coefficients with which to multiply
        if( sp == dg::xspace)
        {
            std::vector<real_type> pxF(px.size(), 0);
            std::vector<real_type> forward =
                dg::DLT<real_type>::forward(g.n());
            for( unsigned l=0; l<g.n(); l++)
                for( unsigned k=0; k<g.n(); k++)
                    pxF[l]+= px[k]*forward[k*g.n() + l];
            px.swap( pxF);
        }
        for ( unsigned l=0; l<g.n(); l++)
        {
            cols.push_back( nn*g.n() + l);
            vals.push_back( negative ? -px[l] : px[l]);
        }
    }
    else //the point already exists
    {
        cols.push_back( idxX);
        vals.push_back( negative ? -1. : 1.);
    }
}


// dG interpolation
template<class host_vector, class real_type>
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation1d(
    dg::space sp,
    const host_vector& x, // can be a view...
    const RealGrid1d<real_type>& g,
    dg::bc bcx )
{
    cusp::array1d<real_type, cusp::host_memory> values;
    cusp::array1d<int, cusp::host_memory> row_indices;
    cusp::array1d<int, cusp::host_memory> column_indices;
    auto ptr = x.begin();
    for( unsigned i=0; i<x.size(); i++)
    {
        unsigned size = values.size();
        interpolation_row( sp, *ptr, g, bcx, column_indices, values);
        for( unsigned u=0; u<values.size()-size; u++)
            row_indices.push_back(i);
        ptr++;
    }
    cusp::coo_matrix<int, real_type, cusp::host_memory> A(
            x.size(), g.size(), values.size());
    A.row_indices = row_indices;
    A.column_indices = column_indices;
    A.values = values;
    return A;
}

// nearest, linear or cubic interpolation
template<class host_vector1, class host_vector2 >
cusp::coo_matrix<int, dg::get_value_type<host_vector2>, cusp::host_memory> interpolation1d(
        const host_vector1& x,
        const host_vector2& abs, // must be sorted
        dg::bc bcx, dg::get_value_type<host_vector2> x0, dg::get_value_type<host_vector2> x1,
        std::string method )
{
    using real_type = dg::get_value_type<host_vector2>;
    // boundary condidions for dg::Box likely won't work | Box is now removed
    // from library ...
    cusp::array1d<real_type, cusp::host_memory> values;
    cusp::array1d<int, cusp::host_memory> row_indices;
    cusp::array1d<int, cusp::host_memory> column_indices;
    unsigned points_per_line = 1;
    if( method == "nearest")
        points_per_line = 1;
    else if( method == "linear")
        points_per_line = 2;
    else if( method == "cubic")
        points_per_line = 4;
    else
        throw std::runtime_error( "Interpolation method "+method+" not recognized!\n");
    auto ptr = x.begin();
    for( unsigned i=0; i<x.size(); i++)
    {
        real_type X = *ptr;
        ptr++;
        bool negative = false;
        detail::shift( negative, X, bcx, x0, x1);
        // Test if point already exists since then no interpolation is needed
        int idxX = -1;
        auto it = std::lower_bound( abs.begin(), abs.end(), X);
        if( fabs( X - *it) < 1e-13)
            idxX = it - abs.begin();
        if( it != abs.begin() and fabs(  X - *(it-1)) < 1e-13)
            idxX = (it - abs.begin())-1;
        // THIS IS A VERY BAD IDEA PERFORMANCE WISE
        //for( unsigned u=0; u<abs.size(); u++)
        //    if( fabs( X - abs[u]) <1e-13)
        //        idxX = u;
        if( idxX < 0) //no corresponding point
        {
            thrust::host_vector<unsigned> cols;
            std::vector<real_type> xs  = detail::choose_1d_abscissas( X,
                    points_per_line, x1-x0, abs, bcx, cols);

            std::vector<real_type> px = detail::lagrange( X, xs);
            // px may have size != points_per_line (at boundary)
            for ( unsigned l=0; l<px.size(); l++)
            {
                row_indices.push_back(i);
                column_indices.push_back( cols[l]);
                values.push_back(negative ? -px[l] : px[l]);
            }
        }
        else //the point already exists
        {
            row_indices.push_back(i);
            column_indices.push_back(idxX);
            values.push_back( negative ? -1. : 1.);
        }
    }
    cusp::coo_matrix<int, real_type, cusp::host_memory> A(
            x.size(), abs.size(), values.size());
    A.row_indices = row_indices;
    A.column_indices = column_indices;
    A.values = values;
    return A;
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
 * cubic polynomial. Pay attention that **linear** and **cubic** entail nearest neighbor
 * **communication in mpi**.
 */

/*! @brief Create interpolation matrix of a list of points in given grid
 *
 * The created matrix has \c g.size() columns and <tt>x[0].size()</tt> rows.
 * Per default it uses polynomial interpolation given by the dG polynomials,
 * i.e. the interpolation has order \c g.n(u) in each direction. When applied
 * to a vector the result contains the interpolated values at the given
 * interpolation points. The given boundary conditions determine how
 * interpolation points outside the grid domain are treated.
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj"
 * target="_blank">Introduction to dg methods</a>
 * @tparam Nd Number of dimensions
 * @param x Must be of size \c Nd coordinates of interpolation points
 * (<tt>x[0]</tt> is the list of x-coordinates, <tt>x[1]</tt> is the list of
 * y-coordinates, etc.
 * @param g The Grid on which to operate
 * @copydoc hide_bcx_doc
 * @copydoc hide_method
 */
template<class RecursiveHostVector, class real_type, size_t Nd>
cusp::csr_matrix<int, real_type, cusp::host_memory> interpolation(
        const RecursiveHostVector& x,
        const aRealTopology<real_type, Nd>& g,
        std::array<dg::bc, Nd> bcx,
        std::string method = "dg")
{

    std::array<cusp::csr_matrix<int,real_type,cusp::host_memory>,Nd> axes;
    for( unsigned u=0; u<Nd; u++)
    {
        if( x[u].size() != x[0].size())
            throw dg::Error( dg::Message(_ping_)<<"All coordinate lists must have same size "<<x[0].size());
        if( method == "dg")
        {
            axes[u] = detail::interpolation1d( dg::xspace, x[u], g.grid(u), bcx[u]);
        }
        else
        {
            axes[u] = detail::interpolation1d( x[u], g.abscissas(u), bcx[u], g.p(u), g.q(u), method);
        }
    }
    for( unsigned u=1; u<Nd; u++)
    {
        axes[0] = dg::tensorproduct_cols( axes[u], axes[0]);
    }
    return axes[0];
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
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 * @param x X-coordinates of interpolation points
 * @param g The Grid on which to operate
 * @copydoc hide_bcx_doc
 * @copydoc hide_method
 *
 * @return interpolation matrix
 * @attention removes explicit zeros in the interpolation matrix
 */
template<class host_vector, class real_type, typename = std::enable_if_t<dg::is_vector_v<host_vector>>>
cusp::csr_matrix<int, real_type, cusp::host_memory> interpolation(
        const host_vector& x,
        const RealGrid1d<real_type>& g,
        dg::bc bcx = dg::NEU,
        std::string method = "dg")
{
    dg::View<const host_vector> vx( x.data(), x.size());
    return interpolation(
        std::vector{vx}, g,
        std::array<dg::bc,1>{bcx}, method);
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
 * @snippet interpolation_t.cpp doxygen
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
template<class host_vector, class real_type>
cusp::csr_matrix<int, real_type, cusp::host_memory> interpolation(
        const host_vector& x,
        const host_vector& y,
        const aRealTopology2d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU,
        std::string method = "dg")
{
    dg::View<const host_vector> vx( x.data(), x.size());
    dg::View<const host_vector> vy( y.data(), y.size());
    return interpolation( std::vector{vx,vy}, g, {bcx,bcy}, method);
}



/**
 * @brief Create interpolation matrix
 *
 * The created matrix has \c g.size() columns and \c x.size() rows. Per default
 * it uses polynomial interpolation given by the dG polynomials, i.e. the
 * interpolation has order \c g.n() .
 * When applied to a vector the result contains the interpolated values at the
 * given interpolation points.
 * @snippet interpolation_t.cpp doxygen3d
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
template<class host_vector, class real_type>
cusp::csr_matrix<int, real_type, cusp::host_memory> interpolation(
        const host_vector& x,
        const host_vector& y,
        const host_vector& z,
        const aRealTopology3d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU, dg::bc bcz = dg::PER,
        std::string method = "dg")
{
    dg::View<const host_vector> vx( x.data(), x.size());
    dg::View<const host_vector> vy( y.data(), y.size());
    dg::View<const host_vector> vz( z.data(), z.size());
    return interpolation( std::vector{vx,vy,vz}, g, {bcx,bcy,bcz}, method);
}
/**
 * @brief Create interpolation between two grids
 *
 * This matrix interpolates vectors on the old grid \c g_old to the %Gaussian
 * nodes of the new grid \c g_new. The interpolation is of the order \c
 * g_old.n()
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj"
 * target="_blank">Introduction to dg methods</a>
 * @sa for integer multiples between old and new %grid you may want to consider
 * the dg::create::fast_interpolation %functions
 *
 * @param g_new The new grid
 * @param g_old The old grid. The boundaries of the new grid must lie within
 * the boundaries of the old grid.
 * @copydoc hide_method
 *
 * @return Interpolation matrix with \c g_old.size() columns and \c
 * g_new.size() rows
 * @attention Explicit zeros in the returned matrix are removed
 */
template<class real_type, size_t Nd>
cusp::csr_matrix<int, real_type, cusp::host_memory> interpolation(
    const aRealTopology<real_type,Nd>& g_new,
    const aRealTopology<real_type,Nd>& g_old, std::string method = "dg")
{
    //assert both grids are on the same box
    for( unsigned u=0; u<Nd; u++)
    {
        if( g_new.p(u) < g_old.p(u))
            std::cerr << "ERROR: New grid boundary number "<<u<<" with value "<<g_new.p(u)<<" lies outside old grid "<<g_old.p(u)<<" "<<g_old.p(u)-g_new.p(u)<<"\n";
        assert( g_new.p(u) >= g_old.p(u));
        if( g_new.q(u) > g_old.q(u))
            std::cerr << "ERROR: New grid boundary number "<<u<<" with value "<<g_new.q(u)<<" lies outside old grid "<<g_old.q(u)<<" "<<g_old.q(u)-g_new.q(u)<<"\n";
        assert( g_new.q(u) <= g_old.q(u));
    }
    std::array<cusp::csr_matrix<int,real_type,cusp::host_memory>,Nd> axes;
    for( unsigned u=0; u<Nd; u++)
    {
        if( method == "dg")
        {
            axes[u] = detail::interpolation1d( dg::xspace, g_new.abscissas(u),
                g_old.grid(u), g_old.bc(u));
        }
        else
        {
            axes[u] = detail::interpolation1d( g_new.abscissas(u),
                g_old.abscissas(u), g_old.bc(u), g_old.p(u), g_old.q(u),
                method);
        }
    }
    for( unsigned u=1; u<Nd; u++)
        axes[0] = dg::tensorproduct( axes[u], axes[0]);
    return axes[0];

}
///@}


}//namespace create


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
template<class host_vector, class real_type>
real_type interpolate(
    dg::space sp,
    const host_vector& v,
    real_type x,
    const RealGrid1d<real_type>& g,
    dg::bc bcx = dg::NEU)
{
    assert( v.size() == g.size());
    cusp::array1d<real_type, cusp::host_memory> vals;
    cusp::array1d<int, cusp::host_memory> cols;
    create::detail::interpolation_row( sp, x, g, bcx, cols, vals);
    //multiply x
    real_type value = 0;
    for( unsigned j=0; j<vals.size(); j++)
        value += v[cols[j]]*vals[j];
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
 * @param v The vector to interpolate in dg::xspace, or dg::lspace s.a. \c dg::forward_transform( )
 * @param x X-coordinate of interpolation point
 * @param y Y-coordinate of interpolation point
 * @param g The Grid on which to operate
 * @copydoc hide_bcx_doc
 * @param bcy analogous to \c bcx, applies to y direction
 *
 * @ingroup interpolation
 * @return interpolated point
 */
template<class host_vector, class real_type>
real_type interpolate(
    dg::space sp,
    const host_vector& v,
    real_type x, real_type y,
    const aRealTopology<real_type, 2>& g,
    dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU )
{
    assert( v.size() == g.size());
    cusp::array1d<real_type, cusp::host_memory> valsx;
    cusp::array1d<int, cusp::host_memory> colsx;
    create::detail::interpolation_row( sp, x, g.gx(), bcx, colsx, valsx);
    cusp::array1d<real_type, cusp::host_memory> valsy;
    cusp::array1d<int, cusp::host_memory> colsy;
    create::detail::interpolation_row( sp, y, g.gy(), bcy, colsy, valsy);
    //multiply x
    real_type value = 0;
    for( unsigned i=0; i<valsy.size(); i++)
        for( unsigned j=0; j<valsx.size(); j++)
            value += v[colsy[i]*g.shape(0) + colsx[j]]*valsx[j]*valsy[i];
    return value;
}


} //namespace dg
