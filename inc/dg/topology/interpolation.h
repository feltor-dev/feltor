#pragma once
//#include <iomanip>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include "dg/backend/typedefs.h"
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
    // abs must be sorted for std::lower_bound to work
    assert( abs.size() >= points_per_line && "There must be more points to interpolate\n");
    //determine which cell (X) lies in
    //real_type xnn = (X-g.x0())/g.h();
    //unsigned n = (unsigned)floor(xnn);
    ////intervall correction
    //if (n==g.N() && bcx != dg::PER) {
    //    n-=1;
    //}
    // look for closest abscissa
    std::vector<real_type> xs( points_per_line, 0);
    // X <= *it
    auto it = std::lower_bound( abs.begin(), abs.end(), X);
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
cusp::coo_matrix<int, real_type, cusp::host_memory> interpolation1d(
        const thrust::host_vector<real_type>& x,
        const RealGrid1d<real_type>& g,
        dg::bc bcx )
{
    cusp::array1d<real_type, cusp::host_memory> values;
    cusp::array1d<int, cusp::host_memory> row_indices;
    cusp::array1d<int, cusp::host_memory> column_indices;
    std::vector<real_type> gauss_nodesx = dg::DLT<real_type>::abscissas(g.n());
    dg::Operator<real_type> forward = dg::DLT<real_type>::forward(g.n());
    for( unsigned i=0; i<x.size(); i++)
    {
        real_type X = x[i];
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
        for( unsigned k=0; k<g.nx(); k++)
        {
            if( fabs( xn - gauss_nodesx[k]) < 1e-14)
                idxX = nn*g.nx() + k; //determine which grid column it is
        }
        if( idxX < 0 ) //there is no corresponding point
        {
            //evaluate 2d Legendre polynomials at (xn, yn)...
            std::vector<real_type> px = detail::legendre( xn, g.n());
            //...these are the matrix coefficients with which to multiply
            std::vector<real_type> pxF(px.size(),0);
            for( unsigned l=0; l<g.n(); l++)
                for( unsigned k=0; k<g.n(); k++)
                    pxF[l]+= px[k]*forward(k,l);
            unsigned cols = nn*g.n();
            for ( unsigned l=0; l<g.n(); l++)
            {
                row_indices.push_back(i);
                column_indices.push_back( cols + l);
                values.push_back(negative ? -pxF[l] : pxF[l]);
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
            x.size(), g.size(), values.size());
    A.row_indices = row_indices;
    A.column_indices = column_indices;
    A.values = values;
    return A;
}

template<class host_vector >
cusp::coo_matrix<int, dg::get_value_type<host_vector>, cusp::host_memory> interpolation1d(
        const host_vector& x,
        const host_vector& abs, // must be sorted
        dg::bc bcx, dg::get_value_type<host_vector> x0, dg::get_value_type<host_vector> x1,
        std::string method )
{
    using real_type = dg::get_value_type<host_vector>;
    // boundary condidions for dg::Box likely won't work
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
    for( unsigned i=0; i<x.size(); i++)
    {
        real_type X = x[i];
        bool negative = false;
        detail::shift( negative, X, bcx, x0, x1);
        // Test if point already exists since then no interpolation is needed
        int idxX = -1;
        for( unsigned u=0; u<abs.size(); u++)
            if( fabs( X - abs[u]) <1e-14)
                idxX = u;
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

template<class real_type, size_t Nd>
cusp::csr_matrix<int, real_type, cusp::host_memory> interpolation(
        const std::array<thrust::host_vector<real_type>,Nd>& x,
        const aRealTopology<real_type, Nd>& g,
        std::array<dg::bc, Nd> bcs,
        std::string method = "dg")
{

    std::array<cusp::csr_matrix<int,real_type,cusp::host_memory>,Nd> axes;
    const auto& abs = g.get_abscissas();
    for( unsigned u=0; u<Nd; u++)
    {
        if( x[u].size() != x[0].size())
            throw dg::Error( dg::Message(_ping_)<<"All coordinate lists must have same size "<<x[0].size());
        if( method == "dg")
        {
            axes[u] = detail::interpolation1d( x[u], g.grid(u), bcs[u]);
        }
        else
        {
            axes[u] = detail::interpolation1d( x[u], abs[u], bcs[u], g.p(u), g.q(u), method);
        }
    }
    for( unsigned u=1; u<Nd; u++)
    {
        axes[0] = dg::tensorproduct_cols( axes[u], axes[0]);
    }
    return axes[0];
}

template<class host_vector, size_t Nd>
cusp::csr_matrix<int, dg::get_value_type<host_vector>, cusp::host_memory> interpolation(
        const std::array<host_vector,Nd>& x,
        const Box<host_vector, Nd>& g,
        std::string method = "linear")
{
    using real_type = dg::get_value_type<host_vector>;
    std::array<cusp::csr_matrix<int,real_type,cusp::host_memory>,Nd> axes;
    const auto& abs = g.get_abscissas();
    for( unsigned u=0; u<Nd; u++)
    {
        if( x[u].size() != x[0].size())
            throw dg::Error( dg::Message(_ping_)<<"All coordinate lists must have same size "<<x[0].size());
        axes[u] = detail::interpolation1d( x[u], abs(u), dg::NEU,
            abs[u][0], abs[u][abs[u].size()-1], method);
    }
    for( unsigned u=1; u<Nd; u++)
        axes[0] = dg::tensorproduct_cols( axes[u], axes[0]);
    return axes[0];
}
///@cond
///@endcond

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
template<class real_type>
cusp::csr_matrix<int, real_type, cusp::host_memory> interpolation(
        const thrust::host_vector<real_type>& x,
        const RealGrid1d<real_type>& g,
        dg::bc bcx = dg::NEU,
        std::string method = "dg")
{
    // The explicit conversion to std::array prevents the function from calling itself indefinitely
    return interpolation( std::array<thrust::host_vector<real_type>,1>{x}, g, std::array<dg::bc,1>{bcx}, method);
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
cusp::csr_matrix<int, real_type, cusp::host_memory> interpolation(
        const thrust::host_vector<real_type>& x,
        const thrust::host_vector<real_type>& y,
        const aRealTopology2d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU,
        std::string method = "dg")
{
    return interpolation( {x,y}, g, {bcx,bcy}, method);
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
cusp::csr_matrix<int, real_type, cusp::host_memory> interpolation(
        const thrust::host_vector<real_type>& x,
        const thrust::host_vector<real_type>& y,
        const thrust::host_vector<real_type>& z,
        const aRealTopology3d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU, dg::bc bcz = dg::PER,
        std::string method = "dg")
{
    return interpolation( {x,y,z}, g, {bcx,bcy,bcz}, method);
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
 * @copydoc hide_method
 *
 * @return Interpolation matrix with \c g_old.size() columns and \c g_new.size() rows
 * @attention Explicit zeros in the returned matrix are removed
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 * @note When interpolating a 2d grid to a 3d grid the third coordinate is simply ignored, i.e. the 2d vector will be trivially copied Nz times into the 3d vector
 * @note also check the transformation matrix, which is the more general solution
 */
template<class real_type, size_t Nd>
cusp::csr_matrix<int, real_type, cusp::host_memory> interpolation(
    const aRealTopology<real_type,Nd>& g_new,
    const aRealTopology<real_type,Nd>& g_old, std::string method = "dg")
{
    //assert both grids are on the same box
    for( unsigned u=0; u<Nd; u++)
    {
        assert( g_new.p(u) >= g_old.p(u));
        assert( g_new.q(u) <= g_old.q(u));
    }
    auto x = g_new.get_abscissas();
    const auto& abs = g_old.get_abscissas();
    std::array<cusp::csr_matrix<int,real_type,cusp::host_memory>,Nd> axes;
    for( unsigned u=0; u<Nd; u++)
    {
        if( method == "dg")
        {
            axes[u] = detail::interpolation1d( x[u], g_old.grid(u), g_old.bc(u));
        }
        else
        {
            axes[u] = detail::interpolation1d( x[u], abs[u], g_old.bc(u), g_old.p(u), g_old.q(u), method);
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
    create::detail::shift( negative, x, bcx, g.x0(), g.x1());

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
    std::vector<real_type> px = create::detail::legendre( xn, g.n());
    if( sp == dg::xspace)
    {
        dg::Operator<real_type> forward = dg::DLT<real_type>::forward(g.n());
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
    create::detail::shift( negative, x, bcx, g.x0(), g.x1());
    create::detail::shift( negative, y, bcy, g.y0(), g.y1());

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
    std::vector<real_type> px = create::detail::legendre( xn, g.nx()),
                           py = create::detail::legendre( yn, g.ny());
    if( sp == dg::xspace)
    {
        dg::Operator<real_type> forwardx = dg::DLT<real_type>::forward(g.nx());
        dg::Operator<real_type> forwardy = dg::DLT<real_type>::forward(g.ny());
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
