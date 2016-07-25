#pragma once

#include "grid_refined.h"
#include "dg/backend/gridX.h"
#include "dg/backend/interpolation.cuh"
#include "cusp/transpose.h"


namespace dg
{
namespace refined
{

/**
 * @brief Constant refinement in one direction 
 */
enum direction
{
    XDIR, //!< x-direction
    YDIR, //!< y-direction
    XYDIR //!< both directions
};

namespace detail
{

/**
 * @brief Create 1d refinement weights for the exponential refinement at a corner
 *
 * @param add_x number of additional cells 
 * @param n number of polynomial coefficients in the grid
 * @param Nx original number of cells
 * @param side 0 is left-side, 1 is right-side, rest is both sides
 *
 * @return A 1d vector of size n*(Nx+add_x) for one-sided refinement and n*(Nx+2*add_x)) for two-sided refinement
 */
thrust::host_vector<double> exponential_ref( const unsigned add_x, const GridX1d<double>& g)
{
    //there are add_x+1 finer cells per refined cell ...
    thrust::host_vector< double> left( n*(Nx_old+add_x), 1), right(left);
    if( add_x == 0) { return left; }
    for( unsigned k=0; k<n; k++)//the original cell and the additional ones
        left[k] = pow( 2, add_x);
    for( unsigned i=0; i<add_x; i++) 
    for( unsigned k=0; k<n; k++)
        left[(i+1)*n+k] = pow( 2, add_x-i);
    //mirror left into right
    for( unsigned i=0; i<right.size(); i++)
        right[i] = left[ (left.size()-1)-i];
    thrust::host_vector< double> both( n*(Nx_old+2*add_x), 1);
    for( unsigned i=0; i<left.size(); i++)
        both[i] *= left[i];
    for( unsigned i=0; i<right.size(); i++)
        both[i+n*add_x] *= right[i];

    if( side == 0) { return left; }
    else if( side == 1) { return right; }
    return both;
}

thrust::host_vector<double> ref_abscissas( double x0, double x1, unsigned n, unsigned Nx_new, thrust::host_vector<double>& weights)
{
    assert( weights.size() == n*Nx_new);
    Grid1d<double> g(x0, x1, n, Nx_new);
    thrust::host_vector<double> boundaries(Nx_new+1), abs(n*Nx_new);
    double lx = x1-x0;
    double Nx_old = 0.;
    for( unsigned i=0; i<weights.size(); i++)
        Nx_old += 1./weights[i]/(double)n;
    boundaries[0] = x0;
    for( unsigned i=0; i<Nx_new; i++)
    {
        boundaries[i+1] = boundaries[i] + lx/Nx_old/weights[n*i];
        for( unsigned j=0; j<n; j++)
        {
            abs[i*n+j] =  (boundaries[i+1]+boundaries[i])/2. + 
                (boundaries[i+1]-boundaries[i])/2.*g.dlt().abscissas()[j];
        }
    }
    return abs;

}

}//namespace detail

/**
 * @brief Refined grid 
 */
struct GridX2d : public dg::GridX2d
{
    /**
     * @brief Refine a corner of a grid
     *
     * @param c
     * @param add_x Add number of cells to the existing one
     * @param add_y Add number of cells to the existing one
     * @param x0
     * @param x1
     * @param y0
     * @param y1
     * @param n
     * @param Nx
     * @param Ny
     * @param bcx
     * @param bcy
     */
    GridX2d( unsigned add_x, unsigned add_y, 
            double x0, double x1, double y0, double y1, 
            double fx, double fy, 
            unsigned n, unsigned Nx, unsigned Ny, bc bcx = dg::PER, bc bcy = dg::PER) : dg::GridX2d<double>( x0, x1, y0, y1, 
                fx_new(Nx, add_x, fx), fy_new(Ny, add_y, fy), n, nx_new(Nx, add_x, fx), ny_new(Ny, add_y, fy), bcx, bcy), 
        g_assoc_( x0, x1, y0, y1, fx, fy, n, Nx, Ny, bcx, bcy)
    {
        wx_.resize( this->size()), wy_.resize( this->size());
        absX_.resize( this->size()), absY_.resize( this->size());
        thrust::host_vector<double> weightsX, weightsY, absX, absY;
        if( bcx != dg::PER && bcy != dg::PER)
        {
            if( c == CORNER_LL)
            {
                 weightsX = detail::exponential_ref( add_x, n, Nx, 0);
                 weightsY = detail::exponential_ref( add_y, n, Ny, 1);
            }
            else if( c == CORNER_LR)
            {
                 weightsX = detail::exponential_ref( add_x, n, Nx, 1);
                 weightsY = detail::exponential_ref( add_y, n, Ny, 1);
            }
            else if( c == CORNER_UR)
            {
                 weightsX = detail::exponential_ref( add_x, n, Nx, 1);
                 weightsY = detail::exponential_ref( add_y, n, Ny, 0);
            }
            else if( c == CORNER_UL)
            {
                 weightsX = detail::exponential_ref( add_x, n, Nx, 0);
                 weightsY = detail::exponential_ref( add_y, n, Ny, 0);
            }
        }
        else if( bcx == dg::PER && bcy != dg::PER)
        {
            if( c == CORNER_LL || c == CORNER_LR)
            {
                 weightsX = detail::exponential_ref( add_x, n, Nx, 2);
                 weightsY = detail::exponential_ref( add_y, n, Ny, 1);
            }
            else if( c == CORNER_UL || c == CORNER_UR)
            {
                 weightsX = detail::exponential_ref( add_x, n, Nx, 2);
                 weightsY = detail::exponential_ref( add_y, n, Ny, 0);
            }
        }
        else if( bcx != dg::PER && bcy == dg::PER)
        {
            if( c == CORNER_LL || c == CORNER_UL)
            {
                 weightsX = detail::exponential_ref( add_x, n, Nx, 0);
                 weightsY = detail::exponential_ref( add_y, n, Ny, 2);
            }
            else if( c == CORNER_UR || c == CORNER_LR)
            {
                 weightsX = detail::exponential_ref( add_x, n, Nx, 1);
                 weightsY = detail::exponential_ref( add_y, n, Ny, 2);
            }
        }
        else if( bcx == dg::PER && bcy == dg::PER)
        {
            weightsX = detail::exponential_ref( add_x, n, Nx, 2);
            weightsY = detail::exponential_ref( add_y, n, Ny, 2);
        }
        absX = detail::ref_abscissas( x0, x1, n, n_new( Nx, add_x, bcx), weightsX);
        absY = detail::ref_abscissas( y0, y1, n, n_new( Ny, add_y, bcy), weightsY);
        for( unsigned i=0; i<weightsY.size(); i++)
        for( unsigned j=0; j<weightsX.size(); j++)
        {
            wx_[i*weightsX.size()+j] = weightsX[j];
            wy_[i*weightsX.size()+j] = weightsY[i];
            absX_[i*weightsX.size()+j] = absX[j];
            absY_[i*weightsX.size()+j] = absY[i];
        }
        //normalize weights
        dg::blas1::scal( wx_, (double)Nx/(double)this->Nx() );
        dg::blas1::scal( wy_, (double)Ny/(double)this->Ny() );
    }

    /**
     * @brief The grid that this object refines
     *
     * @return  2d grid
     */
    dg::GridX2d associated()const {return g_assoc_;}
    /**
     * @brief Return the abscissas in X-direction 
     *
     * @return A 2d vector
     */
    const thrust::host_vector<double>& abscissasX() const {return absX_;} 
    /**
     * @brief Return the abscissas in Y-direction 
     *
     * @return A 2d vector
     */
    const thrust::host_vector<double>& abscissasY() const {return absY_;} 
    /**
     * @brief Return the weights in X-direction 
     *
     * @return A 2d vector
     */
    const thrust::host_vector<double>& weightsX() const {return wx_;} 
    /**
     * @brief Return the weights in Y-direction 
     *
     * @return A 2d vector
     */
    const thrust::host_vector<double>& weightsY() const {return wy_;} 

    private:
    unsigned nx_new( unsigned Nx, unsigned add_x, unsigned fx)
    {
        if( fx==0 ) return Nx; 
        return Nx + 2*add_x;
    }
    unsigned ny_new( unsigned Ny, unsigned add_y, unsigned fy)
    {
        if( fy==0 ) return Ny; 
        return Ny + 4*add_y;
    }
    unsigned fx_new( unsigned Nx, unsigned add_x, unsigned fx)
    {
        if( fx==0 ) 0.; 
        return (fx*(double)Nx + (double)add_x)/((double)Nx+2.*add_x);
    }
    unsigned fy_new( unsigned Ny, unsigned add_y, unsigned fy)
    {
        if( fy==0 ) 0.; 
        return (fy*(double)Ny + (double)add_y)/((double)Ny+4.*add_y);
    }
    thrust::host_vector<double> wx_, wy_; //weights
    thrust::host_vector<double> absX_, absY_; //abscissas 
    dg::GridX2d g_assoc_;
};

struct GridX3d : public dg::GridX3d
{

};
}//namespace refined


cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const dg::refined::Grid2d& g_fine)
{
    dg::Grid2d<double> g = g_fine.associated();
    //determine number of refined cells
    thrust::host_vector<int> idxX = g_fine.idxX();
    thrust::host_vector<int> idxY = g_fine.idxY();
    thrust::host_vector<double> x = g_fine.abscissasX();
    thrust::host_vector<double> y = g_fine.abscissasY();
    
    return interpolation( x,y, g);

}

cusp::coo_matrix<int, double, cusp::host_memory> projection( const dg::refined::Grid2d& g_fine)
{
    cusp::coo_matrix<int, double, cusp::host_memory> temp = interpolation( g_fine), A;
    cusp::transpose( temp, A);
    return A;
}

cusp::coo_matrix<int, double, cusp::host_memory> smoothing( const dg::refined::Grid2d& g)
{
    cusp::coo_matrix<int, double, cusp::host_memory> A = interpolation(g);
    cusp::coo_matrix<int, double, cusp::host_memory> B = projection(g);
    cusp::coo_matrix<int, double, cusp::host_memory> C;
    cusp::multiply( A, B, C);
    C.sort_by_row_and_column();
    return C; 
}

}//namespace dg
