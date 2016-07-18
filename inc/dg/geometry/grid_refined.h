#pragma once

#include "dg/backend/grid.h"
#include "dg/backend/interpolation.cuh"


namespace dg
{
namespace refined
{

/**
 * @brief Corner mode of grid - refinement using exponential refinement
 */
enum corner
{
    CORNER_LL, //!< lower left corner
    CORNER_LR, //!< lower right corner
    CORNER_UL, //!< upper left corner
    CORNER_UR //!< upper right corner
};

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
thrust::host_vector<double> exponential_ref( unsigned add_x, unsigned n, unsigned Nx, int side)
{
    thrust::host_vector< double> left( n*(Nx+add_x), 1), right(left);
    for( unsigned k=0; k<n; k++)//the original cell and the additional ones
        left[k] = pow( 2, add_x);
    for( unsigned i=0; i<add_x; i++) 
    for( unsigned k=0; k<n; k++)
        left[(i+1)*n+k] = pow( 2, add_x-i);
    //mirror left into right
    for( unsigned i=0; i<right.size(); i++)
        right[i] = left[ (left.size()-1)-i];
    thrust::host_vector< double> both( n*(Nx+2*add_x), 1);
    for( unsigned i=0; i<left.size(); i++)
        both[i] *= left[i];
    for( unsigned i=0; i<right.size(); i++)
        both[i+n*add_x] *= right[i];
    if( side == 0)
        return left;
    else if( side == 1)
        return right;
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
struct Grid2d : public dg::Grid2d<double>
{
    Grid2d( corner c, unsigned add_x, unsigned add_y, double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = dg::PER, bc bcy = dg::PER): 
        dg::Grid2d<double>( x0, x1, y0, y1, n, n_new(Nx, add_x, bcx), n_new(Ny, add_y, bcy), bcx, bcy), 
        g_assoc_( x0, x1, y0, y1, n, Nx, Ny, bcx, bcy)
    {
        thrust::host_vector< double> wx2d = dg::evaluate( dg::one, *this), wy2d(wx2d);
        for( unsigned i=0; i<this->Ny(); i++)
        for( unsigned k=0; k<this->n(); k++)
        for( unsigned j=0; j<this->Nx(); j++)
        for( unsigned l=0; l<this->n(); l++)
        {

        }




    }
    dg::Grid2d<double> associated()const {return g_assoc_;}

    private:
    unsigned n_new( unsigned N, unsigned factor, dg::bc bc)
    {
        if( bc == dg::PER) return N + 2*factor; 
        return N + factor;
    }
    thrust::host_vector<double> wx_, wy_;
    std::vector<bool> coincide_;
    dg::Grid2d<double> g_assoc_;
};

template< class container>
struct Grid3d : public dg::Grid3d<double>
{

};
}//namespace refined


cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const dg::refined::Grid2d& g_new, const dg::Grid2d<double>& g_old)
{
    //assert( g_new.associated() == g_old); //make sure the associated grid is the same
    return cusp::coo_matrix<int, double, cusp::host_memory>();

}

cusp::coo_matrix<int, double, cusp::host_memory> projection( const dg::Grid2d<double>& g_new, const dg::refined::Grid2d& g_old)
{
    //assert( g_new == g_old.associated());
    return cusp::coo_matrix<int, double, cusp::host_memory>();

}
cusp::coo_matrix<int, double, cusp::host_memory> smoothing( const dg::refined::Grid2d& g)
{
    return cusp::coo_matrix<int, double, cusp::host_memory>();

}

}//namespace dg
