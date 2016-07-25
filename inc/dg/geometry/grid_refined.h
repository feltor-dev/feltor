#pragma once

#include "dg/backend/grid.h"
#include "dg/backend/interpolation.cuh"
#include "cusp/transpose.h"


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
 * @brief Create 1d refinement weights and abscissas for the exponential refinement around a node 
 *
 * There will be two refined cells at the end except if a corner node is 
 * given and the boundary condition is not periodic. We count nodes from
 * 0 (left corner) to N (right corner). 
 * @param add_x number of additional cells in the cells idx-1 and idx
 * @param node The cells node-1 and node will be refined
 * @param g The 1d grid to refine
 *
 * @param weights A 1d vector of size n*(Nx+add_x) for one-sided refinement and n*(Nx+2*add_x)) for two-sided refinement
 * @param abscissas A 1d vector of size n*(Nx+add_x) for one-sided refinement and n*(Nx+2*add_x)) for two-sided refinement
 * @return the new number of cells
 */
int exponential_ref( unsigned add_x, unsigned node, const Grid1d<double>& g, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas)
{
    if( add_x == 0)
    {
        thrust::host_vector<double> w_( g.size(), 1);
        thrust::host_vector<double> abs_= dg::create::abscissas(g);
        weights = w_; abscissas = abs_; 
        return g.N();
    }
    assert( node <= g.N());
    //there are add_x+1 finer cells per refined cell ...
    thrust::host_vector< double> left( g.size()+g.n()*add_x, 1), right(left);
    for( unsigned k=0; k<g.n(); k++)//the original cell and the additional ones
        left[k] = pow( 2, add_x);
    for( unsigned i=0; i<add_x; i++) 
    for( unsigned k=0; k<g.n(); k++)
        left[(i+1)*g.n()+k] = pow( 2, add_x-i);
    //mirror left into right
    for( unsigned i=0; i<right.size(); i++)
        right[i] = left[ (left.size()-1)-i];
    thrust::host_vector< double> both( g.size()+2*g.n()*add_x, 1);
    for( unsigned i=0; i<left.size(); i++)
        both[i] *= left[i];
    for( unsigned i=0; i<right.size(); i++)
        both[i+g.n()*add_x] *= right[i];
    if(      node == 0     && g.bcx() != dg::PER) { weights = left; }
    else if( node == g.N() && g.bcx() != dg::PER) { weights = right; }
    else if((node == g.N() || node == 0) && g.bcx() == dg::PER) { weights = both; }
    else 
    {
        thrust::host_vector<double> w_ = both;
        //now shift indices so that refinement is around nodes
        for( unsigned i=0; i<both.size(); i++)
            w_[((add_x+node)*g.n()+i)%both.size()] = both[i];
        weights = w_;
    }

    //normalize weights
    unsigned Nx_new = weights.size()/g.n();
    for( unsigned i=0;i<weights.size(); i++)
        weights[i] *= (double)g.N()/(double)Nx_new;

    thrust::host_vector<double> boundaries(Nx_new+1), abs(g.n()*Nx_new);
    boundaries[0] = g.x0();
    for( unsigned i=0; i<Nx_new; i++)
    {
        boundaries[i+1] = boundaries[i] + g.lx()/(double)Nx_new/weights[g.n()*i];
        for( unsigned j=0; j<g.n(); j++)
        {
            abs[i*g.n()+j] =  (boundaries[i+1]+boundaries[i])/2. + 
                (boundaries[i+1]-boundaries[i])/2.*g.dlt().abscissas()[j];
        }
    }
    abscissas = abs;
    return Nx_new;

}

}//namespace detail

/**
 * @brief Refined grid 
 */
struct Grid2d : public dg::Grid2d<double>
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
    Grid2d( corner c, unsigned add_x, unsigned add_y, 
            double x0, double x1, double y0, double y1, 
            unsigned n, unsigned Nx, unsigned Ny, bc bcx = dg::PER, bc bcy = dg::PER) : dg::Grid2d<double>( x0, x1, y0, y1, n, n_new(Nx, add_x, bcx), n_new(Ny, add_y, bcy), bcx, bcy), 
        g_assoc_( x0, x1, y0, y1, n, Nx, Ny, bcx, bcy)
    {
        /*
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
        */
    }

    /**
     * @brief The grid that this object refines
     *
     * @return  2d grid
     */
    dg::Grid2d<double> associated()const {return g_assoc_;}
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
    unsigned n_new( unsigned N, unsigned factor, dg::bc bc)
    {
        if( bc == dg::PER) return N + 2*factor; 
        return N + factor;
    }
    thrust::host_vector<double> wx_, wy_; //weights
    thrust::host_vector<double> absX_, absY_; //abscissas 
    dg::Grid2d<double> g_assoc_;
};

template< class container>
struct Grid3d : public dg::Grid3d<double>
{

};
}//namespace refined


cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const dg::refined::Grid2d& g_fine)
{
    dg::Grid2d<double> g = g_fine.associated();
    //determine number of refined cells
    thrust::host_vector<double> x = g_fine.abscissasX();
    thrust::host_vector<double> y = g_fine.abscissasY();
    
    return dg::create::interpolation( x,y, g);

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
