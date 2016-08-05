#pragma once

#include "cusp/transpose.h"
#include "dg/backend/grid.h"
#include "dg/backend/weights.cuh"
#include "dg/backend/interpolation.cuh"


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
 * @brief Divide cells in an equally distributed number of new cells
 *
 * @param add_x the number of cells to add 
 * @param node the node around which to refine
 * @param n polynomials
 * @param N # of cells
 * @param bcx boundary condition
 * @param howmany number of cells to refine
 *
 * @return Weights
 */
thrust::host_vector<double> equidist_ref( unsigned add_x, unsigned node, unsigned n, unsigned N, dg::bc bcx, unsigned howmany)
{
    assert( howmany <= N);
    assert( node <= N);
    if( node != 0 && node != N) assert( howmany <= node && howmany <= N-node);
    if( add_x == 0 || howmany == 0)
    {
        thrust::host_vector<double> w_( n*N, 1);
        return w_;
    }
    assert( node <= N);
    //there are add_x+1 finer cells per refined cell ...
    thrust::host_vector< double> left( n*N+n*add_x*howmany, 1), right(left);
    for( unsigned i=0; i<(add_x+1)*howmany; i++)//the original cell and the additional ones
        for( unsigned k=0; k<n; k++)
            left[i*n+k] = add_x + 1;
    //mirror left into right
    for( unsigned i=0; i<right.size(); i++)
        right[i] = left[ (left.size()-1)-i];
    thrust::host_vector< double> both( n*N+2*n*add_x*howmany, 1);
    for( unsigned i=0; i<left.size(); i++)
        both[i] *= left[i];
    for( unsigned i=0; i<right.size(); i++)
        both[i+n*add_x*howmany] *= right[i];
    if(      node == 0     && bcx != dg::PER) { return left; }
    else if( node == N && bcx != dg::PER) { return right; }
    else if((node == N || node == 0) && bcx == dg::PER) { return both; }
    else 
    {
        thrust::host_vector<double> w_ = both;
        //now shift indices so that refinement is around nodes
        for( unsigned i=0; i<both.size(); i++)
            w_[((howmany*add_x+node)*n+i)%both.size()] = both[i];
        return w_;
    }
}

/**
 * @brief Divide cells in an equally distributed number of new cells
 *
 * @param add_x the number of cells to add 
 * @param node the node around which to refine
 * @param n polynomials
 * @param N # of cells
 * @param bcx boundary condition
 *
 * @return Weights
 */
thrust::host_vector<double> equidist_ref( unsigned add_x, unsigned node, unsigned n, unsigned N, dg::bc bcx)
{
    return equidist_ref(add_x, node, n, N, bcx, 1);
}

thrust::host_vector<double> exponential_ref( unsigned add_x, unsigned node, unsigned n, unsigned N, dg::bc bcx)
{
    if( add_x == 0)
    {
        thrust::host_vector<double> w_( n*N, 1);
        return w_;
    }
    assert( node <= N);
    //there are add_x+1 finer cells per refined cell ...
    thrust::host_vector< double> left( n*N+n*add_x, 1), right(left);
    for( unsigned k=0; k<n; k++)//the original cell and the additional ones
        left[k] = pow( 2, add_x);
    for( unsigned i=0; i<add_x; i++) 
        for( unsigned k=0; k<n; k++)
            left[(i+1)*n+k] = pow( 2, add_x-i);
    //mirror left into right
    for( unsigned i=0; i<right.size(); i++)
        right[i] = left[ (left.size()-1)-i];
    thrust::host_vector< double> both( n*N+2*n*add_x, 1);
    for( unsigned i=0; i<left.size(); i++)
        both[i] *= left[i];
    for( unsigned i=0; i<right.size(); i++)
        both[i+n*add_x] *= right[i];
    if(      node == 0     && bcx != dg::PER) { return left; }
    else if( node == N && bcx != dg::PER) { return right; }
    else if((node == N || node == 0) && bcx == dg::PER) { return both; }
    else 
    {
        thrust::host_vector<double> w_ = both;
        //now shift indices so that refinement is around nodes
        for( unsigned i=0; i<both.size(); i++)
            w_[((add_x+node)*n+i)%both.size()] = both[i];
        return w_;
    }
}

/**
 * @brief Normalize the given weights and compute the abscissas of the grid
 *
 * @param g The grid to be refined
 * @param weights the unnormalized weights
 *
 * @return The abscissas of the new grid
 */
thrust::host_vector<double> normalize_weights_and_compute_abscissas( const Grid1d<double>& g, thrust::host_vector<double>& weights)
{
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
    return abs;
}

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
    weights = exponential_ref( add_x, node, g.n(), g.N(), g.bcx());
    unsigned Nx_new = weights.size()/g.n();
    abscissas = normalize_weights_and_compute_abscissas( g, weights);
    return Nx_new;
}

/**
 * @brief Create 1d refinement weights and abscissas for the equidist refinement around a node 
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
int equidist_ref( unsigned add_x, unsigned node, const Grid1d<double>& g, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas, unsigned howmany)
{
    if( add_x == 0 || howmany == 0)
    {
        thrust::host_vector<double> w_( g.size(), 1);
        thrust::host_vector<double> abs_= dg::create::abscissas(g);
        weights = w_; abscissas = abs_; 
        return g.N();
    }
    weights = equidist_ref( add_x, node, g.n(), g.N(), g.bcx(), howmany);
    unsigned Nx_new = weights.size()/g.n();
    abscissas = normalize_weights_and_compute_abscissas( g, weights);
    return Nx_new;
}
int equidist_ref( unsigned add_x, unsigned node, const Grid1d<double>& g, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas)
{
    return equidist_ref( add_x, node, g, weights, abscissas, 1);
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
    Grid2d( unsigned node_x, unsigned node_y, unsigned add_x, unsigned add_y, 
            unsigned howmanyX, unsigned howmanyY,
            double x0, double x1, double y0, double y1, 
            unsigned n, unsigned Nx, unsigned Ny, bc bcx = dg::PER, bc bcy = dg::PER) : dg::Grid2d<double>( x0, x1, y0, y1, n, n_new(Nx, add_x*howmanyX, bcx), n_new(Ny, add_y*howmanyY, bcy), bcx, bcy), 
        wx_(size()), wy_(size()), absX_(size()), absY_(size()),
        g_assoc_( x0, x1, y0, y1, n, Nx, Ny, bcx, bcy)
    {
        //assert( howmanyX <= node_x && howmanyX <= Nx - node_x);
        //assert( howmanyY <= node_y && howmanyY <= Ny - node_y);
        Grid1d<double> gx( x0, x1, n, Nx, bcx);
        Grid1d<double> gy( y0, y1, n, Ny, bcy);
        thrust::host_vector<double> wx, ax, wy, ay;
        detail::equidist_ref( add_x, node_x, gx, wx, ax, howmanyX);
        detail::equidist_ref( add_y, node_y, gy, wy, ay, howmanyY);
        //now make product space
        for( unsigned i=0; i<wy.size(); i++)
            for( unsigned j=0; j<wx.size(); j++)
            {
                wx_[i*wx.size()+j] = wx[j];
                wy_[i*wx.size()+j] = wy[i];
                absX_[i*wx.size()+j] = ax[j];
                absY_[i*wx.size()+j] = ay[i];
            }
    }

    /**
     * @brief The grid that this object refines
     *
     * This function is vitual so that derived classes can also construct the associated grid 
     * @return  2d grid
     */
    virtual const dg::Grid2d<double>& associated()const {return g_assoc_;}
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


namespace create{

cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const dg::refined::Grid2d& g_fine)
{
    dg::Grid2d<double> g_coarse = g_fine.associated();
    thrust::host_vector<double> x = g_fine.abscissasX();
    thrust::host_vector<double> y = g_fine.abscissasY();
    return dg::create::interpolation( x,y, g_coarse);

}
cusp::coo_matrix<int, double, cusp::host_memory> interpolationT( const dg::refined::Grid2d& g_fine)
{
    cusp::coo_matrix<int, double, cusp::host_memory> temp = interpolation( g_fine), A;
    cusp::transpose( temp, A);
    return A;
}

/**
 * @brief Create the adjoint of the interpolation 
 *
 * The adjoint is formed with respect to the volume form of the fine grid
 * @param g_fine The fine grid
 *
 * @return 
 */
cusp::coo_matrix<int, double, cusp::host_memory> projection( const dg::refined::Grid2d& g_fine)
{
    //form the adjoint
    thrust::host_vector<double> w_f = dg::create::weights( g_fine);
    thrust::host_vector<double> v_c = dg::create::inv_weights( g_fine.associated() );
    cusp::coo_matrix<int, double, cusp::host_memory> Wf( w_f.size(), w_f.size(), w_f.size());
    cusp::coo_matrix<int, double, cusp::host_memory> Vc( v_c.size(), v_c.size(), v_c.size());
    for( int i =0; i<(int)w_f.size(); i++)
    {
        Wf.row_indices[i] = Wf.column_indices[i] = i;
        Wf.values[i] = w_f[i]/g_fine.weightsX()[i]/g_fine.weightsY()[i];
    }
    for( int i =0; i<(int)v_c.size(); i++)
    {
        Vc.row_indices[i] = Vc.column_indices[i] = i;
        Vc.values[i] = v_c[i];
    }
    cusp::coo_matrix<int, double, cusp::host_memory> temp = interpolation( g_fine), A;
    cusp::transpose( temp, A);
    cusp::multiply( A, Wf, temp);
    cusp::multiply( Vc, temp, A);
    A.sort_by_row_and_column();
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
}//namespace create

}//namespace dg
