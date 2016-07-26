#pragma once

#include "grid_refined.h"
#include "dg/backend/gridX.h"


namespace dg
{
namespace refined
{

namespace detail
{

/**
 * @brief Create 1d refinement weights and abscissas for the X-point
 *
 * There will be four refined cells at the end except if the X-point is a corner node.
 * @param add_x number of additional cells in the cells around the X-point
 * @param g The 1d Xgrid to refine
 *
 * @param weights   A 1d vector of size n*(Nx+2*add_x) for one-sided refinement and n*(Nx+4*add_x)) for two-sided refinement
 * @param abscissas A 1d vector of size n*(Nx+2*add_x) for one-sided refinement and n*(Nx+4*add_x)) for two-sided refinement
 * @return the new number of cells
 */
int exponential_Xref( unsigned add_x, const GridX1d& g, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas)
{
    if( add_x == 0) { return exponential_ref( add_x, 0, g.grid(), weights, abscissas); }
    if( g.f() == 0) { return exponential_ref( add_x, 0, Grid1d<double>( g.x0(), g.x1(), g.n(), g.N(), dg::PER), weights, abscissas); }
    thrust::host_vector<double> w1, w2, w3;
    unsigned node1 = g.outer_N();
    w1 = exponential_ref( add_x, node1, g.n(), g.outer_N(), dg::DIR); //left side
    w2 = exponential_ref( add_x, 0, g.n(), g.inner_N(), dg::PER);//inner side
    w3 = exponential_ref( add_x, 0, g.n(), g.outer_N(), dg::DIR);//right side
    //now combine unnormalized weights
    thrust::host_vector<double> wtot( g.size() + 4*g.n()*add_x);
    for( unsigned i=0; i<w1.size() ; i++)
        wtot[i] = w1[i];
    for( unsigned i=0; i<w2.size(); i++)
        wtot[w1.size()+i] = w2[i];
    for( unsigned i=0; i<w3.size(); i++)
        wtot[w1.size()+w2.size()+i] = w3[i];
    weights = wtot;

    unsigned Nx_new = weights.size()/g.n();
    abscissas = normalize_weights_and_compute_abscissas( g.grid(), weights);
    return Nx_new;
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
            unsigned n, unsigned Nx, unsigned Ny, 
            bc bcx = dg::PER, bc bcy = dg::PER) : dg::GridX2d( x0, x1, y0, y1, 
                fx_new(Nx, add_x, fx), fy_new(Ny, add_y, fy), n, nx_new(Nx, add_x, fx), ny_new(Ny, add_y, fy), bcx, bcy), 
        wx_(size()), wy_(size()), absX_(size()), absY_(size()),
        g_assoc_( x0, x1, y0, y1, fx, fy, n, Nx, Ny, bcx, bcy)
    {
        Grid1d<double>  gx( x0, x1, n, Nx, bcx);
        GridX1d         gy( y0, y1, fy, n, Ny, bcy);
        thrust::host_vector<double> wx, ax, wy, ay;
        detail::exponential_ref(  add_x, g_assoc_.inner_Nx(), gx, wx, ax);
        detail::exponential_Xref( add_y, gy, wy, ay);
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
        if( fx==0 ) return 0; 
        return (fx*(double)Nx + (double)add_x)/((double)Nx+2.*add_x);
    }
    unsigned fy_new( unsigned Ny, unsigned add_y, unsigned fy)
    {
        if( fy==0 ) return 0; 
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

namespace create{

cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const dg::refined::GridX2d& g_fine)
{
    dg::GridX2d g = g_fine.associated();
    //determine number of refined cells
    thrust::host_vector<double> x = g_fine.abscissasX();
    thrust::host_vector<double> y = g_fine.abscissasY();
    
    return dg::create::interpolation( x,y, g.grid());

}

cusp::coo_matrix<int, double, cusp::host_memory> projection( const dg::refined::GridX2d& g_fine)
{
    cusp::coo_matrix<int, double, cusp::host_memory> temp = interpolation( g_fine), A;
    cusp::transpose( temp, A);
    return A;
}

cusp::coo_matrix<int, double, cusp::host_memory> smoothing( const dg::refined::GridX2d& g)
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
