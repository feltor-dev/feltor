#pragma once

#include "cusp/transpose.h"
#include "dg/backend/weightsX.cuh"
#include "dg/backend/gridX.h"
#include "refined_grid.h"


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


struct GridX3d;
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
     * @brief Reduce from a  3d grid 
     *
     * This is possible because all our grids are product space grids. 
     *
     * @param g The 3d grid
     */
    GridX2d( const dg::refined::GridX3d& g);

    /**
     * @brief The grid that this object refines
     *
     * This function is vitual so that derived classes can also construct the associated grid 
     * @return  2d grid
     */
    virtual const dg::GridX2d& associated()const {return g_assoc_;}
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
    protected:
    void init_X_boundaries( double x0, double x1)
    {
        double alpha = (x1-x0)/this->lx();
        double beta = (x0*this->x1()-x1*this->x0())/this->lx();
        //weights are invariant
        for( unsigned i=0; i<absX_.size(); i++)
            absX_[i]=alpha*absX_[i]+beta;
        dg::GridX2d::init_X_boundaries( x0, x1);

    }

    private:
    unsigned nx_new( unsigned Nx, unsigned add_x, double fx)
    {
        if( fx==0 ) return Nx + add_x ; 
        return Nx + 2*add_x;
    }
    unsigned ny_new( unsigned Ny, unsigned add_y, double fy)
    {
        if( fy==0 ) return Ny * 2*add_y; 
        return Ny + 4*add_y;
    }
    double fx_new( unsigned Nx, unsigned add_x, double fx)
    {
        if( fx==0 ) return 0; 
        return (fx*(double)Nx + (double)add_x)/((double)Nx+2.*add_x);
    }
    double fy_new( unsigned Ny, unsigned add_y, double fy)
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
    /**
     * @brief Refine the X-point
     *
     * No refinement in the third dimension
     * @param c
     * @param add_x Add number of cells to the existing one
     * @param add_y Add number of cells to the existing one
     * @param x0
     * @param x1
     * @param y0
     * @param y1
     * @param z0
     * @param y1
     * @param n
     * @param Nx
     * @param Ny
     * @param Nz
     * @param bcx
     * @param bcy
     * @param bcz
     */
    GridX3d( unsigned add_x, unsigned add_y, 
            double x0, double x1, double y0, double y1, double z0, double z1,
            double fx, double fy, 
            unsigned n, unsigned Nx, unsigned Ny, unsigned Nz,
            bc bcx = dg::PER, bc bcy = dg::NEU, bc bcz = dg::PER) : 
               dg::GridX3d( x0, x1, y0, y1, z0, z1,
                fx_new(Nx, add_x, fx), fy_new(Ny, add_y, fy), n, nx_new(Nx, add_x, fx), ny_new(Ny, add_y, fy), Nz, bcx, bcy, bcz), 
        wx_(size()), wy_(size()), absX_(size()), absY_(size()),
        g_assoc_( x0, x1, y0, y1,z0,z1, fx, fy, n, Nx, Ny, Nz, bcx, bcy, bcz)
    {
        Grid1d<double>  gx( x0, x1, n, Nx, bcx);
        GridX1d         gy( y0, y1, fy, n, Ny, bcy);
        thrust::host_vector<double> wx, ax, wy, ay;
        detail::exponential_ref(  add_x, g_assoc_.inner_Nx(), gx, wx, ax);
        detail::exponential_Xref( add_y, gy, wy, ay);
        //now make product space
        for( unsigned s=0; s<Nz; s++)
            for( unsigned i=0; i<wy.size(); i++)
                for( unsigned j=0; j<wx.size(); j++)
                {
                    wx_[(s*wy.size()+i)*wx.size()+j] = wx[j];
                    wy_[(s*wy.size()+i)*wx.size()+j] = wy[i];
                    absX_[(s*wy.size()+i)*wx.size()+j] = ax[j];
                    absY_[(s*wy.size()+i)*wx.size()+j] = ay[i];
                }
    }

    /**
     * @brief The grid that this object refines
     *
     * This function is vitual so that derived classes can also construct the associated grid 
     * @return  2d grid
     */
    virtual const dg::GridX3d& associated()const {return g_assoc_;}
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
    protected:
    void init_X_boundaries( double x0, double x1)
    {
        double alpha = (x1-x0)/this->lx();
        double beta = (x0*this->x1()-x1*this->x0())/this->lx();
        //weights are invariant under linear transformation
        for( unsigned i=0; i<absX_.size(); i++)
            absX_[i]=alpha*absX_[i]+beta;
        dg::GridX3d::init_X_boundaries( x0, x1);
    }

    private:
    unsigned nx_new( unsigned Nx, unsigned add_x, double fx)
    {
        if( fx==0 ) return Nx + add_x; 
        return Nx + 2*add_x;
    }
    unsigned ny_new( unsigned Ny, unsigned add_y, double fy)
    {
        if( fy==0 ) return Ny + 2*add_y; 
        return Ny + 4*add_y;
    }
    double fx_new( unsigned Nx, unsigned add_x, double fx)
    {
        if( fx==0 ) return 0; 
        return (fx*(double)Nx + (double)add_x)/(double)(Nx+2.*add_x);
    }
    double fy_new( unsigned Ny, unsigned add_y, double fy)
    {
        if( fy==0 ) return 0; 
        return (fy*(double)Ny + (double)add_y)/(double)(Ny+4.*add_y);
    }
    thrust::host_vector<double> wx_, wy_; //weights
    thrust::host_vector<double> absX_, absY_; //abscissas 
    dg::GridX3d g_assoc_;
};

GridX2d::GridX2d( const dg::refined::GridX3d& g) : 
    dg::GridX2d( g.x0(), g.x1(), g.y0(), g.y1(), g.fx(), g.fy(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy()),
    wx_( this->size()), wy_(this->size()), absX_(this->size()), absY_(this->size()),
    g_assoc_( g.associated())
{
    for(unsigned i=0; i<this->size(); i++)
    {
        wx_[i] = g.weightsX()[i];
        wy_[i] = g.weightsY()[i];
        absX_[i] = g.abscissasX()[i];
        absY_[i] = g.abscissasY()[i];
    }
}
}//namespace refined

namespace create{

cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const dg::refined::GridX2d& g_fine)
{
    dg::GridX2d g = g_fine.associated();
    thrust::host_vector<double> x = g_fine.abscissasX();
    thrust::host_vector<double> y = g_fine.abscissasY();
    return dg::create::interpolation( x,y, g.grid());

}
cusp::coo_matrix<int, double, cusp::host_memory> interpolationT( const dg::refined::GridX2d& g_fine)
{
    cusp::coo_matrix<int, double, cusp::host_memory> temp = interpolation( g_fine), A;
    cusp::transpose( temp, A);
    return A;
}

cusp::coo_matrix<int, double, cusp::host_memory> projection( const dg::refined::GridX2d& g_fine)
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

cusp::coo_matrix<int, double, cusp::host_memory> smoothing( const dg::refined::GridX2d& g)
{
    cusp::coo_matrix<int, double, cusp::host_memory> A = interpolation(g);
    cusp::coo_matrix<int, double, cusp::host_memory> B = projection(g);
    cusp::coo_matrix<int, double, cusp::host_memory> C;
    cusp::multiply( A, B, C);
    C.sort_by_row_and_column();
    return C; 
}

cusp::coo_matrix<int, double, cusp::host_memory> interpolation( const dg::refined::GridX3d& g_fine)
{
    dg::GridX3d g = g_fine.associated();
    thrust::host_vector<double> x = g_fine.abscissasX();
    thrust::host_vector<double> y = g_fine.abscissasY();
    return dg::create::interpolation( x,y, g.grid());
}

cusp::coo_matrix<int, double, cusp::host_memory> interpolationT( const dg::refined::GridX3d& g_fine)
{
    cusp::coo_matrix<int, double, cusp::host_memory> temp = interpolation( g_fine), A;
    cusp::transpose( temp, A);
    return A;
}

cusp::coo_matrix<int, double, cusp::host_memory> projection( const dg::refined::GridX3d& g_fine)
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

cusp::coo_matrix<int, double, cusp::host_memory> smoothing( const dg::refined::GridX3d& g)
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
