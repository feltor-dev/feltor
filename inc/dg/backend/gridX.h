#pragma once
#include <cassert>
#include <cmath> //floor function
#include "dlt.h"
#include "grid.h"
#include "../enums.h"

/*! @file 
  
  Grid objects
  */


namespace dg{

///@addtogroup grid
///@{
/**
* @brief 1D grid for X-point topology
*
* The grid looks like 
@code
|------x----------x------|
@endcode
* , where the x indicates the periodicity and - the number of points
* The left boundary is x0 and the right x1, the inner boundaries lie at
* x0 + f*Lx and x1-f*Lx
* therefore f must be smaller than 0.5
*/
struct GridX1d
{
    typedef SharedTag memory_category;
    typedef OneDimensionalTag dimensionality;
    /**
     * @brief 1D grid
     * 
     @param x0 left boundary
     @param x1 right boundary
     @param f factor 0<f<0.5 divides the domain
     @param n # of polynomial coefficients
     @param N # of cells
     @param bcx boundary conditions
     */
    GridX1d( double x0, double x1, double f, unsigned n, unsigned N, bc bcx = NEU):
        x0_(x0), x1_(x1), f_(f),
        n_(n), Nx_(N), bcx_(bcx), dlt_(n)
    {
        assert( (f >= 0) && (f < 0.5) );
        assert( fabs(outer_N() - f*(double)N) < 1e-15); 
        assert( x1 > x0 );
        assert( N > 0  );
        assert( n != 0 );
        assert( bcx != PER);
        lx_ = (x1-x0);
        hx_ = lx_/(double)Nx_;
    }
    /**
     * @brief left boundary
     *
     * @return 
     */
    double x0() const {return x0_;}
    /**
     * @brief right boundary
     *
     * @return 
     */
    double x1() const {return x1_;}
    /**
     * @brief Factor
     *
     * @return 
     */
    double f() const {return f_;}
    /**
     * @brief total length of interval
     *
     * @return 
     */
    double lx() const {return lx_;}
    /**
     * @brief cell size
     *
     * @return 
     */
    double h() const {return hx_;}
    /**
     * @brief number of cells
     *
     * @return 
     */
    unsigned N() const {return Nx_;}
    /**
     * @brief number of cells in one of the outer regions
     *
     * @return 
     */
    unsigned outer_N() const {return (unsigned)(floor(f_*(double)Nx_+0.5));}
    /**
     * @brief number of cells in the inner region
     *
     * @return 
     */
    unsigned inner_N() const {return N()-2*outer_N();}
    /**
     * @brief number of polynomial coefficients
     *
     * @return 
     */
    unsigned n() const {return n_;}
    /**
     * @brief boundary conditions
     *
     * @return 
     */
    bc bcx() const {return bcx_;}
    /**
     * @brief the total number of points
     *
     * @return n*Nx
     */
    unsigned size() const { return n_*Nx_;}
    /**
     * @brief the discrete legendre transformation
     *
     * @return 
     */
    /**
     * @brief Display 
     *
     * @param os output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "Grid parameters are: \n"
            <<"    n  = "<<n_<<"\n"
            <<"    N  = "<<Nx_<<"\n"
            <<"    inner N = "<<inner_N()<<"\n"
            <<"    outer N = "<<outer_N()<<"\n"
            <<"    h  = "<<hx_<<"\n"
            <<"    x0 = "<<x0_<<"\n"
            <<"    x1 = "<<x1_<<"\n"
            <<"    lx = "<<lx_<<"\n"
            <<"Boundary conditions in x are: \n"
            <<"    "<<bc2str(bcx_)<<"\n";
    }
    const DLT<double>& dlt() const {return dlt_;}
    Grid1d grid() const{return Grid1d( x0_, x1_, n_, Nx_, bcx_);}

    /**
     * @brief Shifts a point coordinate due to topology
     *
     * If you want to construct a point by adding delta X to a given point
     * x0 then the resulting coordinate x1 might be incorrect due to topologic reasons (periodic boundaries). This function corrects this coordinate
     * @param x0 starting point (must lie inside of the grid)
     * @param x1 end point (inout)
     */
    void shift_topologic( double x0, double& x1) const
    {
        assert( contains(x0));
        double deltaX;
        double xleft = x0_ + f_*lx_;
        double xright = x1_ - f_*lx_;
        if( x0 >= xleft && x0<xright)
        {
            if( x1 > xleft) deltaX = (x1 -xleft);
            else deltaX = xright - x1;
            unsigned N = floor(deltaX/(xright-xleft));
            if( x1  > xright ) x1 -= N*lx_;
            if( x1  < xleft ) x1 += N*lx_;
        }
        else if( x0 < xleft && x1 >=xleft)
            x1 += (xright-xleft);
        else if( x0 >= xright  && x1 < xright)
            x1 -= (xright-xleft);

    }

    /**
     * @brief Check if the grid contains a point
     *
     * @note doesn't check periodicity!!
     * @param x point to check
     *
     * @return true if x is between x0 and x1, false else
     */
    bool contains( double x) const
    {
        if( (x>=x0_ && x <= x1_)) return true; 
        return false;
    }
  private:
    double x0_, x1_, f_;
    double lx_;
    unsigned n_, Nx_;
    double hx_;
    bc bcx_;
    DLT<double> dlt_;
};

struct GridX3d; //forward declare 3d version

/**
 * @brief A 2D grid class with X-point topology
 *
 * is of the form
 @code
 | -----> y 
 |  |---x----------x---|
 |  |---x----------x---|
 v  |--- ---------- ---|
 x  |--- ---------- ---| fx*Lx
    |--- ---------- ---|
    fy*Ly
 @endcode
 *
 * @tparam double scalar value type 
 */
struct GridX2d
{
    typedef SharedTag memory_category;
    typedef TwoDimensionalTag dimensionality;
    /**
     * @brief Construct a 2D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param fx factor for x-direction (fx*Nx must be a natural number)
     * @param fy factor for y-direction (fy*Ny must be a natural number)
     * @param n  # of polynomial coefficients per dimension
     * @param Nx # of points in x 
     * @param Ny # of points in y
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     */
    GridX2d( double x0, double x1, double y0, double y1, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = NEU):
        x0_(x0), x1_(x1), y0_(y0), y1_(y1), fx_(fx), fy_(fy),
        n_(n), Nx_(Nx), Ny_(Ny), bcx_(bcx), bcy_( bcy), dlt_(n)
    {
        assert( (fy_ >= 0.) && (fy_ < 0.5) );
        assert( (fx_ >= 0.) && (fx_ < 1.) );
        assert( fabs(outer_Nx() - fx_*(double)Nx) < 1e-15); 
        assert( fabs(outer_Ny() - fy_*(double)Ny) < 1e-15); 
        assert( n != 0);
        assert( x1 > x0 && y1 > y0);
        assert( Nx_ > 0  && Ny > 0 );
        assert( bcy != PER);
        lx_ = (x1_-x0_), ly_ = (y1_-y0_);
        hx_ = lx_/(double)Nx_, hy_ = ly_/(double)Ny_;
    }
    /**
     * @brief Reduce from a 3d grid
     *
     * This takes the x and y dimension from the 3d grid.
     * Note that this is possible because all our grids are product space grids  and only the first two dimensions are dG discretized.
     *
     * @param g The 3d counterpart
     */
    GridX2d( const GridX3d& g);

    /**
     * @brief Left boundary in x
     *
     * @return 
     */
    double x0() const {return x0_;}
    /**
     * @brief Right boundary in x
     *
     * @return 
     */
    double x1() const {return x1_;}
    /**
     * @brief left boundary in y
     *
     * @return 
     */
    double y0() const {return y0_;}
    /**
     * @brief Right boundary in y 
     *
     * @return 
     */
    double y1() const {return y1_;}
    /**
     * @brief length of x 
     *
     * @return 
     */
    double lx() const {return lx_;}
    /**
     * @brief length of y
     *
     * @return 
     */
    double ly() const {return ly_;}
    /**
     * @brief cell size in x 
     *
     * @return 
     */
    double hx() const {return hx_;}
    /**
     * @brief cell size in y
     *
     * @return 
     */
    double hy() const {return hy_;}
    /**
     * @brief Factor
     *
     * @return 
     */
    double fx() const {return fx_;}
    /**
     * @brief Factor
     *
     * @return 
     */
    double fy() const {return fy_;}
    /**
     * @brief number of polynomial coefficients in x and y
     *
     * @return 
     */
    unsigned n() const {return n_;}
    /**
     * @brief number of cells in x
     *
     * @return 
     */
    unsigned Nx() const {return Nx_;}
    /**
     * @brief number of topological cells in x
     *
     * @return 
     */
    unsigned inner_Nx() const {return Nx_ - outer_Nx();}
    /**
     * @brief number of smooth rows in x
     *
     * @return 
     */
    unsigned outer_Nx() const {return (unsigned)floor(fx_*(double)Nx_+0.5);}
    /**
     * @brief number of cells in y
     *
     * @return 
     */
    unsigned Ny() const {return Ny_;}
    /**
     * @brief number of cells in the inner region of y
     *
     * @return 
     */
    unsigned inner_Ny() const {return Ny_-2*outer_Ny();}
    /**
     * @brief number of cells in one of the outer regions of y
     *
     * @return 
     */
    unsigned outer_Ny() const {return (unsigned)floor(fy_*(double)Ny_+0.5);}
    /**
     * @brief boundary conditions in x
     *
     * @return 
     */
    bc bcx() const {return bcx_;}
    /**
     * @brief boundary conditions in y
     *
     * @return 
     */
    bc bcy() const {return bcy_;}
    /**
     * @brief Return a copy
     *
     * @return 
     */
    GridX2d local_grid() const {return *this;}
    /**
     * @brief Return a copy without topology
     *
     * @return 
     */
    Grid2d grid() const {return Grid2d( x0_,x1_,y0_,y1_,n_,Nx_,Ny_,bcx_,bcy_);}
    /**
     * @brief discrete legendre trafo
     *
     * @return 
     */
    const DLT<double>& dlt() const{return dlt_;}
    /**
     * @brief doublehe total number of points
     *
     * @return n*n*Nx*Ny
     */
    unsigned size() const { return n_*n_*Nx_*Ny_;}
    /**
     * @brief Display 
     *
     * @param os output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "Grid parameters are: \n"
            <<"    n  = "<<n_<<"\n"
            <<"    Nx = "<<Nx_<<"\n"
            <<"    inner Nx = "<<inner_Nx()<<"\n"
            <<"    outer Nx = "<<outer_Nx()<<"\n"
            <<"    Ny = "<<Ny_<<"\n"
            <<"    inner Ny = "<<inner_Ny()<<"\n"
            <<"    outer Ny = "<<outer_Ny()<<"\n"
            <<"    hx = "<<hx_<<"\n"
            <<"    hy = "<<hy_<<"\n"
            <<"    x0 = "<<x0_<<"\n"
            <<"    x1 = "<<x1_<<"\n"
            <<"    y0 = "<<y0_<<"\n"
            <<"    y1 = "<<y1_<<"\n"
            <<"    lx = "<<lx_<<"\n"
            <<"    ly = "<<ly_<<"\n"
            <<"Boundary conditions in x are: \n"
            <<"    "<<bc2str(bcx_)<<"\n"
            <<"Boundary conditions in y are: \n"
            <<"    "<<bc2str(bcy_)<<"\n";
    }

    /**
     * @brief Shifts a point coordinate due to topology
     *
     * If you want to construct a point by adding (delta X, delta Y) to a given point
     * (x0, y0) then the resulting coordinate x1 might be incorrect due to topologic reasons (periodic boundaries). This function corrects this coordinate
     * @param x0 starting x-point (must lie inside of the grid)
     * @param y0 starting y-point (must lie inside of the grid)
     * @param x1 end x-point (inout)
     * @param y1 end y-point (inout)
     */
    void shift_topologic( double x0, double y0, double& x1, double& y1) const
    {
        assert( contains(x0, y0));
        double deltaX;
        if( x1 > x0_) deltaX = (x1 -x0_);
        else deltaX = x1_ - x1;
        unsigned N = floor(deltaX/lx_);
        if( x1  > x1_ && bcx_ == dg::PER) x1 -= N*lx_;
        if( x1  < x0_ && bcx_ == dg::PER) x1 += N*lx_;

        if( x0 < x1_ - fx_*(x1_-x0_) ) //if x0 is  one of the inner points
        {
            double deltaY;
            double yleft = y0_ + fy_*ly_;
            double yright = y1_ - fy_*ly_;
            if( y0 >= yleft && y0<yright)
            {
                if( y1 > yleft) deltaY = (y1 -yleft);
                else deltaY = yright - y1;
                unsigned N = floor(deltaY/(yright-yleft));
                if( y1  > yright ) y1 -= N*ly_;
                if( y1  < yleft ) y1 += N*ly_;
            }
            else if( y0 < yleft && y1 >=yleft)
                y1 += (yright-yleft);
            else if( y0 >= yright  && y1 < yright)
                y1 -= (yright-yleft);
        }

    }

    /**
     * @brief Check if the grid contains a point
     *
     * @note doesn't check periodicity!!
     * @param x x-point to check
     * @param y y-point to check
     *
     * @return true if (x,y) is inside the grid, false else
     */
    bool contains( double x, double y)const
    {
        if( (x>=x0_ && x <= x1_) && (y>=y0_ && y <= y1_)) return true; 
        return false;
    }
  protected:
    virtual void init_X_boundaries( double x0, double x1)
    {
        x0_ = x0, x1_ = x1;
        assert( x1 > x0 );
        lx_ = (x1_-x0_);
        hx_ = lx_/(double)Nx_;
    }
  private:
    double x0_, x1_, y0_, y1_;
    double fx_, fy_;
    double lx_, ly_;
    unsigned n_, Nx_, Ny_;
    double hx_, hy_;
    bc bcx_, bcy_;
    DLT<double> dlt_;
};

/**
 * @brief A 3D grid class  for cartesian coordinates
 *
 * In the third dimension only 1 polynomial coefficient is used,
 * not n. In 2d it looks like
 @code
 | -----> y 
 |  |---x----------x---|
 |  |---x----------x---|
 v  |--- ---------- ---|
 x  |--- ---------- ---| fx*Lx
    |--- ---------- ---|
    fy*Ly
 @endcode
 * @tparam double scalar value type 
 */
struct GridX3d
{
    typedef SharedTag memory_category;
    typedef ThreeDimensionalTag dimensionality;
    /**
     * @brief Construct a 3D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param z0 lower boundary in z
     * @param z1 upper boundary in z 
     * @param fx factor for x-direction
     * @param fy factor for y-direction
     * @param n  # of polynomial coefficients per (x-,y-) dimension
     * @param Nx # of points in x 
     * @param Ny # of points in y
     * @param Nz # of points in z
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     * @param bcz boundary condition in z
     * @attention # of polynomial coefficients in z direction is always 1
     */
    GridX3d( double x0, double x1, double y0, double y1, double z0, double z1, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = NEU, bc bcz = PER):
        x0_(x0), x1_(x1), y0_(y0), y1_(y1), z0_(z0), z1_(z1), fx_(fx), fy_(fy),
        n_(n), Nx_(Nx), Ny_(Ny), Nz_(Nz), bcx_(bcx), bcy_( bcy), bcz_( bcz), dlt_(n)
    {
        assert( (fy_ >= 0.) && (fy_ < 0.5) );
        assert( (fx_ >= 0.) && (fx_ < 1.) );
        assert( fabs(outer_Nx() - fx_*(double)Nx) < 1e-15); 
        assert( fabs(outer_Ny() - fy_*(double)Ny) < 1e-15); 
        assert( n != 0);
        assert( x1 > x0 && y1 > y0 ); assert( z1 > z0 );         
        assert( Nx_ > 0  && Ny > 0); assert( Nz > 0);
        lx_ = (x1-x0), ly_ = (y1-y0), lz_ = (z1-z0);
        hx_ = lx_/(double)Nx_, hy_ = ly_/(double)Ny_, hz_ = lz_/(double)Nz_;
    }
    /**
     * @brief left boundary in x
     *
     * @return 
     */
    double x0() const {return x0_;}
    /**
     * @brief right boundary in x
     *
     * @return 
     */
    double x1() const {return x1_;}

    /**
     * @brief left boundary in y 
     *
     * @return 
     */
    double y0() const {return y0_;}
    /**
     * @brief right boundary in y
     *
     * @return 
     */
    double y1() const {return y1_;}

    /**
     * @brief left boundary in z
     *
     * @return 
     */
    double z0() const {return z0_;}
    /**
     * @brief right boundary in z
     *
     * @return 
     */
    double z1() const {return z1_;}

    /**
     * @brief length in x
     *
     * @return 
     */
    double lx() const {return lx_;}
    /**
     * @brief length in y
     *
     * @return 
     */
    double ly() const {return ly_;}
    /**
     * @brief length in z
     *
     * @return 
     */
    double lz() const {return lz_;}
    
    /**
     * @brief cell size in x
     *
     * @return 
     */
    double hx() const {return hx_;}
    /**
     * @brief cell size in y
     *
     * @return 
     */
    double hy() const {return hy_;}
    /**
     * @brief cell size in z
     *
     * @return 
     */
    double hz() const {return hz_;}
    /**
     * @brief Factor
     *
     * @return 
     */
    double fx() const {return fx_;}
    /**
     * @brief Factor
     *
     * @return 
     */
    double fy() const {return fy_;}
    /**
     * @brief number of polynomial coefficients in x and y
     *
     * @return 
     */
    unsigned n() const {return n_;}
    /**
     * @brief number of points in x
     *
     * @return 
     */
    unsigned Nx() const {return Nx_;}
    /**
     * @brief number of topological cells in x
     *
     * @return 
     */
    unsigned inner_Nx() const {return Nx_ - outer_Nx();}
    /**
     * @brief number of smooth rows in x
     *
     * @return 
     */
    unsigned outer_Nx() const {return (unsigned)floor(fx_*(double)Nx_+0.5);}
    /**
     * @brief number of cells in y
     *
     * @return 
     */
    unsigned Ny() const {return Ny_;}
    /**
     * @brief number of cells in the inner region of y
     *
     * @return 
     */
    unsigned inner_Ny() const {return Ny_-2*outer_Ny();}
    /**
     * @brief number of cells in one of the outer regions of y
     *
     * @return 
     */
    unsigned outer_Ny() const {return (unsigned)floor(fy_*(double)Ny_+0.5);}
    /**
     * @brief number of points in z
     *
     * @return 
     */
    unsigned Nz() const {return Nz_;}
    /**
     * @brief boundary conditions in x 
     *
     * @return 
     */
    bc bcx() const {return bcx_;}
    /**
     * @brief boundary conditions in y
     *
     * @return 
     */
    bc bcy() const {return bcy_;}
    /**
     * @brief boundary conditions in z 
     *
     * @return 
     */
    bc bcz() const {return bcz_;}
    /**
     * @brief Return a copy without topology
     *
     * @return 
     */
    Grid3d grid() const {return Grid3d( x0_,x1_,y0_,y1_,z0_,z1_,n_,Nx_,Ny_,Nz_,bcx_,bcy_,bcz_);}
    /**
     * @brief discrete legendre transformation
     *
     * @return 
     */
    const DLT<double>& dlt() const{return dlt_;}
    /**
     * @brief doublehe total number of points
     *
     * @return n*n*Nx*Ny*Nz
     */
    unsigned size() const { return n_*n_*Nx_*Ny_*Nz_;}
    /**
     * @brief Display 
     *
     * @param os output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "Grid parameters are: \n"
            <<"    n  = "<<n_<<"\n"
            <<"    Nx = "<<Nx_<<"\n"
            <<"    inner Nx = "<<inner_Nx()<<"\n"
            <<"    outer Nx = "<<outer_Nx()<<"\n"
            <<"    Ny = "<<Ny_<<"\n"
            <<"    inner Ny = "<<inner_Ny()<<"\n"
            <<"    outer Ny = "<<outer_Ny()<<"\n"
            <<"    Nz = "<<Nz_<<"\n"
            <<"    hx = "<<hx_<<"\n"
            <<"    hy = "<<hy_<<"\n"
            <<"    hz = "<<hz_<<"\n"
            <<"    x0 = "<<x0_<<"\n"
            <<"    x1 = "<<x1_<<"\n"
            <<"    y0 = "<<y0_<<"\n"
            <<"    y1 = "<<y1_<<"\n"
            <<"    z0 = "<<z0_<<"\n"
            <<"    z1 = "<<z1_<<"\n"
            <<"    lx = "<<lx_<<"\n"
            <<"    ly = "<<ly_<<"\n"
            <<"    lz = "<<lz_<<"\n"
            <<"Boundary conditions in x are: \n"
            <<"    "<<bc2str(bcx_)<<"\n"
            <<"Boundary conditions in y are: \n"
            <<"    "<<bc2str(bcy_)<<"\n"
            <<"Boundary conditions in z are: \n"
            <<"    "<<bc2str(bcz_)<<"\n";
    }
    /**
     * @brief Check if the grid contains a point
     *
     * @note doesn't check periodicity!!
     * @param x x-point to check
     * @param y y-point to check
     * @param z z-point to check
     *
     * @return true if x is between x0 and x1, false else
     */
    bool contains( double x, double y, double z)const
    {
        if( (x>=x0_ && x <= x1_) && (y>=y0_ && y <= y1_) && (z>=z0_ && z<=z1_)) 
            return true; 
        return false;
    }
  protected:
    virtual void init_X_boundaries( double x0, double x1)
    {
        x0_ = x0, x1_ = x1;
        assert( x1 > x0 );
        lx_ = (x1_-x0_);
        hx_ = lx_/(double)Nx_;
    }
  private:
    double x0_, x1_, y0_, y1_, z0_, z1_;
    double fx_,fy_;
    double lx_, ly_, lz_;
    unsigned n_, Nx_, Ny_, Nz_;
    double hx_, hy_, hz_;
    bc bcx_, bcy_, bcz_;
    DLT<double> dlt_;
};
///@}

///@cond
GridX2d::GridX2d( const GridX3d& g) : x0_(g.x0()), x1_(g.x1()), y0_(g.y0()), y1_(g.y1()), fx_(g.fx()), fy_(g.fy()), n_(g.n()), Nx_(g.Nx()), Ny_(g.Ny()), bcx_(g.bcx()), bcy_(g.bcy()), dlt_(g.n())
{}
///@endcond

}// namespace dg
