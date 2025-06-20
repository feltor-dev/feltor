#pragma once
#include <cassert>
#include <cmath> //floor function
#include "dlt.h"
#include "grid.h"
#include "../enums.h"

/*! @file
  @brief base X-point topology classes
  */

/*!@class hide_gridX_parameters2d
 * @brief Construct a 2D X-point grid
 *
 * @param x0 left boundary in x
 * @param x1 right boundary in x
 * @param y0 lower boundary in y
 * @param y1 upper boundary in y
 * @param fx factor for the partition in x-direction (fx*Nx will be rounded)
 * @param fy factor for the partition in y-direction (fy*Ny will be rounded)
 * @param n  # of polynomial coefficients per dimension
 *   (1<=n<=20, note that the library is optimized for n=3 )
 * @param Nx # of points in x
 * @param Ny # of points in y
 */

/*!@class hide_gridX_parameters3d
 * @brief Construct a 3D X-point grid
 * @param x0 left boundary in x
 * @param x1 right boundary in x
 * @param y0 lower boundary in y
 * @param y1 upper boundary in y
 * @param z0 lower boundary in z
 * @param z1 upper boundary in z
 * @param fx factor for the partition in x-direction
 * @param fy factor for the partition in y-direction
 * @param n  # of polynomial coefficients per (x-,y-) dimension
*   (1<=n<=20, note that the library is optimized for n=3 )
* @attention # of polynomial coefficients in z direction is always 1
 * @param Nx # of points in x
 * @param Ny # of points in y
 * @param Nz # of points in z
 */

namespace dg{
///@cond
template<class real_type>
struct RealGridX2d;
template<class real_type>
struct RealGridX3d;
///@endcond

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
* @ingroup basicXtopology
*/
template<class real_type>
struct RealGridX1d
{
    using value_type = real_type;
    /// The host vector type used by host functions like evaluate
    using host_vector = thrust::host_vector<real_type>;
    using host_grid = RealGridX1d<real_type>;
    /// @brief number of dimensions : 1
    constexpr static unsigned ndim() {return 1;}
    RealGridX1d() = default;
    /**
     * @brief 1D X-point grid
     *
     * @param x0 left boundary
     * @param x1 right boundary
     * @param f factor 0<f<0.5 divides the domain
     * @param n # of polynomial coefficients
     * @param N # of cells
     * @param bcx boundary conditions
     */
    RealGridX1d( real_type x0, real_type x1, real_type f, unsigned n, unsigned N, bc bcx = NEU):
        x0_(x0), x1_(x1), f_(f),
        n_(n), Nx_(N), bcx_(bcx)
    {
        assert( (f >= 0) && (f < 0.5) );
        assert( fabs(outer_N() - f*(real_type)N) < 1e-14);
        assert( x1 > x0 );
        assert( N > 0  );
        assert( n != 0 );
        assert( bcx != PER);
    }
    host_vector abscissas(unsigned =0) const { return grid().abscissas(0);}
    host_vector weights(unsigned =0) const { return grid().weights(0);}
    unsigned shape(unsigned =0) const { return grid().shape(0);}
    auto get_shape() const { return grid().get_shape();}
    /**
     * @brief left boundary
     *
     * @return
     */
    real_type x0() const {return x0_;}
    /**
     * @brief right boundary
     *
     * @return
     */
    real_type x1() const {return x1_;}
    /**
     * @brief Factor
     *
     * @return
     */
    real_type f() const {return f_;}
    /**
     * @brief total length of interval
     *
     * @return
     */
    real_type lx() const {return x1_-x0_;}
    /**
     * @brief cell size
     *
     * @return
     */
    real_type h() const {return lx()/(real_type)Nx_;}
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
    unsigned outer_N() const {return (unsigned)(round(f_*(real_type)Nx_));}
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
     * @brief Display
     *
     * @param os output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "RealGrid parameters are: \n"
            <<"    n  = "<<n_<<"\n"
            <<"    N  = "<<Nx_<<"\n"
            <<"    inner N = "<<inner_N()<<"\n"
            <<"    outer N = "<<outer_N()<<"\n"
            <<"    h  = "<<h()<<"\n"
            <<"    x0 = "<<x0_<<"\n"
            <<"    x1 = "<<x1_<<"\n"
            <<"    lx = "<<lx()<<"\n"
            <<"Boundary conditions in x are: \n"
            <<"    "<<bc2str(bcx_)<<"\n";
    }
    RealGrid<real_type,1> grid() const{return RealGrid<real_type,1>( x0_, x1_, n_, Nx_, bcx_);}

    /**
     * @brief Shifts a point coordinate due to topology
     *
     * If you want to construct a point by adding delta X to a given point
     * x0 then the resulting coordinate x1 might be incorrect due to topologic reasons (periodic boundaries). This function corrects this coordinate
     * @param x0 starting point (must lie inside of the grid)
     * @param x1 end point (inout)
     */
    void shift_topologic( real_type x0, real_type& x1) const
    {
        assert( contains(x0));
        real_type deltaX;
        real_type xleft = x0_ + f_*lx();
        real_type xright = x1_ - f_*lx();
        if( x0 >= xleft && x0<xright)
        {
            if( x1 > xleft) deltaX = (x1 -xleft);
            else deltaX = xright - x1;
            unsigned N = floor(deltaX/(xright-xleft));
            if( x1  > xright ) x1 -= N*lx();
            if( x1  < xleft ) x1 += N*lx();
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
     * @return true if x0()<=x<=x1(), false else
     */
    bool contains( real_type x) const
    {
        if( (x>=x0_ && x <= x1_)) return true;
        return false;
    }
  private:
    real_type x0_, x1_, f_;
    unsigned n_, Nx_;
    bc bcx_;
};

//template<class real_type>
//struct aRealTopologyX3d; //forward declare 3d version

/**
 * @brief A 2D grid class with X-point topology
 *
 * is of the form
 @code
 | -----> y
 |  |---x----------x---|
 |  |---x----------x---|
 v  |--- ---------- ---|
 x  |--- ---------- ---| fx*lx
    |--- ---------- ---|
    fy*ly
 @endcode
 *
 * @ingroup basicXtopology
 */
template<class real_type>
struct aRealTopologyX2d
{
    using value_type = real_type;
    /// The host vector type used by host functions like evaluate
    using host_vector = thrust::host_vector<real_type>;
    using host_grid = RealGridX2d<real_type>;
    /// @brief number of dimensions : 2
    constexpr static unsigned ndim() {return 2;}
    host_vector abscissas(unsigned u=0) const { return grid().abscissas(u);}
    host_vector weights(unsigned u=0) const { return grid().weights(u);}
    unsigned shape(unsigned u=0) const { return grid().shape(u);}
    auto get_shape() const { return grid().get_shape();}

    /**
     * @brief Left boundary in x
     *
     * @return
     */
    real_type x0() const {return x0_;}
    /**
     * @brief Right boundary in x
     *
     * @return
     */
    real_type x1() const {return x1_;}
    /**
     * @brief left boundary in y
     *
     * @return
     */
    real_type y0() const {return y0_;}
    /**
     * @brief Right boundary in y
     *
     * @return
     */
    real_type y1() const {return y1_;}
    /**
     * @brief length of x
     *
     * @return
     */
    real_type lx() const {return x1_-x0_;}
    /**
     * @brief length of y
     *
     * @return
     */
    real_type ly() const {return y1_-y0_;}
    /**
     * @brief cell size in x
     *
     * @return
     */
    real_type hx() const {return lx()/(real_type)Nx_;}
    /**
     * @brief cell size in y
     *
     * @return
     */
    real_type hy() const {return ly()/(real_type)Ny_;}
    /**
     * @brief partition factor in x
     *
     * @return
     */
    real_type fx() const {return fx_;}
    /**
     * @brief partition factor in y
     *
     * @return
     */
    real_type fy() const {return fy_;}
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
    unsigned outer_Nx() const {return (unsigned)round(fx_*(real_type)Nx_);}
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
    unsigned outer_Ny() const {return (unsigned)round(fy_*(real_type)Ny_);}
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
     * @brief Return a copy without topology
     *
     * @return
     */
    RealGrid<real_type,2> grid() const {return RealGrid<real_type,2>( x0_,x1_,y0_,y1_,n_,Nx_,Ny_,bcx_,bcy_);}
    /**
     * @brief real_typehe total number of points
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
            <<"    hx = "<<hx()<<"\n"
            <<"    hy = "<<hy()<<"\n"
            <<"    x0 = "<<x0_<<"\n"
            <<"    x1 = "<<x1_<<"\n"
            <<"    y0 = "<<y0_<<"\n"
            <<"    y1 = "<<y1_<<"\n"
            <<"    lx = "<<lx()<<"\n"
            <<"    ly = "<<ly()<<"\n"
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
    void shift_topologic( real_type x0, real_type y0, real_type& x1, real_type& y1) const
    {
        assert( contains(x0, y0));
        real_type deltaX;
        if( x1 > x0_) deltaX = (x1 -x0_);
        else deltaX = x1_ - x1;
        unsigned N = floor(deltaX/lx());
        if( x1  > x1_ && bcx_ == dg::PER) x1 -= N*lx();
        if( x1  < x0_ && bcx_ == dg::PER) x1 += N*lx();

        if( x0 < x1_ - fx_*(x1_-x0_) ) //if x0 is  one of the inner points
        {
            real_type deltaY;
            real_type yleft = y0_ + fy_*ly();
            real_type yright = y1_ - fy_*ly();
            if( y0 >= yleft && y0<yright)
            {
                if( y1 > yleft) deltaY = (y1 -yleft);
                else deltaY = yright - y1;
                unsigned N = floor(deltaY/(yright-yleft));
                if( y1  > yright ) y1 -= N*ly();
                if( y1  < yleft ) y1 += N*ly();
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
     * @return true if x0()<=x<=x1() and y0()<=y<=y1(), false else
     */
    bool contains( real_type x, real_type y)const
    {
        if( (x>=x0_ && x <= x1_) && (y>=y0_ && y <= y1_)) return true;
        return false;
    }
  protected:
    ///disallow destruction through base class pointer
    ~aRealTopologyX2d() = default;
    aRealTopologyX2d() = default;
    ///@copydoc hide_gridX_parameters2d
    ///@copydoc hide_bc_parameters2d
    aRealTopologyX2d( real_type x0, real_type x1, real_type y0, real_type y1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy):
        x0_(x0), x1_(x1), y0_(y0), y1_(y1), fx_(fx), fy_(fy),
        n_(n), Nx_(Nx), Ny_(Ny), bcx_(bcx), bcy_( bcy)
    {
        assert( (fy_ >= 0.) && (fy_ < 0.5) );
        assert( (fx_ >= 0.) && (fx_ < 1.) );
        assert( fabs(outer_Nx() - fx_*(real_type)Nx) < 1e-14);
        assert( fabs(outer_Ny() - fy_*(real_type)Ny) < 1e-14);
        assert( n != 0);
        assert( x1 > x0 && y1 > y0);
        assert( Nx_ > 0  && Ny > 0 );
        assert( bcy != PER);
    }
    ///@copydoc aRealTopology::aRealTopology(const aRealTopology&)
    aRealTopologyX2d(const aRealTopologyX2d& src) = default;
    ///@copydoc aRealTopology::operator=
    aRealTopologyX2d& operator=(const aRealTopologyX2d& src) = default;
  private:
    real_type x0_, x1_, y0_, y1_;
    real_type fx_, fy_;
    unsigned n_, Nx_, Ny_;
    bc bcx_, bcy_;
};
/**
 * @brief The simplest implementation of aRealTopologyX2d
 * @ingroup basicXtopology
 */
template<class real_type>
struct RealGridX2d : public aRealTopologyX2d<real_type>
{
    ///@copydoc hide_gridX_parameters2d
    ///@copydoc hide_bc_parameters2d
    RealGridX2d( real_type x0, real_type x1, real_type y0, real_type y1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, bc bcx=PER, bc bcy=NEU):
        aRealTopologyX2d<real_type>(x0,x1,y0,y1,fx,fy,n,Nx,Ny,bcx,bcy) { }
    ///allow explicit type conversion from any other topology
    explicit RealGridX2d( const aRealTopologyX2d<real_type>& src): aRealTopologyX2d<real_type>(src){}
};

/**
 * @brief A 3D grid class with X-point topology
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
 * @ingroup basicXtopology
 */
template<class real_type>
struct aRealTopologyX3d
{
    using value_type = real_type;
    /// The host vector type used by host functions like evaluate
    using host_vector = thrust::host_vector<real_type>;
    using host_grid = RealGridX3d<real_type>;
    /// @brief number of dimensions : 3
    constexpr static unsigned ndim() {return 3;}
    host_vector abscissas(unsigned u=0) const { return grid().abscissas(u);}
    host_vector weights(unsigned u=0) const { return grid().weights(u);}
    unsigned shape(unsigned u=0) const { return grid().shape(u);}
    auto get_shape() const { return grid().get_shape();}
    /**
     * @brief left boundary in x
     *
     * @return
     */
    real_type x0() const {return x0_;}
    /**
     * @brief right boundary in x
     *
     * @return
     */
    real_type x1() const {return x1_;}

    /**
     * @brief left boundary in y
     *
     * @return
     */
    real_type y0() const {return y0_;}
    /**
     * @brief right boundary in y
     *
     * @return
     */
    real_type y1() const {return y1_;}

    /**
     * @brief left boundary in z
     *
     * @return
     */
    real_type z0() const {return z0_;}
    /**
     * @brief right boundary in z
     *
     * @return
     */
    real_type z1() const {return z1_;}

    /**
     * @brief length in x
     *
     * @return
     */
    real_type lx() const {return x1_-x0_;}
    /**
     * @brief length in y
     *
     * @return
     */
    real_type ly() const {return y1_-y0_;}
    /**
     * @brief length in z
     *
     * @return
     */
    real_type lz() const {return z1_-z0_;}

    /**
     * @brief cell size in x
     *
     * @return
     */
    real_type hx() const {return lx()/(real_type)Nx_;}
    /**
     * @brief cell size in y
     *
     * @return
     */
    real_type hy() const {return ly()/(real_type)Ny_;}
    /**
     * @brief cell size in z
     *
     * @return
     */
    real_type hz() const {return lz()/(real_type)Nz_;}
    /**
     * @brief partition factor in x
     *
     * @return
     */
    real_type fx() const {return fx_;}
    /**
     * @brief partition factor in y
     *
     * @return
     */
    real_type fy() const {return fy_;}
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
    unsigned outer_Nx() const {return (unsigned)round(fx_*(real_type)Nx_);}
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
    unsigned outer_Ny() const {return (unsigned)round(fy_*(real_type)Ny_);}
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
    RealGrid<real_type,3> grid() const {
        return RealGrid<real_type,3>( x0_,x1_,y0_,y1_,z0_,z1_,n_,Nx_,Ny_,Nz_,bcx_,bcy_,bcz_);
    }
    /**
     * @brief real_typehe total number of points
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
            <<"    hx = "<<hx()<<"\n"
            <<"    hy = "<<hy()<<"\n"
            <<"    hz = "<<hz()<<"\n"
            <<"    x0 = "<<x0_<<"\n"
            <<"    x1 = "<<x1_<<"\n"
            <<"    y0 = "<<y0_<<"\n"
            <<"    y1 = "<<y1_<<"\n"
            <<"    z0 = "<<z0_<<"\n"
            <<"    z1 = "<<z1_<<"\n"
            <<"    lx = "<<lx()<<"\n"
            <<"    ly = "<<ly()<<"\n"
            <<"    lz = "<<lz()<<"\n"
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
     * @return true if x0()<=x<=x1() and y0()<=y<=y1() and z0()<=z<=z1() , false else
     */
    bool contains( real_type x, real_type y, real_type z)const
    {
        if( (x>=x0_ && x <= x1_) && (y>=y0_ && y <= y1_) && (z>=z0_ && z<=z1_))
            return true;
        return false;
    }
  protected:
    ///disallow destruction through base class pointer
    ~aRealTopologyX3d() = default;
    aRealTopologyX3d() = default;
    ///@copydoc hide_gridX_parameters3d
    ///@copydoc hide_bc_parameters3d
    aRealTopologyX3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz):
        x0_(x0), x1_(x1), y0_(y0), y1_(y1), z0_(z0), z1_(z1), fx_(fx), fy_(fy),
        n_(n), Nx_(Nx), Ny_(Ny), Nz_(Nz), bcx_(bcx), bcy_( bcy), bcz_( bcz)
    {
        assert( (fy_ >= 0.) && (fy_ < 0.5) );
        assert( (fx_ >= 0.) && (fx_ < 1.) );
        assert( fabs(outer_Nx() - fx_*(real_type)Nx) < 1e-14);
        assert( fabs(outer_Ny() - fy_*(real_type)Ny) < 1e-14);
        assert( n != 0);
        assert( x1 > x0 && y1 > y0 ); assert( z1 > z0 );
        assert( Nx_ > 0  && Ny > 0); assert( Nz > 0);
    }
    ///@copydoc aRealTopology::aRealTopology(const aRealTopology&)
    aRealTopologyX3d(const aRealTopologyX3d& src) = default;
    ///@copydoc aRealTopology::operator=
    aRealTopologyX3d& operator=(const aRealTopologyX3d& src) = default;
  private:
    real_type x0_, x1_, y0_, y1_, z0_, z1_;
    real_type fx_,fy_;
    unsigned n_, Nx_, Ny_, Nz_;
    bc bcx_, bcy_, bcz_;
};

/**
 * @brief The simplest implementation of aRealTopologyX3d
 * @ingroup basicXtopology
 */
template<class real_type>
struct RealGridX3d : public aRealTopologyX3d<real_type>
{
    ///@copydoc hide_gridX_parameters3d
    ///@copydoc hide_bc_parameters3d
    RealGridX3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, real_type fx, real_type fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx=PER, bc bcy=NEU, bc bcz=PER):
        aRealTopologyX3d<real_type>(x0,x1,y0,y1,z0,z1,fx,fy,n,Nx,Ny,Nz,bcx,bcy,bcz) { }
    ///allow explicit type conversion from any other topology
    explicit RealGridX3d( const aRealTopologyX3d<real_type>& src): aRealTopologyX3d<real_type>(src){}
};

///@addtogroup gridtypes
///@{
using GridX1d       = dg::RealGridX1d<double>;
using GridX2d       = dg::RealGridX2d<double>;
using GridX3d       = dg::RealGridX3d<double>;
using aTopologyX2d  = dg::aRealTopologyX2d<double>;
using aTopologyX3d  = dg::aRealTopologyX3d<double>;
///@}

}// namespace dg
