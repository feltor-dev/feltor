#pragma once

#include <cassert>
#include "dlt.h"
#include "../enums.h"

/*! @file 
  
  Grid objects
  */


namespace dg{

struct SharedTag;

///@addtogroup grid
///@{
/**
* @brief 1D grid
*
* @tparam T value type
*/
template <class T>
struct Grid1d
{
    typedef SharedTag memory_category;
    /**
     * @brief 1D grid
     * 
     @param x0 left boundary
     @param x1 right boundary
     @param n # of polynomial coefficients
     @param N # of cells
     @param bcx boundary conditions
     */
    Grid1d( T x0, T x1, unsigned n, unsigned N, bc bcx = PER):
        x0_(x0), x1_(x1),
        n_(n), Nx_(N), bcx_(bcx), dlt_(n)
    {
        assert( x1 > x0 );
        assert( N > 0  );
        assert( n != 0 );
        lx_ = (x1-x0);
        hx_ = lx_/(double)Nx_;
    }
    /**
     * @brief left boundary
     *
     * @return 
     */
    T x0() const {return x0_;}
    /**
     * @brief right boundary
     *
     * @return 
     */
    T x1() const {return x1_;}
    /**
     * @brief total length of interval
     *
     * @return 
     */
    T lx() const {return lx_;}
    /**
     * @brief cell size
     *
     * @return 
     */
    T h() const {return hx_;}
    /**
     * @brief number of cells
     *
     * @return 
     */
    unsigned N() const {return Nx_;}
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
     * @brief The total number of points
     *
     * @return n*Nx
     */
    unsigned size() const { return n_*Nx_;}
    /**
     * @brief the discrete legendre transformation
     *
     * @return 
     */
    const DLT<T>& dlt() const {return dlt_;}
    void display( std::ostream& os = std::cout) const
    {
        os << "Grid parameters are: \n"
            <<"    n  = "<<n_<<"\n"
            <<"    N = "<<Nx_<<"\n"
            <<"    h = "<<hx_<<"\n"
            <<"    x0 = "<<x0_<<"\n"
            <<"    x1 = "<<x1_<<"\n"
            <<"    lx = "<<lx_<<"\n"
            <<"Boundary conditions in x are: \n";
        switch(bcx_)
        {
            case(dg::PER): os << "    PERIODIC \n"; break;
            case(dg::DIR): os << "    DIRICHLET\n"; break;
            default: os << "    Not specified!!\n"; 
        }
    }

    void shift_topologic( double x0, double& x1)
    {
        assert( contains(x0));
        double deltaX;
        if( x1 > x0_) deltaX = x1 -x0_;
        else deltaX = x1_ - x1;
        unsigned N = floor(deltaX/lx_);
        if( x1  > x1_ && bcx == dg::PER) x1 -= N*lx_;
        if( x1  < x0_ && bcx == dg::PER) x1 += N*lx_;
    }
    bool contains( double x)
    {
        if( (x>=x0_ && x <= x1_)) return true; 
        return false;
    }
  private:
    T x0_, x1_;
    T lx_;
    unsigned n_, Nx_;
    T hx_;
    bc bcx_;
    DLT<T> dlt_;
};

/**
 * @brief A 2D grid class 
 *
 * @tparam T scalar value type 
 */
template< class T>
struct Grid2d
{
    typedef SharedTag memory_category;
    /**
     * @brief Construct a 2D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param n  # of polynomial coefficients per dimension
     * @param Nx # of points in x 
     * @param Ny # of points in y
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     */
    Grid2d( T x0, T x1, T y0, T y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER):
        x0_(x0), x1_(x1), y0_(y0), y1_(y1), 
        n_(n), Nx_(Nx), Ny_(Ny), bcx_(bcx), bcy_( bcy), dlt_(n)
    {
        assert( n != 0);
        assert( x1 > x0 && y1 > y0);
        assert( Nx > 0  && Ny > 0);
        lx_ = (x1_-x0_), ly_ = (y1_-y0_);
        hx_ = lx_/(double)Nx_, hy_ = ly_/(double)Ny_;
    }
    /**
     * @brief Construct a 2d grid as the product of two 1d grids
     *
     * @param gx Grid in x - direction
     * @param gy Grid in y - direction
     */
    Grid2d( const Grid1d<T>& gx, const Grid1d<T>& gy): 
        x0_(gx.x0()), x1_(gx.x1()), y0_(gy.x0()), y1_(gy.x1()), 
        n_(gx.n()), Nx_(gx.N()), Ny_(gy.N()), bcx_(gx.bcx()), bcy_( gy.bcx()), dlt_(gx.n())
    {
        assert( gx.n() == gy.n() );
        lx_ = (x1_-x0_), ly_ = (y1_-y0_);
        hx_ = lx_/(double)Nx_, hy_ = ly_/(double)Ny_;
    }
    /**
     * @brief Left boundary in x
     *
     * @return 
     */
    T x0() const {return x0_;}
    /**
     * @brief Right boundary in x
     *
     * @return 
     */
    T x1() const {return x1_;}
    /**
     * @brief left boundary in y
     *
     * @return 
     */
    T y0() const {return y0_;}
    /**
     * @brief Right boundary in y 
     *
     * @return 
     */
    T y1() const {return y1_;}
    /**
     * @brief length of x 
     *
     * @return 
     */
    T lx() const {return lx_;}
    /**
     * @brief length of y
     *
     * @return 
     */
    T ly() const {return ly_;}
    /**
     * @brief cell size in x 
     *
     * @return 
     */
    T hx() const {return hx_;}
    /**
     * @brief cell size in y
     *
     * @return 
     */
    T hy() const {return hy_;}
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
     * @brief number of cells in y
     *
     * @return 
     */
    unsigned Ny() const {return Ny_;}
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
    Grid2d<double> local_grid() const {return *this;}
    /**
     * @brief discrete legendre trafo
     *
     * @return 
     */
    const DLT<T>& dlt() const{return dlt_;}
    /**
     * @brief The total number of points
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
            <<"    Ny = "<<Ny_<<"\n"
            <<"    hx = "<<hx_<<"\n"
            <<"    hy = "<<hy_<<"\n"
            <<"    x0 = "<<x0_<<"\n"
            <<"    x1 = "<<x1_<<"\n"
            <<"    y0 = "<<y0_<<"\n"
            <<"    y1 = "<<y1_<<"\n"
            <<"    lx = "<<lx_<<"\n"
            <<"    ly = "<<ly_<<"\n"
            <<"Boundary conditions in x are: \n";
        switch(bcx_)
        {
            case(dg::PER): os << "    PERIODIC \n"; break;
            case(dg::DIR): os << "    DIRICHLET\n"; break;
            default: os << "    Not specified!!\n"; 
        }
        os <<"Boundary conditions in y are: \n";
        switch(bcy_)
        {
            case(dg::PER): os << "    PERIODIC \n"; break;
            case(dg::DIR): os << "    DIRICHLET\n"; break;
            default: os << "    Not specified!!\n"; 
        }
    }
    void shift_topologic( double x0, double y0, double& x1, double& y1)
    {
        assert( contains(x0, y0));
        double deltaX;
        if( x1 > x0_) deltaX = (x1 -x0_);
        else deltaX = x1_ - x1;
        unsigned N = floor(deltaX/lx_);
        if( x1  > x1_ && bcx == dg::PER) x1 -= N*lx_;
        if( x1  < x0_ && bcx == dg::PER) x1 += N*lx_;
        double deltaY;
        if( y1 > y0_) deltaY = (y1 -y0_);
        else deltaY = y1_ - y1;
        unsigned N = floor(deltaY/ly_);
        if( y1  > y1_ && bcy == dg::PER) y1 -= N*ly_;
        if( y1  < y0_ && bcy == dg::PER) y1 += N*ly_;
    }
    bool contains( double x, double y)
    {
        if( (x>=x0_ && x <= x1_) && (y>=y0_ && y <= y1_)) return true; 
        return false;
    }
  private:
    T x0_, x1_, y0_, y1_;
    T lx_, ly_;
    unsigned n_, Nx_, Ny_;
    T hx_, hy_;
    bc bcx_, bcy_;
    DLT<T> dlt_;
};

/**
 * @brief A 3D grid class  for cartesian coordinates
 *
 * In the third dimension only 1 polynomial coefficient is used,
 * not n.
 * @tparam T scalar value type 
 */
template< class T>
struct Grid3d
{
    typedef SharedTag memory_category;
    /**
     * @brief Construct a 3D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param z0 lower boundary in z
     * @param z1 upper boundary in z 
     * @param n  # of polynomial coefficients per (x-,y-) dimension
     * @param Nx # of points in x 
     * @param Ny # of points in y
     * @param Nz # of points in z
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     * @param bcz boundary condition in z
     * @attention # of polynomial coefficients in z direction is always 1
     */
    Grid3d( T x0, T x1, T y0, T y1, T z0, T z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER):
        x0_(x0), x1_(x1), y0_(y0), y1_(y1), z0_(z0), z1_(z1),
        n_(n), Nx_(Nx), Ny_(Ny), Nz_(Nz), bcx_(bcx), bcy_( bcy), bcz_( bcz), dlt_(n)
    {
        assert( n != 0);
        assert( x1 > x0 && y1 > y0 ); assert( z1 > z0 );         
        assert( Nx > 0  && Ny > 0); assert( Nz > 0);

        lx_ = (x1-x0), ly_ = (y1-y0), lz_ = (z1-z0);
        hx_ = lx_/(double)Nx_, hy_ = ly_/(double)Ny_, hz_ = lz_/(double)Nz_;
    }
    /**
     * @brief Construct a 3d cartesian grid as the product of three 1d grids
     *
     * @param gx Grid in x - direction
     * @param gy Grid in y - direction
     * @param gz Grid in z - direction
     */
    Grid3d( const Grid1d<T>& gx, const Grid1d<T>& gy, const Grid1d<T>& gz): 
        x0_(gx.x0()), x1_(gx.x1()),  
        y0_(gy.x0()), y1_(gy.x1()),
        z0_(gz.x0()), z1_(gz.x1()),
        n_(gx.n()), Nx_(gx.N()), Ny_(gy.N()), Nz_(gz.N()),
        bcx_(gx.bcx()), bcy_( gy.bcx()), bcz_(gz.bcx()), 
        dlt_(gx.n())
    {
        assert( gx.n() == gy.n() );
        lx_ = (x1_-x0_), ly_ = (y1_-y0_), lz_ = (z1_-z0_);
        hx_ = lx_/(double)Nx_, hy_ = ly_/(double)Ny_, hz_ = lz_/(double)Nz_;
    }
    /**
     * @brief left boundary in x
     *
     * @return 
     */
    T x0() const {return x0_;}
    /**
     * @brief right boundary in x
     *
     * @return 
     */
    T x1() const {return x1_;}

    /**
     * @brief left boundary in y 
     *
     * @return 
     */
    T y0() const {return y0_;}
    /**
     * @brief right boundary in y
     *
     * @return 
     */
    T y1() const {return y1_;}

    /**
     * @brief left boundary in z
     *
     * @return 
     */
    T z0() const {return z0_;}
    /**
     * @brief right boundary in z
     *
     * @return 
     */
    T z1() const {return z1_;}

    /**
     * @brief length in x
     *
     * @return 
     */
    T lx() const {return lx_;}
    /**
     * @brief length in y
     *
     * @return 
     */
    T ly() const {return ly_;}
    /**
     * @brief length in z
     *
     * @return 
     */
    T lz() const {return lz_;}
    
    /**
     * @brief cell size in x
     *
     * @return 
     */
    T hx() const {return hx_;}
    /**
     * @brief cell size in y
     *
     * @return 
     */
    T hy() const {return hy_;}
    /**
     * @brief cell size in z
     *
     * @return 
     */
    T hz() const {return hz_;}
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
     * @brief number of points in y
     *
     * @return 
     */
    unsigned Ny() const {return Ny_;}
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
     * @brief discrete legendre transformation
     *
     * @return 
     */
    const DLT<T>& dlt() const{return dlt_;}
    /**
     * @brief The total number of points
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
            <<"    Ny = "<<Ny_<<"\n"
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
            <<"Boundary conditions in x are: \n";
        switch(bcx_)
        {
            case(dg::PER): os << "    PERIODIC \n"; break;
            case(dg::DIR): os << "    DIRICHLET\n"; break;
            default: os << "    Not specified!!\n"; 
        }
        os <<"Boundary conditions in y are: \n";
        switch(bcy_)
        {
            case(dg::PER): os << "    PERIODIC \n"; break;
            case(dg::DIR): os << "    DIRICHLET\n"; break;
            default: os << "    Not specified!!\n"; 
        }
        os <<"Boundary conditions in z are: \n";
        switch(bcz_)
        {
            case(dg::PER): os << "    PERIODIC \n"; break;
            case(dg::DIR): os << "    DIRICHLET\n"; break;
            default: os << "    Not specified!!\n"; 
        }
    }
    void shift_topologic( double x0, double y0, double z0, double& x1, double& y1, double& z1)
    {
        assert( contains(x0, y0, z0));
        double deltaX;
        if( x1 > x0_) deltaX = (x1 -x0_);
        else deltaX = x1_ - x1;
        unsigned N = floor(deltaX/lx_);
        if( x1  > x1_ && bcx == dg::PER) x1 -= N*lx_;
        if( x1  < x0_ && bcx == dg::PER) x1 += N*lx_;
        double deltaY;
        if( y1 > y0_) deltaY = (y1 -y0_);
        else deltaY = y1_ - y1;
        unsigned N = floor(deltaY/ly_);
        if( y1  > y1_ && bcy == dg::PER) y1 -= N*ly_;
        if( y1  < y0_ && bcy == dg::PER) y1 += N*ly_;
        double deltaZ;
        if( z1 > z0_) deltaZ = (z1 -z0_);
        else deltaZ = z1_ - z1;
        unsigned N = floor(deltaZ/lz_);
        if( z1  > z1_ && bcz == dg::PER) z1 -= N*lz_;
        if( z1  < z0_ && bcz == dg::PER) z1 += N*lz_;
    }
    bool contains( double x, double y, double z)
    {
        if( (x>=x0_ && x <= x1_) && (y>=y0_ && y <= y1_) && (z>=z0_ && z<=z1_)) 
            return true; 
        return false;
    }
  private:
    T x0_, x1_, y0_, y1_, z0_, z1_;
    T lx_, ly_, lz_;
    unsigned n_, Nx_, Ny_, Nz_;
    T hx_, hy_, hz_;
    bc bcx_, bcy_, bcz_;
    DLT<T> dlt_;
};

///@}
}// namespace dg
