#ifndef _DG_GRID_CUH_
#define _DG_GRID_CUH_

#include <cassert>

namespace dg{

enum bc{ 
    PER, //!< periodic boundaries
    DIR //!< homogeneous dirichlet boundaries
};

template <class T, size_t n>
struct Grid1d
{
    Grid1d( T x0, T x1, unsigned Nx, bc bcx = PER):
        x0_(x0), x1_(x1),
        Nx_(Nx), bcx_(bcx)
    {
        assert( x1 > x0 );
        assert( Nx > 0  );
        lx_ = (x1-x0);
        hx_ = lx_/(double)Nx_;
    }
    T x0() const {return x0_;}
    T x1() const {return x1_;}
    T lx() const {return lx_;}
    T h() const {return hx_;}
    unsigned N() const {return Nx_;}
    bc bcx() const {return bcx_;}
  private:
    T x0_, x1_;
    T lx_;
    unsigned Nx_;
    T hx_;
    bc bcx_;
};
/**
 * @brief A 2D grid class 
 *
 * @tparam T scalar value type 
 * @tparam n number of polynomial coefficients
 */
template< class T, size_t n>
struct Grid
{
    /**
     * @brief Construct a 2D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param Nx # of points in x 
     * @param Ny # of points in y
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     */
    Grid( T x0, T x1, T y0, T y1, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER):
        x0_(x0), x1_(x1), y0_(y0), y1_(y1), 
        Nx_(Nx), Ny_(Ny), bcx_(bcx), bcy_( bcy)
    {
        assert( x1 > x0 && y1 > y0);
        assert( Nx > 0  && Ny > 0);
        lx_ = (x1-x0), ly_ = (y1-y0);
        hx_ = lx_/(double)Nx_, hy_ = ly_/(double)Ny_;
    }
    T x0() const {return x0_;}
    T x1() const {return x1_;}
    T y0() const {return y0_;}
    T y1() const {return y1_;}
    T lx() const {return lx_;}
    T ly() const {return ly_;}
    T hx() const {return hx_;}
    T hy() const {return hy_;}
    unsigned Nx() const {return Nx_;}
    unsigned Ny() const {return Ny_;}
    bc bcx() const {return bcx_;}
    bc bcy() const {return bcy_;}
  private:
    T x0_, x1_, y0_, y1_;
    T lx_, ly_;
    unsigned Nx_, Ny_;
    T hx_, hy_;
    bc bcx_, bcy_;
};

}// namespace dg
#endif // _DG_GRID_CUH_
