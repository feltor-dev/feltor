#ifndef _DG_GRID_CUH_
#define _DG_GRID_CUH_


#include <cassert>

namespace dg{

enum bc{ 
    PER, 
    DIR
};

template< class T, size_t n>
struct Grid
{
    Grid( T x0, T x1, T y0, T y1, unsigned Nx, unsigned Ny): x0_(x0), x1_(x1), y0_(y0), y1_(y1), 
                                                             Nx_(Nx), Ny_(Ny)
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
  private:
    T x0_, x1_, y0_, y1_;
    T lx_, ly_;
    unsigned Nx_, Ny_;
    T hx_, hy_;
    bc bcx, bcy;
};
}// namespace dg
#endif // _DG_GRID_CUH_
