#ifndef _DG_GRID_CUH_
#define _DG_GRID_CUH_

#include <cassert>

/*! @file Grid objects
  */


namespace dg{


    //future plans: grid object should be the one which computes node coordinates (as vec2d?)

/**
 * @brief Switch between boundary conditions
 * 
 * @ingroup creation
 */
enum bc{ 
    PER, //!< periodic boundaries
    DIR //!< homogeneous dirichlet boundaries
};

///@addtogroup grid
///@{
/**
* @brief 1D grid
*
* @tparam T value type
* @tparam n # of polynomial coefficients
*/
template <class T, size_t n>
struct Grid1d
{
    /**
     * @brief 1D grid
     * 
     @param x0 left boundary
     @param x1 right boundary
     @param N # of cells
     @param bcx boundary conditions
     */
    Grid1d( T x0, T x1, unsigned N, bc bcx = PER):
        x0_(x0), x1_(x1),
        Nx_(N), bcx_(bcx)
    {
        assert( x1 > x0 );
        assert( N > 0  );
        lx_ = (x1-x0);
        hx_ = lx_/(double)Nx_;
    }
    T x0() const {return x0_;}
    T x1() const {return x1_;}
    T lx() const {return lx_;}
    T h() const {return hx_;}
    unsigned N() const {return Nx_;}
    bc bcx() const {return bcx_;}
    /**
     * @brief The total number of points
     *
     * @return n*Nx
     */
    unsigned size() const { return n*Nx_;}
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
    /**
     * @brief The total number of points
     *
     * @return n*n*Nx*Ny
     */
    unsigned size() const { return n*n*Nx_*Ny_;}
    void display( std::ostream& os = std::cout) const
    {
        os << "Grid parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx_<<"\n"
            <<"    Ny = "<<Ny_<<"\n"
            <<"    hx = "<<hx_<<"\n"
            <<"    hy = "<<hy_<<"\n"
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
  private:
    T x0_, x1_, y0_, y1_;
    T lx_, ly_;
    unsigned Nx_, Ny_;
    T hx_, hy_;
    bc bcx_, bcy_;
};

///@}
}// namespace dg
#endif // _DG_GRID_CUH_
