#ifndef _DG_GRID_
#define _DG_GRID_

#include <cassert>
#include "dlt.h"
#include "grid.h"
#include "../enums.h"

/*! @file 
  
  Grid objects
  */


namespace dg{

struct SharedTag;

///@addtogroup grid
///@{
/**
* @brief 1D grid for X-point topology
*
* The grid looks like 
* |------x----------x------|
* , where the x indicates the periodicity and - the number of points
* The left boundary is x0 and the right x1, the inner boundaries lie at
* x0 + f*Lx and x1-f*Lx
* therefore f must be smaller than 0.5
*/
struct GridX1d
{
    typedef SharedTag memory_category;
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
    GridX1d( double x0, double x1, double f, unsigned n, unsigned N, bc bcx = PER):
        x0_(x0), x1_(x1), f_(f),
        n_(n), Nx_(N), bcx_(bcx), dlt_(n)
    {
        assert( (f > 0) && (f < 0.5) );
        assert( floor( f*(double)N ) == f*(double)N); 
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
    unsigned outer_N() const {return (unsigned)(f_*(double)Nx_);}
    /**
     * @brief number of cells in the inner region
     *
     * @return 
     */
    unsigned inner_N() const {return (unsigned)((1.-2.*f_)*(double)Nx_);}
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
    const DLT<double>& dlt() const {return dlt_;}
    Grid1d<double> grid() const{return Grid1d<double>( x0_, x1_, n_, Nx_, bcx_);}
  private:
    double x0_, x1_, f_;
    double lx_;
    unsigned n_, Nx_;
    double hx_;
    bc bcx_;
    DLT<double> dlt_;
};

/**
 * @brief A 2D grid class with X-point topology
 *
 * is of the form
 * |--- ---------- ---|
 * |--- ---------- ---|
 * |--- ---------- ---|
 * |---x----------x---|
 * |---x----------x---|
 *
 * @tparam double scalar value type 
 */
struct GridX2d
{
    typedef SharedTag memory_category;
    /**
     * @brief Construct a 2D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param fy factor for y-direction
     * @param n  # of polynomial coefficients per dimension
     * @param Nx # of points in x 
     * @param Nx_topological number of rows with X-point topology
     * @param Ny # of points in y
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     */
    GridX2d( double x0, double x1, double y0, double y1, double fy, unsigned n, unsigned Nx, unsigned Nx_topologic, unsigned Ny, bc bcx = PER, bc bcy = PER):
        x0_(x0), x1_(x1), y0_(y0), y1_(y1), f_(f), 
        n_(n), Nx_(Nx), Nx_top_(Nx_topologic), Ny_(Ny), bcx_(bcx), bcy_( bcy), dlt_(n)
    {
        assert( (f > 0) && (f < 0.5) );
        assert( floor( f*(double)Ny ) == f*(double)Ny); 
        assert( n != 0);
        assert( x1 > x0 && y1 > y0);
        assert( Nx > 0  && Ny > 0 && Nx_top_ > 0);
        assert( Nx_top_ < Nx);
        lx_ = (x1_-x0_), ly_ = (y1_-y0_);
        hx_ = lx_/(double)Nx_, hy_ = ly_/(double)Ny_;
    }
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
    unsigned Nx_top() const {return Nx_top_;}
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
    unsigned inner_Ny() const {return (unsigned)(f_*(double)Ny_);}
    /**
     * @brief number of cells in one of the outer regions of y
     *
     * @return 
     */
    unsigned outer_Ny() const {return (unsigned)((1.-2.*f)(double)Ny_);}
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
    GridX2d<double> local_grid() const {return *this;}
    /**
     * @brief Return a copy without topology
     *
     * @return 
     */
    Grid2d<double> grid() const {return Grid2d<double>( x0_,x1_,y0_,y1_,n_,Nx_,Ny_,bcx_,bcy_);}
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
  private:
    double x0_, x1_, y0_, y1_;
    double f_;
    double lx_, ly_;
    unsigned n_, Nx_, Nx_top_, Ny_;
    double hx_, hy_;
    bc bcx_, bcy_;
    DLT<double> dlt_;
};

/**
 * @brief A 3D grid class  for cartesian coordinates
 *
 * In the third dimension only 1 polynomial coefficient is used,
 * not n.
 * @tparam double scalar value type 
 */
struct GridX3d
{
    typedef SharedTag memory_category;
    /**
     * @brief Construct a 3D grid
     *
     * @param x0 left boundary in x
     * @param x1 right boundary in x 
     * @param y0 lower boundary in y
     * @param y1 upper boundary in y 
     * @param fy factor for y-direction
     * @param z0 lower boundary in z
     * @param z1 upper boundary in z 
     * @param n  # of polynomial coefficients per (x-,y-) dimension
     * @param Nx # of points in x 
     * @param Nx_topological number of rows with X-point topology
     * @param Ny # of points in y
     * @param Nz # of points in z
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     * @param bcz boundary condition in z
     * @attention # of polynomial coefficients in z direction is always 1
     */
    GridX3d( double x0, double x1, double y0, double y1, double fy, double z0, double z1, unsigned n, unsigned Nx, unsigned Nx_topological, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz = PER):
        x0_(x0), x1_(x1), y0_(y0), y1_(y1), z0_(z0), z1_(z1), f_(fy),
        n_(n), Nx_(Nx), Nx_top_(Nx_topological), Ny_(Ny), Nz_(Nz), bcx_(bcx), bcy_( bcy), bcz_( bcz), dlt_(n)
    {
        assert( (f > 0) && (f < 0.5) );
        assert( floor( f*(double)Ny ) == f*(double)Ny); 
        assert( n != 0);
        assert( x1 > x0 && y1 > y0 ); assert( z1 > z0 );         
        assert( Nx > 0  && Ny > 0); assert( Nz > 0);
        assert( Nx_top_ < Nx);
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
    unsigned Nx_top() const {return Nx_top_;}
    /**
     * @brief number of points in y
     *
     * @return 
     */
    unsigned Ny() const {return Ny_;}
    /**
     * @brief number of cells in the inner region of y
     *
     * @return 
     */
    unsigned inner_Ny() const {return (unsigned)(f_*(double)Ny_);}
    /**
     * @brief number of cells in one of the outer regions of y
     *
     * @return 
     */
    unsigned outer_Ny() const {return (unsigned)((1.-2.*f)(double)Ny_);}
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
    Grid3d<double> grid() const {return Grid3d<double>( x0_,x1_,y0_,y1_,z0_,z1_,n_,Nx_,Ny_,Nz_,bcx_,bcy_,bcz_);}
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
  private:
    double x0_, x1_, y0_, y1_, z0_, z1_;
    double f_;
    double lx_, ly_, lz_;
    unsigned n_, Nx_, Nx_top_, Ny_, Nz_;
    double hx_, hy_, hz_;
    bc bcx_, bcy_, bcz_;
    DLT<double> dlt_;
};

///@}
}// namespace dg
#endif // _DG_GRID_
