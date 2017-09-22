#pragma once

#include <cassert>
#include <cmath>
#include "topological_traits.h"
#include "dlt.h"
#include "../enums.h"

/*! @file 
  @brief base topology classes
  */

/*!@class hide_grid_parameters2d
 * @brief Construct a 2D grid
 *
 * @param x0 left boundary in x
 * @param x1 right boundary in x 
 * @param y0 lower boundary in y
 * @param y1 upper boundary in y 
 * @param n  # of polynomial coefficients per dimension
 *  (1<=n<=20, note that the library is optimized for n=3 )
 * @param Nx # of points in x 
 * @param Ny # of points in y
 */
/*!@class hide_bc_parameters2d
 * @param bcx boundary condition in x
 * @param bcy boundary condition in y
 */

/*!@class hide_grid_parameters3d
 * @brief Construct a 3D grid
 *
 * @param x0 left boundary in x
 * @param x1 right boundary in x 
 * @param y0 lower boundary in y
 * @param y1 upper boundary in y 
 * @param z0 lower boundary in z
 * @param z1 upper boundary in z 
 * @param n  # of polynomial coefficients per (x-,y-) dimension
 *   (1<=n<=20, note that the library is optimized for n=3 )
 * @attention # of polynomial coefficients in z direction is always 1
 * @param Nx # of points in x 
 * @param Ny # of points in y
 * @param Nz # of points in z
 */
/*!@class hide_bc_parameters3d
 * @param bcx boundary condition in x
 * @param bcy boundary condition in y
 * @param bcz boundary condition in z
 */

namespace dg{

/**
* @brief 1D grid
* @ingroup grid
*
*/
struct Grid1d
{
    typedef SharedTag memory_category;
    typedef OneDimensionalTag dimensionality;
    /**
     * @brief construct an empty grid
     * this leaves the access functions undefined
     */
    Grid1d(){}
    /**
     * @brief 1D grid
     * 
     * @param x0 left boundary
     * @param x1 right boundary
     * @param n # of polynomial coefficients 
     *  (1<=n<=20, note that the library is optimized for n=3 )
     * @param N # of cells
     * @param bcx boundary conditions
     */
    Grid1d( double x0, double x1, unsigned n, unsigned N, bc bcx = PER)
    {
        set(x0,x1,bcx);
        set(n,N);
    }
    //////////////////////////////////////////get/////////////////////////////
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
     * @brief total length of interval
     *
     * @return 
     */
    double lx() const {return x1_-x0_;}
    /**
     * @brief cell size
     *
     * @return 
     */
    double h() const {return lx()/(double)Nx_;}
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
    //////////////////////////////////////////set/////////////////////////////
    /**
     * @brief reset the boundaries of the grid
     *
     * @param x0 new left boundary
     * @param x1 new right boundary ( > x0)
     * @param bcx new boundary condition
     */
    void set(double x0, double x1, bc bcx)
    {
        assert( x1 > x0 );
        x0_=x0, x1_=x1;
        bcx_=bcx;
    }
    /**
     * @brief reset the cell numbers in the grid
     *
     * @param n new # of polynomial coefficients (0<n<21)
     * @param N new # of cells (>0)
     */
    void set( unsigned n, unsigned N)
    {
        assert( N > 0  );
        Nx_=N; n_=n;
        dlt_=DLT<double>(n);
    }
    /**
     * @brief Reset all values of the grid
     *
     * @param x0 new left boundary
     * @param x1 new right boundary
     * @param n new # of polynomial coefficients
     * @param N new # of cells
     * @param bcx new boundary condition
     */
    void set( double x0, double x1, unsigned n, unsigned N, bc bcx)
    {
        set(x0,x1,bcx);
        set(n,N);
    }
    /////////////////////////////////////////convencience//////////////////////////////
    unsigned size() const { return n_*Nx_;}
    /**
     * @brief the discrete legendre transformation
     *
     * @return 
     */
    const DLT<double>& dlt() const {return dlt_;}
    void display( std::ostream& os = std::cout) const
    {
        os << "aTopology parameters are: \n"
            <<"    n  = "<<n_<<"\n"
            <<"    N = "<<Nx_<<"\n"
            <<"    h = "<<h()<<"\n"
            <<"    x0 = "<<x0_<<"\n"
            <<"    x1 = "<<x1_<<"\n"
            <<"    lx = "<<lx()<<"\n"
            <<"Boundary conditions in x are: \n"
            <<"    "<<bc2str(bcx_)<<"\n";
    }

    /**
     * @brief Shifts a point coordinate if periodic
     *
     * This function shifts a point coordinate to its value between x0() and x1() if bcx() returns dg::PER
     * @param x0 arbitrary point (irrelevant for the function, it's there to be consistent with GridX1d)
     * @param x1 end point (inout)
     */
    void shift_topologic( double x0, double& x1)const
    {
        double deltaX;
        if( x1 > x0_) deltaX = x1 -x0_;
        else deltaX = x1_ - x1;
        unsigned N = floor(deltaX/lx());
        if( x1  > x1_ && bcx_ == dg::PER) x1 -= N*lx();
        if( x1  < x0_ && bcx_ == dg::PER) x1 += N*lx();
    }

    /**
     * @brief Check if the grid contains a point
     *
     * @note Doesn't check periodicity!!
     * @param x point to check
     *
     * @return true if x0()<=x<=x1(), false else
     */
    bool contains( double x)const
    {
        if( (x>=x0_ && x <= x1_)) return true; 
        return false;
    }

  private:
    double x0_, x1_;
    unsigned n_, Nx_;
    bc bcx_;
    DLT<double> dlt_;
};

/**
 * @brief An abstract base class for two-dimensional grids
 * @note although it is abstract objects are not meant to be hold on the heap via a base class pointer ( we protected the destructor)
 * @ingroup basictopology
 */
struct aTopology2d
{
    typedef SharedTag memory_category; //!< tag for choosing default host vector type
    typedef TwoDimensionalTag dimensionality; 

    /**
     * @brief Left boundary in x
     *
     * @return 
     */
    double x0() const {return gx_.x0();}
    /**
     * @brief Right boundary in x
     *
     * @return 
     */
    double x1() const {return gx_.x1();}
    /**
     * @brief left boundary in y
     *
     * @return 
     */
    double y0() const {return gy_.x0();}
    /**
     * @brief Right boundary in y 
     *
     * @return 
     */
    double y1() const {return gy_.x1();}
    /**
     * @brief length of x 
     *
     * @return 
     */
    double lx() const {return gx_.lx();}
    /**
     * @brief length of y
     *
     * @return 
     */
    double ly() const {return gy_.lx();}
    /**
     * @brief cell size in x 
     *
     * @return 
     */
    double hx() const {return gx_.h();}
    /**
     * @brief cell size in y
     *
     * @return 
     */
    double hy() const {return gy_.h();}
    /**
     * @brief number of polynomial coefficients in x and y
     *
     * @return 
     */
    unsigned n() const {return gx_.n();}
    /**
     * @brief number of cells in x
     *
     * @return 
     */
    unsigned Nx() const {return gx_.N();}
    /**
     * @brief number of cells in y
     *
     * @return 
     */
    unsigned Ny() const {return gy_.N();}
    /**
     * @brief boundary conditions in x
     *
     * @return 
     */
    bc bcx() const {return gx_.bcx();}
    /**
     * @brief boundary conditions in y
     *
     * @return 
     */
    bc bcy() const {return gy_.bcx();}
    /**
     * @brief discrete legendre trafo
     *
     * @return 
     */
    const DLT<double>& dlt() const{return gx_.dlt();}

    /**
    * @brief Multiply the number of cells with a given factor
    *
    * With this function you can resize the grid ignorantly of its current size
    * the number of polynomial coefficients is left as is
    * @param fx new number of cells is the nearest integer to fx*Nx()
    * @param fy new number of cells is the nearest integer to fy*Ny()
    */
    void multiplyCellNumbers( double fx, double fy){
        do_set(n(), round(fx*(double)Nx()), round(fy*(double)Ny()));
    }
    /**
    * @brief Set the number of polynomials and cells
    *
    * @param new_n new number of %Gaussian nodes
    * @param new_Nx new number of cells in x 
    * @param new_Ny new number of cells in y
    */
    void set( unsigned new_n, unsigned new_Nx, unsigned new_Ny) {
        if( !( new_n==n() && new_Nx==Nx() && new_Ny == Ny() ) ) 
            do_set(new_n,new_Nx,new_Ny);
    }


    /**
     * @brief The total number of points
     *
     * @return n*n*Nx*Ny
     */
    unsigned size() const { return gx_.size()*gy_.size();}
    /**
     * @brief Display 
     *
     * @param os output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "aTopology parameters are: \n"
            <<"    n  = "<<n()<<"\n"
            <<"    Nx = "<<Nx()<<"\n"
            <<"    Ny = "<<Ny()<<"\n"
            <<"    hx = "<<hx()<<"\n"
            <<"    hy = "<<hy()<<"\n"
            <<"    x0 = "<<x0()<<"\n"
            <<"    x1 = "<<x1()<<"\n"
            <<"    y0 = "<<y0()<<"\n"
            <<"    y1 = "<<y1()<<"\n"
            <<"    lx = "<<lx()<<"\n"
            <<"    ly = "<<ly()<<"\n"
            <<"Boundary conditions in x are: \n"
            <<"    "<<bc2str(bcx())<<"\n"
            <<"Boundary conditions in y are: \n"
            <<"    "<<bc2str(bcy())<<"\n";
    }
    /**
     * @brief Shifts point coordinates if periodic
     *
     * This function shifts point coordinates to its values inside
     the domain if the respective boundary condition is periodic
     * @param x0 arbitrary coordinate (irrelevant for the function, it's there to be consistent with aTopologyX2d)
     * @param y0 arbitrary coordinate (irrelevant for the function, it's there to be consistent with aTopologyX2d)
     * @param x1 x-coordinate to shift (inout)
     * @param y1 y-coordinate to shift (inout)
     */
    void shift_topologic( double x0, double y0, double& x1, double& y1)const
    {
        gx_.shift_topologic( x0,x1);
        gy_.shift_topologic( y0,y1);
    }
    /**
     * @brief Check if the grid contains a point
     *
     * @note doesn't check periodicity!!
     * @param x x-coordinate to check
     * @param y y-coordinate to check
     *
     * @return true if x0()<=x<=x1() and y0()<=y<=y1(), false else
     */
    bool contains( double x, double y)const
    {
        if( gx_.contains(x) && gy_.contains(y)) return true; 
        return false;
    }
    protected:
    ///disallow destruction through base class pointer
    ~aTopology2d(){}
    /**
     *@copydoc hide_grid_parameters2d
     *@copydoc hide_bc_parameters2d
     */
    aTopology2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy):
        gx_(x0,x1,n,Nx,bcx), gy_(y0,y1,n,Ny,bcy) { }
    /**
     * @brief Construct a 2d grid as the product of two 1d grids
     *
     * @param gx a Grid in x - direction
     * @param gy a Grid in y - direction
     * @note gx and gy must have the same n
     */
    aTopology2d( const Grid1d& gx, const Grid1d& gy): gx_(gx),gy_(gy)
    {
        assert( gx.n() == gy.n() );
    }

    ///explicit copy constructor (default)
    ///@param src source 
    aTopology2d(const aTopology2d& src){gx_=src.gx_, gy_=src.gy_;}
    ///explicit assignment operator (default)
    ///@param src source 
    aTopology2d& operator=(const aTopology2d& src){
        gx_=src.gx_;
        gy_=src.gy_;
        return *this;
    }
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny)=0;
    private:
    Grid1d gx_, gy_; 
};



/**
 * @brief An abstract base class for three-dimensional grids
 * @note although it is abstract objects are not meant to be hold on the heap via a base class pointer ( we protected the destructor)
 * @ingroup basictopology
 */
struct aTopology3d
{
    typedef SharedTag memory_category;
    typedef ThreeDimensionalTag dimensionality;
    ///@copydoc aTopology2d::multiplyCellNumbers()
    void multiplyCellNumbers( double fx, double fy){
        set(n(), round(fx*(double)Nx()), round(fy*(double)Ny()), Nz());
    }
    /**
    * @brief Set the number of polynomials and cells
    *
    * @param new_n new number of %Gaussian nodes
    * @param new_Nx new number of cells in x 
    * @param new_Ny new number of cells in y
    * @param new_Nz new number of cells in z
    */
    void set( unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) {
        if(!( new_n==n() && new_Nx ==Nx() && new_Ny == Ny() && new_Nz==Nz())) 
            do_set(new_n,new_Nx,new_Ny,new_Nz);
    }

    /**
     * @brief left boundary in x
     *
     * @return 
     */
    double x0() const {return gx_.x0();}
    /**
     * @brief right boundary in x
     *
     * @return 
     */
    double x1() const {return gx_.x1();}

    /**
     * @brief left boundary in y 
     *
     * @return 
     */
    double y0() const {return gy_.x0();}
    /**
     * @brief right boundary in y
     *
     * @return 
     */
    double y1() const {return gy_.x1();}

    /**
     * @brief left boundary in z
     *
     * @return 
     */
    double z0() const {return gz_.x0();}
    /**
     * @brief right boundary in z
     *
     * @return 
     */
    double z1() const {return gz_.x1();}

    /**
     * @brief length in x
     *
     * @return 
     */
    double lx() const {return gx_.lx();}
    /**
     * @brief length in y
     *
     * @return 
     */
    double ly() const {return gy_.lx();}
    /**
     * @brief length in z
     *
     * @return 
     */
    double lz() const {return gz_.lx();}
    
    /**
     * @brief cell size in x
     *
     * @return 
     */
    double hx() const {return gx_.h();}
    /**
     * @brief cell size in y
     *
     * @return 
     */
    double hy() const {return gy_.h();}
    /**
     * @brief cell size in z
     *
     * @return 
     */
    double hz() const {return gz_.h();}
    /**
     * @brief number of polynomial coefficients in x and y
     *
     * @return 
     */
    unsigned n() const {return gx_.n();}
    /**
     * @brief number of points in x
     *
     * @return 
     */
    unsigned Nx() const {return gx_.N();}
    /**
     * @brief number of points in y
     *
     * @return 
     */
    unsigned Ny() const {return gy_.N();}
    /**
     * @brief number of points in z
     *
     * @return 
     */
    unsigned Nz() const {return gz_.N();}
    /**
     * @brief boundary conditions in x 
     *
     * @return 
     */
    bc bcx() const {return gx_.bcx();}
    /**
     * @brief boundary conditions in y
     *
     * @return 
     */
    bc bcy() const {return gy_.bcx();}
    /**
     * @brief boundary conditions in z 
     *
     * @return 
     */
    bc bcz() const {return gz_.bcx();}
    /**
     * @brief discrete legendre transformation
     *
     * @return 
     */
    const DLT<double>& dlt() const{return gx_.dlt();}
    /**
     * @brief The total number of points
     *
     * @return n*n*Nx*Ny*Nz
     */
    unsigned size() const { return gx_.size()*gy_.size()*gz_.size();}
    /**
     * @brief Display 
     *
     * @param os output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "aTopology parameters are: \n"
            <<"    n  = "<<n()<<"\n"
            <<"    Nx = "<<Nx()<<"\n"
            <<"    Ny = "<<Ny()<<"\n"
            <<"    Nz = "<<Nz()<<"\n"
            <<"    hx = "<<hx()<<"\n"
            <<"    hy = "<<hy()<<"\n"
            <<"    hz = "<<hz()<<"\n"
            <<"    x0 = "<<x0()<<"\n"
            <<"    x1 = "<<x1()<<"\n"
            <<"    y0 = "<<y0()<<"\n"
            <<"    y1 = "<<y1()<<"\n"
            <<"    z0 = "<<z0()<<"\n"
            <<"    z1 = "<<z1()<<"\n"
            <<"    lx = "<<lx()<<"\n"
            <<"    ly = "<<ly()<<"\n"
            <<"    lz = "<<lz()<<"\n"
            <<"Boundary conditions in x are: \n"
            <<"    "<<bc2str(bcx())<<"\n"
            <<"Boundary conditions in y are: \n"
            <<"    "<<bc2str(bcy())<<"\n"
            <<"Boundary conditions in z are: \n"
            <<"    "<<bc2str(bcz())<<"\n";
    }

    /**
     * @brief Shifts point coordinates if periodic
     *
     * This function shifts point coordinates to its values inside
     the domain if the respective boundary condition is periodic
     * @param x0 arbitrary x-coordinate (irrelevant for the function, it's there to be consistent with aTopologyX3d)
     * @param y0 arbitrary y-coordinate (irrelevant for the function, it's there to be consistent with aTopologyX3d)
     * @param z0 arbitrary z-coordinate (irrelevant for the function, it's there to be consistent with aTopologyX3d)
     * @param x1 x-coordinate to shift (inout)
     * @param y1 y-coordinate to shift (inout)
     * @param z1 z-coordinate to shift (inout)
     */
    void shift_topologic( double x0, double y0, double z0, double& x1, double& y1, double& z1)const
    {
        gx_.shift_topologic( x0,x1);
        gy_.shift_topologic( y0,y1);
        gz_.shift_topologic( z0,z1);
    }

    /**
     * @brief Check if the grid contains a point
     *
     * @note doesn't check periodicity!!
     * @param x x-coordinate to check
     * @param y y-coordinate to check
     * @param z z-coordinate to check
     *
     * @return true if x0()<=x<=x1() and y0()<=y<=y1() and z0()<=z<=z1() , false else
     */
    bool contains( double x, double y, double z)const
    {
        if( gx_.contains(x) && gy_.contains(y) && gz_.contains(z)) 
            return true; 
        return false;
    }
    protected:
    ///disallow deletion through base class pointer
    ~aTopology3d(){}
    /**
    @copydoc hide_grid_parameters3d
    @copydoc hide_bc_parameters3d
     */
    aTopology3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz): 
        gx_(x0,x1,n,Nx,bcx),
        gy_(y0,y1,n,Ny,bcy),
        gz_(z0,z1,1,Nz,bcz){}
    /**
     * @brief Construct a 3d topology as the product of three 1d grids
     *
     * @param gx a Grid1d in x - direction
     * @param gy a Grid1d in y - direction
     * @param gz a Grid1d in z - direction
     * @note gx and gy must have the same n and gz.n() must return 1
     */
    aTopology3d( const Grid1d& gx, const Grid1d& gy, const Grid1d& gz): gx_(gx),gy_(gy),gz_(gz){
        assert( gx.n() == gy.n());
        assert( gz.n() == 1);
    }
    ///explicit copy constructor (default)
    ///@param src source 
    aTopology3d(const aTopology3d& src):gx_(src.gx_),gy_(src.gy_),gz_(src.gz_){}
    ///explicit assignment operator (default)
    ///@param src source 
    aTopology3d& operator=(const aTopology3d& src){ //use default in C++11
        gx_=src.gx_; gy_=src.gy_; gz_=src.gz_;
        return *this;
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx,unsigned new_Ny, unsigned new_Nz)=0;
  private:
    Grid1d gx_,gy_,gz_;
};

/**
 * @brief The simplest implementation of aTopology2d
 * @ingroup grid
 */
struct Grid2d : public aTopology2d
{

    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_bc_parameters2d
    Grid2d( double x0, double x1, double y0, double y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER):
        aTopology2d(x0,x1,y0,y1,n,Nx,Ny,bcx,bcy) { }
    ///@copydoc aTopology2d::aTopology2d(const Grid1d&,const Grid1d&)
    Grid2d( const Grid1d& gx, const Grid1d& gy): aTopology2d(gx,gy){ }
    ///allow explicit type conversion from any other topology
    explicit Grid2d( const aTopology2d& src): aTopology2d(src){}
    private:
    virtual void do_set( unsigned n, unsigned Nx, unsigned Ny){ 
        aTopology2d::do_set(n,Nx,Ny);
    }

};

/**
 * @brief The simplest implementation of aTopology3d
 * @ingroup grid
 */
struct Grid3d : public aTopology3d
{
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    Grid3d( double x0, double x1, double y0, double y1, double z0, double z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz=PER):
        aTopology3d(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz) { }
    ///@copydoc aTopology3d::aTopology3d(const Grid1d&,const Grid1d&,const Grid1d&)
    Grid3d( const Grid1d& gx, const Grid1d& gy, const Grid1d& gz): aTopology3d(gx,gy,gz){ }
    ///allow explicit type conversion from any other topology
    ///@param src source
    explicit Grid3d( const aTopology3d& src): aTopology3d(src){ }
    private:
    virtual void do_set( unsigned n, unsigned Nx, unsigned Ny, unsigned Nz){ 
        aTopology3d::do_set(n,Nx,Ny,Nz);
    }
};
//
///@cond
void aTopology2d::do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny)
{
    gx_.set(new_n, new_Nx);
    gy_.set(new_n, new_Ny);
}
void aTopology3d::do_set(unsigned new_n, unsigned new_Nx,unsigned new_Ny, unsigned new_Nz)
{
    gx_.set(new_n, new_Nx);
    gy_.set(new_n, new_Ny);
    gz_.set(1,new_Nz);
}
///@endcond

}// namespace dg
