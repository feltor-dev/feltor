#pragma once

#include <cassert>
#include <cmath>
#include <thrust/host_vector.h>
#include "topological_traits.h"
#include "dlt.h"
#include "../enums.h"

/*! @file
  @brief base topology classes
  */

/*!@class hide_grid_parameters2d
 * @brief Equal polynomial coefficients
 *
 * @param x0 left boundary in x
 * @param x1 right boundary in x
 * @param y0 lower boundary in y
 * @param y1 upper boundary in y
 * @param n  # of polynomial coefficients for both x and y dimension
 *  (1<=n<=20)
 * @param Nx # of points in x
 * @param Ny # of points in y
 */
/*!@class hide_bc_parameters2d
 * @param bcx boundary condition in x
 * @param bcy boundary condition in y
 */
/*!@class hide_grid_parameters3d
 * @brief Equal polynomial coefficients
 *
 * @param x0 left boundary in x
 * @param x1 right boundary in x
 * @param y0 lower boundary in y
 * @param y1 upper boundary in y
 * @param z0 lower boundary in z
 * @param z1 upper boundary in z
 * @param n  # of polynomial coefficients for x and y dimension ( z-direction
 *  is set to 1) (1<=n<=20)
 * @param Nx # of points in x
 * @param Ny # of points in y
 * @param Nz # of points in z
 */
/*!@class hide_bc_parameters3d
 * @param bcx boundary condition in x
 * @param bcy boundary condition in y
 * @param bcz boundary condition in z
 */
/*!@class hide_shift_doc
 * @brief Shift any point coordinate to a corresponding grid coordinate according to the boundary condition
 *
 * If the given point is already inside the grid, the function does nothing, else along each dimension the following happens: check the boundary condition.
 *If \c dg::PER, the point will be shifted topologically back onto the domain (modulo operation). Else the
 * point will be mirrored at the closest boundary. If the boundary is a Dirichlet boundary (happens for \c dg::DIR, \c dg::DIR_NEU and \c dg::NEU_DIR; the latter two apply \c dg::DIR to the respective left or right boundary )
 * an additional sign flag is swapped. This process is repeated until the result lies inside the grid. This function forms the basis for extending/periodifying a
 * function discretized on the grid beyond the grid boundaries.
 * @sa interpolate
 * @note For periodic boundaries the right boundary point is considered outside the grid and is shifted to the left boundary point.
 * @param negative swap value if there was a sign swap (happens when a point is mirrored along a Dirichlet boundary)
 * @param x point to shift (inout) the result is guaranteed to lie inside the grid
 */

namespace dg{

///@cond
template<class real_type>
struct RealGrid2d;
template<class real_type>
struct RealGrid3d;
///@endcond

/**
* @brief 1D grid
* @ingroup grid
* @copydoc hide_code_evaluate1d
*/
template<class real_type>
struct RealGrid1d
{
    using value_type = real_type;
    /// The host vector type used by host functions like evaluate
    using host_vector = thrust::host_vector<real_type>;
    using host_grid = RealGrid1d<real_type>;
    /**
     * @brief construct an empty grid
     * this leaves the access functions undefined
     */
    RealGrid1d(){}
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
    RealGrid1d( real_type x0, real_type x1, unsigned n, unsigned N, bc bcx = PER)
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
    real_type x0() const {return x0_;}
    /**
     * @brief right boundary
     *
     * @return
     */
    real_type x1() const {return x1_;}
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
    //////////////////////////////////////////set/////////////////////////////
    /**
     * @brief reset the boundaries of the grid
     *
     * @param x0 new left boundary
     * @param x1 new right boundary ( > x0)
     * @param bcx new boundary condition
     */
    void set(real_type x0, real_type x1, bc bcx)
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
        dlt_=DLT<real_type>(n);
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
    void set( real_type x0, real_type x1, unsigned n, unsigned N, bc bcx)
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
    const DLT<real_type>& dlt() const {return dlt_;}
    void display( std::ostream& os = std::cout) const
    {
        os << "Topology parameters are: \n"
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
     * @copydoc hide_shift_doc
     */
    void shift( bool& negative, real_type& x)const
    {
        shift( negative, x, bcx_);
    }
    /**
     * @copydoc hide_shift_doc
     * @param bcx overrule grid internal boundary condition with this value
     */
    void shift( bool& negative, real_type &x, bc bcx)const
    {
        if( bcx == dg::PER)
        {
            real_type N = floor((x-x0_)/(x1_-x0_)); // ... -2[ -1[ 0[ 1[ 2[ ...
            x = x - N*(x1_-x0_); //shift
        }
        //mirror along boundary as often as necessary
        while( (x<x0_) || (x>x1_) )
        {
            if( x < x0_){
                x = 2.*x0_ - x;
                //every mirror swaps the sign if Dirichlet
                if( bcx == dg::DIR || bcx == dg::DIR_NEU)
                    negative = !negative;//swap sign
            }
            if( x > x1_){
                x = 2.*x1_ - x;
                if( bcx == dg::DIR || bcx == dg::NEU_DIR) //notice the different boundary NEU_DIR to the above DIR_NEU !
                    negative = !negative; //swap sign
            }
        }
    }

    /**
     * @brief Check if the grid contains a point
     *
     * @note Does not consider periodicity!!
     * @param x point to check
     *
     * @return true if x0()<=x<=x1(), false else
     * @attention returns false if x is NaN or INF
     */
    bool contains( real_type x)const
    {
        if( !std::isfinite(x) ) return false;
        //should we catch the case x1_==x && dg::PER?
        if( (x>=x0_ && x <= x1_)) return true;
        return false;
    }

  private:
    real_type x0_, x1_;
    unsigned n_, Nx_;
    bc bcx_;
    DLT<real_type> dlt_;
};

/**
 * @brief An abstract base class for two-dimensional grids
 * @note although it is abstract, objects are not meant to be hold on the heap via a base class pointer ( we protected the destructor)
 * @ingroup basictopology
 */
template<class real_type>
struct aRealTopology2d
{
    using value_type = real_type;
    /// The host vector type used by host functions like evaluate
    using host_vector = thrust::host_vector<real_type>;
    using host_grid = RealGrid2d<real_type>;

    /**
     * @brief Left boundary in x
     *
     * @return
     */
    real_type x0() const {return gx_.x0();}
    /**
     * @brief Right boundary in x
     *
     * @return
     */
    real_type x1() const {return gx_.x1();}
    /**
     * @brief left boundary in y
     *
     * @return
     */
    real_type y0() const {return gy_.x0();}
    /**
     * @brief Right boundary in y
     *
     * @return
     */
    real_type y1() const {return gy_.x1();}
    /**
     * @brief length of x
     *
     * @return
     */
    real_type lx() const {return gx_.lx();}
    /**
     * @brief length of y
     *
     * @return
     */
    real_type ly() const {return gy_.lx();}
    /**
     * @brief cell size in x
     *
     * @return
     */
    real_type hx() const {return gx_.h();}
    /**
     * @brief cell size in y
     *
     * @return
     */
    real_type hy() const {return gy_.h();}
    /**
     * @brief number of polynomial coefficients in x
     *
     * @return
     */
    unsigned n() const {return gx_.n();}
    /// number of polynomial coefficients in x
    unsigned nx() const {return gx_.n();}
    /// number of polynomial coefficients in y
    unsigned ny() const {return gy_.n();}
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
    //const DLT<real_type>& dlt() const{return gx_.dlt();}
    /// discrete legendre transformation in x
    const DLT<real_type>& dltx() const{return gx_.dlt();}
    /// discrete legendre transformation in y
    const DLT<real_type>& dlty() const{return gy_.dlt();}

    /// The x-axis grid
    const RealGrid1d<real_type>& gx() const {return gx_;}
    /// The y-axis grid
    const RealGrid1d<real_type>& gy() const {return gy_;}

    /**
    * @brief Multiply the number of cells with a given factor
    *
    * With this function you can resize the grid ignorantly of its current size
    * the number of polynomial coefficients is left as is
    * @param fx new number of cells is the nearest integer to fx*Nx()
    * @param fy new number of cells is the nearest integer to fy*Ny()
    */
    void multiplyCellNumbers( real_type fx, real_type fy){
        if( fx != 1 || fy != 1)
            do_set(nx(), round(fx*(real_type)Nx()), ny(), round(fy*(real_type)Ny()));
    }
    /**
    * @brief Set the number of polynomials and cells
    *
    * Same as \c set(new_n,new_Nx,new_n,new_Ny)
    * @param new_n new number of %Gaussian nodes for both x and y
    * @param new_Nx new number of cells in x
    * @param new_Ny new number of cells in y
    */
    void set( unsigned new_n, unsigned new_Nx, unsigned new_Ny) {
        set( new_n, new_Nx, new_n, new_Ny);
    }
    /**
    * @brief Set the number of polynomials and cells
    *
    * @param new_nx new number of %Gaussian nodes in x
    * @param new_Nx new number of cells in x
    * @param new_ny new number of %Gaussian nodes in y
    * @param new_Ny new number of cells in y
    */
    void set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny) {
        if( new_nx==nx() && new_Nx==Nx() && new_ny==ny() && new_Ny == Ny())
            return;
        do_set(new_nx,new_Nx,new_ny,new_Ny);
    }


    /**
     * @brief The total number of points
     *
     * @return nx*ny*Nx*Ny
     */
    unsigned size() const { return gx_.size()*gy_.size();}
    /**
     * @brief Display
     *
     * @param os output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "Topology parameters are: \n"
            <<"    nx = "<<nx()<<"\n"
            <<"    ny = "<<ny()<<"\n"
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
     * @copydoc hide_shift_doc
     * @param y point (y) to shift (inout) the result is guaranteed to lie inside the grid
     */
    void shift( bool& negative, real_type& x, real_type& y)const
    {
        shift( negative, x, y, bcx(), bcy());
    }
    /**
     * @copydoc hide_shift_doc
     * @param y point (y) to shift (inout) the result is guaranteed to lie inside the grid
     * @param bcx overrule grid internal boundary condition with this value
     * @param bcy overrule grid internal boundary condition with this value
     */
    void shift( bool& negative, real_type& x, real_type& y, bc bcx, bc bcy)const
    {
        gx_.shift( negative, x,bcx);
        gy_.shift( negative, y,bcy);
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
    bool contains( real_type x, real_type y)const
    {
        if( gx_.contains(x) && gy_.contains(y)) return true;
        return false;
    }
    /// Shortcut for contains( x[0], x[1])
    template<class Vector>
    bool contains( const Vector& x) const{
        return contains( x[0], x[1]);
    }
    protected:
    ///disallow destruction through base class pointer
    ~aRealTopology2d() = default;
    /**
     * @brief Construct a 2d grid as the product of two 1d grids
     *
     * @code
     * dg::Grid2d g2d( {x0,x1,nx,Nx,bcx},{y0,y1,ny,Ny,bcy});
     * @endcode
     * @param gx a Grid in x - direction
     * @param gy a Grid in y - direction
     */
    aRealTopology2d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy): gx_(gx),gy_(gy) { }

    ///explicit copy constructor (default)
    ///@param src source
    aRealTopology2d(const aRealTopology2d& src) = default;
    ///explicit assignment operator (default)
    ///@param src source
    aRealTopology2d& operator=(const aRealTopology2d& src) = default;
    virtual void do_set( unsigned new_nx, unsigned new_Nx, unsigned new_ny,
            unsigned new_Ny)=0;
    private:
    RealGrid1d<real_type> gx_, gy_;
};



/**
 * @brief An abstract base class for three-dimensional grids
 * @note although it is abstract, objects are not meant to be hold on the heap via a base class pointer ( we protected the destructor)
 * @ingroup basictopology
 */
template<class real_type>
struct aRealTopology3d
{
    using value_type = real_type;
    /// The host vector type used by host functions like evaluate
    using host_vector = thrust::host_vector<real_type>;
    using host_grid = RealGrid3d<real_type>;

    /**
     * @brief left boundary in x
     *
     * @return
     */
    real_type x0() const {return gx_.x0();}
    /**
     * @brief right boundary in x
     *
     * @return
     */
    real_type x1() const {return gx_.x1();}

    /**
     * @brief left boundary in y
     *
     * @return
     */
    real_type y0() const {return gy_.x0();}
    /**
     * @brief right boundary in y
     *
     * @return
     */
    real_type y1() const {return gy_.x1();}

    /**
     * @brief left boundary in z
     *
     * @return
     */
    real_type z0() const {return gz_.x0();}
    /**
     * @brief right boundary in z
     *
     * @return
     */
    real_type z1() const {return gz_.x1();}

    /**
     * @brief length in x
     *
     * @return
     */
    real_type lx() const {return gx_.lx();}
    /**
     * @brief length in y
     *
     * @return
     */
    real_type ly() const {return gy_.lx();}
    /**
     * @brief length in z
     *
     * @return
     */
    real_type lz() const {return gz_.lx();}

    /**
     * @brief cell size in x
     *
     * @return
     */
    real_type hx() const {return gx_.h();}
    /**
     * @brief cell size in y
     *
     * @return
     */
    real_type hy() const {return gy_.h();}
    /**
     * @brief cell size in z
     *
     * @return
     */
    real_type hz() const {return gz_.h();}
    /**
     * @brief number of polynomial coefficients in x
     *
     * @return
     */
    unsigned n() const {return gx_.n();}
    /// number of polynomial coefficients in x
    unsigned nx() const {return gx_.n();}
    /// number of polynomial coefficients in y
    unsigned ny() const {return gy_.n();}
    /// number of polynomial coefficients in z
    unsigned nz() const {return gz_.n();}
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
    /// discrete legendre transformation in x
    const DLT<real_type>& dltx() const{return gx_.dlt();}
    /// discrete legendre transformation in y
    const DLT<real_type>& dlty() const{return gy_.dlt();}
    /// discrete legendre transformation in z
    const DLT<real_type>& dltz() const{return gz_.dlt();}
    /// The x-axis grid
    const RealGrid1d<real_type>& gx() const {return gx_;}
    /// The y-axis grid
    const RealGrid1d<real_type>& gy() const {return gy_;}
    /// The z-axis grid
    const RealGrid1d<real_type>& gz() const {return gz_;}
    /**
     * @brief The total number of points
     *
     * @return nx*ny*nz*Nx*Ny*Nz
     */
    unsigned size() const { return gx_.size()*gy_.size()*gz_.size();}
    /**
     * @brief Display
     *
     * @param os output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "Topology parameters are: \n"
            <<"    nx = "<<nx()<<"\n"
            <<"    ny = "<<ny()<<"\n"
            <<"    nz = "<<nz()<<"\n"
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
     * @copydoc hide_shift_doc
     * @param y point (y) to shift (inout) the result is guaranteed to lie inside the grid
     * @param z point (z) to shift (inout) the result is guaranteed to lie inside the grid
     */
    void shift( bool& negative, real_type& x, real_type& y, real_type& z)const
    {
        shift( negative, x,y,z, bcx(), bcy(), bcz());
    }
    /**
     * @copydoc hide_shift_doc
     * @param y point (y) to shift (inout) the result is guaranteed to lie inside the grid
     * @param z point (z) to shift (inout) the result is guaranteed to lie inside the grid
     * @param bcx overrule grid internal boundary condition with this value
     * @param bcy overrule grid internal boundary condition with this value
     * @param bcz overrule grid internal boundary condition with this value
     */
    void shift( bool& negative, real_type& x, real_type& y, real_type& z, bc bcx, bc bcy, bc bcz)const
    {
        gx_.shift( negative, x,bcx);
        gy_.shift( negative, y,bcy);
        gz_.shift( negative, z,bcz);
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
    bool contains( real_type x, real_type y, real_type z)const
    {
        if( gx_.contains(x) && gy_.contains(y) && gz_.contains(z))
            return true;
        return false;
    }
    /// Shortcut for contains( x[0], x[1], x[2])
    template<class Vector>
    bool contains( const Vector& x) const{
        return contains( x[0], x[1], x[2]);
    }
    ///@copydoc aRealTopology2d::multiplyCellNumbers()
    void multiplyCellNumbers( real_type fx, real_type fy){
        if( fx != 1 || fy != 1)
            do_set(nx(), round(fx*(real_type)Nx()), ny(),
                    round(fy*(real_type)Ny()), nz(), Nz());
    }
    /**
    * @brief Set the number of polynomials and cells
    *
    * Set \c nz to 1
    * Same as \c set(new_n,new_Nx,new_n,new_Ny,1,new_Nz)
    * @param new_n new number of %Gaussian nodes in x and y
    * @param new_Nx new number of cells in x
    * @param new_Ny new number of cells in y
    * @param new_Nz new number of cells in z
    */
    void set( unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) {
        set(new_n,new_Nx,new_n,new_Ny,1,new_Nz);
    }
    /**
    * @brief Set the number of polynomials and cells
    *
    * @param new_nx new number of %Gaussian nodes in x
    * @param new_Nx new number of cells in x
    * @param new_ny new number of %Gaussian nodes in y
    * @param new_Ny new number of cells in y
    * @param new_nz new number of %Gaussian nodes in z
    * @param new_Nz new number of cells in z
    */
    void set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny, unsigned new_nz, unsigned new_Nz) {
        if( new_nx==nx() && new_Nx ==Nx() && new_ny == ny() && new_Ny == Ny() && new_nz == nz() && new_Nz==Nz())
            return;
        do_set(new_nx,new_Nx,new_ny,new_Ny,new_nz,new_Nz);
    }
    protected:
    ///disallow deletion through base class pointer
    ~aRealTopology3d() = default;
    /**
     * @brief Construct a 3d topology as the product of three 1d grids
     *
     * @code
     * dg::Grid3d g3d( {x0,x1,nx,Nx,bcx},{y0,y1,ny,Ny,bcy},{z0,z1,nz,Nz,bcz});
     * @endcode
     * @param gx a Grid1d in x - direction
     * @param gy a Grid1d in y - direction
     * @param gz a Grid1d in z - direction
     */
    aRealTopology3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz):
        gx_(gx),gy_(gy),gz_(gz){
    }
    ///explicit copy constructor (default)
    ///@param src source
    aRealTopology3d(const aRealTopology3d& src) = default;
    ///explicit assignment operator (default)
    ///@param src source
    aRealTopology3d& operator=(const aRealTopology3d& src) = default;
    virtual void do_set(unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny, unsigned new_nz, unsigned new_Nz)=0;
  private:
    RealGrid1d<real_type> gx_,gy_,gz_;
};

/**
 * @brief The simplest implementation of aRealTopology2d
 * @ingroup grid
 * @copydoc hide_code_evaluate2d
 */
template<class real_type>
struct RealGrid2d : public aRealTopology2d<real_type>
{
    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_bc_parameters2d
    RealGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx = PER, bc bcy = PER):
        aRealTopology2d<real_type>({x0,x1,n,Nx,bcx},{y0,y1,n,Ny,bcy}) { }

    ///@copydoc aRealTopology2d()
    RealGrid2d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy): aRealTopology2d<real_type>(gx,gy){ }

    ///@brief allow explicit type conversion from any other topology
    ///@param src source
    explicit RealGrid2d( const aRealTopology2d<real_type>& src): aRealTopology2d<real_type>(src){}
    private:
    virtual void do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny) override final{
        aRealTopology2d<real_type>::do_set(nx,Nx,ny,Ny);
    }

};

/**
 * @brief The simplest implementation of aRealTopology3d
 * @ingroup grid
 * @copydoc hide_code_evaluate3d
 */
template<class real_type>
struct RealGrid3d : public aRealTopology3d<real_type>
{
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    RealGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = PER, bc bcy = PER, bc bcz=PER):
        aRealTopology3d<real_type>({x0,x1,n,Nx,bcx},{y0,y1,n,Ny,bcy},{z0,z1,1,Nz,bcz}) { }
    ///@copydoc aRealTopology3d::aRealTopology3d(RealGrid1d,RealGrid1d,RealGrid1d)
    RealGrid3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz): aRealTopology3d<real_type>(gx,gy,gz){ }

    ///@brief allow explicit type conversion from any other topology
    ///@param src source
    explicit RealGrid3d( const aRealTopology3d<real_type>& src): aRealTopology3d<real_type>(src){ }
    private:
    virtual void do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny,
            unsigned nz, unsigned Nz) override final{
        aRealTopology3d<real_type>::do_set(nx,Nx,ny,Ny,nz,Nz);
    }
};

///@cond
template<class real_type>
void aRealTopology2d<real_type>::do_set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny)
{
    gx_.set(new_nx, new_Nx);
    gy_.set(new_ny, new_Ny);
}
template<class real_type>
void aRealTopology3d<real_type>::do_set(unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny, unsigned new_nz, unsigned new_Nz)
{
    gx_.set(new_nx, new_Nx);
    gy_.set(new_ny, new_Ny);
    gz_.set(new_nz, new_Nz);
}

template<class Topology>
using get_host_vector = typename Topology::host_vector;

template<class Topology>
using get_host_grid = typename Topology::host_grid;

///@endcond

///@addtogroup gridtypes
///@{
using Grid1d        = dg::RealGrid1d<double>;
using Grid2d        = dg::RealGrid2d<double>;
using Grid3d        = dg::RealGrid3d<double>;
using aTopology2d   = dg::aRealTopology2d<double>;
using aTopology3d   = dg::aRealTopology3d<double>;
#ifndef MPI_VERSION
namespace x {
using Grid1d        = Grid1d      ;
using Grid2d        = Grid2d      ;
using Grid3d        = Grid3d      ;
using aTopology2d   = aTopology2d ;
using aTopology3d   = aTopology3d ;
} //namespace x
#endif
///@}

}// namespace dg
