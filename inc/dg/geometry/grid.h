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
* @copydoc hide_code_evaluate1d
*/
template<class real_type>
struct RealGrid1d
{
    typedef SharedTag memory_category;
    typedef OneDimensionalTag dimensionality;
    typedef real_type value_type;
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
     * @brief Shifts a point coordinate if periodic
     *
     * This function shifts a point coordinate to its value between x0() and x1() if bcx() returns dg::PER
     * @param x0 arbitrary point (irrelevant for the function, it's there to be consistent with GridX1d)
     * @param x1 end point (inout)
     */
    void shift_topologic( real_type x0, real_type& x1)const
    {
        real_type deltaX;
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
    bool contains( real_type x)const
    {
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
 * @note although it is abstract objects are not meant to be hold on the heap via a base class pointer ( we protected the destructor)
 * @ingroup basictopology
 */
template<class real_type>
struct aRealTopology2d
{
    typedef SharedTag memory_category; //!< tag for choosing default host vector type
    typedef TwoDimensionalTag dimensionality;
    typedef real_type value_type;

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
    const DLT<real_type>& dlt() const{return gx_.dlt();}

    /**
    * @brief Multiply the number of cells with a given factor
    *
    * With this function you can resize the grid ignorantly of its current size
    * the number of polynomial coefficients is left as is
    * @param fx new number of cells is the nearest integer to fx*Nx()
    * @param fy new number of cells is the nearest integer to fy*Ny()
    */
    void multiplyCellNumbers( real_type fx, real_type fy){
        do_set(n(), round(fx*(real_type)Nx()), round(fy*(real_type)Ny()));
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
        os << "Topology parameters are: \n"
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
     * @param x0 arbitrary coordinate (irrelevant for the function, it's there to be consistent with aRealTopologyX2d)
     * @param y0 arbitrary coordinate (irrelevant for the function, it's there to be consistent with aRealTopologyX2d)
     * @param x1 x-coordinate to shift (inout)
     * @param y1 y-coordinate to shift (inout)
     */
    void shift_topologic( real_type x0, real_type y0, real_type& x1, real_type& y1)const
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
    bool contains( real_type x, real_type y)const
    {
        if( gx_.contains(x) && gy_.contains(y)) return true;
        return false;
    }
    protected:
    ///disallow destruction through base class pointer
    ~aRealTopology2d() = default;
    /**
     *@copydoc hide_grid_parameters2d
     *@copydoc hide_bc_parameters2d
     */
    aRealTopology2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy):
        gx_(x0,x1,n,Nx,bcx), gy_(y0,y1,n,Ny,bcy) { }
    /**
     * @brief Construct a 2d grid as the product of two 1d grids
     *
     * @param gx a Grid in x - direction
     * @param gy a Grid in y - direction
     * @note gx and gy must have the same n
     */
    aRealTopology2d( const RealGrid1d<real_type>& gx, const RealGrid1d<real_type>& gy): gx_(gx),gy_(gy)
    {
        assert( gx.n() == gy.n() );
    }

    ///explicit copy constructor (default)
    ///@param src source
    aRealTopology2d(const aRealTopology2d& src) = default;
    ///explicit assignment operator (default)
    ///@param src source
    aRealTopology2d& operator=(const aRealTopology2d& src) = default;
    virtual void do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny)=0;
    private:
    RealGrid1d<real_type> gx_, gy_;
};



/**
 * @brief An abstract base class for three-dimensional grids
 * @note although it is abstract objects are not meant to be hold on the heap via a base class pointer ( we protected the destructor)
 * @ingroup basictopology
 */
template<class real_type>
struct aRealTopology3d
{
    typedef SharedTag memory_category;
    typedef ThreeDimensionalTag dimensionality;
    typedef real_type value_type;

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
    const DLT<real_type>& dlt() const{return gx_.dlt();}
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
        os << "Topology parameters are: \n"
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
     * @param x0 arbitrary x-coordinate (irrelevant for the function, it's there to be consistent with aRealTopologyX3d)
     * @param y0 arbitrary y-coordinate (irrelevant for the function, it's there to be consistent with aRealTopologyX3d)
     * @param z0 arbitrary z-coordinate (irrelevant for the function, it's there to be consistent with aRealTopologyX3d)
     * @param x1 x-coordinate to shift (inout)
     * @param y1 y-coordinate to shift (inout)
     * @param z1 z-coordinate to shift (inout)
     */
    void shift_topologic( real_type x0, real_type y0, real_type z0, real_type& x1, real_type& y1, real_type& z1)const
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
    bool contains( real_type x, real_type y, real_type z)const
    {
        if( gx_.contains(x) && gy_.contains(y) && gz_.contains(z))
            return true;
        return false;
    }
    ///@copydoc aRealTopology2d::multiplyCellNumbers()
    void multiplyCellNumbers( real_type fx, real_type fy){
        set(n(), round(fx*(real_type)Nx()), round(fy*(real_type)Ny()), Nz());
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
    protected:
    ///disallow deletion through base class pointer
    ~aRealTopology3d() = default;
    /**
    @copydoc hide_grid_parameters3d
    @copydoc hide_bc_parameters3d
     */
    aRealTopology3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz):
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
    aRealTopology3d( const RealGrid1d<real_type>& gx, const RealGrid1d<real_type>& gy, const RealGrid1d<real_type>& gz): gx_(gx),gy_(gy),gz_(gz){
        assert( gx.n() == gy.n());
        assert( gz.n() == 1);
    }
    ///explicit copy constructor (default)
    ///@param src source
    aRealTopology3d(const aRealTopology3d& src) = default;
    ///explicit assignment operator (default)
    ///@param src source
    aRealTopology3d& operator=(const aRealTopology3d& src) = default;
    virtual void do_set(unsigned new_n, unsigned new_Nx,unsigned new_Ny, unsigned new_Nz)=0;
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
        aRealTopology2d<real_type>(x0,x1,y0,y1,n,Nx,Ny,bcx,bcy) { }
    ///@copydoc aRealTopology2d::aRealTopology2d(const RealGrid1d&,const RealGrid1d&)
    RealGrid2d( const RealGrid1d<real_type>& gx, const RealGrid1d<real_type>& gy): aRealTopology2d<real_type>(gx,gy){ }

    ///allow explicit type conversion from any other topology
    ///@param src source
    explicit RealGrid2d( const aRealTopology2d<real_type>& src): aRealTopology2d<real_type>(src){}
    private:
    virtual void do_set( unsigned n, unsigned Nx, unsigned Ny) override final{
        aRealTopology2d<real_type>::do_set(n,Nx,Ny);
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
        aRealTopology3d<real_type>(x0,x1,y0,y1,z0,z1,n,Nx,Ny,Nz,bcx,bcy,bcz) { }
    ///@copydoc aRealTopology3d::aRealTopology3d(const RealGrid1d&,const RealGrid1d&,const RealGrid1d&)
    RealGrid3d( const RealGrid1d<real_type>& gx, const RealGrid1d<real_type>& gy, const RealGrid1d<real_type>& gz): aRealTopology3d<real_type>(gx,gy,gz){ }

    ///allow explicit type conversion from any other topology
    ///@param src source
    explicit RealGrid3d( const aRealTopology3d<real_type>& src): aRealTopology3d<real_type>(src){ }
    private:
    virtual void do_set( unsigned n, unsigned Nx, unsigned Ny, unsigned Nz) override final{
        aRealTopology3d<real_type>::do_set(n,Nx,Ny,Nz);
    }
};

///@cond
template<class real_type>
void aRealTopology2d<real_type>::do_set( unsigned new_n, unsigned new_Nx, unsigned new_Ny)
{
    gx_.set(new_n, new_Nx);
    gy_.set(new_n, new_Ny);
}
template<class real_type>
void aRealTopology3d<real_type>::do_set(unsigned new_n, unsigned new_Nx,unsigned new_Ny, unsigned new_Nz)
{
    gx_.set(new_n, new_Nx);
    gy_.set(new_n, new_Ny);
    gz_.set(1,new_Nz);
}
template<class MemoryTag, class DimensionalityTag, class real_type>
struct MemoryTraits { };

template<class real_type>
struct MemoryTraits< SharedTag, OneDimensionalTag, real_type> {
    using host_vector = thrust::host_vector<real_type>;
    using host_grid   = RealGrid1d<real_type>;
};
template<class real_type>
struct MemoryTraits< SharedTag, TwoDimensionalTag, real_type> {
    using host_vector = thrust::host_vector<real_type>;
    using host_grid   = RealGrid2d<real_type>;
};
template<class real_type>
struct MemoryTraits< SharedTag, ThreeDimensionalTag,real_type> {
    using host_vector = thrust::host_vector<real_type>;
    using host_grid   = RealGrid3d<real_type>;
};

template<class Topology>
using get_host_vector = typename MemoryTraits< typename TopologyTraits<Topology>::memory_category, typename TopologyTraits<Topology>::dimensionality, typename TopologyTraits<Topology>::value_type>::host_vector;
template<class Topology>
using get_host_grid = typename MemoryTraits< typename TopologyTraits<Topology>::memory_category, typename TopologyTraits<Topology>::dimensionality, typename TopologyTraits<Topology>::value_type>::host_grid;

///@endcond

///@addtogroup gridtypes
///@{
using Grid1d        = dg::RealGrid1d<double>;
using Grid2d        = dg::RealGrid2d<double>;
using Grid3d        = dg::RealGrid3d<double>;
using aTopology2d   = dg::aRealTopology2d<double>;
using aTopology3d   = dg::aRealTopology3d<double>;
///@}

}// namespace dg
