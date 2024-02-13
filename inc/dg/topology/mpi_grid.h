#pragma once

#include <cmath>
#include "dg/backend/mpi_vector.h"
#include "dg/enums.h"
#include "grid.h"

/*! @file
  @brief MPI Grid objects
  */

namespace dg
{

/*! @class hide_comm_parameters1d
 * @param comm a one-dimensional Cartesian communicator
 * @note the parameters given in the constructor are global parameters
 */
/*! @class hide_comm_parameters2d
 * @param comm a two-dimensional Cartesian communicator
 * @note the parameters given in the constructor are global parameters
 */
/*! @class hide_comm_parameters3d
 * @param comm a three-dimensional Cartesian communicator
 * @note the parameters given in the constructor are global parameters
 */

///@cond
template<class real_type>
struct RealMPIGrid2d;
template<class real_type>
struct RealMPIGrid3d;
///@endcond
/// Just a tag (used e.g. in netcdf output to indicate 0-dimensional (point) data)
template<class real_type>
struct RealMPIGrid0d{
    /// @brief number of dimensions : 0
    constexpr static unsigned ndim() {return 0;}
};

/**
* @brief 1D MPI grid
* @ingroup grid
* @copydoc hide_code_evaluate1d
*/
template<class real_type>
struct RealMPIGrid1d
{
    using value_type = real_type;
    /// The host vector type used by host functions like evaluate
    using host_vector = MPI_Vector<thrust::host_vector<real_type>>;
    using host_grid = RealMPIGrid1d<real_type>;
    /// @brief number of dimensions : 1
    constexpr static unsigned ndim() {return 1;}

    ///@copydoc RealGrid1d::RealGrid1d
    RealMPIGrid1d() = default;
    /**
     * @brief 1D grid
     *
     * @param x0 left boundary
     * @param x1 right boundary
     * @param n # of polynomial coefficients
     *  (1<=n<=20, note that the library is optimized for n=3 )
     * @param N # of cells
     * @copydoc hide_comm_parameters1d
     */
    RealMPIGrid1d( real_type x0, real_type x1, unsigned n, unsigned N, MPI_Comm comm): m_comm(comm)
    {
        set(x0,x1,n,N,dg::PER);
    }
    ///@copydoc RealGrid1d::RealGrid1d(real_type,real_type,unsigned,unsigned,bc)
    ///@copydoc hide_comm_parameters1d
    RealMPIGrid1d( real_type x0, real_type x1, unsigned n, unsigned N, bc bcx, MPI_Comm comm): m_comm(comm)
    {
        set(x0,x1,n,N,bcx);
    }
    //////////////////////////////////////////get/////////////////////////////

    ///@copydoc aRealMPITopology3d::x0()
    real_type x0() const {return m_g.x0();}
    ///@copydoc aRealMPITopology3d::x1()
    real_type x1() const {return m_g.x1();}
    ///@copydoc aRealMPITopology3d::lx()
    real_type lx() const {return m_g.lx();}
    ///@copydoc aRealMPITopology3d::hx()
    real_type h() const {return m_g.h();}
    ///@copydoc aRealMPITopology3d::Nx()
    unsigned N() const {return m_g.N();}
    ///@copydoc aRealMPITopology3d::n()
    unsigned n() const {return m_g.n();}
    ///@copydoc aRealMPITopology3d::bcx()
    bc bcx() const {return m_g.bcx();}
    //////////////////////////////////////////set/////////////////////////////

    ///@copydoc RealGrid1d::set(real_type,real_type,bc)
    void set(real_type x0, real_type x1, bc bcx)
    {
        m_g.set( x0, x1, bcx);
        update_local();
    }
    ///@copydoc RealGrid1d::set(unsigned,unsigned)
    void set( unsigned n, unsigned N)
    {
        check_division( N, m_g.bcx());
        m_g.set( n, N);
        update_local();
    }
    ///@copydoc RealGrid1d::set(real_type,real_type,unsigned,unsigned,bc)
    void set( real_type x0, real_type x1, unsigned n, unsigned N, bc bcx)
    {
        check_division( N, bcx);
        m_g.set( x0, x1, n, N, bcx);
        update_local();
    }
    /////////////////////////////////////////convencience//////////////////////////////
    /**
     * @brief Return one-dimensional Cartesian mpi communicator that is used in this grid
     *
     * @return Communicator
     */
    MPI_Comm communicator() const{return m_comm;}

    /// @brief <tt> n()* N()</tt> (Global Total number of grid points)
    unsigned size() const { return m_g.size();}
    /// @brief Local number of grid points
    unsigned local_size() const { return m_l.size();}
    ///@copydoc RealGrid1d::dlt()const
    const DLT<real_type>& dlt() const {return m_g.dlt();}

    ///@copydoc aRealMPITopology3d::display(std::ostream&)const
    void display( std::ostream& os = std::cout) const
    {
        os << "GLOBAL GRID \n";
        m_g.display();
        os << "LOCAL GRID \n";
        m_l.display();
    }
    /**
     * @brief Returns the pid of the process that holds the local grid surrounding the given point
     *
     * @param x X-coord
     *
     * @return pid of a process, or -1 if non of the grids matches
     */
    int pidOf( real_type x) const;

    ///@copydoc aRealMPITopology2d::local2globalIdx(int,int,int&)const
    bool local2globalIdx( int localIdx, int PID, int& globalIdx)const
    {
        if( localIdx < 0 || localIdx >= (int)size()) return -1;
        int coords[1];
        if( MPI_Cart_coords( m_comm, PID, 1, coords) != MPI_SUCCESS)
            return false;
        globalIdx = coords[0]*m_l.nx()*m_l.Nx()+localIdx;
        return true;
    }
    ///@copydoc aRealMPITopology2d::global2localIdx(int,int&,int&)const
    bool global2localIdx( int globalIdx, int& localIdx, int& PID)const
    {
        if( globalIdx < 0 || globalIdx >= (int)m_g.size()) return -1;
        int coords[1];
        coords[0] = globalIdx/(m_l.nx()*m_l.Nx());
        localIdx  = globalIdx%(m_l.nx()*m_l.Nx());
        if( MPI_Cart_rank( m_comm, coords, &PID) == MPI_SUCCESS )
            return true;
        else
        {
            std::cout<<"Failed "<<PID<<"\n";
            return false;
        }
    }
    ///@copydoc aRealMPITopology2d::local()const
    const RealGrid1d<real_type>& local() const {return m_l;}
     ///@copydoc aRealMPITopology2d::global()const
    const RealGrid1d<real_type>& global() const {return m_g;}

  private:
    void check_division( unsigned Nx, bc bcx)
    {
        int rank, dims[1], periods[1], coords[1];
        MPI_Cart_get( m_comm, 1, dims, periods, coords);
        MPI_Comm_rank( m_comm, &rank);
        if( rank == 0)
        {
            if(Nx%dims[0]!=0)
                std::cerr << "Nx "<<Nx<<" npx "<<dims[0]<<std::endl;
            assert( Nx%dims[0] == 0);
            if( bcx == dg::PER) assert( periods[0] == true);
            else assert( periods[0] == false);
        }
    }
    void update_local(){
        int dims[1], periods[1], coords[1];
        MPI_Cart_get( m_comm, 1, dims, periods, coords);
        real_type x0 = m_g.x0() + m_g.lx()/(real_type)dims[0]*(real_type)coords[0];
        real_type x1 = m_g.x0() + m_g.lx()/(real_type)dims[0]*(real_type)(coords[0]+1);
        if( coords[0] == dims[0]-1)
            x1 = m_g.x1();
        unsigned Nx = m_g.N()/dims[0];
        m_l = RealGrid1d<real_type> { x0, x1, m_g.n(), Nx, m_g.bcx()};
    }
    RealGrid1d<real_type> m_g, m_l; // global and local grid
    MPI_Comm m_comm;
};


/**
 * @brief 2D MPI abstract grid class
 *
 * Represents the global grid coordinates and the process topology.
 * It just divides the given (global) box into nonoverlapping (local) subboxes that are attributed to each process
 * @note a single cell is never divided across processes.
 * @note although it is abstract, objects are not meant to be hold on the heap via a base class pointer ( we protected the destructor)
 * @attention
 * The access functions \c nx() \c Nx() ,... all return the global parameters. If you want to have the local ones call the \c local() function.
 * @ingroup basictopology
 */
template<class real_type>
struct aRealMPITopology2d
{
    using value_type = real_type;
    /// The host vector type used by host functions like evaluate
    using host_vector = MPI_Vector<thrust::host_vector<real_type>>;
    using host_grid = RealMPIGrid2d<real_type>;
    /// @brief number of dimensions : 2
    constexpr static unsigned ndim() {return 2;}

    /**
     * @brief Return global x0
     *
     * @return global left boundary
     */
    real_type x0() const { return g.x0();}
    /**
     * @brief Return global x1
     *
     * @return global right boundary
     */
    real_type x1() const { return g.x1(); }
    /**
     * @brief Return global y0
     *
     * @return global left boundary
     */
    real_type y0() const { return g.y0();}
    /**
     * @brief Return global y1
     *
     * @return global right boundary
     */
    real_type y1() const { return g.y1();}
    /**
     * @brief Return global lx
     *
     * @return global length
     */
    real_type lx() const {return g.lx();}
    /**
     * @brief Return global ly
     *
     * @return global length
     */
    real_type ly() const {return g.ly();}
    /**
     * @brief Return global hx
     *
     * @return global grid constant
     */
    real_type hx() const {return g.hx();}
    /**
     * @brief Return global hy
     *
     * @return global grid constant
     */
    real_type hy() const {return g.hy();}
    /**
     * @brief Return n
     *
     * @return number of polynomial coefficients
     */
    unsigned n() const {return g.n();}
    /// number of polynomial coefficients in x
    unsigned nx() const {return g.nx();}
    /// number of polynomial coefficients in y
    unsigned ny() const {return g.ny();}
    /**
     * @brief Return the global number of cells
     *
     * @return number of cells
     */
    unsigned Nx() const { return g.Nx();}
    /**
     * @brief Return the global number of cells
     *
     * @return number of cells
     */
    unsigned Ny() const { return g.Ny();}
    /**
     * @brief global x boundary
     *
     * @return boundary condition
     */
    bc bcx() const {return g.bcx();}
    /**
     * @brief global y boundary
     *
     * @return boundary condition
     */
    bc bcy() const {return g.bcy();}
    /**
     * @brief Return mpi cartesian communicator that is used in this grid
     *
     * @return Communicator
     */
    MPI_Comm communicator() const{return comm;}
    /// discrete legendre transformation in x
    const DLT<real_type>& dltx() const{return g.dltx();}
    /// discrete legendre transformation in y
    const DLT<real_type>& dlty() const{return g.dlty();}
    /**
     * @brief The total global number of points
     * @return equivalent to <tt> nx()*ny()*Nx()*Ny() </tt>
     */
    unsigned size() const { return g.size();}
    /**
     * @brief The total local number of points
     * @return equivalent to \c local.size()
     */
    unsigned local_size() const { return l.size();}
    /**
     * @brief Display global and local grid
     *
     * @param os output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "GLOBAL GRID \n";
        g.display();
        os << "LOCAL GRID \n";
        l.display();
    }

    /**
     * @brief Returns the pid of the process that holds the local grid surrounding the given point
     *
     * @param x X-coord
     * @param y Y-coord
     *
     * @return pid of a process, or -1 if non of the grids matches
     */
    int pidOf( real_type x, real_type y) const;
    /**
    * @brief Multiply the number of cells with a given factor
    *
    * With this function you can resize the grid ignorantly of its current size
    * @param fx new global number of cells is fx*global().Nx()
    * @param fy new global number of cells is fy*global().Ny()
    */
    void multiplyCellNumbers( real_type fx, real_type fy){
        if( fx != 1 || fy != 1)
            do_set(nx(), round(fx*(real_type)Nx()), ny(), round(fy*(real_type)Ny()));
    }
    ///@copydoc Grid2d::set(unsigned,unsigned,unsigned)
    void set( unsigned new_n, unsigned new_Nx, unsigned new_Ny) {
        set( new_n, new_Nx, new_n, new_Ny);
    }
    ///@copydoc Grid2d::set(unsigned,unsigned,unsigned,unsigned)
    void set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny) {
        check_division( new_Nx, new_Ny, g.bcx(), g.bcy());
        if( new_nx == nx() && new_Nx == Nx() && new_ny == ny() && new_Ny == Ny())
            return;
        do_set( new_nx,new_Nx,new_ny,new_Ny);
    }
    /**
    * @brief Map a local index plus the PID to a global vector index
    *
    * @param localIdx a local vector index
    * @param PID a PID in the communicator
    * @param globalIdx the corresponding global vector Index (contains result on output)
    * @return true if successful, false if localIdx or PID is not part of the grid
    */
    bool local2globalIdx( int localIdx, int PID, int& globalIdx)const
    {
        if( localIdx < 0 || localIdx >= (int)size()) return -1;
        int coords[2];
        if( MPI_Cart_coords( comm, PID, 2, coords) != MPI_SUCCESS)
            return false;
        int lIdx0 = localIdx %(l.nx()*l.Nx());
        int lIdx1 = localIdx /(l.nx()*l.Nx());
        int gIdx0 = coords[0]*l.nx()*l.Nx()+lIdx0;
        int gIdx1 = coords[1]*l.ny()*l.Ny()+lIdx1;
        globalIdx = gIdx1*g.nx()*g.Nx() + gIdx0;
        return true;
    }
    /**
    * @brief Map a global vector index to a local vector Index and the corresponding PID
    *
    * @param globalIdx a global vector Index
    * @param localIdx contains local vector index on output
    * @param PID contains corresponding PID in the communicator on output
    * @return true if successful, false if globalIdx is not part of the grid
    */
    bool global2localIdx( int globalIdx, int& localIdx, int& PID)const
    {
        if( globalIdx < 0 || globalIdx >= (int)g.size()) return -1;
        int coords[2];
        int gIdx0 = globalIdx%(g.nx()*g.Nx());
        int gIdx1 = globalIdx/(g.nx()*g.Nx());
        coords[0] = gIdx0/(l.nx()*l.Nx());
        coords[1] = gIdx1/(l.ny()*l.Ny());
        int lIdx0 = gIdx0%(l.nx()*l.Nx());
        int lIdx1 = gIdx1%(l.ny()*l.Ny());
        localIdx = lIdx1*l.nx()*l.Nx() + lIdx0;
        if( MPI_Cart_rank( comm, coords, &PID) == MPI_SUCCESS )
            return true;
        else
        {
            std::cout<<"Failed "<<PID<<"\n";
            return false;
        }
    }
    /**
     * @brief Return a non-MPI grid local for the calling process
     *
     * The local grid contains the boundaries and cell numbers the calling process sees and is in charge of.
     * @return Grid object
     * @note the boundary conditions in the local grid are not well defined since there might not actually be any boundaries
     */
    const RealGrid2d<real_type>& local() const {return l;}
    /**
     * @brief Return the global non-MPI grid
     *
     * The global grid contains the global boundaries and cell numbers.
     * This is the grid that we would have to use in a non-MPI implementation.
     * The global grid returns the same values for x0(), x1(), ..., Nx(), Ny(), ... as the grid
     * class itself
     * @return non-MPI Grid object
     */
    const RealGrid2d<real_type>& global() const {return g;}
    protected:
    ///disallow deletion through base class pointer
    ~aRealMPITopology2d() = default;

    ///@copydoc aRealTopology2d::aRealTopology2d(RealGrid1d,RealGrid1d)
    ///@copydoc hide_comm_parameters2d
    aRealMPITopology2d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, MPI_Comm comm): g( gx, gy), l(gx, gy), comm(comm){
        check_division( gx.N(), gy.N(), gx.bcx(), gy.bcx());
        update_local();
    }
    ///@copydoc aRealTopology2d::aRealTopology2d(const aRealTopology2d&)
    aRealMPITopology2d(const aRealMPITopology2d& src) = default;
    ///@copydoc aRealTopology2d::operator=(const aRealTopology2d&)
    aRealMPITopology2d& operator=(const aRealMPITopology2d& src) = default;
    ///This function has an implementation
    virtual void do_set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny)=0;
    private:
    void check_division( unsigned Nx, unsigned Ny, bc bcx, bc bcy)
    {
        int rank, dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        MPI_Comm_rank( comm, &rank);
        if( rank == 0)
        {
            if(Nx%dims[0]!=0)
                std::cerr << "Nx "<<Nx<<" npx "<<dims[0]<<std::endl;
            assert( Nx%dims[0] == 0);
            if(Ny%dims[1]!=0)
                std::cerr << "Ny "<<Ny<<" npy "<<dims[1]<<std::endl;
            assert( Ny%dims[1] == 0);
            if( bcx == dg::PER) assert( periods[0] == true);
            else assert( periods[0] == false);
            if( bcy == dg::PER) assert( periods[1] == true);
            else assert( periods[1] == false);
        }
    }
    void update_local(){
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        real_type x0 = g.x0() + g.lx()/(real_type)dims[0]*(real_type)coords[0];
        real_type x1 = g.x0() + g.lx()/(real_type)dims[0]*(real_type)(coords[0]+1);
        if( coords[0] == dims[0]-1)
            x1 = g.x1();
        real_type y0 = g.y0() + g.ly()/(real_type)dims[1]*(real_type)coords[1];
        real_type y1 = g.y0() + g.ly()/(real_type)dims[1]*(real_type)(coords[1]+1);
        if( coords[1] == dims[1]-1)
            y1 = g.y1();
        unsigned Nx = g.Nx()/dims[0];
        unsigned Ny = g.Ny()/dims[1];
        l = RealGrid2d<real_type>(
                { x0, x1, g.nx(), Nx, g.bcx()},
                { y0, y1, g.ny(), Ny, g.bcy()});
    }
    RealGrid2d<real_type> g, l; //global and local grid
    MPI_Comm comm; //just an integer...
};


/**
 * @brief 3D MPI Grid class
 *
 * @copydetails aRealMPITopology2d
 * @ingroup basictopology
 */
template<class real_type>
struct aRealMPITopology3d
{
    using value_type = real_type;
    /// The host vector type used by host functions like evaluate
    using host_vector = MPI_Vector<thrust::host_vector<real_type>>;
    using host_grid = RealMPIGrid3d<real_type>;
    /// @brief number of dimensions : 3
    constexpr static unsigned ndim() {return 3;}

    /**
     * @brief Return global x0
     *
     * @return global left boundary
     */
    real_type x0() const { return g.x0();}
    /**
     * @brief Return global x1
     *
     * @return global right boundary
     */
    real_type x1() const { return g.x1();}
    /**
     * @brief Return global y0
     *
     * @return global left boundary
     */
    real_type y0() const { return g.y0();}
    /**
     * @brief Return global y1
     *
     * @return global right boundary
     */
    real_type y1() const { return g.y1();}
    /**
     * @brief Return global z0
     *
     * @return global left boundary
     */
    real_type z0() const { return g.z0();}
    /**
     * @brief Return global z1
     *
     * @return global right boundary
     */
    real_type z1() const { return g.z1();}
    /**
     * @brief Return global lx
     *
     * @return global length
     */
    real_type lx() const {return g.lx();}
    /**
     * @brief Return global ly
     *
     * @return global length
     */
    real_type ly() const {return g.ly();}
    /**
     * @brief Return global lz
     *
     * @return global length
     */
    real_type lz() const {return g.lz();}
    /**
     * @brief Return global hx
     *
     * @return global grid constant
     */
    real_type hx() const {return g.hx();}
    /**
     * @brief Return global hy
     *
     * @return global grid constant
     */
    real_type hy() const {return g.hy();}
    /**
     * @brief Return global hz
     *
     * @return global grid constant
     */
    real_type hz() const {return g.hz();}
    /**
     * @brief Return n
     *
     * @return number of polynomial coefficients
     */
    unsigned n() const {return g.n();}
    /// number of polynomial coefficients in x
    unsigned nx() const {return g.nx();}
    /// number of polynomial coefficients in y
    unsigned ny() const {return g.ny();}
    /// number of polynomial coefficients in z
    unsigned nz() const {return g.nz();}
    /**
     * @brief Return the global number of cells
     *
     * @return number of cells
     */
    unsigned Nx() const { return g.Nx();}
    /**
     * @brief Return the global number of cells
     *
     * @return number of cells
     */
    unsigned Ny() const { return g.Ny();}
    /**
     * @brief Return the global number of cells
     *
     * @return number of cells
     */
    unsigned Nz() const { return g.Nz();}
    /**
     * @brief global x boundary
     *
     * @return boundary condition
     */
    bc bcx() const {return g.bcx();}
    /**
     * @brief global y boundary
     *
     * @return boundary condition
     */
    bc bcy() const {return g.bcy();}
    /**
     * @brief global z boundary
     *
     * @return boundary condition
     */
    bc bcz() const {return g.bcz();}
    /**
     * @brief Return mpi cartesian communicator that is used in this grid
     * @return Communicator
     */
    MPI_Comm communicator() const{return comm;}
    /**
     * @brief MPI Cartesian communicator in the first two dimensions (x and y)
     * @return 2d Cartesian Communicator
     */
    MPI_Comm get_perp_comm() const {return planeComm;}
    /**
     * @brief The Discrete Legendre Transformation
     *
     * @return DLT corresponding to n given in the constructor
     */
    const DLT<real_type>& dlt() const{return g.dlt();}
    const DLT<real_type>& dltx() const{return g.dltx();}
    const DLT<real_type>& dlty() const{return g.dlty();}
    const DLT<real_type>& dltz() const{return g.dltz();}
    /**
     * @brief The total global number of points
     * @return equivalent to <tt> nx()*ny()*nz()*Nx()*Ny()*Nz() </tt>
     */
    unsigned size() const { return g.size();}
    /**
     * @brief The total local number of points
     * @return equivalent to \c local.size()
     */
    unsigned local_size() const { return l.size();}
    /**
     * @brief Display global and local grid paramters
     *
     * @param os output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "GLOBAL GRID \n";
        g.display();
        os << "LOCAL GRID \n";
        l.display();
    }
    /**
     * @brief Returns the pid of the process that holds the local grid surrounding the given point
     *
     * @param x X-coord
     * @param y Y-coord
     * @param z Z-coord
     *
     * @return pid of a process, or -1 if non of the grids matches
     */
    int pidOf( real_type x, real_type y, real_type z) const;
    ///@copydoc aRealMPITopology2d::multiplyCellNumbers()
    void multiplyCellNumbers( real_type fx, real_type fy){
        if( fx != 1 || fy != 1)
            do_set(nx(), round(fx*(real_type)Nx()), ny(),
                    round(fy*(real_type)Ny()), nz(), Nz());
    }
    ///@copydoc Grid3d::set(unsigned,unsigned,unsigned,unsigned)
    void set( unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) {
        set(new_n,new_Nx,new_n,new_Ny,1,new_Nz);
    }
    ///@copydoc Grid3d::set(unsigned,unsigned,unsigned,unsigned,unsigned,unsigned)
    void set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny, unsigned new_nz, unsigned new_Nz) {
        check_division( new_Nx,new_Ny,new_Nz,g.bcx(),g.bcy(),g.bcz());
        if( new_nx == nx() && new_Nx == Nx() && new_ny == ny() && new_Ny == Ny() && new_nz == nz() && new_Nz == Nz())
            return;
        do_set(new_nx,new_Nx,new_ny,new_Ny,new_nz,new_Nz);
    }
    ///@copydoc aRealMPITopology2d::local2globalIdx(int,int,int&)const
    bool local2globalIdx( int localIdx, int PID, int& globalIdx)const
    {
        if( localIdx < 0 || localIdx >= (int)size()) return false;
        int coords[3];
        if( MPI_Cart_coords( comm, PID, 3, coords) != MPI_SUCCESS)
            return false;
        int lIdx0 = localIdx %(l.nx()*l.Nx());
        int lIdx1 = (localIdx /(l.nx()*l.Nx())) % (l.ny()*l.Ny());
        int lIdx2 = localIdx / (l.nx()*l.ny()*l.Nx()*l.Ny());
        int gIdx0 = coords[0]*l.nx()*l.Nx()+lIdx0;
        int gIdx1 = coords[1]*l.ny()*l.Ny()+lIdx1;
        int gIdx2 = coords[2]*l.nz()*l.Nz()+lIdx2;
        globalIdx = (gIdx2*g.ny()*g.Ny() + gIdx1)*g.nx()*g.Nx() + gIdx0;
        return true;
    }
    ///@copydoc aRealMPITopology2d::global2localIdx(int,int&,int&)const
    bool global2localIdx( int globalIdx, int& localIdx, int& PID)const
    {
        if( globalIdx < 0 || globalIdx >= (int)g.size()) return false;
        int coords[3];
        int gIdx0 = globalIdx%(g.nx()*g.Nx());
        int gIdx1 = (globalIdx/(g.nx()*g.Nx())) % (g.ny()*g.Ny());
        int gIdx2 = globalIdx/(g.nx()*g.ny()*g.Nx()*g.Ny());
        coords[0] = gIdx0/(l.nx()*l.Nx());
        coords[1] = gIdx1/(l.ny()*l.Ny());
        coords[2] = gIdx2/(l.nz()*l.Nz());
        int lIdx0 = gIdx0%(l.nx()*l.Nx());
        int lIdx1 = gIdx1%(l.ny()*l.Ny());
        int lIdx2 = gIdx2%(l.nz()*l.Nz());
        localIdx = (lIdx2*l.ny()*l.Ny() + lIdx1)*l.nx()*l.Nx() + lIdx0;
        if( MPI_Cart_rank( comm, coords, &PID) == MPI_SUCCESS )
            return true;
        else
            return false;
    }
    ///@copydoc aRealMPITopology2d::local()const
    const RealGrid3d<real_type>& local() const {return l;}
     ///@copydoc aRealMPITopology2d::global()const
    const RealGrid3d<real_type>& global() const {return g;}
    protected:
    ///disallow deletion through base class pointer
    ~aRealMPITopology3d() = default;

    ///@copydoc aRealTopology3d::aRealTopology3d(RealGrid1d,RealGrid1d,RealGrid1d)
    ///@copydoc hide_comm_parameters3d
    aRealMPITopology3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz, MPI_Comm comm): g( gx, gy, gz), l(gx, gy, gz), comm(comm){
        check_division( gx.N(), gy.N(), gz.N(), gx.bcx(), gy.bcx(), gz.bcx());
        update_local();
        int remain_dims[] = {true,true,false}; //true true false
        MPI_Cart_sub( comm, remain_dims, &planeComm);
    }
    ///explicit copy constructor (default)
    ///@param src source
    aRealMPITopology3d(const aRealMPITopology3d& src) = default;
    ///explicit assignment operator (default)
    ///@param src source
    aRealMPITopology3d& operator=(const aRealMPITopology3d& src) = default;
    virtual void do_set( unsigned new_nx, unsigned new_Nx, unsigned new_ny, unsigned new_Ny, unsigned new_nz, unsigned new_Nz)=0;
    private:
    void check_division( unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz)
    {
        int rank, dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        MPI_Comm_rank( comm, &rank);
        if( rank == 0)
        {
            if(!(Nx%dims[0]==0))
                std::cerr << "Nx "<<Nx<<" npx "<<dims[0]<<std::endl;
            assert( Nx%dims[0] == 0);
            if( !(Ny%dims[1]==0))
                std::cerr << "Ny "<<Ny<<" npy "<<dims[1]<<std::endl;
            assert( Ny%dims[1] == 0);
            if( !(Nz%dims[2]==0))
                std::cerr << "Nz "<<Nz<<" npz "<<dims[2]<<std::endl;
            assert( Nz%dims[2] == 0);
            if( bcx == dg::PER) assert( periods[0] == true);
            else assert( periods[0] == false);
            if( bcy == dg::PER) assert( periods[1] == true);
            else assert( periods[1] == false);
            if( bcz == dg::PER) assert( periods[2] == true);
            else assert( periods[2] == false);
        }
    }
    void update_local(){
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( comm, 3, dims, periods, coords);
        real_type x0 = g.x0() + g.lx()/(real_type)dims[0]*(real_type)coords[0];
        real_type x1 = g.x0() + g.lx()/(real_type)dims[0]*(real_type)(coords[0]+1);
        if( coords[0] == dims[0]-1)
            x1 = g.x1();

        real_type y0 = g.y0() + g.ly()/(real_type)dims[1]*(real_type)coords[1];
        real_type y1 = g.y0() + g.ly()/(real_type)dims[1]*(real_type)(coords[1]+1);
        if( coords[1] == dims[1]-1)
            y1 = g.y1();

        real_type z0 = g.z0() + g.lz()/(real_type)dims[2]*(real_type)coords[2];
        real_type z1 = g.z0() + g.lz()/(real_type)dims[2]*(real_type)(coords[2]+1);
        if( coords[2] == dims[2]-1)
            z1 = g.z1();
        unsigned Nx = g.Nx()/dims[0];
        unsigned Ny = g.Ny()/dims[1];
        unsigned Nz = g.Nz()/dims[2];

        l = RealGrid3d<real_type>(
                { x0, x1, g.nx(), Nx, g.bcx()},
                { y0, y1, g.ny(), Ny, g.bcy()},
                { z0, z1, g.nz(), Nz, g.bcz()});
    }
    RealGrid3d<real_type> g, l; //global grid
    MPI_Comm comm, planeComm; //just an integer...
};
///@cond
template<class real_type>
int RealMPIGrid1d<real_type>::pidOf( real_type x) const
{
    int dims[1], periods[1], coords[1];
    MPI_Cart_get( m_comm, 1, dims, periods, coords);
    coords[0] = (unsigned)floor( (x-m_g.x0())/m_g.lx()*(real_type)dims[0] );
    //if point lies on or over boundary of last cell shift into current cell (not so good for periodic boundaries)
    coords[0]=(coords[0]==dims[0]) ? coords[0]-1 :coords[0];
    int rank;
    if( MPI_Cart_rank( m_comm, coords, &rank) == MPI_SUCCESS )
        return rank;
    else
        return -1;
}
template<class real_type>
int aRealMPITopology2d<real_type>::pidOf( real_type x, real_type y) const
{
    int dims[2], periods[2], coords[2];
    MPI_Cart_get( comm, 2, dims, periods, coords);
    coords[0] = (unsigned)floor( (x-g.x0())/g.lx()*(real_type)dims[0] );
    coords[1] = (unsigned)floor( (y-g.y0())/g.ly()*(real_type)dims[1] );
    //if point lies on or over boundary of last cell shift into current cell (not so good for periodic boundaries)
    coords[0]=(coords[0]==dims[0]) ? coords[0]-1 :coords[0];
    coords[1]=(coords[1]==dims[1]) ? coords[1]-1 :coords[1];
    int rank;
    if( MPI_Cart_rank( comm, coords, &rank) == MPI_SUCCESS )
        return rank;
    else
        return -1;
}
template<class real_type>
int aRealMPITopology3d<real_type>::pidOf( real_type x, real_type y, real_type z) const
{
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( comm, 3, dims, periods, coords);
    coords[0] = (unsigned)floor( (x-g.x0())/g.lx()*(real_type)dims[0] );
    coords[1] = (unsigned)floor( (y-g.y0())/g.ly()*(real_type)dims[1] );
    coords[2] = (unsigned)floor( (z-g.z0())/g.lz()*(real_type)dims[2] );
    //if point lies on or over boundary of last cell shift into current cell (not so good for periodic boundaries)
    coords[0]=(coords[0]==dims[0]) ? coords[0]-1 :coords[0];
    coords[1]=(coords[1]==dims[1]) ? coords[1]-1 :coords[1];
    coords[2]=(coords[2]==dims[2]) ? coords[2]-1 :coords[2];
    int rank;
    if( MPI_Cart_rank( comm, coords, &rank) == MPI_SUCCESS )
        return rank;
    else
        return -1;
}
template<class real_type>
void aRealMPITopology2d<real_type>::do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny) {
    g.set(nx,Nx,ny,Ny);
    update_local();
}
template<class real_type>
void aRealMPITopology3d<real_type>::do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny, unsigned nz, unsigned Nz) {
    g.set(nx,Nx,ny,Ny,nz,Nz);
    update_local();
}

template<class real_type>
std::vector<unsigned> shape( const dg::RealMPIGrid0d<real_type>& g) { return {1};}
template<class real_type>
std::vector<unsigned> shape( const dg::RealMPIGrid1d<real_type>& g) { return {g.size()};}
template<class real_type>
std::vector<unsigned> shape( const dg::aRealMPITopology2d<real_type>& g) { return {g.nx()*g.Nx(), g.ny()*g.Ny()};}
template<class real_type>
std::vector<unsigned> shape( const dg::aRealMPITopology3d<real_type>& g) { return {g.nx()*g.Nx(), g.ny()*g.Ny(), g.nz()*g.Nz()};}

template<class Grid>
using is_mpi_grid = std::is_same< get_host_vector<Grid>, MPI_Vector<thrust::host_vector<typename Grid::value_type>>>;

///@endcond

/**
 * @brief The simplest implementation of aRealMPITopology2d
 * @ingroup grid
 * @copydoc hide_code_mpi_evaluate2d
 */
template<class real_type>
struct RealMPIGrid2d: public aRealMPITopology2d<real_type>
{
    /**
     * @copydoc hide_grid_parameters2d
     * @copydoc hide_comm_parameters2d
     */
    RealMPIGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm):
        aRealMPITopology2d<real_type>( {x0,x1,n,Nx,dg::PER},
                {y0,y1,n,Ny,dg::PER}, comm)
    { }

    /**
     * @copydoc hide_grid_parameters2d
     * @copydoc hide_bc_parameters2d
     * @copydoc hide_comm_parameters2d
     */
    RealMPIGrid2d( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, bc bcx, bc bcy, MPI_Comm comm):
        aRealMPITopology2d<real_type>( {x0,x1,n,Nx,bcx}, {y0,y1,n,Ny,bcy},comm)
    { }
    ///@copydoc aRealTopology2d::aRealTopology2d(RealGrid1d,RealGrid1d)
    ///@copydoc hide_comm_parameters2d
    RealMPIGrid2d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, MPI_Comm comm): aRealMPITopology2d<real_type>(gx,gy,comm){ }
    ///allow explicit type conversion from any other topology
    explicit RealMPIGrid2d( const aRealMPITopology2d<real_type>& src): aRealMPITopology2d<real_type>(src){}
    private:
    virtual void do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny) override final{
        aRealMPITopology2d<real_type>::do_set(nx,Nx,ny,Ny);
    }
};

/**
 * @brief The simplest implementation of aRealMPITopology3d
 * @ingroup grid
 * @copydoc hide_code_mpi_evaluate3d
 */
template<class real_type>
struct RealMPIGrid3d : public aRealMPITopology3d<real_type>
{
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_comm_parameters3d
    RealMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, MPI_Comm comm):
        aRealMPITopology3d<real_type>( {x0, x1, n, Nx, dg::PER}, {y0, y1, n, Ny, dg::PER}, {z0, z1, 1, Nz, dg::PER}, comm )
    { }

    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@copydoc hide_comm_parameters3d
    RealMPIGrid3d( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx, bc bcy, bc bcz, MPI_Comm comm):
        aRealMPITopology3d<real_type>( {x0, x1, n, Nx, bcx}, {y0, y1, n, Ny, bcy}, {z0, z1, 1, Nz, bcz}, comm )
    { }
    ///@copydoc aRealTopology3d::aRealTopology3d(RealGrid1d,RealGrid1d,RealGrid1d)
    ///@copydoc hide_comm_parameters3d
    RealMPIGrid3d( RealGrid1d<real_type> gx, RealGrid1d<real_type> gy, RealGrid1d<real_type> gz, MPI_Comm comm): aRealMPITopology3d<real_type>(gx,gy,gz,comm){ }
    ///allow explicit type conversion from any other topology
    ///@param src source
    explicit RealMPIGrid3d( const aRealMPITopology3d<real_type>& src): aRealMPITopology3d<real_type>(src){ }
    private:
    virtual void do_set( unsigned nx, unsigned Nx, unsigned ny, unsigned Ny, unsigned nz, unsigned Nz) override final{
        aRealMPITopology3d<real_type>::do_set(nx,Nx,ny,Ny,nz,Nz);
    }
};

///@addtogroup gridtypes
///@{
using MPIGrid0d         = dg::RealMPIGrid0d<double>;
using MPIGrid1d         = dg::RealMPIGrid1d<double>;
using MPIGrid2d         = dg::RealMPIGrid2d<double>;
using MPIGrid3d         = dg::RealMPIGrid3d<double>;
using aMPITopology2d    = dg::aRealMPITopology2d<double>;
using aMPITopology3d    = dg::aRealMPITopology3d<double>;
namespace x{
using Grid0d          = MPIGrid0d      ;
using Grid1d          = MPIGrid1d      ;
using Grid2d          = MPIGrid2d      ;
using Grid3d          = MPIGrid3d      ;
using aTopology2d     = aMPITopology2d ;
using aTopology3d     = aMPITopology3d ;
}//namespace x
///@}

}//namespace dg
