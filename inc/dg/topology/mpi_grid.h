#pragma once

#include <cmath>
#include "dg/backend/mpi_vector.h"
#include "dg/enums.h"
#include "grid.h"

/*! @file
  @brief MPI Grid objects
  */

//    /*
//     * The rationale behind this is that \c shape is used to create / construct
//     * MPI distributed vectors and for this we need the local shape
//     * Since a MPI communicator is carried by \c MPI_Vector this allows to write
//     * device independent code
//     * @param u axis number
//     */
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
template<class real_type, size_t Nd>
struct RealMPIGrid;
///@endcond

/**
 * @brief MPI abstract grid class
 *
 * Represents the global grid coordinates and the process topology.
 * It just divides the given (global) box into nonoverlapping (local) subboxes that are attributed to each process
 * @note a single cell is never divided across processes.
 * @note although it is abstract, objects are not meant to be hold on the heap via a base class pointer ( we protected the destructor)
 * @ingroup basictopology
 */
template<class real_type, size_t Nd>
struct aRealMPITopology
{
    /////////////////// TYPE TRAITS ////////////////////////////
    /// value type of abscissas and weights
    using value_type = real_type;
    /// vector type of abscissas and weights
    using host_vector = MPI_Vector<thrust::host_vector<real_type>>;
    using host_grid = RealMPIGrid<real_type, Nd>;
    /// Dimensionality == Nd
    constexpr static unsigned ndim() { return Nd;}

    /////////////////// TOPOLOGY CONCEPT ////////////////////////////

    unsigned shape(unsigned u=0) const
    {
        return m_g.shape(u);
    }
    host_vector weights(unsigned u=0) const
    {
        auto local_weights = m_l.weights(u);
        return host_vector{local_weights, m_comms[u]};
    }
    host_vector abscissas(unsigned u=0) const
    {
        int dims[Nd], periods[Nd], coords[Nd];
        MPI_Cart_get( m_comm, Nd, dims, periods, coords);
        real_type global_x0 = m_g.p(u);
        real_type global_hx = m_g.h(u);
        int mpi_coord = coords[u];
        thrust::host_vector<real_type> abs(m_l.shape(u));
        for( unsigned i=0; i<m_l.N(u); i++)
        {
            auto aa = dg::DLT<real_type>::abscissas(m_l.n(u));
            for( unsigned j=0; j<m_l.n(u); j++)
            {
                unsigned coord = i+m_l.N(u)*mpi_coord;
                real_type xmiddle = DG_FMA( global_hx, (real_type)(coord), global_x0);
                real_type h2 = global_hx/2.;
                real_type absj = 1.+aa[j];
                abs[i*m_l.n(u)+j] = DG_FMA( h2, absj, xmiddle);
            }
        }
        return host_vector{abs, m_comms[u]};
    }
    /////////////////// GETTERS ////////////////////////////
    /// Global shape of grid
    std::array<unsigned,Nd> get_shape() const{
        m_g.get_shape();
    }
    std::array<host_vector,Nd> get_abscissas() const{
        std::array<host_vector,Nd> abs;
        for( unsigned u=0; u<Nd; u++)
            abs[u] = abscissas(u);
        return abs;
    }
    std::array<host_vector,Nd> get_weights() const{
        std::array<host_vector,Nd> w;
        for( unsigned u=0; u<Nd; u++)
            w[u] = weights(u);
        return w;
    }
    std::array<real_type,Nd> get_p() const{
        return m_g.get_p();
    }
    std::array<real_type,Nd> get_q() const{
        return m_g.get_q();
    }
    std::array<real_type,Nd> get_l() const{
        return m_g.get_l();
    }
    std::array<real_type,Nd> get_h() const{
        return m_g.get_h();
    }
    std::array<unsigned, Nd> get_N() const
    {
        return m_g.get_N();
    }
    /// Unambiguous <tt> local.get_n() == global.get_n() </tt>
    std::array<unsigned, Nd> get_n() const
    {
        return m_g.get_n();
    }
    /// Unambiguous <tt> local.get_bc() == global.get_bc() </tt>
    std::array<dg::bc, Nd> get_bc() const
    {
        return m_g.get_bc();
    }
    std::array<MPI_Comm, Nd> get_comms() const { return m_comms;}

    real_type p( unsigned u=0) const { return m_g.p(u);}
    real_type q( unsigned u=0) const { return m_g.q(u);}
    real_type h( unsigned u=0) const { return m_g.h(u);}
    real_type l( unsigned u=0) const { return m_g.l(u);}
    unsigned n( unsigned u=0) const  { return m_g.n(u);}
    unsigned N( unsigned u=0) const  { return m_g.N(u);}
    dg::bc bc( unsigned u=0) const   { return m_g.bc(u);}
    template<size_t Md = Nd>
    std::enable_if_t<Md==1,MPI_Comm> comm() const { return m_comms.at(0);}
    MPI_Comm comm(unsigned u) const { return m_comms.at(u);}
    /**
     * @brief Return Nd dimensional MPI cartesian communicator that is used in this grid
     * @return Communicator
     */
    MPI_Comm communicator() const{return m_comm;}
    /**
     * @brief MPI Cartesian communicator in the first two dimensions (x and y)
     * @return 2d Cartesian Communicator
     */
    template<size_t Md = Nd>
    std::enable_if_t<(Md >= 2), MPI_Comm> get_perp_comm() const
    {
        return mpi_cart_kron( {m_comms[0], m_comms[1]});
    }


    /// Get the u-th axis as a 1d Grid
    RealMPIGrid<real_type,1> grid(unsigned u ) const{
        if( u < Nd)
            return RealMPIGrid<real_type,1>{ m_g.p(u), m_g.q(u), m_g.n(u), m_g.N(u), m_g.bc(u), m_comms[u]};
        else
            throw Error( Message(_ping_)<<"u>Nd not allowed! You typed: "<<u<<" while Nd is "<<Nd);
    }
    ///@copydoc aRealMPITopology2d::local()const
    const RealGrid<real_type,Nd>& local() const {return m_l;}
     ///@copydoc aRealMPITopology2d::global()const
    const RealGrid<real_type, Nd>& global() const {return m_g;}

    /// Equivalent to <tt> p(0) </tt>
    template<size_t Md = Nd>
    real_type x0() const {return m_g.x0();}
    /// Equivalent to <tt> p(1) </tt>
    template<size_t Md = Nd>
    real_type x1() const {return m_g.x1();}
    /// Equivalent to <tt> p(2) </tt>
    template<size_t Md = Nd>
    real_type y0() const {return m_g.y0();}
    /// Equivalent to <tt> q(0) </tt>
    template<size_t Md = Nd>
    real_type y1() const {return m_g.y1();}
    /// Equivalent to <tt> q(1) </tt>
    template<size_t Md = Nd>
    real_type z0() const {return m_g.z0();}
    /// Equivalent to <tt> q(2) </tt>
    template<size_t Md = Nd>
    real_type z1() const {return m_g.z1();}

    /// Equivalent to <tt> l(0) </tt>
    template<size_t Md = Nd>
    real_type lx() const {return m_g.lx();}
    /// Equivalent to <tt> l(1) </tt>
    template<size_t Md = Nd>
    real_type ly() const {return m_g.ly();}
    /// Equivalent to <tt> l(2) </tt>
    template<size_t Md = Nd>
    real_type lz() const {return m_g.lz();}

    /// Equivalent to <tt> h(0) </tt>
    template<size_t Md = Nd>
    real_type hx() const {return m_g.hx();}
    /// Equivalent to <tt> h(1) </tt>
    template<size_t Md = Nd>
    real_type hy() const {return m_g.hy();}
    /// Equivalent to <tt> h(2) </tt>
    template<size_t Md = Nd>
    real_type hz() const {return m_g.hz();}

    /// Equivalent to <tt> n(0) </tt>
    template<size_t Md = Nd>
    unsigned nx() const {return m_g.nx();}
    /// Equivalent to <tt> n(1) </tt>
    template<size_t Md = Nd>
    unsigned ny() const {return m_g.ny();}
    /// Equivalent to <tt> n(2) </tt>
    template<size_t Md = Nd>
    unsigned nz() const {return m_g.nz();}

    /// Equivalent to <tt> N(0) </tt>
    template<size_t Md = Nd>
    unsigned Nx() const {return m_g.Nx();}
    /// Equivalent to <tt> N(1) </tt>
    template<size_t Md = Nd>
    unsigned Ny() const {return m_g.Ny();}
    /// Equivalent to <tt> N(2) </tt>
    template<size_t Md = Nd>
    unsigned Nz() const {return m_g.Nz();}

    /// Equivalent to <tt> bc(0) </tt>
    template<size_t Md = Nd>
    dg::bc bcx() const {return m_g.bcx();}
    /// Equivalent to <tt> bc(1) </tt>
    template<size_t Md = Nd>
    dg::bc bcy() const {return m_g.bcy();}
    /// Equivalent to <tt> bc(2) </tt>
    template<size_t Md = Nd>
    dg::bc bcz() const {return m_g.bcz();}

    template<size_t Md = Nd>
    RealMPIGrid<real_type,1> gx() const {
        static_assert( Nd > 0);
        return grid(0);
    }
    template<size_t Md = Nd>
    RealMPIGrid<real_type,1> gy() const {
        static_assert( Nd > 1);
        return grid(1);
    }
    template<size_t Md = Nd>
    RealMPIGrid<real_type,1> gz() const {
        static_assert( Nd > 2);
        return grid(2);
    }

    ////////////////////SETTERS/////////////////////////////
    /**
    * @brief Multiply the number of cells in the first two dimensions with a given factor
    *
    * With this function you can resize the grid ignorantly of its current size
    * @param fx new global number of cells is fx*Nx()
    * @param fy new global number of cells is fy*Ny()
    * The remaining dimensions are left unchanged
    */
    void multiplyCellNumbers( real_type fx, real_type fy){
        auto Ns = m_g.get_N();
        Ns[0] = round(fx*(real_type)m_g.N(0));
        Ns[1] = round(fy*(real_type)m_g.N(1));
        if( fx != 1 || fy != 1)
            set( m_g.get_n(), Ns);
    }
    template<size_t Md = Nd>
    std::enable_if_t<(Md == 1), void> set( unsigned new_n, unsigned new_Nx)
    {
        set({new_n}, {new_Nx});
    }

    template<size_t Md = Nd>
    std::enable_if_t<(Md == 2), void> set( unsigned new_n, unsigned new_Nx,
        unsigned new_Ny)
    {
        set({new_n,new_n}, {new_Nx,new_Ny});
    }
    template<size_t Md = Nd>
    std::enable_if_t<(Md == 3), void> set( unsigned new_n, unsigned new_Nx,
        unsigned new_Ny, unsigned new_Nz)
    {
        set({new_n,new_n,1}, {new_Nx,new_Ny,new_Nz});
    }
    /// Same as <tt> set( {new_n, new_n,...}, new_N);</tt>
    void set( unsigned new_n, std::array<unsigned,Nd> new_N)
    {
        std::array<unsigned , Nd> tmp;
        for( unsigned u=0; u<Nd; u++)
            tmp[u] = new_n;
        set( tmp, new_N);
    }
    /**
    * @brief Set the number of polynomials and cells
    *
    * @param new_n new number of %Gaussian nodes in each dimension
    * @param new_N new number of cells in each dimension
    */
    void set( std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N)
    {
        if( new_n==m_g.get_n() && new_N == m_g.get_N())
            return;
        do_set(new_n, new_N);
    }
    /**
     * @brief reset the boundaries of the grid
     *
     * @param p new left boundary
     * @param q new right boundary ( > x0)
     */
    void set_pq( std::array<real_type,Nd> p, std::array<real_type,Nd> q)
    {
        do_set( p, q);
    }
    /**
     * @brief reset the boundary conditions of the grid
     *
     * @param bcs new boundary condition
     */
    void set_bcs( std::array<dg::bc,Nd> bcs)
    {
        do_set( bcs);
    }

    ////////////////////UTILITY/////////////////////////////
    /**
     * @brief The total global number of points
     * @return global size
     */
    unsigned size() const { return m_g.size();}
    /**
     * @brief The total local number of points
     * @return equivalent to \c local.size()
     */
    unsigned local_size() const { return m_l.size();}
    // used in conversion policy in interpolation

    /**
     * @brief Display global and local grid paramters
     *
     * @param os output stream
     */
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
     * @param y Y-coord
     * @param z Z-coord
     *
     * @return pid of a process, or -1 if non of the grids matches
     */
    int pidOf( std::array<real_type,Nd> x) const;
    ///@copydoc aRealMPITopology2d::local2globalIdx(int,int,int&)const
    bool local2globalIdx( int localIdx, int PID, int& globalIdx)const
    {
        if( localIdx < 0 || localIdx >= (int)m_g.size()) return false;

        int coords[Nd];
        if( MPI_Cart_coords( m_comm, PID, Nd, coords) != MPI_SUCCESS)
            return false;
        int gIdx[Nd];
        int current = localIdx;
        for( unsigned u=0; u<Nd; u++)
        {
            int lIdx = current %(m_l.n(u)*m_l.N(u));
            current = current / (m_l.n(u)*m_l.N(u));
            gIdx[u] = coords[u]*m_l.n(u)*m_l.N(u)+lIdx;
        }
        globalIdx = gIdx[Nd-1];
        for( int u=Nd-2; u>=0; u--)
            globalIdx = globalIdx*m_g.n(u)*m_g.N(u) + gIdx[u];
        return true;
    }
    ///@copydoc aRealMPITopology2d::global2localIdx(int,int&,int&)const
    bool global2localIdx( int globalIdx, int& localIdx, int& PID)const
    {
        if( globalIdx < 0 || globalIdx >= (int)m_g.size()) return false;

        int coords[Nd];
        int lIdx[Nd];
        int current = globalIdx;
        for( unsigned u=0; u<Nd; u++)
        {
            int gIdx = current%(m_g.n(u)*m_g.N(u));
            current = current / (m_g.n(u)*m_g.N(u));
            coords[u] = gIdx/(m_l.n(u)*m_l.N(u));
            lIdx[u] = gIdx % (m_l.n(u)*m_l.N(u));
        }
        localIdx = lIdx[Nd-1];
        for( int u=Nd-2; u>=0; u--)
            localIdx = localIdx*m_l.n(u)*m_l.N(u) + lIdx[u];

        if( MPI_Cart_rank( m_comm, coords, &PID) == MPI_SUCCESS )
            return true;
        else
            return false;
    }
    protected:
    ///disallow deletion through base class pointer
    ~aRealMPITopology() = default;

    aRealMPITopology() = default;

    aRealMPITopology( std::array<real_type,Nd> p, std::array<real_type,Nd> q,
        std::array<unsigned,Nd> n, std::array<unsigned,Nd> N,
        std::array<dg::bc,Nd> bc, std::array<MPI_Comm, Nd> comms) :
        m_g(p,q,n,N,bc), m_comms(comms)
    {
        // assert dimensionality of Cartesian communicators
        int ndims;
        for ( unsigned u=0; u<Nd;u++)
        {
            MPI_Cartdim_get( m_comms[u], &ndims);
            assert( (unsigned)ndims == 1);
        }
        m_comm = dg::mpi_cart_kron( {m_comms.begin(), m_comms.end()});
        MPI_Cartdim_get( m_comm, &ndims);
        assert( (unsigned)ndims == Nd);
        check_division( N, bc);
        update_local();
    }
    aRealMPITopology( const std::array< RealMPIGrid<real_type, 1>, Nd>& grids)
    {
        std::array<RealGrid<real_type,1>,Nd> globals, locals;
        for( unsigned u=0; u<Nd; u++)
        {
            globals[u] = grids[u].global();
            locals[u] = grids[u].local();
            m_comms[u] = grids[u].communicator();
        }
        m_g = RealGrid<real_type,Nd>( globals);
        m_l = RealGrid<real_type,Nd>( locals);
        m_comm = dg::mpi_cart_kron( {m_comms.begin(), m_comms.end()});
    }

    // MW: The shared version of this constructor causes nvcc-12.4 to segfault when constructing a Geometry
    // Funnily the mpi version works (but let's kill it for now
    //template< size_t M0, size_t ...Ms>
    //aRealMPITopology( const aRealMPITopology<real_type,M0>& g0,
    //        const aRealMPITopology<real_type,Ms> & ...gs)
    //{
    //    auto grid = aRealMPITopology<real_type, Nd - M0>( gs ...);
    //    *this = aRealMPITopology<real_type, Nd>( g0, grid);
    //}
    //template< size_t M0, size_t M1>
    //aRealMPITopology( const aRealMPITopology<real_type,M0>& g0,
    //        const aRealMPITopology<real_type,M1>& g1) : m_g( g0.global(),g1.global()),
    //    m_l( g0.local(), g1.local())
    //{
    //    static_assert( (M0 + M1) == Nd);
    //    for( unsigned u=0; u<M0; u++)
    //    {
    //        m_comms[u] = g0.comm(u);
    //    }
    //    for( unsigned u=0; u<M1; u++)
    //    {
    //        m_comms[M0+u] = g1.comm(u);
    //    }
    //    m_comm = dg::mpi_cart_kron( {m_comms.begin(), m_comms.end()});

    //}

    //We do not want that because we cannot distinguish if g is meant to be the local or the global grid...
    //aRealMPITopology( const RealGrid<real_type,Nd> & g, MPI_Comm comm);
    ///explicit copy constructor (default)
    ///@param src source
    aRealMPITopology(const aRealMPITopology& src) = default;
    ///explicit assignment operator (default)
    ///@param src source
    aRealMPITopology& operator=(const aRealMPITopology& src) = default;
    virtual void do_set(std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N) =0;
    virtual void do_set_pq( std::array<real_type, Nd> x0, std::array<real_type,Nd> x1) =0;
    virtual void do_set( std::array<dg::bc, Nd> bcs) =0;
    private:
    void check_division( std::array<unsigned , Nd> N, std::array<dg::bc, Nd> bc)
    {
        int rank, dims[Nd], periods[Nd], coords[Nd];
        MPI_Cart_get( m_comm, Nd, dims, periods, coords);
        MPI_Comm_rank( m_comm, &rank);
        if( rank == 0)
        {
            for( unsigned u=0; u<Nd; u++)
            {
                if(!(N[u]%dims[u]==0))
                    std::cerr << "N"<<u<<" "<<N[u]<<" np "<<dims[u]<<std::endl;
                assert( N[u]%dims[u] == 0);
                if( bc[u] == dg::PER) assert( periods[u] == true);
                else assert( periods[u] == false);
            }
        }
    }
    void update_local(){
        int dims[Nd], periods[Nd], coords[Nd];
        MPI_Cart_get( m_comm, Nd, dims, periods, coords);
        std::array<real_type,Nd> p, q;
        std::array<unsigned, Nd> N;
        for( unsigned u=0;u<Nd; u++)
        {
            p[u] = m_g.p(u) + m_g.l(u)/(real_type)dims[u]*(real_type)coords[u];
            q[u] = m_g.p(u) + m_g.l(u)/(real_type)dims[u]*(real_type)(coords[u]+1);
            if( coords[u] == dims[u]-1)
                q[u] = m_g.q(u);
            N[u] = m_g.N(u)/dims[u];
        }

        m_l.set( p, q, m_g.get_n(), N, m_g.get_bc());
    }
    RealGrid<real_type, Nd> m_g, m_l; //global grid, local grid
    std::array<MPI_Comm, Nd> m_comms; // 1d comms
    MPI_Comm m_comm; //just an integer...(No, more like an address)
};
///@cond
template<class real_type, size_t Nd>
int aRealMPITopology<real_type,Nd>::pidOf( std::array<real_type,Nd> x) const
{
    int dims[Nd], periods[Nd], coords[Nd];
    MPI_Cart_get( m_comm, Nd, dims, periods, coords);
    for( unsigned u=0; u<Nd; u++)
    {
        coords[u] = (unsigned)floor( (x[u]-m_g.p(u))/m_g.l(u)*(real_type)dims[u] );
        //if point lies on or over boundary of last cell shift into current cell (not so good for periodic boundaries)
        coords[u]=(coords[u]==dims[u]) ? coords[u]-1 :coords[u];
        int rank;
        if( MPI_Cart_rank( m_comm, coords, &rank) == MPI_SUCCESS )
            return rank;
        else
            return -1;
    }
}
// pure virtual implementations must be declared outside class
template<class real_type,size_t Nd>
void aRealMPITopology<real_type,Nd>::do_set( std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N)
{
    check_division( new_N,m_g.get_bc());
    m_g.set(new_n, new_N);
    update_local();
}
template<class real_type,size_t Nd>
void aRealMPITopology<real_type,Nd>::do_set_pq( std::array<real_type, Nd> x0, std::array<real_type,Nd> x1)
{
    m_g.set_pq( x0, x1);
    update_local();
}
template<class real_type,size_t Nd>
void aRealMPITopology<real_type,Nd>::do_set( std::array<dg::bc, Nd> bcs)
{
    m_g.set_bcs(bcs);
    update_local();
}

/// Used to recognize MPI specialisation of interpolation and projection functions
template<class Grid>
using is_mpi_grid = std::is_base_of< dg::MPIVectorTag, dg::get_tensor_category< get_host_vector<Grid> > >;

///@endcond

/**
 * @brief The simplest implementation of aRealMPITopology3d
 * @ingroup grid
 * @copydoc hide_code_mpi_evaluate3d
 */
template<class real_type, size_t Nd>
struct RealMPIGrid : public aRealMPITopology<real_type,Nd>
{
    RealMPIGrid() = default;
    template<size_t Md = Nd>
    RealMPIGrid( real_type x0, real_type x1, unsigned n, unsigned N, MPI_Comm
            comm): aRealMPITopology<real_type,1>( {x0}, {x1},
                {n}, {N}, {dg::PER}, {comm})
    { }
    ///@copydoc RealGrid1d::RealGrid1d(real_type,real_type,unsigned,unsigned,bc)
    ///@copydoc hide_comm_parameters1d
    template<size_t Md = Nd>
    RealMPIGrid( real_type x0, real_type x1, unsigned n, unsigned N, dg::bc bcx, MPI_Comm
            comm): aRealMPITopology<real_type,1>( {x0}, {x1},
                {n}, {N}, {bcx}, {comm})
    {}
    /**
     * @copydoc hide_grid_parameters2d
     * @copydoc hide_comm_parameters2d
     */
    template<size_t Md = Nd>
    RealMPIGrid( real_type x0, real_type x1, real_type y0, real_type y1,
            unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm):
        aRealMPITopology<real_type,2>(
                {x0,y0},{x1,y1},{n,n},{Nx,Ny},{dg::PER,dg::PER},
                dg::mpi_cart_split<2>(comm))
    { }

    /**
     * @copydoc hide_grid_parameters2d
     * @copydoc hide_bc_parameters2d
     * @copydoc hide_comm_parameters2d
     */
    template<size_t Md = Nd>
    RealMPIGrid( real_type x0, real_type x1, real_type y0, real_type y1,
            unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, dg::bc bcy, MPI_Comm
            comm):
        aRealMPITopology<real_type,2>(
                {x0,y0},{x1,y1},{n,n},{Nx,Ny},{bcx,bcy},
                dg::mpi_cart_split<2>(comm))
    { }
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_comm_parameters3d
    template<size_t Md = Nd>
    RealMPIGrid( real_type x0, real_type x1, real_type y0, real_type y1,
            real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny,
            unsigned Nz, MPI_Comm comm):
        aRealMPITopology<real_type,3>(
                {x0,y0,z0},{x1,y1,z1},{n,n,1},{Nx,Ny,Nz},{dg::PER,dg::PER,dg::PER},
                dg::mpi_cart_split<3>(comm))
    { }

    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    ///@copydoc hide_comm_parameters3d
    template<size_t Md = Nd>
    RealMPIGrid( real_type x0, real_type x1, real_type y0, real_type y1,
            real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny,
            unsigned Nz, dg::bc bcx, dg::bc bcy, dg::bc bcz, MPI_Comm comm):
        aRealMPITopology<real_type,3>(
                {x0,y0,z0},{x1,y1,z1},{n,n,1},{Nx,Ny,Nz},{bcx,bcy,bcz},
                dg::mpi_cart_split<3>(comm))
    { }

    RealMPIGrid( const std::array<RealMPIGrid<real_type,1>,Nd>& grids) :
        aRealMPITopology<real_type,Nd>( grids){}

    RealMPIGrid( std::initializer_list<RealMPIGrid<real_type,1>> grids) :
        RealMPIGrid( std::array<RealMPIGrid<real_type,1>,Nd>{grids}){}

    RealMPIGrid( std::array<real_type,Nd> p, std::array<real_type,Nd> q,
        std::array<unsigned,Nd> n, std::array<unsigned,Nd> N,
        std::array<dg::bc,Nd> bcs, std::array<dg::bc,Nd> comms) :
        aRealMPITopology<real_type,Nd>( p,q,n,N,bcs,comms)
    {}

    ///allow explicit type conversion from any other topology
    ///@param src source
    explicit RealMPIGrid( const aRealMPITopology<real_type,Nd>& src):
        aRealMPITopology<real_type,Nd>(src){ }
    private:
    virtual void do_set( std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N) override final{
        aRealMPITopology<real_type,Nd>::do_set(new_n, new_N);
    }
    virtual void do_set_pq( std::array<real_type,Nd> new_x0, std::array<real_type,Nd> new_x1) override final{
        aRealMPITopology<real_type,Nd>::do_set_pq(new_x0,new_x1);
    }
    virtual void do_set( std::array<dg::bc,Nd> new_bcs) override final{
        aRealMPITopology<real_type,Nd>::do_set(new_bcs);
    }
};

///@addtogroup gridtypes
///@{
using MPIGrid0d         = dg::RealMPIGrid<double,0>;
using MPIGrid1d         = dg::RealMPIGrid<double,1>;
using MPIGrid2d         = dg::RealMPIGrid<double,2>;
using MPIGrid3d         = dg::RealMPIGrid<double,3>;
template<size_t Nd>
using MPIGrid           = dg::RealMPIGrid<double,Nd>;
using aMPITopology2d    = dg::aRealMPITopology<double,2>;
using aMPITopology3d    = dg::aRealMPITopology<double,3>;
template<class T>
using aRealMPITopology2d   = dg::aRealMPITopology<T,2>;
template<class T>
using aRealMPITopology3d   = dg::aRealMPITopology<T,3>;
template<class T>
using RealMPIGrid0d   = dg::RealMPIGrid<T,0>;
template<class T>
using RealMPIGrid1d   = dg::RealMPIGrid<T,1>;
template<class T>
using RealMPIGrid2d   = dg::RealMPIGrid<T,2>;
template<class T>
using RealMPIGrid3d   = dg::RealMPIGrid<T,3>;
namespace x{
using Grid0d          = MPIGrid0d      ;
using Grid1d          = MPIGrid1d      ;
using Grid2d          = MPIGrid2d      ;
using Grid3d          = MPIGrid3d      ;
template<size_t Nd>
using Grid            = MPIGrid<Nd>    ;
using aTopology2d     = aMPITopology2d ;
using aTopology3d     = aMPITopology3d ;
template<class T>
using aRealTopology2d   = aRealMPITopology<T,2>;
template<class T>
using aRealTopology3d   = aRealMPITopology<T,3>;
template<class T>
using RealGrid0d   = RealMPIGrid<T,0>;
template<class T>
using RealGrid1d   = RealMPIGrid<T,1>;
template<class T>
using RealGrid2d   = RealMPIGrid<T,2>;
template<class T>
using RealGrid3d   = RealMPIGrid<T,3>;
}//namespace x
///@}

}//namespace dg
