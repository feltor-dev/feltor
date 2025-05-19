#pragma once

#include <cmath>
#include "../backend/mpi_vector.h"
#include "../backend/mpi_kron.h"
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
//     TODO I think it does not matter which boundary condition the communicator has!?
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
 * @brief An abstract base class for MPI distributed Nd-dimensional dG grids
 *
 * @section Shared grids vs MPI grids
 * In the dg library we have MPI grids as a separate type from shared memory
 * grids. The design goal is that MPI grids generally can be used the same
 * way as just a single global shared memory grid. A notable exception is
 * the constructor, when MPI grids depend on a Cartesian MPI communicator
 * (that has to be constructed beforehand with e.g. \c dg::mpi_cart_create)
 * @subsection shared_mpi_grid Shared memory considerations
 * @copydoc hide_grid_description
 * @ingroup basictopology
 * @subsection mpi_mpi_grid Distributed memory considerations
 * In MPI we want to distribute the above hypercube among processes in a
 * Cartesian communicator of same dimensionality \f$ N_d\f$. This is done by
 * evenly distributing the **global** number of cells \f$ N_u\f$ in each
 * dimension among the processes in the corresponding dimension in the
 * Cartesian communicator. Each process gets a **local** number of cells \f$
 * N_{ur} \approx N_u / s_u\f$, where  \f$ s_u \f$ is the size of the Cartesian
 * communicator in direction \f$ u\f$. The approximation becomes an equality if
 * \f$ s_u\f$ evenly divides \f$ N_u\f$ otherwise the remainder is distributed
 * among the participating processes (such that some have one cell more or less
 * than others).  We have \f[ N_u = \sum_{r=0}^{s-1} N_{ur}\f] The number of
 * polynomial coefficients and the boundary condition is the same for all
 * processes of an axis.  Each axis among processes is thus
 * \f[
 * [p_u,q_u] = [p_{u0}, q_{u0}],[p_{u1}, q_{u1}],...,[p_{us-1}, q_{us-1}]
 * \f]
 * The **local boundaries** are determined such that the local \f$ h_{ur} =
 * \frac{q_{ur}-p_{ur}}{N_{ur}} = \frac{q_u-p_u}{N_u}\f$
 *
 * @subsection mpi_overview Overview
 * This class in essence provides a collection of getters and setters for the
 * aforementioned parameters together with the \c abscissas and \c weights
 * members that are necessary for \c dg::evaluate and \c dg::create::weights.
 *
 * @copydoc hide_grid_xyz_description
 * @note Unless otherwise indicate the getters and setters refer to the
 * **global** grid quantities. If access to local quantities is needed
 * we have the \c local() function, which provides a grid with all local
 * quantities
 *
 * Lastly, we provide \c start and \c count members such that the grid can be
 * used as a \c dg::file::MPINcHyperslab in NetCDF output in dg::file
 * @tparam real_type Determines value type of abscissas and weights
 * @tparam Nd The number of dimensions \f$ N_d\f$
 */
template<class real_type, size_t Nd>
struct aRealMPITopology
{
    // ///////////////// TYPE TRAITS ////////////////////////////

    /// value type of abscissas and weights
    using value_type = real_type;
    /// vector type of abscissas and weights; Can be used to recognize MPI grid
    ///via:
    /// <tt>dg::is_vector_v< typename Topology::host_vector, MPIVectorTag></tt>
    using host_vector = MPI_Vector<thrust::host_vector<real_type>>;
    using host_grid = RealMPIGrid<real_type, Nd>;
    /// Dimensionality == Nd
    constexpr static unsigned ndim() { return Nd;}

    // ///////////////// TOPOLOGY CONCEPT ////////////////////////////

    ///@copydoc aRealTopology::shape
    unsigned shape(unsigned u=0) const
    {
        return m_g.shape(u);
    }
    /*! @brief Get the grid abscissas of the \c u axis
     *
     * @note In MPI there is a strong guarantee that the returned abscissas
     * correspond binary exactly to the shared memory abscissas returned
     * by \c global() grid.
     * @param u Axis number \c u<Nd
     * @return Vector containing abscissas
     * @sa dg::evaluate
     */
    host_vector abscissas(unsigned u=0) const
    {
        // We want to be binary exact
        // Therefore we can't just call local abscissas
        int dims[Nd], periods[Nd], coords[Nd];
        MPI_Cart_get( m_comm, Nd, dims, periods, coords);
        real_type global_x0 = m_g.p(u);
        real_type global_hx = m_g.h(u);
        thrust::host_vector<real_type> abs(m_l.shape(u));
        auto aa = dg::DLT<real_type>::abscissas(m_l.n(u));
        auto idx = increment( partition( m_g.N(u), dims[u]));
        for( unsigned i=0; i<m_l.N(u); i++)
        {
            for( unsigned j=0; j<m_l.n(u); j++)
            {
                unsigned coord = idx[coords[u]] + i;
                real_type xmiddle = DG_FMA( global_hx, (real_type)(coord), global_x0);
                real_type h2 = global_hx/2.;
                real_type absj = 1.+aa[j];
                abs[i*m_l.n(u)+j] = DG_FMA( h2, absj, xmiddle);
            }
        }
        return host_vector{abs, m_comms[u]};
    }
    /*! @brief Get the weights of the \c u axis
     *
     * @note In MPI there is a strong guarantee that the returned weights
     * correspond binary exactly to the shared memory weights returned
     * by \c global() grid.
     * @param u Axis number \c u<Nd
     * @return Vector containing weights
     * @sa dg::create::weights dg::DLT
     */
    host_vector weights(unsigned u=0) const
    {
        if( u >= Nd)
            throw Error( Message(_ping_)<<"u>Nd not allowed! You typed: "<<u<<" while Nd is "<<Nd);
        thrust::host_vector<real_type> v( m_l.shape(u));
        auto ww = dg::DLT<real_type>::weights(m_l.n(u));
        real_type hu = m_g.h(u); // We need global h here to be binary exact
        for( unsigned i=0; i<m_l.N(u); i++)
            for( unsigned j=0; j<m_l.n(u); j++)
                v[i*m_l.n(u) + j] = hu/2.*ww[j];
        return host_vector{v, m_comms[u]};
    }
    // ///////////////// GETTERS ////////////////////////////
    /// @copydoc aRealTopology::get_shape
    std::array<unsigned,Nd> get_shape() const{
        m_g.get_shape();
    }
    /// @copydoc aRealTopology::get_abscissas
    std::array<host_vector,Nd> get_abscissas() const{
        std::array<host_vector,Nd> abs;
        for( unsigned u=0; u<Nd; u++)
            abs[u] = abscissas(u);
        return abs;
    }
    /// @copydoc aRealTopology::get_weights
    std::array<host_vector,Nd> get_weights() const{
        std::array<host_vector,Nd> w;
        for( unsigned u=0; u<Nd; u++)
            w[u] = weights(u);
        return w;
    }
    /// @copydoc aRealTopology::get_p
    std::array<real_type,Nd> get_p() const{
        return m_g.get_p();
    }
    /// @copydoc aRealTopology::get_q
    std::array<real_type,Nd> get_q() const{
        return m_g.get_q();
    }
    /// @copydoc aRealTopology::get_l
    std::array<real_type,Nd> get_l() const{
        return m_g.get_l();
    }
    /// @copydoc aRealTopology::get_h
    std::array<real_type,Nd> get_h() const{
        return m_g.get_h();
    }
    /// @copydoc aRealTopology::get_N
    std::array<unsigned, Nd> get_N() const
    {
        return m_g.get_N();
    }
    /// @copydoc aRealTopology::get_n
    std::array<unsigned, Nd> get_n() const
    {
        return m_g.get_n();
    }
    /// @copydoc aRealTopology::get_bc
    std::array<dg::bc, Nd> get_bc() const
    {
        return m_g.get_bc();
    }
    ///@brief Get 1d Cartesian communicator \f$ c_u\f$ for all axes
    ///@return 1d Cartesian communicator \f$ c_u\f$ for all \c u
    std::array<MPI_Comm, Nd> get_comms() const { return m_comms;}

    /// @copydoc aRealTopology::p
    real_type p( unsigned u=0) const { return m_g.p(u);}
    /// @copydoc aRealTopology::q
    real_type q( unsigned u=0) const { return m_g.q(u);}
    /// @copydoc aRealTopology::h
    real_type h( unsigned u=0) const { return m_g.h(u);}
    /// @copydoc aRealTopology::l
    real_type l( unsigned u=0) const { return m_g.l(u);}
    /// @copydoc aRealTopology::n
    unsigned n( unsigned u=0) const  { return m_g.n(u);}
    /// @copydoc aRealTopology::N
    unsigned N( unsigned u=0) const  { return m_g.N(u);}
    /// @copydoc aRealTopology::bc
    dg::bc bc( unsigned u=0) const   { return m_g.bc(u);}

    /**
     * @brief Get 1d Cartesian communicator \f$ c_u\f$ for axis \c u
     * @param u Axis number \c u<Nd
     * @return MPI Cartesian communicator
     */
    MPI_Comm comm(unsigned u) const { return m_comms.at(u);}
    /// Equivalent to <tt>comm(0)</tt>
    template<size_t Md = Nd>
    std::enable_if_t<Md==1,MPI_Comm> comm() const { return m_comms.at(0);}
    /**
     * @brief Return Nd dimensional MPI cartesian communicator that is used in
     * this grid
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


    /// @copydoc aRealTopology::grid
    RealMPIGrid<real_type,1> grid(unsigned u ) const{
        if( u < Nd)
            return RealMPIGrid<real_type,1>{ m_g.p(u), m_g.q(u), m_g.n(u), m_g.N(u), m_g.bc(u), m_comms[u]};
        else
            throw Error( Message(_ping_)<<"u>Nd not allowed! You typed: "<<u<<" while Nd is "<<Nd);
    }
    /// @copydoc aRealTopology::axis
    RealMPIGrid<real_type,1> axis(unsigned u ) const{ return grid(u);} // because we can't decide how to name it ...
    /*!
     * @brief The local grid as a shared memory grid
     * @return Local grid
     */
    const RealGrid<real_type,Nd>& local() const {return m_l;}
    /*!
     * @brief The global grid as a shared memory grid
     * @return Global grid
     */
    const RealGrid<real_type, Nd>& global() const {return m_g;}

    /// Equivalent to <tt>p(0)</tt>
    template<size_t Md = Nd>
    real_type x0() const {return m_g.x0();}
    /// Equivalent to <tt>p(1)</tt>
    template<size_t Md = Nd>
    real_type x1() const {return m_g.x1();}
    /// Equivalent to <tt>p(2)</tt>
    template<size_t Md = Nd>
    real_type y0() const {return m_g.y0();}
    /// Equivalent to <tt>q(0)</tt>
    template<size_t Md = Nd>
    real_type y1() const {return m_g.y1();}
    /// Equivalent to <tt>q(1)</tt>
    template<size_t Md = Nd>
    real_type z0() const {return m_g.z0();}
    /// Equivalent to <tt>q(2)</tt>
    template<size_t Md = Nd>
    real_type z1() const {return m_g.z1();}

    /// Equivalent to <tt>l(0)</tt>
    template<size_t Md = Nd>
    real_type lx() const {return m_g.lx();}
    /// Equivalent to <tt>l(1)</tt>
    template<size_t Md = Nd>
    real_type ly() const {return m_g.ly();}
    /// Equivalent to <tt>l(2)</tt>
    template<size_t Md = Nd>
    real_type lz() const {return m_g.lz();}

    /// Equivalent to <tt>h(0)</tt>
    template<size_t Md = Nd>
    real_type hx() const {return m_g.hx();}
    /// Equivalent to <tt>h(1)</tt>
    template<size_t Md = Nd>
    real_type hy() const {return m_g.hy();}
    /// Equivalent to <tt>h(2)</tt>
    template<size_t Md = Nd>
    real_type hz() const {return m_g.hz();}

    /// Equivalent to <tt>n(0)</tt>
    template<size_t Md = Nd>
    unsigned nx() const {return m_g.nx();}
    /// Equivalent to <tt>n(1)</tt>
    template<size_t Md = Nd>
    unsigned ny() const {return m_g.ny();}
    /// Equivalent to <tt>n(2)</tt>
    template<size_t Md = Nd>
    unsigned nz() const {return m_g.nz();}

    /// Equivalent to <tt>N(0)</tt>
    template<size_t Md = Nd>
    unsigned Nx() const {return m_g.Nx();}
    /// Equivalent to <tt>N(1)</tt>
    template<size_t Md = Nd>
    unsigned Ny() const {return m_g.Ny();}
    /// Equivalent to <tt>N(2)</tt>
    template<size_t Md = Nd>
    unsigned Nz() const {return m_g.Nz();}

    /// Equivalent to <tt>bc(0)</tt>
    template<size_t Md = Nd>
    dg::bc bcx() const {return m_g.bcx();}
    /// Equivalent to <tt>bc(1)</tt>
    template<size_t Md = Nd>
    dg::bc bcy() const {return m_g.bcy();}
    /// Equivalent to <tt>bc(2)</tt>
    template<size_t Md = Nd>
    dg::bc bcz() const {return m_g.bcz();}

    /// Equivalent to <tt>grid(0)</tt>
    template<size_t Md = Nd>
    RealMPIGrid<real_type,1> gx() const {
        static_assert( Nd > 0);
        return grid(0);
    }
    /// Equivalent to <tt>grid(1)</tt>
    template<size_t Md = Nd>
    RealMPIGrid<real_type,1> gy() const {
        static_assert( Nd > 1);
        return grid(1);
    }
    /// Equivalent to <tt>grid(2)</tt>
    template<size_t Md = Nd>
    RealMPIGrid<real_type,1> gz() const {
        static_assert( Nd > 2);
        return grid(2);
    }

    // //////////////////SETTERS/////////////////////////////
    ///@copydoc aRealTopology::multiplyCellNumbers
    template<size_t Md = Nd>
    std::enable_if_t< (Md>=2),void> multiplyCellNumbers( real_type fx, real_type fy)
    {
        auto Ns = m_g.get_N();
        Ns[0] = round(fx*(real_type)m_g.N(0));
        Ns[1] = round(fy*(real_type)m_g.N(1));
        if( fx != 1 or fy != 1)
            set( m_g.get_n(), Ns);
    }
    ///@copydoc aRealTopology::set(unsigned,unsigned)
    template<size_t Md = Nd>
    std::enable_if_t<(Md == 1), void> set( unsigned new_n, unsigned new_Nx)
    {
        set(std::array{new_n}, std::array{new_Nx});
    }
    ///@copydoc aRealTopology::set(unsigned,unsigned,unsigned)
    template<size_t Md = Nd>
    std::enable_if_t<(Md == 2), void> set( unsigned new_n, unsigned new_Nx,
        unsigned new_Ny)
    {
        set(std::array{new_n,new_n}, std::array{new_Nx,new_Ny});
    }
    ///@copydoc aRealTopology::set(unsigned,unsigned,unsigned,unsigned)
    template<size_t Md = Nd>
    std::enable_if_t<(Md == 3), void> set( unsigned new_n, unsigned new_Nx,
        unsigned new_Ny, unsigned new_Nz)
    {
        set(std::array{new_n,new_n,1u}, std::array{new_Nx,new_Ny,new_Nz});
    }
    ///@copydoc aRealTopology::set(unsigned,std::array<unsigned,Nd>)
    void set( unsigned new_n, std::array<unsigned,Nd> new_N)
    {
        std::array<unsigned , Nd> tmp;
        for( unsigned u=0; u<Nd; u++)
            tmp[u] = new_n;
        set( tmp, new_N);
    }
    ///@copydoc aRealTopology::set_axis
    void set_axis( unsigned coord, unsigned new_n , unsigned new_N)
    {
        auto n = m_g.get_n(), N = m_g.get_N();
        n[coord] = new_n;
        N[coord] = new_N;
        set( n, N);
    }
    ///@copydoc aRealTopology::set(std::array<unsigned,Nd>,std::array<unsigned,Nd>)
    void set( std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N)
    {
        if( new_n==m_g.get_n() && new_N == m_g.get_N())
            return;
        do_set(new_n, new_N);
    }
    ///@copydoc aRealTopology::set_pq
    void set_pq( std::array<real_type,Nd> new_p, std::array<real_type,Nd> new_q)
    {
        do_set_pq( new_p, new_q);
    }
    ///@copydoc aRealTopology::set_bcs
    void set_bcs( std::array<dg::bc,Nd> new_bcs)
    {
        do_set( new_bcs);
    }

    // //////////////////UTILITY/////////////////////////////
    /**
     * @brief The total global number of points
     * @return \f$ \prod_{i=0}^{N-1} n_i N_i\f$
     */
    unsigned size() const { return m_g.size();}
    /**
     * @brief The total local number of points
     * @return \c local().size()
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

    ///@copydoc aRealTopology::contains(const Vector&)const
    template<size_t Md = Nd>
    std::enable_if_t<(Md == 1), bool> contains( real_type x) const
    {
        return m_g.contains( x);
    }

    ///@copydoc aRealTopology::contains(const Vector&)const
    template<class Vector>
    bool contains( const Vector& x)const { return m_g.contains(x);}

    /*!
     * @brief Convert the index of a local vector and the rank of the
     * containing process to a global index
     * 
     * @note All calling processes will get the same result
     * @param localIdx Index in a local chunk of a vector
     * @param rank Rank of the process that contains the localIdx
     * @param globalIdx (write only) Global index of the element
     * @sa The main usage of this function is dg::make_mpi_matrix and dg::global2local
     */
    bool local2globalIdx( int localIdx, int rank, int& globalIdx)const
    {
        // TODO shouldn't this test for m_l.size() ? How is this used?
        // ATTENTION This function cannot depend on who is calling it
        // so it cannot depend on m_l
        if( localIdx < 0 || localIdx >= (int)m_g.size()) return false;

        int dims[Nd], periods[Nd], coords[Nd]; // we need the dims
        if( MPI_Cart_get( m_comm, Nd, dims, periods, coords) != MPI_SUCCESS)
            return false;
        // and the coords associated to rank
        if( MPI_Cart_coords( m_comm, rank, Nd, coords) != MPI_SUCCESS)
            return false;
        int gIdx[Nd];
        int current = localIdx;
        for( unsigned u=0; u<Nd; u++)
        {
            auto idx = increment(partition( m_g.N(u), dims[u]));
            unsigned shapeu = (idx[coords[u]+1] - idx[coords[u]])*m_g.n(u);
            int lIdx = current %shapeu; // 1d idx
            current = current / shapeu;
            gIdx[u] = lIdx + idx[coords[u]]*m_g.n(u);
        }
        globalIdx = gIdx[int(Nd)-1]; // prevent overflow if Nd == 0
        for( int u=int(Nd)-2; u>=0; u--)
            globalIdx = globalIdx*m_g.shape(u) + gIdx[u];
        return true;
    }

    /*!
     * @brief Convert the global index of a vector to a local index and the
     * rank of the containing process
     * 
     * @note All calling processes will get the same result
     * @param globalIdx Global index of the element
     * @param localIdx (write only) Index in a local chunk of a vector
     * @param rank (write only) Rank of the process that contains the globalIdx
     * @sa The main usage of this function is dg::make_mpi_matrix and dg::global2local
     */
    bool global2localIdx( int globalIdx, int& localIdx, int& rank)const
    {
        // an exercise in flattening and unflattening indices
        if( globalIdx < 0 || globalIdx >= (int)m_g.size()) return false;

        int dims[Nd], periods[Nd], coords[Nd];
        if( MPI_Cart_get( m_comm, Nd, dims, periods, coords) != MPI_SUCCESS)
            return false;

        int lIdx[Nd] = {0}, local_shape[Nd] = {0};
        int current = globalIdx;
        // ATTENTION This function cannot depend on who is calling it
        // so it cannot depend on m_l or current coords
        for( unsigned u=0; u<Nd; u++)
        {
            int gIdx = current%(m_g.shape(u)); // 1d idx
            current = current / (m_g.shape(u));
            auto idx = increment(partition( m_g.N(u), dims[u]));
            // Find coord
            for( unsigned c=0; c<idx.size()-1; c++)
                if( unsigned(gIdx)< idx[c+1]*m_g.n(u))
                {
                    coords[u] = c;
                    lIdx[u] = gIdx - idx[c]*m_g.n(u);
                    local_shape[u] = (idx[c+1]-idx[c])*m_g.n(u);
                    break;
                }
        }
        localIdx = lIdx[int(Nd)-1];
        for( int u=int(Nd)-2; u>=0; u--)
            localIdx = localIdx*local_shape[u] + lIdx[u];

        if( MPI_Cart_rank( m_comm, coords, &rank) == MPI_SUCCESS )
            return true;
        else
            return false;
    }

    /*! @brief The global start coordinate in C-order of \c
     * dg::file::MPINcHyperslab that the local grid represents
     *
     * Used to construct \c dg::file::MPINcHyperslab together with \c count()
     * @return global start coordinates of the local grid in C-order
     * @note In C-order the fastest dimension is the last one while our \c
     * dg::evaluate and \c dg::kronecker make the 0 dimension/ 1st argument the
     * fastest varying.
     */
    std::array<unsigned, Nd> start() const
    {
        int dims[Nd], periods[Nd], coords[Nd];
        MPI_Cart_get( m_comm, Nd, dims, periods, coords);
        std::array<unsigned, Nd> start;
        for( unsigned u=0;u<Nd; u++)
        {
            auto idx = increment(partition( m_g.N(u), dims[u]));
            start[Nd-1-u] = idx[coords[u]]*m_g.n(u);
        }
        return start;
    }
    /*! @brief Count vector in C-order for \c dg::file::MPINcHyperslab
     *
     * Used to construct \c dg::file::MPINcHyperslab together with \c start()
     * @return \c local().count()
     * @note In C-order the fastest dimension is the last one while our \c
     * dg::evaluate and \c dg::kronecker make the 0 dimension/ 1st argument the
     * fastest varying, so we return the reverse order of \c local().get_shape()
     */
    std::array<unsigned, Nd> count() const { return m_l.count(); }
    protected:
    ///@copydoc aRealTopology::~aRealTopology
    ~aRealMPITopology() = default;

    ///@copydoc aRealTopology::aRealTopology()
    aRealMPITopology() = default;

    /**
     * @brief Construct a topology directly from points and dimensions
     *
     * @param p left boundary point
     * @param q right boundary point
     * @param n number of polynomial coefficients for each axis
     * @param N number of cells for each axis
     * @param bcs boundary condition for each axis
     * @param comms The 1d MPI Cartesian communicators for each axis that make
     * up an Nd dimensional Cartesian communicator (see \c dg::mpi_cart_split,
     * \c dg::mpi_cart_kron)
     */
    aRealMPITopology( std::array<real_type,Nd> p, std::array<real_type,Nd> q,
        std::array<unsigned,Nd> n, std::array<unsigned,Nd> N,
        std::array<dg::bc,Nd> bcs, std::array<MPI_Comm, Nd> comms) :
        m_g(p,q,n,N,bcs), m_comms(comms)
    {
        // assert dimensionality of Cartesian communicators
        int ndims;
        for ( unsigned u=0; u<Nd;u++)
        {
            MPI_Cartdim_get( m_comms[u], &ndims);
            assert( (unsigned)ndims == 1);
        }
        m_comm = dg::mpi_cart_kron( m_comms);
        MPI_Cartdim_get( m_comm, &ndims);
        assert( (unsigned)ndims == Nd);
        // The idea is that every grid gets the same amount and the
        // rest is distributed to the lowest rank grids
        int dims[Nd], periods[Nd], coords[Nd];
        MPI_Cart_get( m_comm, Nd, dims, periods, coords);
        for( unsigned u=0;u<Nd; u++)
        {
            auto idx = increment(partition( m_g.N(u), dims[u]));
            N[u] = idx[coords[u]+1]-idx[coords[u]] ;

            p[u] = m_g.p(u) + m_g.h(u)*idx[coords[u]];
            q[u] = m_g.p(u) + m_g.h(u)*idx[coords[u] +1];
            // The local right boundary should be the same as the global right boundary
            if( coords[u] == dims[u]-1)
                q[u] = m_g.q(u);
        }
        m_l = { p, q, m_g.get_n(), N, m_g.get_bc()};
    }
    ///@copydoc aRealTopology::aRealTopology(const std::array<RealGrid<real_type,1>,Nd>&)
    aRealMPITopology( const std::array< RealMPIGrid<real_type, 1>, Nd>& axes)
    {
        std::array<RealGrid<real_type,1>,Nd> globals, locals;
        for( unsigned u=0; u<Nd; u++)
        {
            globals[u] = axes[u].global();
            locals[u] = axes[u].local();
            m_comms[u] = axes[u].communicator();
        }
        m_g = RealGrid<real_type,Nd>( globals);
        m_l = RealGrid<real_type,Nd>( locals);
        m_comm = dg::mpi_cart_kron( {m_comms.begin(), m_comms.end()});
    }

    ///@copydoc aRealTopology::aRealTopology(const aRealTopology&)
    aRealMPITopology(const aRealMPITopology& src) = default;
    ///@copydoc aRealTopology::operator=
    aRealMPITopology& operator=(const aRealMPITopology& src) = default;
    ///@copydoc aRealTopology::set(std::array<unsigned,Nd>,std::array<unsigned,Nd>)
    virtual void do_set(std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N) =0;
    ///@copydoc aRealTopology::set_pq
    virtual void do_set_pq( std::array<real_type, Nd> new_p, std::array<real_type,Nd> new_q) =0;
    ///@copydoc aRealTopology::set_bcs(std::array<dg::bc,Nd>)
    virtual void do_set( std::array<dg::bc, Nd> new_bcs) =0;

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
    private:
    void check_periods( std::array<dg::bc, Nd> bc) const
    {
        if constexpr ( Nd > 0) // avoid zero sized array warnings
        {
            int rank, dims[Nd], periods[Nd], coords[Nd];
            MPI_Cart_get( m_comm, Nd, dims, periods, coords);
            MPI_Comm_rank( m_comm, &rank);
            if( rank == 0)
            {
                for( unsigned u=0; u<Nd; u++)
                {
                    if( bc[u] == dg::PER) assert( periods[u] == true);
                    else assert( periods[u] == false);
                }
            }
        }
    }
    std::vector<unsigned> partition( unsigned N, unsigned r) const
    {
        // Divide N points as equally as possible among participants r
        std::vector<unsigned> points(r, N/r );
        for( unsigned u=0; u<N%r; u++)
            points[u]++;
        return points;
    }

    std::vector<unsigned> increment( const std::vector<unsigned>& partition) const
    {
        // replace with std::inclusive_scan ?
        // return global starting idx and end idex
        // start = inc[coord], end = inc[coord+1]
        std::vector<unsigned> inc( partition.size()+1, 0);
        for( unsigned u=0; u<inc.size(); u++)
            for( unsigned k=0; k<u; k++)
                inc[u] += partition[k];
        return inc;
    }

    RealGrid<real_type, Nd> m_g, m_l; //global grid, local grid
    std::array<MPI_Comm, Nd> m_comms; // 1d comms
    MPI_Comm m_comm; //just an integer...(No, more like an address)
};
///@cond

// pure virtual implementations must be declared outside class
template<class real_type,size_t Nd>
void aRealMPITopology<real_type,Nd>::do_set( std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N)
{
    if constexpr ( Nd > 0) // avoid zero sized array warnings
    {
        m_g.set(new_n, new_N);
        int dims[Nd], periods[Nd], coords[Nd];
        MPI_Cart_get( m_comm, Nd, dims, periods, coords);
        std::array<unsigned, Nd> N;
        for( unsigned u=0;u<Nd; u++)
        {
            auto idx = increment(partition( m_g.N(u), dims[u]));
            N[u] = idx[coords[u]+1]-idx[coords[u]] ;
        }
        m_l.set( new_n, N);
    }

}
template<class real_type,size_t Nd>
void aRealMPITopology<real_type,Nd>::do_set_pq( std::array<real_type, Nd> x0, std::array<real_type,Nd> x1)
{
    if constexpr ( Nd > 0) // avoid zero sized array warnings
    {
        m_g.set_pq( x0, x1);
        int dims[Nd], periods[Nd], coords[Nd];
        MPI_Cart_get( m_comm, Nd, dims, periods, coords);
        std::array<real_type,Nd> p, q;
        for( unsigned u=0;u<Nd; u++)
        {
            auto idx = increment(partition( m_g.N(u), dims[u]));
            p[u] = m_g.p(u) + m_g.h(u)*idx[coords[u]];
            q[u] = m_g.p(u) + m_g.h(u)*idx[coords[u] +1];
        }
        m_l.set_pq( p, q);
    }
}
template<class real_type,size_t Nd>
void aRealMPITopology<real_type,Nd>::do_set( std::array<dg::bc, Nd> bcs)
{
    check_periods( bcs);
    m_g.set_bcs( bcs);
    m_l.set_bcs( bcs);
}

///@endcond

/**
 * @brief The simplest implementation of aRealMPITopology3d
 * @ingroup basictopology
 * @snippet{trimleft} evaluation_t.cpp evaluate2d
 */
template<class real_type, size_t Nd>
struct RealMPIGrid : public aRealMPITopology<real_type,Nd>
{
    ///@copydoc RealGrid::RealGrid()
    RealMPIGrid() = default;
    template<size_t Md = Nd>
    RealMPIGrid( real_type x0, real_type x1, unsigned n, unsigned N, MPI_Comm
            comm): aRealMPITopology<real_type,1>( {x0}, {x1},
                {n}, {N}, {dg::PER}, {comm})
    { }
    ///@copydoc RealGrid::RealGrid(real_type,real_type,unsigned,unsigned,bc)
    ///@copydoc hide_comm_parameters1d
    template<size_t Md = Nd>
    RealMPIGrid( real_type x0, real_type x1, unsigned n, unsigned Nx, dg::bc bcx, MPI_Comm
            comm): aRealMPITopology<real_type,1>( {x0}, {x1},
                {n}, {Nx}, {bcx}, {comm})
    {}
    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_comm_parameters2d
    template<size_t Md = Nd>
    RealMPIGrid( real_type x0, real_type x1, real_type y0, real_type y1,
            unsigned n, unsigned Nx, unsigned Ny, MPI_Comm comm):
        aRealMPITopology<real_type,2>(
                {x0,y0},{x1,y1},{n,n},{Nx,Ny},{dg::PER,dg::PER},
                dg::mpi_cart_split_as<2>(comm))
    { }

    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_bc_parameters2d
    ///@copydoc hide_comm_parameters2d
    template<size_t Md = Nd>
    RealMPIGrid( real_type x0, real_type x1, real_type y0, real_type y1,
            unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, dg::bc bcy, MPI_Comm
            comm):
        aRealMPITopology<real_type,2>(
                {x0,y0},{x1,y1},{n,n},{Nx,Ny},{bcx,bcy},
                dg::mpi_cart_split_as<2>(comm))
    { }
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_comm_parameters3d
    template<size_t Md = Nd>
    RealMPIGrid( real_type x0, real_type x1, real_type y0, real_type y1,
            real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny,
            unsigned Nz, MPI_Comm comm):
        aRealMPITopology<real_type,3>(
                {x0,y0,z0},{x1,y1,z1},{n,n,1},{Nx,Ny,Nz},{dg::PER,dg::PER,dg::PER},
                dg::mpi_cart_split_as<3>(comm))
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
                dg::mpi_cart_split_as<3>(comm))
    { }

    ///@copydoc aRealTopology::aRealTopology(const std::array<RealGrid<real_type,1>,Nd>&)
    RealMPIGrid( const std::array<RealMPIGrid<real_type,1>,Nd>& axes) :
        aRealMPITopology<real_type,Nd>( axes){}

    /**
     * @brief Construct from given 1d grids
     * Equivalent to <tt>RealMPIGrid( std::array{g0,gs...})</tt>
     * @param g0 Axis 0 grid
     * @param gs more axes
     */
    template<class ...Grid1ds>
    RealMPIGrid( const RealMPIGrid<real_type,1>& g0, const Grid1ds& ...gs) :
        aRealMPITopology<real_type,Nd>( std::array<RealMPIGrid<real_type,1>,Nd>{g0, gs...}){}

    ///@copydoc aRealMPITopology::aRealMPITopology(std::array<real_type,Nd>,std::array<real_type,Nd>,std::array<unsigned,Nd>,std::array<unsigned,Nd>,std::array<dg::bc,Nd>,std::array<MPI_Comm,Nd>)
    RealMPIGrid( std::array<real_type,Nd> p, std::array<real_type,Nd> q,
        std::array<unsigned,Nd> n, std::array<unsigned,Nd> N,
        std::array<dg::bc,Nd> bcs, std::array<MPI_Comm,Nd> comms) :
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
