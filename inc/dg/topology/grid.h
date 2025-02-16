#pragma once
#include <array>
#include <cmath>
#include <thrust/host_vector.h>
#include "../backend/tensor_traits.h" // for get_value_type
#include "../backend/config.h" // for DG_FMA
#include "dlt.h"
#include "../enums.h"

/*! @file
  @brief base topology classes
  */
/*!@class hide_grid_parameters2d
 * @brief Construct with equal polynomial coefficients
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
 * @brief Construct with equal polynomial coefficients
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

namespace dg{

///@cond
template<class real_type, size_t Nd>
struct RealGrid;
///@endcond
///
/**
 * @class hide_grid_description
 * This grid defines a discretization of the \f$N_d\f$ dimensional hypercube
 * given by
 * \f[ [\vec p, \vec q] = [p_0, p_1] \times [p_1,q_1] \times ... \times
 * [p_{N_d-1}, q_{N_d-1}]\f]
 * Each axis \f$ [p_i, q_i] \f$ is discretized using \f$ N_i\f$ equidistant
 * cells. Each cells is then further discretized using \f$ n_i\f$
 * Gauss-Legendre nodes. The Gauss-Legendre nodes are tabulated by the \c
 * dg::DLT class for \f$ 1\leq n_i \leq 20\f$.  Each axis further can have a
 * boundary condition \f$ b_i \f$ that is given by \c dg::bc
 * For more information dG methods see
 * @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
 *
 * @class hide_grid_xyz_description
 * For code readability in many physical contexts the first 3 dimensions get
 * special names \f$ x,\ y,\ z\f$ i.e. \f$ \vec p = (x_0, y_0, z_0)\f$ and \f$
 * \vec q = (x_1, y_1, z_1)\f$. The class provides coresponding getters and
 * setters like e.g. \c x0() or \c Nx() for \c p(0) and \c N(0)
 */

/**
 * @brief An abstract base class for Nd-dimensional dG grids
 *
 * @copydoc hide_grid_description
 *
 * This class in essence provides a collection of getters and setters for the
 * aforementioned parameters together with the \c abscissas and \c weights
 * members that are necessary for \c dg::evaluate and \c dg::create::weights.
 * Lastly, we provide \c start and \c count members such that the grid can be
 * used as a \c dg::file::NcHyperslab in NetCDF output in dg::file
 * @copydoc hide_grid_xyz_description
 * @ingroup basictopology
 * @tparam real_type Determines value type of abscissas and weights
 * @tparam Nd The number of dimensions \f$ N_d\f$
 */
template<class real_type, size_t Nd>
struct aRealTopology
{
    // ///////////////// TYPE TRAITS ////////////////////////////
    /// value type of abscissas and weights
    using value_type = real_type;
    /// vector type of abscissas and weights; Is used to recognise shared
    ///topology (vs MPI)
    /// <tt>dg::is_vector_v< typename Topology::host_vector, SharedVectorTag></tt>
    using host_vector = thrust::host_vector<real_type>;
    /// Associated realisation
    using host_grid = RealGrid<real_type, Nd>;
    /// Dimensionality == Nd
    constexpr static unsigned ndim() { return Nd;}

    // ///////////////// TOPOLOGY CONCEPT ////////////////////////////
    /**
     * @brief \f$ n_u N_u\f$ the total number of points of an axis
     * @param u Axis number \c u<Nd
     * @return \f$ n_u N_u\f$
     */
    unsigned shape(unsigned u=0) const
    {
        if( u >= Nd)
            throw Error( Message(_ping_)<<"u>=Nd not allowed! You typed: "
                    <<u<<" while Nd is "<<Nd);
        return m_n[u]*m_N[u];
    }

    /*! @brief Construct grid abscissas of the \c u axis
     *
     * @param u Axis number \c u<Nd
     * @return Vector \f$ \vec a_u\f$ containing abscissas for axis \c u
     * @sa dg::evaluate dg::DLT
     */
    host_vector abscissas(unsigned u=0) const
    {
        // On getting n,N,x0,x1 given the abscissas:
        // Unfortunately, we cannot invert this to binary precision
        // at least not with our pyfeltor implementation
        if( u >= Nd)
            throw Error( Message(_ping_)<<"u>=Nd not allowed! You typed: "
                    <<u<<" while Nd is "<<Nd);
        host_vector abs(m_n[u]*m_N[u]);
        real_type hu = h(u);
        auto aa = dg::DLT<real_type>::abscissas(m_n[u]);
        for( unsigned i=0; i<m_N[u]; i++)
            for( unsigned j=0; j<m_n[u]; j++)
            {
                real_type xmiddle = DG_FMA( hu, (real_type)(i), m_x0[u]);
                real_type h2 = hu/2.;
                real_type absj = 1.+aa[j];
                abs[i*m_n[u]+j] = DG_FMA( h2, absj, xmiddle);
            }
        return abs;
    }
    /*! @brief Get the weights of the \c u axis
     *
     * @param u Axis number \c u<Nd
     * @return Vector \f$ \vec w_u\f$ containing weights for axis \c u
     * @sa dg::create::weights dg::DLT
     */
    host_vector weights(unsigned u=0) const
    {
        if( u >= Nd)
            throw Error( Message(_ping_)<<"u>Nd not allowed! You typed: "<<u<<" while Nd is "<<Nd);
        host_vector v( m_n[u]*m_N[u]);
        auto ww = dg::DLT<real_type>::weights(m_n[u]);
        real_type hu = h(u);
        for( unsigned i=0; i<m_N[u]; i++)
            for( unsigned j=0; j<m_n[u]; j++)
                v[i*m_n[u] + j] = hu/2.*ww[j];
        return v;
    }

    // ///////////////// GETTERS ////////////////////////////
    /**
     * @brief \f$ n_u N_u\f$ the total number of points of an axis
     * @return \f$ n_u N_u\f$ for all \f$ u < N_d\f$
     */
    std::array<unsigned,Nd> get_shape() const{
        std::array<unsigned,Nd> ss;
        for( unsigned u=0; u<Nd; u++)
            ss[u] = shape(u);
        return ss;
    }
    /**
     * @brief Construct abscissas for all axes
     * @return \f$ \vec a_u\f$ for all \f$ u < N_d\f$
     */
    std::array<host_vector,Nd> get_abscissas() const{
        std::array<host_vector,Nd> abs;
        for( unsigned u=0; u<Nd; u++)
            abs[u] = abscissas(u);
        return abs;
    }
    /**
     * @brief Construct weights for all axes
     * @return \f$ \vec w_u\f$ for all \f$ u < N_d\f$
     */
    std::array<host_vector,Nd> get_weights() const{
        std::array<host_vector,Nd> w;
        for( unsigned u=0; u<Nd; u++)
            w[u] = weights(u);
        return w;
    }

    ///@brief Get left boundary point \f$ \vec p\f$
    ///@return Left boundary \f$ \vec p\f$
    std::array<real_type,Nd> get_p() const{
        return m_x0;
    }
    ///@brief Get right boundary point \f$ \vec q\f$
    ///@return right boundary \f$ \vec q\f$
    std::array<real_type,Nd> get_q() const{
        return m_x1;
    }
    ///@brief Get grid length \f$ l_u = q_u - p_u\f$ for all axes
    ///@return Grid length \f$ l_u\f$ for all \c u
    std::array<real_type,Nd> get_l() const{
        std::array<real_type, Nd> p;
        for( unsigned u=0; u<Nd; u++)
            p[u] = l(u);
        return p;
    }
    ///@brief Get grid constant \f$ h_u = \frac{q_u - p_u}{N_u}\f$ for all axes
    ///@return Grid constant \f$ h_u\f$ for all \c u
    std::array<real_type,Nd> get_h() const{
        std::array<real_type, Nd> hh;
        for( unsigned u=0; u<Nd; u++)
            hh[u] = h(u);
        return hh;
    }
    ///@brief Get number of cells \f$ N_u\f$ for all axes
    ///@return Number of cells \f$ N_u\f$ for all \c u
    std::array<unsigned, Nd> get_N() const
    {
        return m_N;
    }
    ///@brief Get number of polynomial coefficients \f$ n_u\f$ for all axes
    ///@return Number of polynomial coefficients \f$ n_u\f$ for all \c u
    std::array<unsigned, Nd> get_n() const
    {
        return m_n;
    }
    ///@brief Get boundary condition \f$ b_u\f$ for all axes
    ///@return Boundary condition \f$ b_u\f$ for all \c u
    std::array<dg::bc, Nd> get_bc() const
    {
        return m_bcs;
    }

    /*! 
     * @brief Get left boundary point \f$ p_u\f$ for axis \c u
     * @param u Axis number \c u<Nd
     * @return Value for axis \c u
     */
    real_type p( unsigned u=0) const { return m_x0.at(u);}
    ///@brief Get right boundary point \f$ q_u\f$ for axis \c u
    ///@copydetails p(unsigned)const
    real_type q( unsigned u=0) const { return m_x1.at(u);}
    ///@brief Get grid constant \f$ h_u = \frac{q_u - p_u}{N_u}\f$ for axis \c u
    ///@copydetails p(unsigned)const
    real_type h( unsigned u=0) const { return (m_x1.at(u) - m_x0.at(u))/(real_type)m_N.at(u);}
    ///@brief Get grid length \f$ l_u = q_u - p_u\f$ for axis \c u
    ///@copydetails p(unsigned)const
    real_type l( unsigned u=0) const { return m_x1.at(u) - m_x0.at(u);}
    ///@brief Get number of polynomial coefficients \f$ n_u\f$ for axis \c u
    ///@copydetails p(unsigned)const
    unsigned n( unsigned u=0) const { return m_n.at(u);}
    ///@brief Get number of cells \f$ N_u\f$ for axis \c u
    ///@copydetails p(unsigned)const
    unsigned N( unsigned u=0) const { return m_N.at(u);}
    ///@brief Get boundary condition \f$ b_u\f$ for axis \c u
    ///@copydetails p(unsigned)const
    dg::bc bc( unsigned u=0) const { return m_bcs.at(u);}
    /*!
     * @brief Get axis \c u as a 1d grid
     * @param u Axis number \c u<Nd
     * @return One dimensional grid
     */
    RealGrid<real_type,1> grid(unsigned u ) const{
        if( u < Nd)
            return RealGrid<real_type,1>{ m_x0[u], m_x1[u], m_n[u], m_N[u], m_bcs[u]};
        else
            throw Error( Message(_ping_)<<"u>Nd not allowed! You typed: "<<u<<" while Nd is "<<Nd);
    }
    /// @brief An alias for "grid"
    /// @copydetails grid(unsigned)const
    RealGrid<real_type,1> axis(unsigned u ) const{ return grid(u);}
    /// Equivalent to <tt>p(0)</tt>
    template<size_t Md = Nd>
    real_type x0() const {return std::get<0>(m_x0);}
    /// Equivalent to <tt>p(1)</tt>
    template<size_t Md = Nd>
    real_type x1() const {return std::get<0>(m_x1);}
    /// Equivalent to <tt>p(2)</tt>
    template<size_t Md = Nd>
    real_type y0() const {return std::get<1>(m_x0);}
    /// Equivalent to <tt>q(0)</tt>
    template<size_t Md = Nd>
    real_type y1() const {return std::get<1>(m_x1);}
    /// Equivalent to <tt>q(1)</tt>
    template<size_t Md = Nd>
    real_type z0() const {return std::get<2>(m_x0);}
    /// Equivalent to <tt>q(2)</tt>
    template<size_t Md = Nd>
    real_type z1() const {return std::get<2>(m_x1);}

    /// Equivalent to <tt>l(0)</tt>
    template<size_t Md = Nd>
    real_type lx() const {return std::get<0>(get_l());}
    /// Equivalent to <tt>l(1)</tt>
    template<size_t Md = Nd>
    real_type ly() const {return std::get<1>(get_l());}
    /// Equivalent to <tt>l(2)</tt>
    template<size_t Md = Nd>
    real_type lz() const {return std::get<2>(get_l());}

    /// Equivalent to <tt>h(0)</tt>
    template<size_t Md = Nd>
    real_type hx() const {return std::get<0>(get_h());}
    /// Equivalent to <tt>h(1)</tt>
    template<size_t Md = Nd>
    real_type hy() const {return std::get<1>(get_h());}
    /// Equivalent to <tt>h(2)</tt>
    template<size_t Md = Nd>
    real_type hz() const {return std::get<2>(get_h());}

    /// Equivalent to <tt>n(0)</tt>
    template<size_t Md = Nd>
    unsigned nx() const {return std::get<0>(m_n);}
    /// Equivalent to <tt>n(1)</tt>
    template<size_t Md = Nd>
    unsigned ny() const {return std::get<1>(m_n);}
    /// Equivalent to <tt>n(2)</tt>
    template<size_t Md = Nd>
    unsigned nz() const {return std::get<2>(m_n);}

    /// Equivalent to <tt>N(0)</tt>
    template<size_t Md = Nd>
    unsigned Nx() const {return std::get<0>(m_N);}
    /// Equivalent to <tt>N(1)</tt>
    template<size_t Md = Nd>
    unsigned Ny() const {return std::get<1>(m_N);}
    /// Equivalent to <tt>N(2)</tt>
    template<size_t Md = Nd>
    unsigned Nz() const {return std::get<2>(m_N);}

    /// Equivalent to <tt>bc(0)</tt>
    template<size_t Md = Nd>
    dg::bc bcx() const {return std::get<0>(m_bcs);}
    /// Equivalent to <tt>bc(1)</tt>
    template<size_t Md = Nd>
    dg::bc bcy() const {return std::get<1>(m_bcs);}
    /// Equivalent to <tt>bc(2)</tt>
    template<size_t Md = Nd>
    dg::bc bcz() const {return std::get<2>(m_bcs);}

    /// Equivalent to <tt>grid(0)</tt>
    template<size_t Md = Nd>
    RealGrid<real_type,1> gx() const {
        static_assert( Nd > 0);
        return grid(0);
    }
    /// Equivalent to <tt>grid(1)</tt>
    template<size_t Md = Nd>
    RealGrid<real_type,1> gy() const {
        static_assert( Nd > 1);
        return grid(1);
    }
    /// Equivalent to <tt>grid(2)</tt>
    template<size_t Md = Nd>
    RealGrid<real_type,1> gz() const {
        static_assert( Nd > 2);
        return grid(2);
    }

    /*! @brief Start coordinate in C-order for \c dg::file::NcHyperslab
     *
     * Used to construct \c dg::file::NcHyperslab together with \c count()
     * @return \c {0}
     */
    std::array<unsigned, Nd> start() const { return {0};}
    /*! @brief Count vector in C-order for \c dg::file::NcHyperslab
     *
     * Used to construct \c dg::file::NcHyperslab together with \c start()
     * @return <tt> reverse( get_shape())</tt>
     * @note In C-order the fastest dimension is the last one while our \c
     * dg::evaluate and \c dg::kronecker make the 0 dimension/ 1st argument the
     * fastest varying, so we return the reverse order of \c get_shape()
     */
    std::array<unsigned, Nd> count() const
    {
        std::array<unsigned,Nd> ss;
        for( unsigned u=0; u<Nd; u++)
            ss[Nd-1-u] = m_n[u]*m_N[u];
        return ss;
    }

    // //////////////////SETTERS/////////////////////////////
    /**
    * @brief Multiply the number of cells in the first two dimensions with a
    * given factor
    *
    * With this function you can resize the grid ignorantly of its current size
    * @param fx new global number of cells is fx*Nx()
    * @param fy new global number of cells is fy*Ny()
    * The remaining dimensions are left unchanged
    */
    template<size_t Md = Nd>
    std::enable_if_t< (Md>=2),void> multiplyCellNumbers( real_type fx, real_type fy){
        auto Ns = m_N;
        Ns[0] = round(fx*(real_type)m_N[0]);
        Ns[1] = round(fy*(real_type)m_N[1]);
        if( fx != 1 or fy != 1)
            set( m_n, Ns);
    }
    /**
     * @brief Set n and N in a 1-dimensional grid
     *
    * @param new_n new number of %Gaussian nodes
    * @param new_Nx new number of cells
     */
    template<size_t Md = Nd>
    std::enable_if_t<(Md == 1), void> set( unsigned new_n, unsigned new_Nx)
    {
        set({new_n}, {new_Nx});
    }

    /**
     * @brief Set n and N in a 2-dimensional grid
     *
    * @param new_n new number of %Gaussian nodes in x and y
    * @param new_Nx new number of cells
    * @param new_Ny new number of cells
     */
    template<size_t Md = Nd>
    std::enable_if_t<(Md == 2), void> set( unsigned new_n, unsigned new_Nx,
        unsigned new_Ny)
    {
        set({new_n,new_n}, {new_Nx,new_Ny});
    }
    /**
    * @brief Set n and N in a 3-dimensional grid
    *
    * Same as <tt>set({new_n,new_n,1}, {new_Nx,new_Ny,new_Nz})</tt>
    * @param new_n new number of %Gaussian nodes in x and y
    * @attention Set \c nz to 1
    * @param new_Nx new number of cells in x
    * @param new_Ny new number of cells in y
    * @param new_Nz new number of cells in z
    */
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
     * @brief Set n and N for axis \c coord
     *
     * @param coord Axis <tt>coord<Nd</tt>
     * @param new_n new number of %Gaussian nodes of axis \c coord
     * @param new_N new number of cells
     */
    void set_axis( unsigned coord, unsigned new_n , unsigned new_N)
    {
        std::array<unsigned,Nd> n = m_n, N = m_N;
        n[coord] = new_n;
        N[coord] = new_N;
        set( n, N);
    }

    /**
    * @brief Set the number of polynomials and cells
    *
    * @param new_n new number of %Gaussian nodes in each dimension
    * @param new_N new number of cells in each dimension
    */
    void set( std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N)
    {
        if( new_n==m_n && new_N == m_N)
            return;
        do_set(new_n, new_N);
    }

    /**
     * @brief Reset the boundaries of the grid
     *
     * @param new_p new left boundary
     * @param new_q new right boundary ( > x0)
     */
    void set_pq( std::array<real_type,Nd> new_p, std::array<real_type,Nd>
            new_q)
    {
        do_set_pq( new_p, new_q);
    }
    /**
     * @brief Reset the boundary conditions of the grid
     *
     * @param new_bcs new boundary condition
     */
    void set_bcs( std::array<dg::bc,Nd> new_bcs)
    {
        do_set( new_bcs);
    }

    /**
     * @brief Reset the entire grid
     *
     * @param new_p new left boundary
     * @param new_q new right boundary ( > x0)
     * @param new_n new number of %Gaussian nodes in each dimension
     * @param new_N new number of cells in each dimension
     * @param new_bcs new boundary condition
     */
    void set( std::array<real_type,Nd> new_p, std::array<real_type,Nd> new_q,
        std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N,
        std::array<dg::bc,Nd> new_bcs)
    {
        set_pq( new_p,new_q);
        set( new_n, new_N);
        set_bcs( new_bcs);
    }
    // //////////////////UTILITY/////////////////////////////
    /**
     * @brief The total number of points
     *
     * @return \f$ \prod_{i=0}^{N-1} n_i N_i\f$
     */
    unsigned size() const {
        unsigned size=1;
        for( unsigned u=0; u<Nd; u++)
            size *= m_n[u]*m_N[u];
        return size;
    }


    /**
     * @brief Display
     *
     * @param os output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        for( unsigned u=0; u<Nd; u++)
        {
            os << "Topology parameters for Grid "<<u<<" are: \n"
                <<"    n  = "<<m_n[u]<<"\n"
                <<"    N  = "<<m_N[u]<<"\n"
                <<"    x0 = "<<m_x0[u]<<"\n"
                <<"    x1 = "<<m_x1[u]<<"\n"
                <<"    h  = "<<h(u)<<"\n"
                <<"    l  = "<<l(u)<<"\n"
                <<"    bc = "<<bc2str(m_bcs[u])<<"\n";
        }
    }

    ///@copydoc contains(const Vector&)const
    template<size_t Md = Nd>
    std::enable_if_t<(Md == 1), bool> contains( real_type x) const
    {
        return contains( std::array<real_type,1>{x});
    }

    /**
     * @brief Check if the grid contains a point
     *
     * Used for example in \c integrate_in_domain method of  \c dg::AdaptiveTimeloop
     * @note doesn't check periodicity!!
     * @param x point to check
     *
     * @return true if p0[u]<=x[u]<=p1[u] for all u, false else
     * @attention returns false if x[u] is NaN or INF
     */
    template<class Vector>
    bool contains( const Vector& x)const
    {
        for( unsigned u=0; u<Nd; u++)
        {
            if( !std::isfinite(x[u]) ) return false;
            //should we catch the case x1==x && dg::PER?
            if( x[u] < m_x0[u]) return false;
            if( x[u] > m_x1[u]) return false;
        }
        return true;
    }

    protected:
    ///disallow deletion through base class pointer
    ~aRealTopology() = default;
    /// default constructor
    aRealTopology() = default;
    /**
     * @brief Construct a topology directly from points and dimensions
     *
     * @param p left boundary point
     * @param q right boundary point
     * @param n number of polynomial coefficients for each axis
     * @param N number of cells for each axis
     * @param bcs boundary condition for each axis
     */
    aRealTopology(
        std::array<real_type,Nd> p,
        std::array<real_type,Nd> q,
        std::array<unsigned,Nd> n,
        std::array<unsigned,Nd> N,
        std::array<dg::bc, Nd> bcs) : m_x0(p), m_x1(q), m_n(n), m_N(N), m_bcs(bcs)
    {}
    /**
     * @brief Construct a topology as the product of 1d axes grids
     *
     * @param axes One-dimensional grids for each dimension
     */
    aRealTopology( const std::array< RealGrid<real_type, 1>, Nd>& axes)
    {
        for( unsigned u=0; u<Nd; u++)
        {
            m_x0[u] = axes[u].p();
            m_x1[u] = axes[u].q();
            m_n[u] = axes[u].n();
            m_N[u] = axes[u].N();
            m_bcs[u] = axes[u].bc();
        }
    }

    ///explicit copy constructor (default)
    ///@param src source
    aRealTopology(const aRealTopology& src) = default;
    ///explicit assignment operator (default)
    ///@param src source
    aRealTopology& operator=(const aRealTopology& src) = default;

    ///@copydoc set(std::array<unsigned,Nd>,std::array<unsigned,Nd>)
    virtual void do_set(std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N) = 0;
    ///@copydoc set_pq
    virtual void do_set_pq( std::array<real_type, Nd> new_p, std::array<real_type,Nd> new_q) = 0;
    ///@copydoc set_bcs(std::array<dg::bc,Nd>)
    virtual void do_set( std::array<dg::bc, Nd> new_bcs) = 0;

    // MW: This constructor causes nvcc-12.4 to segfault when constructing a Geometry
    // Funnily the mpi version works (but let's kill it for now
    // Maybe in the future a free function "make_grid" ...
    //template< size_t M0, size_t M1, size_t ...Ms>
    //aRealTopology( const aRealTopology<real_type,M0>& g0, const aRealTopology<real_type,M1>& g1, const aRealTopology<real_type,Ms>& ...gs)
    //{
    //    auto grid = aRealTopology<real_type, Nd - M0>( g1, gs ...);
    //    *this = aRealTopology<real_type, Nd>( g0, grid);
    //}
    //template< size_t M0, size_t M1>
    //aRealTopology( const aRealTopology<real_type,M0>& g0, const aRealTopology<real_type,M1>& g1)
    //{
    //    static_assert( (M0 + M1) == Nd);

    //    for( unsigned u=0; u<M0; u++)
    //    {
    //        m_n[u] = g0.n(u);
    //        m_N[u] = g0.N(u);
    //        m_x0[u] = g0.p(u);
    //        m_x1[u] = g0.q(u);
    //        m_bcs[u] = g0.bc(u);
    //    }
    //    for( unsigned u=0; u<M1; u++)
    //    {
    //        m_n[M0+u] = g1.n(u);
    //        m_N[M0+u] = g1.N(u);
    //        m_x0[M0+u] = g1.p(u);
    //        m_x1[M0+u] = g1.q(u);
    //        m_bcs[M0+u] = g1.bc(u);
    //    }
    //}
  private:
    std::array<real_type,Nd> m_x0;
    std::array<real_type,Nd> m_x1;
    std::array<unsigned,Nd> m_n;
    std::array<unsigned,Nd> m_N;
    std::array<dg::bc,Nd> m_bcs;
};
///@cond
// pure virtual implementations must be declared outside class
template<class real_type,size_t Nd>
void aRealTopology<real_type,Nd>::do_set( std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N)
{
    m_n = new_n;
    m_N = new_N;
}
template<class real_type,size_t Nd>
void aRealTopology<real_type,Nd>::do_set_pq( std::array<real_type, Nd> x0, std::array<real_type,Nd> x1)
{
    m_x0 = x0;
    m_x1 = x1;
}
template<class real_type,size_t Nd>
void aRealTopology<real_type,Nd>::do_set( std::array<dg::bc, Nd> bcs)
{
    m_bcs = bcs;
}

///@endcond

/**
 * @brief The simplest implementation of aRealTopology
 *
 * @ingroup basictopology
 * @snippet{trimleft} evaluation_t.cpp evaluate2d
 */
template<class real_type, size_t Nd>
struct RealGrid : public aRealTopology<real_type, Nd>
{
    /**
     * @brief construct an empty grid
     * this leaves the access functions undefined
     */
    RealGrid() = default;
    /**
     * @brief 1D grid
     *
     * @param x0 left boundary
     * @param x1 right boundary
     * @param n # of polynomial coefficients
     *  (1<=n<=20)
     * @param Nx # of cells
     * @param bcx boundary conditions
     */
    template<size_t Md = Nd>
    RealGrid( real_type x0, real_type x1, unsigned n, unsigned Nx, dg::bc bcx = dg::PER ):
        aRealTopology<real_type,1>{{x0}, {x1}, {n}, {Nx}, {bcx}}
    {
    }
    ///@copydoc hide_grid_parameters2d
    ///@copydoc hide_bc_parameters2d
    template<size_t Md = Nd>
    RealGrid( real_type x0, real_type x1, real_type y0, real_type y1, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx = PER, dg::bc bcy = PER):
        aRealTopology<real_type,2>({x0,y0}, {x1,y1}, {n,n}, {Nx, Ny}, {bcx,bcy})
    {
    }
    ///@copydoc hide_grid_parameters3d
    ///@copydoc hide_bc_parameters3d
    template<size_t Md = Nd>
    RealGrid( real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx = PER, dg::bc bcy = PER, dg::bc bcz=PER):
        aRealTopology<real_type,3>({x0,y0,z0}, {x1,y1,z1}, {n,n,1}, {Nx, Ny,Nz}, {bcx,bcy, bcz})
        {}

    ///@copydoc aRealTopology::aRealTopology(const std::array<RealGrid<real_type,1>,Nd>&)
    RealGrid( const std::array<RealGrid<real_type,1>,Nd>& axes) :
        aRealTopology<real_type,Nd>( axes){}

    /**
     * @brief Construct from given 1d grids
     * Equivalent to <tt>RealGrid( std::array{g0,gs...})</tt>
     * @param g0 Axis 0 grid
     * @param gs more axes
     */
    template<class ...Grid1ds>
    RealGrid( const RealGrid<real_type,1>& g0, const Grid1ds& ...gs) :
        aRealTopology<real_type,Nd>( std::array<RealGrid<real_type,1>,Nd>{g0, gs...}){}

    ///@copydoc aRealTopology::aRealTopology(std::array<real_type,Nd>,std::array<real_type,Nd>,std::array<unsigned,Nd>,std::array<unsigned,Nd>,std::array<dg::bc,Nd>)
    RealGrid( std::array<real_type,Nd> p, std::array<real_type,Nd> q,
        std::array<unsigned,Nd> n, std::array<unsigned,Nd> N,
        std::array<dg::bc,Nd> bcs) : aRealTopology<real_type,Nd>( p,q,n,N,bcs)
    {}

    ///@brief allow explicit type conversion from any other topology
    ///@param src source
    explicit RealGrid( const aRealTopology<real_type,Nd>& src): aRealTopology<real_type,Nd>(src){}
    private:
    virtual void do_set( std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N) override final{
        aRealTopology<real_type,Nd>::do_set(new_n,new_N);
    }
    virtual void do_set_pq( std::array<real_type,Nd> new_x0, std::array<real_type,Nd> new_x1) override final{
        aRealTopology<real_type,Nd>::do_set_pq(new_x0,new_x1);
    }
    virtual void do_set( std::array<dg::bc,Nd> new_bcs) override final{
        aRealTopology<real_type,Nd>::do_set(new_bcs);
    }

};

///@addtogroup gridtypes
///@{
using Grid0d        = dg::RealGrid<double,0>;
using Grid1d        = dg::RealGrid<double,1>;
using Grid2d        = dg::RealGrid<double,2>;
using Grid3d        = dg::RealGrid<double,3>;
template<size_t Nd>
using Grid          = dg::RealGrid<double,Nd>;
using aTopology2d   = dg::aRealTopology<double,2>;
using aTopology3d   = dg::aRealTopology<double,3>;
template<class T>
using aRealTopology2d   = dg::aRealTopology<T,2>;
template<class T>
using aRealTopology3d   = dg::aRealTopology<T,3>;
template<class T>
using RealGrid0d   = dg::RealGrid<T,0>;
template<class T>
using RealGrid1d   = dg::RealGrid<T,1>;
template<class T>
using RealGrid2d   = dg::RealGrid<T,2>;
template<class T>
using RealGrid3d   = dg::RealGrid<T,3>;
#ifndef MPI_VERSION
namespace x {
using Grid0d        = Grid0d      ;
using Grid1d        = Grid1d      ;
using Grid2d        = Grid2d      ;
using Grid3d        = Grid3d      ;
template<size_t Nd>
using Grid          = Grid<Nd>    ;
using aTopology2d   = aTopology2d ;
using aTopology3d   = aTopology3d ;
template<class T>
using aRealTopology2d   = aRealTopology<T,2>;
template<class T>
using aRealTopology3d   = aRealTopology<T,3>;
template<class T>
using RealGrid0d   = RealGrid<T,0>;
template<class T>
using RealGrid1d   = RealGrid<T,1>;
template<class T>
using RealGrid2d   = RealGrid<T,2>;
template<class T>
using RealGrid3d   = RealGrid<T,3>;
} //namespace x
#endif
///@}

}// namespace dg
