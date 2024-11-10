#pragma once

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
template<class real_type, size_t Nd>
struct RealGrid;
///@endcond

/**
 * @brief A utility class representing the most basic product space grid
 *
 * Each dimension just is a list of abscissas.
 * Can be used in \c dg::evaluate
 * @tparam ContainerType Vector type that holds the abscissas
 * @tparam Nd The number of dimensions
 */
template<class ContainerType, size_t Nd>
struct Box
{
    /// == Nd
    constexpr static unsigned ndim() {return Nd;}
    Box( std::array<ContainerType, Nd> abs) : m_abs(abs)
    {
        for( unsigned u=0; u<Nd; u++)
            m_shape[u] = abs[u].size();
    }
    const std::array<ContainerType, Nd>& abscissas() const {return m_abs;}
    const std::array<unsigned, Nd>& shape() const {return m_shape;}

    private:
    std::array<ContainerType, Nd> m_abs;
    std::array<unsigned, Nd> m_shape;
};

/**
 * @brief An abstract base class for N-dimensional grids
 * @note although it is abstract, objects are not meant to be hold on the heap via a base class pointer ( we protected the destructor)
 * @ingroup basictopology
 * @tparam real_type Determines value type of abscissas and weights on this grid
 * @tparam Nd The number of dimensions
 */
template<class real_type, size_t Nd>
struct aRealTopology
{
    using value_type = real_type;
    /// The host vector type used by host functions like evaluate
    using host_vector = thrust::host_vector<real_type>;
    using host_grid = RealGrid<real_type, Nd>;
    constexpr static unsigned ndim() { return Nd;}

    std::array<unsigned, Nd> shape() const
    {
        std::array<unsigned, Nd> shape;
        for( unsigned u=0; u<Nd; u++)
            shape[u] = m_n[u]*m_N[u];
        return shape;
    }

    std::array<host_vector,Nd> abscissas() const
    {
        std::array<host_vector,Nd> aas;
        for( unsigned u=0; u<Nd; u++)
        {
            host_vector abs(m_n[u]*m_N[u]);
            double hu = h(u);
            auto aa = dg::DLT<real_type>::abscissas(m_n[u]);
            for( unsigned i=0; i<m_N[u]; i++)
                for( unsigned j=0; j<m_n[u]; j++)
                {
                    real_type xmiddle = DG_FMA( hu, (real_type)(i), m_x0[u]);
                    real_type h2 = hu/2.;
                    real_type absj = 1.+aa[j];
                    abs[i*m_n[u]+j] = DG_FMA( h2, absj, xmiddle);
                }
            aas[u] = abs;
        }
        return aas;
    }
    std::array<host_vector,Nd> weights() const
    {
        std::array<host_vector,Nd> aa;
        for( unsigned u=0; u<Nd; u++)
        {
            host_vector v( m_n[u]*m_N[u]);
            auto ww = dg::DLT<real_type>::weights(m_n[u]);
            double hu = h(u);
            for( unsigned i=0; i<m_N[u]; i++)
                for( unsigned j=0; j<m_n[u]; j++)
                    v[i*m_n[u] + j] = hu/2.*ww[j];
             aa[u] = v;
        }
        return aa;
    }
    std::array<real_type,Nd> pp() const{
        m_x0;
    }
    std::array<real_type,Nd> qq() const{
        m_x1;
    }
    std::array<real_type,Nd> ll() const{
        std::array<real_type, Nd> p;
        for( unsigned u=0; u<Nd; u++)
            p[u] = m_x1[u] - m_x0[u];
        return p;
    }
    std::array<real_type,Nd> hh() const{
        std::array<real_type, Nd> hh;
        for( unsigned u=0; u<Nd; u++)
            hh[u] = (m_x1[u] - m_x0[u])/(real_type)m_N[u];
        return hh;
    }
    std::array<unsigned, Nd> NN() const
    {
        return m_N;
    }
    std::array<unsigned, Nd> nn() const
    {
        return m_n;
    }
    std::array<dg::bc, Nd> bb() const
    {
        return m_bcs;
    }

    real_type p( unsigned u=0) const { return m_x0[u];}
    real_type q( unsigned u=0) const { return m_x1[u];}
    real_type h( unsigned u=0) const { return hh()[u];}
    real_type l( unsigned u=0) const { return ll()[u];}
    unsigned n( unsigned u=0) const { return m_n[u];}
    unsigned N( unsigned u=0) const { return m_N[u];}
    dg::bc bc( unsigned u=0) const { return m_bcs[u];}
    /// Equivalent to <tt> std::get<0>( p0()) </tt>
    template<size_t Md = Nd>
    real_type x0() const {return std::get<0>(m_x0);}
    /// Equivalent to <tt> std::get<0>( p1()) </tt>
    template<size_t Md = Nd>
    real_type x1() const {return std::get<0>(m_x1);}
    /// Equivalent to <tt> std::get<1>( p0()) </tt>
    template<size_t Md = Nd>
    real_type y0() const {return std::get<1>(m_x0);}
    /// Equivalent to <tt> std::get<1>( p1()) </tt>
    template<size_t Md = Nd>
    real_type y1() const {return std::get<1>(m_x1);}
    /// Equivalent to <tt> std::get<2>( p0()) </tt>
    template<size_t Md = Nd>
    real_type z0() const {return std::get<2>(m_x0);}
    /// Equivalent to <tt> std::get<2>( p1()) </tt>
    template<size_t Md = Nd>
    real_type z1() const {return std::get<2>(m_x1);}

    /// Equivalent to <tt> std::get<0>( ll()) </tt>
    template<size_t Md = Nd>
    real_type lx() const {return std::get<0>(ll());}
    /// Equivalent to <tt> std::get<1>( ll()) </tt>
    template<size_t Md = Nd>
    real_type ly() const {return std::get<1>(ll());}
    /// Equivalent to <tt> std::get<2>( ll()) </tt>
    template<size_t Md = Nd>
    real_type lz() const {return std::get<2>(ll());}

    /// Equivalent to <tt> std::get<0>( hh()) </tt>
    template<size_t Md = Nd>
    real_type hx() const {return std::get<0>(hh());}
    /// Equivalent to <tt> std::get<1>( hh()) </tt>
    template<size_t Md = Nd>
    real_type hy() const {return std::get<1>(hh());}
    /// Equivalent to <tt> std::get<2>( hh()) </tt>
    template<size_t Md = Nd>
    real_type hz() const {return std::get<2>(hh());}

    /// Equivalent to <tt> std::get<0>( nn()) </tt>
    template<size_t Md = Nd>
    unsigned nx() const {return std::get<0>(m_n);}
    /// Equivalent to <tt> std::get<1>( nn()) </tt>
    template<size_t Md = Nd>
    unsigned ny() const {return std::get<1>(m_n);}
    /// Equivalent to <tt> std::get<2>( nn()) </tt>
    template<size_t Md = Nd>
    unsigned nz() const {return std::get<2>(m_n);}

    /// Equivalent to <tt> std::get<0>( NN()) </tt>
    template<size_t Md = Nd>
    unsigned Nx() const {return std::get<0>(m_N);}
    /// Equivalent to <tt> std::get<1>( NN()) </tt>
    template<size_t Md = Nd>
    unsigned Ny() const {return std::get<1>(m_N);}
    /// Equivalent to <tt> std::get<2>( NN()) </tt>
    template<size_t Md = Nd>
    unsigned Nz() const {return std::get<2>(m_N);}

    /// Equivalent to <tt> std::get<0>( bb()) </tt>
    template<size_t Md = Nd>
    dg::bc bcx() const {return std::get<0>(m_bcs);}
    /// Equivalent to <tt> std::get<1>( bb()) </tt>
    template<size_t Md = Nd>
    dg::bc bcy() const {return std::get<1>(m_bcs);}
    /// Equivalent to <tt> std::get<2>( bb()) </tt>
    template<size_t Md = Nd>
    dg::bc bcz() const {return std::get<2>(m_bcs);}

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
                <<"    h  = "<<hh()[u]<<"\n"
                <<"    l  = "<<ll()[u]<<"\n"
                <<"    bc = "<<bc2str(m_bcs[u])<<"\n";
        }
    }

    /**
     * @copydoc hide_shift_doc
     */
    void shift( bool& negative, std::array<real_type, Nd>& x)const
    {
        shift( negative, x, m_bcs);
    }
    /**
     * @copydoc hide_shift_doc
     * @param bcs overrule grid internal boundary condition with this value
     */
    void shift( bool& negative, std::array<real_type,Nd>& x, std::array<dg::bc,Nd> bcs)const
    {
        for( unsigned u=0; u<Nd; u++)
        {
            if( bcs[u] == dg::PER)
            {
                real_type N0 = floor((x[u]-m_x0[u])/(m_x1[u]-m_x0[u])); // ... -2[ -1[ 0[ 1[ 2[ ...
                x[u] = x[u] - N0*(m_x1[u]-m_x0[u]); //shift
            }
            //mirror along boundary as often as necessary
            while( (x[u]<m_x0[u]) || (x[u]>m_x1[u]) )
            {
                if( x[u] < m_x0[u]){
                    x[u] = 2.*m_x0[u] - x[u];
                    //every mirror swaps the sign if Dirichlet
                    if( bcs[u] == dg::DIR || bcs[u] == dg::DIR_NEU)
                        negative = !negative;//swap sign
                }
                if( x[u] > m_x1[u]){
                    x[u] = 2.*m_x1[u] - x[u];
                    if( bcs[u] == dg::DIR || bcs[u] == dg::NEU_DIR) //notice the different boundary NEU_DIR to the above DIR_NEU !
                        negative = !negative; //swap sign
                }
            }
        }
    }

    /**
     * @brief Check if the grid contains a point
     *
     * @note doesn't check periodicity!!
     * @param x point to check
     *
     * @return true if p0[u]<=x[u]<=p1[u] for all u, false else
     * @attention returns false if x[u] is NaN or INF
     */
    bool contains( const std::array<real_type,Nd>& x)const
    {
        for( unsigned u=0; u<Nd; u++)
        {
            if( !std::isfinite(x[u]) ) return false;
            //should we catch the case x1==x && dg::PER?
            if( x[u] < m_x0[u]) return false;
            if( x[u] > m_x1[u]) return true;
        }
        return true;
    }
    /**
    * @brief Multiply the number of cells in the first two dimensions with a given factor
    *
    * With this function you can resize the grid ignorantly of its current size
    * @param fx new global number of cells is fx*Nx()
    * @param fy new global number of cells is fy*Ny()
    * The remaining dimensions are left unchanged
    */
    template<size_t Md = Nd>
    std::enable_if_t< (Md>=2),void> multiplyCellNumbers( real_type fx, real_type fy){
        auto Ns = NN();
        Ns[0] = round(fx*(real_type)NN()[0]);
        Ns[1] = round(fy*(real_type)NN()[1]);
        if( fx != 1 || fy != 1)
            set( nn(), Ns);
    }
    /**
    * @brief Set the number of polynomials and cells
    *
    * Set \c nz to 1
    * Same as <tt> set({new_n,new_n,1}, {new_Nx,new_Ny,new_Nz}) </tt>
    * @param new_n new number of %Gaussian nodes in x and y
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
    /**
    * @brief Set the number of polynomials and cells
    *
    * @param new_n new number of %Gaussian nodes in each dimension
    * @param new_N new number of cells in each dimension
    */
    void set( std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N)
    {
        if( new_n==nn() && new_N == NN())
            return;
        do_set(new_n, new_N);
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
     * @brief reset the boundaries of the grid
     *
     * @param p new left boundary
     * @param q new right boundary ( > x0)
     */
    void set( std::array<real_type,Nd> p, std::array<real_type,Nd> q)
    {
        do_set( p, q);
    }
    /**
     * @brief reset the boundary conditions of the grid
     *
     * @param bcs new boundary condition
     */
    void set( std::array<dg::bc,Nd> bcs)
    {
        do_set( bcs);
    }

    void set( std::array<real_type,Nd> p, std::array<real_type,Nd> q, std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N, std::array<dg::bc,Nd> bcs)
    {
        set( p,q);
        set( new_n, new_N);
        set( bcs);
    }

    protected:
    ///disallow deletion through base class pointer
    ~aRealTopology() = default;
    /// default constructor
    aRealTopology() = default;
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
    aRealTopology(
        std::array<real_type,Nd> p,
        std::array<real_type,Nd> q,
        std::array<unsigned,Nd> n,
        std::array<unsigned,Nd> N,
        std::array<dg::bc, Nd> bcs) : m_x0(p), m_x1(q), m_n(n), m_N(N), m_bcs(bcs)
    {}

    template< size_t M0, size_t ...Ms>
    aRealTopology( aRealTopology<real_type,M0> g0, aRealTopology<real_type,Ms> ...gs)
    {
        auto grid = aRealTopology<real_type, Nd - M0>( gs ...);
        *this = aRealTopology<real_type, Nd>( g0, grid);
    }
    template< size_t M0, size_t M1>
    aRealTopology( aRealTopology<real_type,M0> g0, aRealTopology<real_type,M1> g1)
    {
        static_assert( (M0 + M1) == Nd);

        for( unsigned u=0; u<M0; u++)
        {
            m_n[u] = g0.n(u);
            m_N[u] = g0.N(u);
            m_x0[u] = g0.p(u);
            m_x1[u] = g0.q(u);
            m_bcs[u] = g0.bc(u);
        }
        for( unsigned u=0; u<M1; u++)
        {
            m_n[M0+u] = g1.n(u);
            m_N[M0+u] = g1.N(u);
            m_x0[M0+u] = g1.p(u);
            m_x1[M0+u] = g1.q(u);
            m_bcs[M0+u] = g1.bc(u);
        }
    }

    ///explicit copy constructor (default)
    ///@param src source
    aRealTopology(const aRealTopology& src) = default;
    ///explicit assignment operator (default)
    ///@param src source
    aRealTopology& operator=(const aRealTopology& src) = default;

    virtual void do_set(std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N)
    {
        m_n = new_n;
        m_N = new_N;
    }
    virtual void do_set( std::array<real_type, Nd> x0, std::array<real_type,Nd> x1)
    {
        m_x0 = x0;
        m_x1 = x1;
    }
    virtual void do_set( std::array<dg::bc, Nd> bcs)
    {
        m_bcs = bcs;
    }
  private:
    std::array<real_type,Nd> m_x0;
    std::array<real_type,Nd> m_x1;
    std::array<unsigned,Nd> m_n;
    std::array<unsigned,Nd> m_N;
    std::array<dg::bc,Nd> m_bcs;
};

/**
 * @brief The simplest implementation of aRealTopology
 *
 * @ingroup grid
 * @copydoc hide_code_evaluate2d
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
     * @param N # of cells
     * @param bcx boundary conditions
     */
    template<size_t Md = Nd>
    RealGrid( real_type x0, real_type x1, unsigned n, unsigned Nx, dg::bc bbbb ):
        aRealTopology<real_type,1>{{x0}, {x1}, {n}, {Nx}, {bbbb}}
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

    RealGrid( std::array<real_type,Nd> p, std::array<real_type,Nd> q,
        std::array<unsigned,Nd> n, std::array<unsigned,Nd> N,
        std::array<dg::bc,Nd> bcs) : aRealTopology<real_type,Nd>( p,q,n,N,bcs)
    {}


    template<size_t M0, size_t ...Ms>
    RealGrid( RealGrid<real_type,M0> g0, RealGrid<real_type,Ms> ...gs) :
        aRealTopology<real_type,Nd>(g0,gs...)
    {
    }

    RealGrid<real_type,1> grid(unsigned i ) const{
        if( i < Nd)
            return RealGrid<real_type,1>{ this->p0()[i], this->p1()[i], this->nn()[i], this->NN()[i], this->bb()[i]};
        else
            throw Error( Message(_ping_)<<"i>N not allowed! You typed: "<<i<<" while N is "<<Nd);
    }

    ///@brief allow explicit type conversion from any other topology
    ///@param src source
    explicit RealGrid( const aRealTopology<real_type,Nd>& src): aRealTopology<real_type,Nd>(src){}
    private:
    virtual void do_set( std::array<unsigned,Nd> new_n, std::array<unsigned,Nd> new_N) override final{
        aRealTopology<real_type,Nd>::do_set(new_n,new_N);
    }
    virtual void do_set( std::array<real_type,Nd> new_x0, std::array<real_type,Nd> new_x1) override final{
        aRealTopology<real_type,Nd>::do_set(new_x0,new_x1);
    }
    virtual void do_set( std::array<dg::bc,Nd> new_bcs) override final{
        aRealTopology<real_type,Nd>::do_set(new_bcs);
    }

};
//template<class Topology>
//using get_host_vector = typename Topology::host_vector;
//
//template<class Topology>
//using get_host_grid = typename Topology::host_grid;
//
//template<class Grid>
//using is_shared_grid = std::is_same< get_host_vector<Grid>, thrust::host_vector<typename Grid::value_type>>;

///@addtogroup gridtypes
///@{
using Grid0d        = dg::RealGrid<double,0>;
using Grid1d        = dg::RealGrid<double,1>;
using Grid2d        = dg::RealGrid<double,2>;
using Grid3d        = dg::RealGrid<double,3>;
using aTopology2d   = dg::aRealTopology<double,2>;
using aTopology3d   = dg::aRealTopology<double,3>;
template<class T>
using aRealTopology2d   = dg::aRealTopology<T,2>;
template<class T>
using aRealTopology3d   = dg::aRealTopology<T,3>;
#ifndef MPI_VERSION
namespace x {
using Grid0d        = Grid0d      ;
using Grid1d        = Grid1d      ;
using Grid2d        = Grid2d      ;
using Grid3d        = Grid3d      ;
using aTopology2d   = aTopology2d ;
using aTopology3d   = aTopology3d ;
template<class T>
using aRealTopology2d   = dg::aRealTopology<T,2>;
template<class T>
using aRealTopology3d   = dg::aRealTopology<T,3>;
} //namespace x
#endif
///@}

}// namespace dg
