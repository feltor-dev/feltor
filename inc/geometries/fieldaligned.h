#pragma once
#include <cmath>
#include <cusp/csr_matrix.h>

#include "dg/backend/grid.h"
#include "dg/blas.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/projection.cuh"
#include "dg/backend/functions.h"
#include "dg/backend/typedefs.cuh"
#include "dg/backend/split_and_join.h"
#include "dg/backend/transpose.h"

#include "dg/geometry/geometry.h"
#include "dg/functors.h"
#include "dg/nullstelle.h"
#include "dg/runge_kutta.h"
#include "magnetic_field.h"
#include "fluxfunctions.h"
#include "curvilinear.h"

namespace dg{
namespace geo{

///@brief Enum for the use in Fieldaligned
///@ingroup fieldaligned
enum whichMatrix
{
    einsPlus = 0,   /// plus interpolation in next plane
    einsPlusT = 1,  /// transposed plus interpolation in previous plane
    einsMinus = 2,  /// minus interpolation in previous plane
    einsMinusT = 3, /// transposed minus interpolation in next plane
};

///@brief Full Limiter means there is a limiter everywhere
///@ingroup fieldaligned
typedef ONE FullLimiter;

///@brief No Limiter 
///@ingroup fieldaligned
typedef ZERO NoLimiter;
///@cond
namespace detail{

struct DZField
{
    void operator()( const dg::HVec& y, dg::HVec& yp)
    {
        yp[0] = yp[1] = 0;
        yp[2] = 1.;
    }
    double error( const dg::HVec& x0, const dg::HVec& x1)
    {
        return sqrt( (x0[0]-x1[0])*(x0[0]-x1[0]) +(x0[1]-x1[1])*(x0[1]-x1[1])+(x0[2]-x1[2])*(x0[2]-x1[2]));
    }
    bool monitor( const dg::HVec& end){ return true;}
    double operator()( double x, double y) //1/B
    {
        return 1.;
    }
    double operator()( double x, double y, double z)
    {
        return 1.;
    }

};

struct DSFieldCylindrical
{
    DSFieldCylindrical( const dg::geo::BinaryVectorLvl0& v):v_(v) { }
    void operator()( const dg::HVec& y, dg::HVec& yp) const {
        double vz = v_.z()(y[0], y[1]);
        yp[0] = v_.x()(y[0], y[1])/vz; 
        yp[1] = v_.y()(y[0], y[1])/vz;
        yp[2] = 1./vz;
    }

    double error( const dg::HVec& x0, const dg::HVec& x1)const {
        return sqrt( (x0[0]-x1[0])*(x0[0]-x1[0]) +(x0[1]-x1[1])*(x0[1]-x1[1])+(x0[2]-x1[2])*(x0[2]-x1[2]));
    }
    bool monitor( const dg::HVec& end)const{ 
        if ( std::isnan(end[0]) || std::isnan(end[1]) || std::isnan(end[2]) ) 
        {
            return false;
        }
        //if new integrated point outside domain
        if ((1e-5 > end[0]  ) || (1e10 < end[0])  ||(-1e10  > end[1]  ) || (1e10 < end[1])||(-1e10 > end[2]  ) || (1e10 < end[2])  )
        {
            return false;
        }
        return true;
    }
    
    private:
    dg::geo::BinaryVectorLvl0 v_;
};

struct DSField
{
    //z component of v may not vanish
    DSField( const dg::geo::BinaryVectorLvl0& v, const dg::aGeometry2d& g): g_(g)
    {
        thrust::host_vector<double> v_zeta, v_eta;
        dg::pushForwardPerp( v.x(), v.y(), v_zeta, v_eta, g);
        thrust::host_vector<double> v_phi = dg::pullback( v.z(), g);
        dg::blas1::pointwiseDivide(v_zeta, v_phi, v_zeta);
        dg::blas1::pointwiseDivide(v_eta, v_phi, v_eta);
        dg::blas1::transform(v_phi, v_phi, dg::INVERT<double>());
        dzetadphi_  = dg::create::forward_transform( v_zeta, g );
        detadphi_   = dg::create::forward_transform( v_eta, g );
        dsdphi_     = dg::create::forward_transform( v_phi, g );

    }
    //interpolate the vectors given in the constructor on the given point
    //if point lies outside of grid boundaries zero is returned
    void operator()(const thrust::host_vector<double>& y, thrust::host_vector<double>& yp) const
    {
        double R = y[0], Z = y[1];
        g_.get().shift_topologic( y[0], y[1], R, Z); //shift R,Z onto domain
        if( !g_.get().contains( R, Z)) yp[0] = yp[1]= yp[2] = 0;
        else
        {
            //else interpolate
            yp[0] = interpolate( R, Z, dzetadphi_, g_.get());
            yp[1] = interpolate( R, Z, detadphi_,  g_.get());
            yp[2] = interpolate( R, Z, dsdphi_,    g_.get());
        }
    }

    ///take the sum of the absolute errors perp and parallel
    double error( const dg::HVec& x0, const dg::HVec& x1) const {
        //here, we don't need to shift coordinates since x0 and x1 are both end points
        return sqrt( (x0[0]-x1[0])*(x0[0]-x1[0]) +(x0[1]-x1[1])*(x0[1]-x1[1]))+sqrt((x0[2]-x1[2])*(x0[2]-x1[2]));
    }
    bool monitor( const dg::HVec& end)const{ 
        if ( std::isnan(end[0]) || std::isnan(end[1]) || std::isnan(end[2]) ) 
        {
            return false;
        }
        //if new integrated point far outside domain
        if ((end[0] < g_.get().x0()-1e4 ) || (g_.get().x1()+1e4 < end[0])  ||(end[1] < g_.get().y0()-1e4 ) || (g_.get().y1()+1e4 < end[1])||(-1e10 > end[2]  ) || (1e10 < end[2])  )
        {
            return false;
        }
        return true;
    }
    private:
    thrust::host_vector<double> dzetadphi_, detadphi_, dsdphi_;
    dg::Handle<dg::aGeometry2d> g_;

};

void clip_to_boundary( double& x, double& y, const dg::aTopology2d* grid)
{
    if (!(x > grid->x0())) { x=grid->x0();}
    if (!(x < grid->x1())) { x=grid->x1();}
    if (!(y > grid->y0())) { y=grid->y0();}
    if (!(y < grid->y1())) { y=grid->y1();}
}
void clip_to_boundary( thrust::host_vector<double>& x, const dg::aTopology2d* grid)
{
    clip_to_boundary(x[0], x[1], grid);
}

void interpolate_and_clip( const dg::IHMatrix& interpolate, const dg::aTopology2d* g2dFine, const dg::aTopology2d* boundary_ptr, //2 different grid on account of the MPI implementation
        const std::vector<thrust::host_vector<double> >& yp_coarse,
        const std::vector<thrust::host_vector<double> >& ym_coarse,
        std::vector<thrust::host_vector<double> >& yp_fine,
        std::vector<thrust::host_vector<double> >& ym_fine
        )
{
    std::vector<thrust::host_vector<double> > yp( 3, dg::evaluate(dg::zero, *g2dFine)), ym(yp); 
    for( unsigned i=0; i<3; i++)
    {
        dg::blas2::symv( interpolate, yp_coarse[i], yp[i]);
        dg::blas2::symv( interpolate, ym_coarse[i], ym[i]);
    }
    for( unsigned i=0; i<yp[0].size(); i++)
    {
        boundary_ptr->shift_topologic( yp[0][i], yp[1][i], yp[0][i], yp[1][i]);
        boundary_ptr->shift_topologic( ym[0][i], ym[1][i], ym[0][i], ym[1][i]);
        detail::clip_to_boundary( yp[0][i], yp[1][i], boundary_ptr);
        detail::clip_to_boundary( ym[0][i], ym[1][i], boundary_ptr);
    }
    yp_fine=yp, ym_fine=ym;
}

/**
 * @brief Integrate a field line to find whether the result lies inside or outside of the box
 * @tparam Field Must be usable in the integrateRK() functions
 * @tparam Topology must provide 2d contains function
 */
template < class Field, class Topology>
struct BoxIntegrator
{
    /**
     * @brief Construct from a given Field and Grid and accuracy
     *
     * @param field field must overload operator() with dg::HVec for three entries
     * @param g The 2d or 3d grid
     * @param eps the accuracy of the runge kutta integrator
     */
    BoxIntegrator( const Field& field, const Topology& g, double eps): field_(field), g_(g), coords_(3), coordsp_(3), eps_(eps) {}
    /**
     * @brief Set the starting coordinates for next field line integration
     * @param coords the new coords (must have size = 3)
     */
    void set_coords( const thrust::host_vector<double>& coords){ coords_ = coords;}

    /**
     * @brief Integrate from 0 to deltaPhi
     * @param deltaPhi upper integration boundary
     * @return 1 if result is inside the box, -1 else
     */
    double operator()( double deltaPhi)
    {
        dg::integrateRK4( field_, coords_, coordsp_, deltaPhi, eps_);
        if( !g_.contains( coordsp_[0], coordsp_[1]) ) return -1;
        return +1;
    }
    private:
    const Field& field_;
    const Topology& g_;
    thrust::host_vector<double> coords_, coordsp_;
    double eps_;
};

/**
 * @brief Integrate one field line in a given box, Result is guaranteed to lie inside the box modulo periodic boundary conditions
 *
 * @tparam Field Must be usable in the integrateRK function
 * @tparam Topology must provide 2d contains and shift_topologic function
 * @param field The field to use
 * @param grid instance of the Grid class 
 * @param coords0 The initial condition
 * @param coords1 The resulting points (write only) guaranteed to lie inside the grid
 * @param phi1 The angle (read/write) contains maximum phi on input and resulting phi on output
 * @param eps error
 */
template< class Field, class Topology>
void boxintegrator( const Field& field, const Topology& grid, 
        const thrust::host_vector<double>& coords0, 
        thrust::host_vector<double>& coords1, 
        double& phi1, double eps)
{
    dg::integrateRK4( field, coords0, coords1, phi1, eps); //integration
    double R = coords1[0], Z=coords1[1];
    //First, catch periodic domain
    grid.shift_topologic( coords0[0], coords0[1], R, Z);
    if ( !grid.contains( R, Z))   //point still outside domain
    {
#ifdef DG_DEBUG
        std::cerr << "point "<<coords1[0]<<" "<<coords1[1]<<" is somewhere else!\n";
#endif //DG_DEBUG
        double deltaPhi = phi1;
        BoxIntegrator<Field, Topology> boxy( field, grid, eps);//stores references to field and grid
        boxy.set_coords( coords0); //nimm alte koordinaten
        if( phi1 > 0)
        {
            double dPhiMin = 0, dPhiMax = phi1;
            dg::bisection1d( boxy, dPhiMin, dPhiMax,eps); //suche 0 stelle 
            phi1 = (dPhiMin+dPhiMax)/2.;
            dg::integrateRK4( field, coords0, coords1, dPhiMax, eps); //integriere bis über 0 stelle raus damit unten Wert neu gesetzt wird
        }
        else // phi1 < 0 
        {
            double dPhiMin = phi1, dPhiMax = 0;
            dg::bisection1d( boxy, dPhiMin, dPhiMax,eps);
            phi1 = (dPhiMin+dPhiMax)/2.;
            dg::integrateRK4( field, coords0, coords1, dPhiMin, eps);
        }
        detail::clip_to_boundary( coords1, &grid);
        //now assume the rest is purely toroidal
        double deltaS = coords1[2];
        thrust::host_vector<double> temp=coords0;
        field(coords1, temp); //we are just interested in temp[2]
        coords1[2] = deltaS + (deltaPhi-phi1)*temp[2]; // ds + dphi*f[2]
    }
}

//used in constructor of Fieldaligned
void integrate_all_fieldlines2d( const dg::geo::BinaryVectorLvl0& vec, const dg::aGeometry2d* g2dField_ptr, const dg::aTopology2d* evaluate_ptr, std::vector<thrust::host_vector<double> >& yp_result, std::vector<thrust::host_vector<double> >& ym_result , double deltaPhi, double eps)
{
    //g2dField contains the global geometry 
    //evaluate_ptr contains the points to actually integrate
    std::vector<thrust::host_vector<double> > y( 3, dg::evaluate( dg::cooX2d, *evaluate_ptr)); //x
    y[1] = dg::evaluate( dg::cooY2d, *evaluate_ptr); //y
    y[2] = dg::evaluate( dg::zero, *evaluate_ptr); //s
    std::vector<thrust::host_vector<double> > yp( 3, y[0]), ym(yp); 
    //construct field on high polynomial grid, then integrate it
    dg::Timer t;
    t.tic();
    dg::geo::detail::DSField field( vec, *g2dField_ptr);
    t.toc();
    std::cout << "Generation of interpolate grid took "<<t.diff()<<"s\n";
    //field in case of cartesian grid
    dg::geo::detail::DSFieldCylindrical cyl_field(vec);
    for( unsigned i=0; i<g2dField_ptr->size(); i++)
    {
        thrust::host_vector<double> coords(3), coordsP(3), coordsM(3);
        coords[0] = y[0][i], coords[1] = y[1][i], coords[2] = y[2][i]; //x,y,s
        double phi1 = deltaPhi;
        if( dynamic_cast<const dg::CartesianGrid2d*>( g2dField_ptr))
            boxintegrator( cyl_field, *g2dField_ptr, coords, coordsP, phi1, eps);
        else 
            boxintegrator( field, *g2dField_ptr, coords, coordsP, phi1, eps);
        phi1 =  - deltaPhi;
        if( dynamic_cast<const dg::CartesianGrid2d*>( g2dField_ptr))
            boxintegrator( cyl_field, *g2dField_ptr, coords, coordsM, phi1, eps);
        else 
            boxintegrator( field, *g2dField_ptr, coords, coordsM, phi1, eps);
        yp[0][i] = coordsP[0], yp[1][i] = coordsP[1], yp[2][i] = coordsP[2];
        ym[0][i] = coordsM[0], ym[1][i] = coordsM[1], ym[2][i] = coordsM[2];
    }
    yp_result=yp;
    ym_result=ym;
}

}//namespace detail
///@endcond


//////////////////////////////FieldalignedCLASS////////////////////////////////////////////
/**
* @brief Create and manage interpolation matrices from fieldline integration
*
* @ingroup fieldaligned
* @tparam ProductGeometry must be either aProductGeometry3d or aProductMPIGeometry3d or any derivative 
* @tparam IMatrix The type of the interpolation matrix 
    - dg::IHMatrix, or dg::IDMatrix, dg::MIHMatrix, or dg::MIDMatrix
* @tparam container The container-class on which the interpolation matrix operates on
    - dg::HVec, or dg::DVec, dg::MHVec, or dg::MDVec
* @sa The pdf <a href="./parallel.pdf" target="_blank">parallel derivative</a> writeup 
*/
template<class ProductGeometry, class IMatrix, class container >
struct Fieldaligned
{

    typedef IMatrix InterpolationMatrix; //!< typdef to reveal the interpolation matrix
    ///@brief do not allocate memory; no member call except construct is valid
    Fieldaligned(){}

    ///@copydoc construct()
    template <class Limiter>
    Fieldaligned(const dg::geo::BinaryVectorLvl0& vec, const ProductGeometry& grid, unsigned multiplyX, unsigned multiplyY, bool dependsOnX, bool dependsOnY, double eps = 1e-5, dg::bc globalbcx = dg::NEU, dg::bc globalbcy = dg::NEU, Limiter limit = FullLimiter(), double deltaPhi = -1)
    {
        construct( vec, grid, multiplyX, multiplyY, dependsOnX, dependsOnY, eps, limit, globalbcx, globalbcy, limit, deltaPhi);
    }
    /**
    * @brief Construct from a field and a grid
    *
    * @tparam Limiter Class that can be evaluated on a 2d grid, returns 1 if there
        is a limiter and 0 if there isn't. 
        If a field line crosses the limiter in the plane \f$ \phi=0\f$ then the limiter boundary conditions apply. 
    * @param vec The field to integrate
    * @param grid The grid on which to operate defines the parallel boundary condition in case there is a limiter.
    * @param multiplyX defines the resolution in X of the fine grid relative to grid
    * @param multiplyY defines the resolution in Y of the fine grid relative to grid
    * @param dependsOnX indicates, whether the given vector field vec depends on the first coordinate
    * @param dependsOnY indicates, whether the given vector field vec depends on the second coordinate
    * @param eps Desired accuracy of the fieldline integrator
    * @param globalbcx Defines the interpolation behaviour when a fieldline intersects the boundary box in the perpendicular direction
    * @param globalbcy Defines the interpolation behaviour when a fieldline intersects the boundary box in the perpendicular direction
    * @param limit Instance of the limiter class (Default is a limiter everywhere, 
        note that if grid.bcz() is periodic it doesn't matter if there is a limiter or not)
    * @param deltaPhi Is either <0 (then it's ignored), or may differ from grid.hz() if grid.Nz() == 1
    * @note If there is a limiter, the boundary condition on the first/last plane is set 
        by the grid.bcz() variable and can be changed by the set_boundaries function. 
        If there is no limiter, the boundary condition is periodic.
    */
    template <class Limiter>
    void construct(const dg::geo::BinaryVectorLvl0& vec, const ProductGeometry& grid, unsigned multiplyX, unsigned multiplyY, bool dependsOnX, bool dependsOnY, double eps = 1e-5, dg::bc globalbcx = dg::NEU, dg::bc globalbcy = dg::NEU, Limiter limit = FullLimiter(), double deltaPhi = -1);

    bool dependsOnX()const{return m_dependsOnX;}
    bool dependsOnY()const{return m_dependsOnY;}

    /**
    * @brief Set boundary conditions in the limiter region
    *
    * if Dirichlet boundaries are used the left value is the left function
    value, if Neumann boundaries are used the left value is the left derivative value
    * @param bcz boundary condition
    * @param left constant left boundary value
    * @param right constant right boundary value
    */
    void set_boundaries( dg::bc bcz, double left, double right)
    {
        m_bcz = bcz;
        const dg::Grid1d g2d( 0, 1, 1, m_perp_size);
        m_left  = dg::evaluate( dg::CONSTANT(left), g2d);
        m_right = dg::evaluate( dg::CONSTANT(right),g2d);
    }

    /**
    * @brief Set boundary conditions in the limiter region
    *
    * if Dirichlet boundaries are used the left value is the left function
    value, if Neumann boundaries are used the left value is the left derivative value
    * @param bcz boundary condition
    * @param left spatially variable left boundary value
    * @param right spatially variable right boundary value
    */
    void set_boundaries( dg::bc bcz, const container& left, const container& right)
    {
        m_bcz = bcz;
        m_left = left;
        m_right = right;
    }

    /**
     * @brief Set boundary conditions in the limiter region
     *
     * if Dirichlet boundaries are used the left value is the left function
     value, if Neumann boundaries are used the left value is the left derivative value
     * @param bcz boundary condition
     * @param global 3D vector containing boundary values
     * @param scal_left left scaling factor
     * @param scal_right right scaling factor
     */
    void set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right)
    {
        dg::split( global, m_temp);
        dg::blas1::axpby( scal_left,  m_temp[0],      0, m_left);
        dg::blas1::axpby( scal_right, m_temp[m_Nz-1], 0, m_right);
        m_bcz = bcz;
    }

    /**
     * @brief Evaluate a 2d functor and transform to all planes along the fieldlines
     *
     * Evaluates the given functor on a 2d plane and then follows fieldlines to
     * get the values in the 3rd dimension. Uses the grid given in the constructor.
     * @tparam BinaryOp Binary Functor
     * @param binary Functor to evaluate
     * @param p0 The number of the plane to start
     *
     * @return Returns an instance of container
     */
    template< class BinaryOp>
    container evaluate( const BinaryOp& binary, unsigned p0=0) const
    {
        return evaluate( binary, dg::CONSTANT(1), p0, 0);
    }

    /**
     * @brief Evaluate a 2d functor and transform to all planes along the fieldlines
     *
     * Evaluates the given functor on a 2d plane and then follows fieldlines to
     * get the values in the 3rd dimension. Uses the grid given in the constructor.
     * The second functor is used to scale the values along the fieldlines.
     * The fieldlines are assumed to be periodic.
     * @tparam BinaryOp Binary Functor
     * @tparam UnaryOp Unary Functor
     * @param binary Functor to evaluate in x-y
     * @param unary Functor to evaluate in z
     * @param p0 The number of the plane to start
     * @param rounds The number of rounds to follow a fieldline
     * @note g is evaluated such that p0 corresponds to z=0, p0+1 corresponds to z=hz, p0-1 to z=-hz, ...
     *
     * @return Returns an instance of container
     */
    template< class BinaryOp, class UnaryOp>
    container evaluate( const BinaryOp& binary, const UnaryOp& unary, unsigned p0, unsigned rounds) const;

    /**
    * @brief Applies the interpolation 
    * @param which specify what interpolation should be applied
    * @param in input 
    * @param out output may not equal input
    */
    void operator()(enum whichMatrix which, const container& in, container& out);

    ///@brief hz is the distance between the plus and minus planes
    ///@return three-dimensional vector
    const container& hz_inv()const {return m_hz_inv;}
    ///@brief hp is the distance between the plus and current planes
    ///@return three-dimensional vector
    const container& hp_inv()const {return m_hp_inv;}
    ///@brief hm is the distance between the current and minus planes
    ///@return three-dimensional vector
    const container& hm_inv()const {return m_hm_inv;}
    const ProductGeometry& grid()const{return m_g.get();}
    private:
    void ePlus( enum whichMatrix which, const container& in, container& out);
    void eMinus(enum whichMatrix which, const container& in, container& out);
    IMatrix m_plus, m_minus, m_plusT, m_minusT; //2d interpolation matrices
    container m_hz_inv, m_hp_inv, m_hm_inv; //3d size
    container m_hp, m_hm; //2d size
    container m_left, m_right;      //perp_size
    container m_limiter;            //perp_size
    container m_ghostM, m_ghostP;   //perp_size
    unsigned m_Nz, m_perp_size; 
    dg::bc m_bcz;
    std::vector<container> m_f, m_temp; //split 3d vectors
    dg::Handle<ProductGeometry> m_g;
    bool m_dependsOnX, m_dependsOnY;
};

///@cond 
////////////////////////////////////DEFINITIONS////////////////////////////////////////
//


template<class Geometry, class IMatrix, class container>
template <class Limiter>
void Fieldaligned<Geometry, IMatrix, container>::construct(const dg::geo::BinaryVectorLvl0& vec, const Geometry& grid, unsigned mx, unsigned my, bool bx, bool by, double eps, dg::bc globalbcx, dg::bc globalbcy, Limiter limit, double deltaPhi)
{
    m_dependsOnX=bx, m_dependsOnY=by;
    m_Nz=grid.Nz(), m_bcz=grid.bcz(); 
    m_g=grid;
    dg::blas1::transfer( dg::evaluate( dg::zero, grid), m_hz_inv), m_hp_inv= m_hz_inv, m_hm_inv= m_hz_inv;
    dg::split( m_hz_inv, m_temp);
    dg::split( m_hz_inv, m_f);
    if( deltaPhi <=0) deltaPhi = grid.hz();
    else assert( grid.Nz() == 1 || grid.hz()==deltaPhi);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    const dg::aGeometry2d* grid2d_ptr = grid->perp_grid();
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //Resize vector to 2D grid size
    m_perp_size = grid2d_ptr->size();
    m_limiter = dg::pullback( limit, *grid2d_ptr);
    m_right = m_left = dg::evaluate( zero, *grid2d_ptr);
    m_ghostM.resize( m_perp_size); m_ghostP.resize( m_perp_size);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%Set starting points and integrate field lines%%%%%%%%%%%%%%
    std::cout << "Start fieldline integration!\n";
    dg::Timer t;
    std::vector<thrust::host_vector<double> > yp_coarse( 3), ym_coarse(yp_coarse), yp, ym; 
    t.tic();
    
    dg::aGeometry2d* g2dField_ptr = grid2d_ptr->clone();//INTEGRATE HIGH ORDER GRID
    g2dField_ptr->set( 7, g2dField_ptr->Nx(), g2dField_ptr->Ny());
    detail::integrate_all_fieldlines2d( vec, g2dField_ptr, g2dField_ptr, yp_coarse, ym_coarse, deltaPhi, eps);

    dg::Grid2d g2dFine((dg::Grid2d(*grid2d_ptr)));//FINE GRID
    g2dFine.multiplyCellNumbers((double)mx, (double)my);
    dg::IHMatrix interpolate = dg::create::interpolation( g2dFine, *g2dField_ptr);  //INTERPOLATE TO FINE GRID
    dg::geo::detail::interpolate_and_clip( interpolate, &g2dFine, &g2dFine, yp_coarse, ym_coarse, yp, ym);
    delete g2dField_ptr;
    t.toc(); 
    std::cout << "Fieldline integration took "<<t.diff()<<"s\n";
    //%%%%%%%%%%%%%%%%%%Create interpolation and projection%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t.tic();
    dg::IHMatrix plusFine  = dg::create::interpolation( yp[0], yp[1], *grid2d_ptr, globalbcx, globalbcy), plus, plusT;
    dg::IHMatrix minusFine = dg::create::interpolation( ym[0], ym[1], *grid2d_ptr, globalbcx, globalbcy), minus, minusT;
    dg::IHMatrix projection = dg::create::projection( *grid2d_ptr, g2dFine);
    t.toc();
    std::cout <<"Creation of interpolation/projection took "<<t.diff()<<"s\n";
    t.tic();
    cusp::multiply( projection, plusFine, plus);
    cusp::multiply( projection, minusFine, minus);
    t.toc();
    std::cout<< "Multiplication of P*I took: "<<t.diff()<<"s\n";
    //%Transposed matrices work only for csr_matrix due to bad matrix form for ell_matrix!!!
    plusT = dg::transpose( plus);
    minusT = dg::transpose( minus);     
    dg::blas2::transfer( plus, m_plus);
    dg::blas2::transfer( plusT, m_plusT);
    dg::blas2::transfer( minus, m_minus);
    dg::blas2::transfer( minusT, m_minusT);
    //%%%%%%%%%%%%%%%%%%%%%%%project h and copy into h vectors%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    thrust::host_vector<double> hp( m_perp_size), hm(hp), hz(hp);
    dg::blas2::symv( projection, yp[2], hp);
    dg::blas2::symv( projection, ym[2], hm);
    dg::blas1::scal( hm, -1.);
    dg::blas1::axpby(  1., hp, +1., hm, hz);
    dg::blas1::transfer( hp, m_hp);
    dg::blas1::transfer( hm, m_hm);
    dg::blas1::transform( hp, hp, dg::INVERT<double>());
    dg::blas1::transform( hm, hm, dg::INVERT<double>());
    dg::blas1::transform( hz, hz, dg::INVERT<double>());
    dg::join( std::vector<thrust::host_vector<double> >( m_Nz, hp), m_hp_inv);
    dg::join( std::vector<thrust::host_vector<double> >( m_Nz, hm), m_hm_inv);
    dg::join( std::vector<thrust::host_vector<double> >( m_Nz, hz), m_hz_inv);
    
    delete grid2d_ptr;
}

template<class G, class I, class container>
template< class BinaryOp, class UnaryOp>
container Fieldaligned<G, I,container>::evaluate( const BinaryOp& binary, const UnaryOp& unary, unsigned p0, unsigned rounds) const
{
    //idea: simply apply I+/I- enough times on the init2d vector to get the result in each plane
    //unary function is always such that the p0 plane is at x=0
    assert( p0 < m_g.get().Nz());
    const dg::aGeometry2d* g2d_ptr = m_g.get().perp_grid();
    container init2d = dg::pullback( binary, *g2d_ptr); 
    delete g2d_ptr;

    container temp(init2d), tempP(init2d), tempM(init2d);
    container vec3d = dg::evaluate( dg::zero, m_g.get());
    std::vector<container>  plus2d, minus2d, result; 
    dg::split( vec3d, plus2d, m_g.get());
    dg::split( vec3d, minus2d, m_g.get());
    dg::split( vec3d, result, m_g.get());
    unsigned turns = rounds; 
    if( turns ==0) turns++;
    //first apply Interpolation many times, scale and store results
    for( unsigned r=0; r<turns; r++)
        for( unsigned i0=0; i0<m_Nz; i0++)
        {
            dg::blas1::copy( init2d, tempP);
            dg::blas1::copy( init2d, tempM);
            unsigned rep = i0 + r*m_Nz;
            for(unsigned k=0; k<rep; k++)
            {
                dg::blas2::symv( m_plus, tempP, temp);
                temp.swap( tempP);
                dg::blas2::symv( m_minus, tempM, temp);
                temp.swap( tempM);
            }
            dg::blas1::scal( tempP, unary(  (double)rep*m_g.get().hz() ) );
            dg::blas1::scal( tempM, unary( -(double)rep*m_g.get().hz() ) );
            dg::blas1::axpby( 1., tempP, 1., plus2d[i0]);
            dg::blas1::axpby( 1., tempM, 1., minus2d[i0]);
        }
    //now we have the plus and the minus filaments
    if( rounds == 0) //there is a limiter
    {
        for( unsigned i0=0; i0<m_Nz; i0++)
        {
            int idx = (int)i0 - (int)p0;
            if(idx>=0)
                result[i0] = plus2d[idx];
            else
                result[i0] = minus2d[abs(idx)];
            thrust::copy( result[i0].begin(), result[i0].end(), vec3d.begin() + i0*m_perp_size);
        }
    }
    else //sum up plus2d and minus2d
    {
        for( unsigned i0=0; i0<m_Nz; i0++)
        {
            unsigned revi0 = (m_Nz - i0)%m_Nz; //reverted index
            dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
            dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
        }
        dg::blas1::axpby( -1., init2d, 1., result[0]);
        for(unsigned i0=0; i0<m_Nz; i0++)
        {
            int idx = ((int)i0 -(int)p0 + m_Nz)%m_Nz; //shift index
            thrust::copy( result[idx].begin(), result[idx].end(), vec3d.begin() + i0*m_perp_size);
        }
    }
    return vec3d;
}


template<class G, class I, class container>
void Fieldaligned<G, I, container >::operator()(enum whichMatrix which, const container& f, container& fe)
{
    if(which == einsPlus  || which == einsMinusT) ePlus(  which, f, fe);
    if(which == einsMinus || which == einsPlusT ) eMinus( which, f, fe);
}

template< class G, class I, class container>
void Fieldaligned<G, I, container>::ePlus( enum whichMatrix which, const container& f, container& fpe)
{
    dg::split( f, m_f, m_g.get());
    //1. compute 2d interpolation in every plane and store in m_temp
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        unsigned ip = (i0==m_Nz-1) ? 0:i0+1;
        if(which == einsPlus)           dg::blas2::symv( m_plus,   m_f[ip], m_temp[i0]);
        else if(which == einsMinusT)    dg::blas2::symv( m_minusT, m_f[ip], m_temp[i0]);
    }
    //2. apply right boundary conditions in last plane
    unsigned i0=m_Nz-1;
    if( m_bcz != dg::PER)
    {
        if( m_bcz == dg::DIR || m_bcz == dg::NEU_DIR)
            dg::blas1::axpby( 2, m_right, -1., m_f[i0], m_ghostP);
        if( m_bcz == dg::NEU || m_bcz == dg::DIR_NEU)
        {
            dg::blas1::pointwiseDot( m_right, m_hp, m_ghostP);
            dg::blas1::axpby( 1., m_ghostP, 1., m_f[i0], m_ghostP);
        }
        //interlay ghostcells with periodic cells: L*g + (1-L)*fpe
        dg::blas1::axpby( 1., m_ghostP, -1., m_temp[i0], m_ghostP);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostP, 1., m_temp[i0]);
    }
    dg::join( m_temp, fpe, m_g.get());
}

template< class G, class I, class container>
void Fieldaligned<G, I, container>::eMinus( enum whichMatrix which, const container& f, container& fme)
{
    dg::split( f, m_f, m_g.get());
    //1. compute 2d interpolation in every plane and store in m_temp
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        unsigned im = (i0==0) ? m_Nz-1:i0-1;
        if(which == einsPlusT)          dg::blas2::symv( m_plusT, m_f[im], m_temp[i0]);
        else if (which == einsMinus)    dg::blas2::symv( m_minus, m_f[im], m_temp[i0]);
    }
    //2. apply left boundary conditions in first plane
    unsigned i0=0;
    if( m_bcz != dg::PER)
    {
        if( m_bcz == dg::DIR || m_bcz == dg::DIR_NEU)
            dg::blas1::axpby( 2., m_left,  -1., m_f[i0], m_ghostM);
        if( m_bcz == dg::NEU || m_bcz == dg::NEU_DIR)
        {
            dg::blas1::pointwiseDot( m_left, m_hm, m_ghostM);
            dg::blas1::axpby( -1., m_ghostM, 1., m_f[i0], m_ghostM);
        }
        //interlay ghostcells with periodic cells: L*g + (1-L)*fme
        dg::blas1::axpby( 1., m_ghostM, -1., m_temp[i0], m_ghostM);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostM, 1., m_temp[i0]);
    }
    dg::join( m_temp, fme, m_g.get());
}


///@endcond 


}//namespace geo
}//namespace dg
