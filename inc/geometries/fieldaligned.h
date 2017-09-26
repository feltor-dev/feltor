#pragma once
#include <cmath>
#include <cusp/transpose.h>
#include <cusp/csr_matrix.h>

#include "dg/backend/grid.h"
#include "dg/blas.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/projection.cuh"
#include "dg/backend/functions.h"

#include "dg/geometry/geometry.h"
#include "dg/functors.h"
#include "dg/nullstelle.h"
#include "dg/runge_kutta.h"
#include "magnetic_field.h"
#include "fluxfunctions.h"
#include "curvilinear.h"

namespace dg{

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
    DSField( const dg::geo::BinaryVectorLvl0& v, const aGeometry2d& g): g_(g)
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
    dg::Handle<aGeometry2d> g_;

};

void clip_to_boundary( double& x, double& y, const aTopology2d* grid)
{
    if (!(x > grid->x0())) { x=grid->x0();}
    if (!(x < grid->x1())) { x=grid->x1();}
    if (!(y > grid->y0())) { y=grid->y0();}
    if (!(y < grid->y1())) { y=grid->y1();}
}
void clip_to_boundary( thrust::host_vector<double>& x, const aTopology2d* grid)
{
    clip_to_boundary(x[0], x[1], grid);
}


/**
 * @brief Integrate a field line to find whether the result lies inside or outside of the box
 * @tparam Field Must be usable in the integrateRK() functions
 * @tparam Grid must provide 2d contains function
 */
template < class Field, class Grid>
struct BoxIntegrator
{
    /**
     * @brief Construct from a given Field and Grid and accuracy
     *
     * @param field field must overload operator() with dg::HVec for three entries
     * @param g The 2d or 3d grid
     * @param eps the accuracy of the runge kutta integrator
     */
    BoxIntegrator( const Field& field, const Grid& g, double eps): field_(field), g_(g), coords_(3), coordsp_(3), eps_(eps) {}
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
    const Grid& g_;
    thrust::host_vector<double> coords_, coordsp_;
    double eps_;
};

/**
 * @brief Integrate one field line in a given box, Result is guaranteed to lie inside the box modulo periodic boundary conditions
 *
 * @tparam Field Must be usable in the integrateRK function
 * @tparam Grid must provide 2d contains function
 * @param field The field to use
 * @param grid instance of the Grid class 
 * @param coords0 The initial condition
 * @param coords1 The resulting points (write only) guaranteed to lie inside the grid
 * @param phi1 The angle (read/write) contains maximum phi on input and resulting phi on output
 * @param eps error
 */
template< class Field, class Grid>
void boxintegrator( const Field& field, const Grid& grid, 
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
        BoxIntegrator<Field, Grid> boxy( field, grid, eps);//stores references to field and grid
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

//used in constructor of FieldAligned
void integrate_all_fieldlines2d( const dg::geo::BinaryVectorLvl0& vec, const aGeometry2d* g2dField_ptr, std::vector<thrust::host_vector<double> >& yp_result, std::vector<thrust::host_vector<double> >& ym_result , double deltaPhi, double eps)
{
    std::vector<thrust::host_vector<double> > y( 3, dg::evaluate( dg::cooX2d, *g2dField_ptr)); //x
    y[1] = dg::evaluate( dg::cooY2d, *g2dField_ptr); //y
    y[2] = dg::evaluate( dg::zero, *g2dField_ptr); //s
    std::vector<thrust::host_vector<double> > yp( 3, y[0]), ym(yp); 
    //construct field on high polynomial grid, then integrate it
    Timer t;
    t.tic();
    dg::detail::DSField field( vec, *g2dField_ptr);
    t.toc();
    std::cout << "Generation of interpolate grid took "<<t.diff()<<"s\n";
    //field in case of cartesian grid
    dg::detail::DSFieldCylindrical cyl_field(vec);
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
    delete g2dField_ptr;
}

aGeometry2d* clone_3d_to_perp( const aGeometry3d* grid_ptr)
{
    const dg::CartesianGrid3d* grid_cart = dynamic_cast<const dg::CartesianGrid3d*>(grid_ptr);
    const dg::CylindricalGrid3d* grid_cyl = dynamic_cast<const dg::CylindricalGrid3d*>(grid_ptr);
    const dg::geo::CurvilinearProductGrid3d*  grid_curvi = dynamic_cast<const dg::geo::CurvilinearProductGrid3d*>(grid_ptr);
    aGeometry2d* g2d_ptr;
    if( grid_cart) 
    {
        dg::CartesianGrid2d cart = grid_cart->perp_grid();
        g2d_ptr = cart.clone();
    }
    else if( grid_cyl) 
    {
        dg::CartesianGrid2d cart = grid_cyl->perp_grid();
        g2d_ptr = cart.clone();
    }
    else if( grid_curvi) 
    {
        dg::geo::CurvilinearGrid2d curv = grid_curvi->perp_grid();
        g2d_ptr = curv.clone();
    }
    else
        throw dg::Error( dg::Message(_ping_)<<"Grid class not recognized!");
    return g2d_ptr;
}
}//namespace detail
///@endcond


//////////////////////////////FieldAlignedCLASS////////////////////////////////////////////
/**
* @brief Class for the evaluation of a parallel derivative
*
* This class discretizes the operators \f$ \nabla_\parallel = 
\mathbf{b}\cdot \nabla = b_R\partial_R + b_Z\partial_Z + b_\phi\partial_\phi \f$, \f$\nabla_\parallel^\dagger\f$ and \f$\Delta_\parallel=\nabla_\parallel^\dagger\cdot\nabla_\parallel\f$ in
cylindrical coordinates
* @ingroup fieldaligned
* @tparam Geometry The Geometry class 
* @tparam IMatrix The matrix class of the interpolation matrix
* @tparam container The container-class on which the interpolation matrix operates on (does not need to be dg::HVec)
*/
template<class Geometry, class IMatrix, class container >
struct FieldAligned
{

    typedef IMatrix InterpolationMatrix;
    ///@brief do not allocate memory
    FieldAligned(){}

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
    * @param eps Desired accuracy of runge kutta
    * @param limit Instance of the limiter class (Default is a limiter everywhere, 
        note that if grid.bcz() is periodic it doesn't matter if there is a limiter or not)
    * @param globalbcx Defines the interpolation behaviour when a fieldline intersects the boundary box in the perpendicular direction
    * @param globalbcy Defines the interpolation behaviour when a fieldline intersects the boundary box in the perpendicular direction
    * @param deltaPhi Is either <0 (then it's ignored), may differ from hz() only if Nz() == 1
    * @note If there is a limiter, the boundary condition on the first/last plane is set 
        by the grid.bcz() variable and can be changed by the set_boundaries function. 
        If there is no limiter the boundary condition is periodic.
    */
    template <class Limiter>
    FieldAligned(const dg::geo::BinaryVectorLvl0& vec, const Geometry& grid, unsigned multiplyX, unsigned multiplyY, double eps = 1e-4, Limiter limit = FullLimiter(), dg::bc globalbcx = dg::DIR, dg::bc globalbcy = dg::DIR, double deltaPhi = -1);

    /**
    * @brief Set boundary conditions in the limiter region
    *
    * if Dirichlet boundaries are used the left value is the left function
    value, if Neumann boundaries are used the left value is the left derivative value
    * @param bcz boundary condition
    * @param left left boundary value
    * @param right right boundary value
    */
    void set_boundaries( dg::bc bcz, double left, double right)
    {
        bcz_ = bcz;
        const dg::Grid1d g2d( 0, 1, 1, perp_size_);
        left_ = dg::evaluate( dg::CONSTANT(left), g2d);
        right_ = dg::evaluate( dg::CONSTANT(right),g2d);
    }

    ///@copydoc set_boundaries()
    void set_boundaries( dg::bc bcz, const container& left, const container& right)
    {
        bcz_ = bcz;
        left_ = left;
        right_ = right;
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
    void set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right);

    /**
     * @brief Evaluate a 2d functor and transform to all planes along the fieldlines
     *
     * Evaluates the given functor on a 2d plane and then follows fieldlines to
     * get the values in the 3rd dimension. Uses the grid given in the constructor.
     * @tparam BinaryOp Binary Functor
     * @param f Functor to evaluate
     * @param plane The number of the plane to start
     *
     * @return Returns an instance of container
     */
    template< class BinaryOp>
    container evaluate( BinaryOp f, unsigned plane=0) const;

    /**
     * @brief Evaluate a 2d functor and transform to all planes along the fieldlines
     *
     * Evaluates the given functor on a 2d plane and then follows fieldlines to
     * get the values in the 3rd dimension. Uses the grid given in the constructor.
     * The second functor is used to scale the values along the fieldlines.
     * The fieldlines are assumed to be periodic.
     * @tparam BinaryOp Binary Functor
     * @tparam UnaryOp Unary Functor
     * @param f Functor to evaluate in x-y
     * @param g Functor to evaluate in z
     * @param p0 The number of the plane to start
     * @param rounds The number of rounds to follow a fieldline
     * @note g is evaluated such that p0 corresponds to z=0, p0+1 corresponds to z=hz, p0-1 to z=-hz, ...
     *
     * @return Returns an instance of container
     */
    template< class BinaryOp, class UnaryOp>
    container evaluate( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds) const;

    /**
    * @brief Applies the interpolation 
    * @param which specify what interpolation should be applied
    * @param in input 
    * @param out output may not equal input
    */
    void operator()(enum whichMatrix which, const container& in, container& out);

    ///@brief hz is the distance between the plus and minus planes
    ///@return three-dimensional vector
    const container& hz()const {return hz_;}
    ///@brief hp is the distance between the plus and current planes
    ///@return three-dimensional vector
    const container& hp()const {return hp_;}
    ///@brief hm is the distance between the current and minus planes
    ///@return three-dimensional vector
    const container& hm()const {return hm_;}
    private:
    void ePlus( enum whichMatrix which, const container& in, container& out);
    void eMinus(enum whichMatrix which, const container& in, container& out);
    typedef cusp::array1d_view< typename container::iterator> View;
    typedef cusp::array1d_view< typename container::const_iterator> cView;
    IMatrix plus, minus, plusT, minusT; //interpolation matrices
    container hz_, hp_,hm_, ghostM, ghostP;
    unsigned Nz_, perp_size_;
    dg::bc bcz_;
    container left_, right_;
    container limiter_;
    dg::Handle<Geometry> g_;
};

///@cond 
////////////////////////////////////DEFINITIONS////////////////////////////////////////
//


template<class Geometry, class IMatrix, class container>
template <class Limiter>
FieldAligned<Geometry, IMatrix, container>::FieldAligned(const dg::geo::BinaryVectorLvl0& vec, const Geometry& grid, unsigned mx, unsigned my, double eps, Limiter limit, dg::bc globalbcx, dg::bc globalbcy, double deltaPhi):
        hz_( dg::evaluate( dg::zero, grid)), hp_( hz_), hm_( hz_), 
        Nz_(grid.Nz()), bcz_(grid.bcz()), g_(grid)
{
    if( deltaPhi <=0) deltaPhi = grid.hz();
    else assert( grid.Nz() == 1 || grid.hz()==deltaPhi);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //%%%%%%%%%%%downcast grid since we don't have a virtual function perp_grid%%%%%%%%%%%%%
    const aGeometry2d* grid2d_ptr = detail::clone_3d_to_perp(&grid);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //Resize vector to 2D grid size
    perp_size_ = grid2d_ptr->size();
    limiter_ = dg::evaluate( limit, *grid2d_ptr);
    right_ = left_ = dg::evaluate( zero, *grid2d_ptr);
    ghostM.resize( perp_size_); ghostP.resize( perp_size_);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%Set starting points and integrate field lines%%%%%%%%%%%%%%
    std::cout << "Start fieldline integration!\n";
    dg::Timer t;
    std::vector<thrust::host_vector<double> > yp_coarse( 3), ym_coarse(yp_coarse); 
    t.tic();
    
    dg::aGeometry2d* g2dField_ptr = grid2d_ptr->clone();//INTEGRATE HIGH ORDER GRID
    g2dField_ptr->set( 7, g2dField_ptr->Nx(), g2dField_ptr->Ny());
    detail::integrate_all_fieldlines2d( vec, g2dField_ptr, yp_coarse, ym_coarse, deltaPhi, eps);
    delete g2dField_ptr;

    dg::Grid2d g2dFine((dg::Grid2d(*grid2d_ptr)));//FINE GRID
    g2dFine.multiplyCellNumbers((double)mx, (double)my);
    IMatrix interpolate = dg::create::interpolation( g2dFine, *grid2d_ptr);  //INTERPOLATE TO FINE GRID
    std::vector<thrust::host_vector<double> > yp( 3, dg::evaluate(dg::zero, g2dFine)), ym(yp); 
    for( unsigned i=0; i<3; i++)
    {
        dg::blas2::symv( interpolate, yp_coarse[i], yp[i]);
        dg::blas2::symv( interpolate, ym_coarse[i], ym[i]);
    }
    for( unsigned i=0; i<yp[0].size(); i++)
    {
        g2dFine.shift_topologic( yp[0][i], yp[1][i], yp[0][i], yp[1][i]);
        g2dFine.shift_topologic( ym[0][i], ym[1][i], ym[0][i], ym[1][i]);
        detail::clip_to_boundary( yp[0][i], yp[1][i], &g2dFine);
        detail::clip_to_boundary( ym[0][i], ym[1][i], &g2dFine);
    }
    t.toc(); 
    std::cout << "Fieldline integration took "<<t.diff()<<"s\n";
    //%%%%%%%%%%%%%%%%%%Create interpolation and projection%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t.tic();
    IMatrix plusFine  = dg::create::interpolation( yp[0], yp[1], *grid2d_ptr, globalbcx, globalbcy);
    IMatrix minusFine = dg::create::interpolation( ym[0], ym[1], *grid2d_ptr, globalbcx, globalbcy);
    IMatrix projection = dg::create::projection( *grid2d_ptr, g2dFine);
    t.toc();
    std::cout <<"Creation of interpolation/projection took "<<t.diff()<<"s\n";
    t.tic();
    cusp::multiply( projection, plusFine, plus);
    cusp::multiply( projection, minusFine, minus);
    t.toc();
    std::cout<< "Multiplication of P*I took: "<<t.diff()<<"s\n";
    //%Transposed matrices work only for csr_matrix due to bad matrix form for ell_matrix!!!
    cusp::transpose( plus, plusT);
    cusp::transpose( minus, minusT);     
    //%%%%%%%%%%%%%%%%%%%%%%%project h and copy into h vectors%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    thrust::host_vector<double> hp( perp_size_), hm(hp);
    dg::blas2::symv( projection, yp[2], hp);
    dg::blas2::symv( projection, ym[2], hm);
    for( unsigned i=0; i<grid.Nz(); i++)
    {
        thrust::copy( hp.begin(), hp.end(), hp_.begin() + i*perp_size_);
        thrust::copy( hm.begin(), hm.end(), hm_.begin() + i*perp_size_);
    }
    dg::blas1::scal( hm_, -1.);
    dg::blas1::axpby(  1., hp_, +1., hm_, hz_);
    delete grid2d_ptr;
}

template<class G, class I, class container>
void FieldAligned<G, I,container>::set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right)
{
    cView left( global.cbegin(), global.cbegin() + perp_size_);
    cView right( global.cbegin()+(Nz_-1)*perp_size_, global.cbegin() + Nz_*perp_size_);
    View leftView( left_.begin(), left_.end());
    View rightView( right_.begin(), right_.end());
    cusp::copy( left, leftView);
    cusp::copy( right, rightView);
    dg::blas1::scal( left_, scal_left);
    dg::blas1::scal( right_, scal_right);
    bcz_ = bcz;
}

template< class G, class I, class container>
template< class BinaryOp>
container FieldAligned<G, I,container>::evaluate( BinaryOp binary, unsigned p0) const
{
    return evaluate( binary, dg::CONSTANT(1), p0, 0);
}

template<class G, class I, class container>
template< class BinaryOp, class UnaryOp>
container FieldAligned<G, I,container>::evaluate( BinaryOp binary, UnaryOp unary, unsigned p0, unsigned rounds) const
{
    //idea: simply apply I+/I- enough times on the init2d vector to get the result in each plane
    //unary function is always such that the p0 plane is at x=0
    assert( g_.get().Nz() > 1);
    assert( p0 < g_.get().Nz());
    const typename G::perpendicular_grid g2d = g_.get().perp_grid();
    assert( g2d.size() == perp_size_ && Nz_== g_.get().Nz());
    container init2d = dg::pullback( binary, g2d), temp(init2d), tempP(init2d), tempM(init2d);
    container vec3d = dg::evaluate( dg::zero, g_.get());
    std::vector<container>  plus2d( g_.get().Nz(), (container)dg::evaluate(dg::zero, g2d) ), minus2d( plus2d), result( plus2d);
    unsigned turns = rounds; 
    if( turns ==0) turns++;
    //first apply Interpolation many times, scale and store results
    for( unsigned r=0; r<turns; r++)
        for( unsigned i0=0; i0<Nz_; i0++)
        {
            dg::blas1::copy( init2d, tempP);
            dg::blas1::copy( init2d, tempM);
            unsigned rep = i0 + r*Nz_;
            for(unsigned k=0; k<rep; k++)
            {
                dg::blas2::symv( plus, tempP, temp);
                temp.swap( tempP);
                dg::blas2::symv( minus, tempM, temp);
                temp.swap( tempM);
            }
            dg::blas1::scal( tempP, unary(  (double)rep*g_.get().hz() ) );
            dg::blas1::scal( tempM, unary( -(double)rep*g_.get().hz() ) );
            dg::blas1::axpby( 1., tempP, 1., plus2d[i0]);
            dg::blas1::axpby( 1., tempM, 1., minus2d[i0]);
        }
    //now we have the plus and the minus filaments
    if( rounds == 0) //there is a limiter
    {
        for( unsigned i0=0; i0<Nz_; i0++)
        {
            int idx = (int)i0 - (int)p0;
            if(idx>=0)
                result[i0] = plus2d[idx];
            else
                result[i0] = minus2d[abs(idx)];
            thrust::copy( result[i0].begin(), result[i0].end(), vec3d.begin() + i0*perp_size_);
        }
    }
    else //sum up plus2d and minus2d
    {
        for( unsigned i0=0; i0<Nz_; i0++)
        {
            unsigned revi0 = (Nz_ - i0)%Nz_; //reverted index
            dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
            dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
        }
        dg::blas1::axpby( -1., init2d, 1., result[0]);
        for(unsigned i0=0; i0<Nz_; i0++)
        {
            int idx = ((int)i0 -(int)p0 + Nz_)%Nz_; //shift index
            thrust::copy( result[idx].begin(), result[idx].end(), vec3d.begin() + i0*perp_size_);
        }
    }
    return vec3d;
}


template<class G, class I, class container>
void FieldAligned<G, I, container >::operator()(enum whichMatrix which, const container& f, container& fe)
{
    if(which == einsPlus || which == einsMinusT) ePlus( which, f, fe);
    if(which == einsMinus || which == einsPlusT) eMinus( which, f, fe);
}

template< class G, class I, class container>
void FieldAligned<G, I, container>::ePlus( enum whichMatrix which, const container& f, container& fpe)
{
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView rightV( right_.begin(), right_.end());
    for( unsigned i0=0; i0<Nz_; i0++)
    {
        unsigned ip = (i0==Nz_-1) ? 0:i0+1;

        cView fp( f.cbegin() + ip*perp_size_, f.cbegin() + (ip+1)*perp_size_);
        View fP( fpe.begin() + i0*perp_size_, fpe.begin() + (i0+1)*perp_size_);
        if(which == einsPlus) cusp::multiply( plus, fp, fP);
        else if(which == einsMinusT) cusp::multiply( minusT, fp, fP );
        //make ghostcells i.e. modify fpe in the limiter region
        if( i0==Nz_-1 && bcz_ != dg::PER)
        {
            cView f0( f.cbegin() + i0*perp_size_, f.cbegin() + (i0+1)*perp_size_);
            if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
            {
                cusp::blas::axpby( rightV, f0, ghostPV, 2., -1.);
            }
            if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
            {
                thrust::transform( right_.begin(), right_.end(),  hp_.begin(), ghostM.begin(), thrust::multiplies<double>());
                cusp::blas::axpby( ghostMV, f0, ghostPV, 1., 1.);
            }
            //interlay ghostcells with periodic cells: L*g + (1-L)*fpe
            cusp::blas::axpby( ghostPV, fP, ghostPV, 1., -1.);
            dg::blas1::pointwiseDot( limiter_, ghostP, ghostP);
            cusp::blas::axpby(  ghostPV, fP, fP, 1.,1.);
        }
    }
}

template< class G, class I, class container>
void FieldAligned<G, I, container>::eMinus( enum whichMatrix which, const container& f, container& fme)
{
    //note that thrust functions don't work on views
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView leftV( left_.begin(), left_.end());
    for( unsigned i0=0; i0<Nz_; i0++)
    {
        unsigned im = (i0==0) ? Nz_-1:i0-1;
        cView fm( f.cbegin() + im*perp_size_, f.cbegin() + (im+1)*perp_size_);
        View fM( fme.begin() + i0*perp_size_, fme.begin() + (i0+1)*perp_size_);
        if(which == einsPlusT) cusp::multiply( plusT, fm, fM );
        else if (which == einsMinus) cusp::multiply( minus, fm, fM );
        //make ghostcells
        if( i0==0 && bcz_ != dg::PER)
        {
            cView f0( f.cbegin() + i0*perp_size_, f.cbegin() + (i0+1)*perp_size_);
            if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
            {
                cusp::blas::axpby( leftV,  f0, ghostMV, 2., -1.);
            }
            if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
            {
                thrust::transform( left_.begin(), left_.end(),  hm_.begin(), ghostP.begin(), thrust::multiplies<double>());
                cusp::blas::axpby( ghostPV, f0, ghostMV, -1., 1.);
            }
            //interlay ghostcells with periodic cells: L*g + (1-L)*fme
            cusp::blas::axpby( ghostMV, fM, ghostMV, 1., -1.);
            dg::blas1::pointwiseDot( limiter_, ghostM, ghostM);
            cusp::blas::axpby( ghostMV, fM, fM, 1., 1.);

        }
    }
}


///@endcond 


}//namespace dg

