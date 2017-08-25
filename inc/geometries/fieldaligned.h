#pragma once
#include <cmath>
#include <cusp/transpose.h>
#include <cusp/csr_matrix.h>

#include "dg/backend/grid.h"
#include "dg/blas.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/functions.h"

#include "dg/functors.h"
#include "dg/nullstelle.h"
#include "dg/runge_kutta.h"
#include "magnetic_field.h"

namespace dg{

enum whichMatrix
{
    einsPlus = 0,  
    einsPlusT = 1,  
    einsMinus = 2,  
    einsMinusT = 3,  
};

/**
 * @brief With the Default field ds becomes a dz
 */
struct DefaultField
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

/**
 * @brief Integrates the equations for a field line 
 * @tparam MagneticField models aTokamakMagneticField
 * @ingroup misc
 */ 
template<class MagneticField>
struct Field
{
    Field( const MagneticField& c):c_(c), invB_(c), R_0_(c.R_0) { }
    /**
     * @brief \f[ \frac{d \hat{R} }{ d \varphi}  = \frac{\hat{R}}{\hat{I}} \frac{\partial\hat{\psi}_p}{\partial \hat{Z}}, \hspace {3 mm}
     \frac{d \hat{Z} }{ d \varphi}  =- \frac{\hat{R}}{\hat{I}} \frac{\partial \hat{\psi}_p}{\partial \hat{R}} , \hspace {3 mm}
     \frac{d \hat{l} }{ d \varphi}  =\frac{\hat{R}^2 \hat{B}}{\hat{I}  \hat{R}_0}  \f]
     */ 
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double ipol = c_.ipol(y[0],y[1]);
        yp[2] =  y[0]*y[0]/invB_(y[0],y[1])/ipol/R_0_;       //ds/dphi =  R^2 B/I/R_0_hat
        yp[0] =  y[0]*c_.psipZ(y[0],y[1])/ipol;              //dR/dphi =  R/I Psip_Z
        yp[1] = -y[0]*c_.psipR(y[0],y[1])/ipol ;             //dZ/dphi = -R/I Psip_R

    }
    double error( const dg::HVec& x0, const dg::HVec& x1)
    {
        return sqrt( (x0[0]-x1[0])*(x0[0]-x1[0]) +(x0[1]-x1[1])*(x0[1]-x1[1])+(x0[2]-x1[2])*(x0[2]-x1[2]));
    }
    bool monitor( const dg::HVec& end){ 
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
    MagneticField c_;
    InvB invB_;
    double R_0_;
};

template< class GeometryPerp>
struct DSField
{
    DSField( const BinaryVectorLvl0& v, const GeometryPerp& g)
    {
        thrust::host_vector<double> b_zeta, b_eta;
        dg::pushForwardPerp( v.x(), v.y(), b_zeta, b_eta, g);
        FieldP<MagneticField> fieldP(c);
        thrust::host_vector<double> b_phi = dg::pullback( v.z(), g);
        dxdz_ = dg::forward_transform( b_zeta, g );
        dypz_ = dg::forward_transform( b_eta, g );
        dsdz_ = dg::forward_transform( b_phi, g );
    }

    void operator()(const thrust::host_vector<double>& y, thrust::host_vector<double>& yp)
    {
        g_.shift_topologic( y[0], y[1], y[0], y[1]); //shift points onto domain
        if( !g_.contains( y[0], y[1])) yp[0] = yp[1]= yp[2] = 0;
        else
        {
            //else interpolate
            yp[0] = interpolate( y[0], y[1], dzetadphi_, g_);
            yp[1] = interpolate( y[0], y[1], detadphi_, g_);
            yp[2] = interpolate( y[0], y[1], dsphi_, g_);
        }
    }

    double error( const dg::HVec& x0, const dg::HVec& x1)
    {
        return sqrt( (x0[0]-x1[0])*(x0[0]-x1[0]) +(x0[1]-x1[1])*(x0[1]-x1[1])+(x0[2]-x1[2])*(x0[2]-x1[2]));
    }
    bool monitor( const dg::HVec& end){ 
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
    thrust::host_vector<double> dzetadphi_, detadphi_, dsdphi_;
    GeometryPerp g_;

};

/**
 * @brief Default Limiter means there is a limiter everywhere
 */
struct DefaultLimiter
{
    /**
     * @brief return 1
     *
     * @param x x value
     * @param y y value
     * @return 1
     */
    double operator()(double x, double y) { return 1; }
};

/**
 * @brief No Limiter 
 */
struct NoLimiter
{
    /**
     * @brief return 0
     *
     * @param x x value
     * @param y y value
     * @return 0
     */
    double operator()(double x, double y) { return 0.; }
};

/**
 * @brief Integrate a field line to find whether the result lies inside or outside of the box
 *
 * @tparam Field Must be usable in the integrateRK4 function
 * @tparam Grid must provide 2d boundaries x0(), x1(), y0(), and y1()
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
    BoxIntegrator( Field field, const Grid& g, double eps): field_(field), g_(g), coords_(3), coordsp_(3), eps_(eps) {}
    /**
     * @brief Set the starting coordinates for next field line integration
     *
     * @param coords the new coords (must have size = 3)
     */
    void set_coords( const thrust::host_vector<double>& coords){ coords_ = coords;}
    /**
     * @brief Integrate from 0 to deltaPhi
     *
     * @param deltaPhi upper integration boundary
     *
     * @return 1 if point is inside the box, -1 else
     */
    double operator()( double deltaPhi)
    {
        try{
            dg::integrateRK4( field_, coords_, coordsp_, deltaPhi, eps_);
        }
        catch( dg::NotANumber& exception) { return -1;}
        if (!(coordsp_[0] >= g_.x0() && coordsp_[0] <= g_.x1())) {
            return -1;
        }
        if (!(coordsp_[1] >= g_.y0() && coordsp_[1] <= g_.y1())) {
            return -1;
        }
        return +1;
    }
    private:
    Field field_;
    Grid g_;
    thrust::host_vector<double> coords_, coordsp_;
    double eps_;
};

/**
 * @brief Integrate one field line in a given box, Result is guaranteed to lie inside the box
 *
 * @tparam Field Must be usable in the integrateRK4 function
 * @tparam Grid must provide 2d boundaries x0(), x1(), y0(), and y1()
 * @param field The field to use
 * @param grid instance of the Grid class 
 * @param coords0 The initial condition
 * @param coords1 The resulting points (write only) guaranteed to lie inside the grid
 * @param phi1 The angle (read/write) contains maximum phi on input and resulting phi on output
 * @param eps error
 * @param globalbcz boundary condition  (DIR or NEU)
 */
template< class Field, class Grid>
void boxintegrator( Field& field, const Grid& grid, 
        const thrust::host_vector<double>& coords0, 
        thrust::host_vector<double>& coords1, 
        double& phi1, double eps, dg::bc globalbcz)
{
    dg::integrateRK4( field, coords0, coords1, phi1, eps); //+ integration
    //First catch periodic domain
    grid.shift_topologic( coords0[0], coords0[1], coords1[0], coords1[1]);
    if ( !grid.contains( coords1[0], coords1[1]))   //Punkt liegt immer noch außerhalb 
    {
#ifdef DG_DEBUG
        std::cerr << "point "<<coords1[0]<<" "<<coords1[1]<<" "<<coords1[3]<<" "<<coords1[4]<<" is somewhere else!\n";
#endif //DG_DEBUG
        if( globalbcz == dg::DIR)
        {
            //idea: maybe we should take the "wrong" long Delta s instead of the short one to avoid deteriorated CFL condition
            BoxIntegrator<Field, Grid> boxy( field, grid, eps);
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
            if (!(coords1[0] > grid.x0())) { coords1[0]=grid.x0();}
            if (!(coords1[0] < grid.x1())) { coords1[0]=grid.x1();}
            if (!(coords1[1] > grid.y0())) { coords1[1]=grid.y0();}
            if (!(coords1[1] < grid.y1())) { coords1[1]=grid.y1();}
        }
        else if (globalbcz == dg::NEU )
        {
             coords1[0] = coords0[0]; coords1[1] = coords0[1];  //this is clearly wrong -> makes ds a d\varphi
        }
        else if (globalbcz == DIR_NEU )std::cerr << "DIR_NEU NOT IMPLEMENTED "<<std::endl;
        else if (globalbcz == NEU_DIR )std::cerr << "NEU_DIR NOT IMPLEMENTED "<<std::endl;
        else if (globalbcz == dg::PER )std::cerr << "PER NOT IMPLEMENTED "<<std::endl;
    }
}
////////////////////////////////////FieldAlignedCLASS////////////////////////////////////////////
/**
* @brief Class for the evaluation of a parallel derivative
*
* This class discretizes the operators \f$ \nabla_\parallel = 
\mathbf{b}\cdot \nabla = b_R\partial_R + b_Z\partial_Z + b_\phi\partial_\phi \f$, \f$\nabla_\parallel^\dagger\f$ and \f$\Delta_\parallel=\nabla_\parallel^\dagger\cdot\nabla_\parallel\f$ in
cylindrical coordinates
* @ingroup utilities
* @tparam Matrix The matrix class of the interpolation matrix
* @tparam container The container-class on which the interpolation matrix operates on (does not need to be dg::HVec)
*/
template<class IMatrix, class container >
struct FieldAligned
{

    typedef IMatrix InterpolationMatrix;
    FieldAligned(){}

    /**
    * @brief Construct from a field and a grid
    *
    * @tparam Field The Fieldlines to be integrated: 
        Has to provide void operator()( const std::vector<dg::HVec>&, std::vector<dg::HVec>&) 
        where the first index is R, the second Z and the last s (the length of the field line)
    * @tparam Limiter Class that can be evaluated on a 2d grid, returns 1 if there
        is a limiter and 0 if there isn't. 
        If a field line crosses the limiter in the plane \f$ \phi=0\f$ then the limiter boundary conditions apply. 
    * @param field The field to integrate
    * @param grid The grid on which to operate
    * @param eps Desired accuracy of runge kutta
    * @param limit Instance of the limiter class (Default is a limiter everywhere, 
        note that if bcz is periodic it doesn't matter if there is a limiter or not)
    * @param globalbcz Choose NEU or DIR. Defines BC in parallel on bounding box
    * @param deltaPhi Is either <0 (then it's ignored), may differ from hz() only if Nz() == 1
    * @note If there is a limiter, the boundary condition on the first/last plane is set 
        by the bcz variable from the grid and can be changed by the set_boundaries function. 
        If there is no limiter the boundary condition is periodic.
    */
    template <class Geometry, class Limiter>
    FieldAligned(const BinaryVectorLvl0& vec, const Geometry& grid, unsigned multiplyX, unsigned multiplyY, double eps = 1e-4, Limiter limit = DefaultLimiter(), dg::bc globalbcz = dg::DIR, bool dependsOnX=true, bool dependsOnY=true, double deltaPhi = -1);

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

    /**
     * @brief Set boundary conditions in the limiter region
     *
     * if Dirichlet boundaries are used the left value is the left function
     value, if Neumann boundaries are used the left value is the left derivative value
     * @param bcz boundary condition
     * @param left left boundary value
     * @param right right boundary value
    */
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

    void operator()(enum whichMatrix which, const container& in, container& out);
    /**
    * @brief Applies the interpolation to the next planes 
    *
    * @param in input 
    * @param out output may not equal intpu
    */
    void einsPlus( const container& in, container& out);
    /**
    * @brief Applies the interpolation to the previous planes
    *
    * @param in input 
    * @param out output may not equal intpu
    */
    void einsMinus( const container& in, container& out);

    /**
    * @brief hz is the distance between the plus and minus planes
    *
    * @return three-dimensional vector
    */
    const container& hz()const {return hz_;}
    /**
    * @brief hp is the distance between the plus and current planes
    *
    * @return three-dimensional vector
    */
    const container& hp()const {return hp_;}
    /**
    * @brief hm is the distance between the current and minus planes
    *
    * @return three-dimensional vector
    */
    const container& hm()const {return hm_;}
    private:
    typedef cusp::array1d_view< typename container::iterator> View;
    typedef cusp::array1d_view< typename container::const_iterator> cView;
    IMatrix plus, minus, plusT, minusT; //interpolation matrices
    container hz_, hp_,hm_, ghostM, ghostP;
    unsigned Nz_, perp_size_;
    dg::bc bcz_;
    container left_, right_;
    container limiter_;
};

///@cond 
////////////////////////////////////DEFINITIONS////////////////////////////////////////

template<class I, class container>
template <class MagneticField, class Geometry, class Limiter>
FieldAligned<I, container>::FieldAligned(const BinaryVectorLvl0& mag, const Geometry& grid, unsigned mx, unsigned my, double eps, Limiter limit, dg::bc globalbcz, bool dependsOnX, bool dependsOnY, double deltaPhi):
        hz_( dg::evaluate( dg::zero, grid)), hp_( hz_), hm_( hz_), 
        Nz_(grid.Nz()), bcz_(grid.bcz())
{
    //Resize vector to 2D grid size

    typename Geometry::perpendicular_grid g2dCoarse = grid.perp_grid( );
    perp_size_ = g2dCoarse.size();
    limiter_ = dg::evaluate( limit, g2dCoarse);
    right_ = left_ = dg::evaluate( zero, g2dCoarse);
    ghostM.resize( perp_size_); ghostP.resize( perp_size_);
    //Set starting points
    typename Geometry::perpendicular_grid g2dFine = g2dCoarse.muliplyCellNumbers( (double)mX, (double)mY);
    
    std::vector<thrust::host_vector<double> > y( 3, dg::evaluate( dg::cooX2d, g2dFine)); //x
    y[1] = dg::evaluate( dg::cooY2d, g2dFine); //y
    y[2] = dg::evaluate( dg::zero, g2dFine); //s
    std::vector<thrust::host_vector<double> > yp( 3, dg::evaluate(dg::zero, g2dFine)), ym(yp); 
    if( deltaPhi <=0) deltaPhi = grid.hz();
    else assert( grid.Nz() == 1 || grid.hz()==deltaPhi);
    dg::Timer t;
    t.tic();
    //construct field on high polynomial grid, then integrate it
    typename Geometry::perpendicular_grid g2dField = g2dCoarse;
    g2dField.set( 11, g2dField.Nx(), g2dField.Ny());
    dg::DSField<typename Geometry::perpendicular_grid> field( mag, g2dField);
#ifdef _OPENMP
#pragma omp parallel for shared(field)
#endif //_OPENMP
    for( unsigned i=0; i<g2dF.size(); i++)
    {
        thrust::host_vector<double> coords(3), coordsP(3), coordsM(3);
        coords[0] = y[0][i], coords[1] = y[1][i], coords[2] = y[2][i]; //x,y,s
        double phi1 = deltaPhi;
        boxintegrator( field, g2d, coords, coordsP, phi1, eps, globalbcz);
        phi1 =  - deltaPhi;
        boxintegrator( field, g2d, coords, coordsM, phi1, eps, globalbcz);
        yp[0][i] = coordsP[0], yp[1][i] = coordsP[1], yp[2][i] = coordsP[2];
        ym[0][i] = coordsM[0], ym[1][i] = coordsM[1], ym[2][i] = coordsM[2];
    }
    t.toc(); 
    std::cout << "Fieldline integration took "<<t.diff()<<"s\n";
    t.tic();
    IMatrix plusFine  = dg::create::interpolation( yp[0], yp[1], g2dCoarse, globalbcz);
    IMatrix minusFine = dg::create::interpolation( ym[0], ym[1], g2dCoarse, globalbcz);
    IMatrix interpolation = dg::create::interpolation( g2dFine, g2dCoarse);
    IMatrix projection = dg::create::projection( g2dCoarse, g2dFine);
    t.toc();
    std::cout <<"Creation of interpolation/projection took "<<t.diff()<<"s\n";
    t.tic();
    cusp::multiply( projection, plusFine, plus);
    cusp::multiply( projection, minusFine, minus);
    t.toc();
    std::cout<< "Multiplication of PI took: "<<t.diff()<<"s\n";
    //Transposed matrices work only for csr_matrix due to bad matrix form for ell_matrix!!!
    cusp::transpose( plus, plusT);
    cusp::transpose( minus, minusT);     
//     copy into h vectors
    
    thrust::host_vector<double> hp( perp_size_), hm(hp);
    dg::blas2::symv( projection, yp[2], hp);
    dg::blas2::symv( projection, ym[2], hm);
    for( unsigned i=0; i<grid.Nz(); i++)
    {
        thrust::copy( hp.begin(), hp.end(), hp_.begin() + i*perp_size_);
        thrust::copy( hm.begin(), hm.end(), hm_.begin() + i*perp_size_);
    }
    dg::blas1::scal( hm_, -1.);
    dg::blas1::axpby(  1., hp_, +1., hm_, hz_);    //
 
}

template<class I, class container>
void FieldAligned<I,container>::set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right)
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

template< class I, class container>
template< class BinaryOp, class Grid>
container FieldAligned<I,container>::evaluate( BinaryOp binary, unsigned p0, Grid grid) const
{
    return evaluate( binary, dg::CONSTANT(1), p0, 0, grid);
}

template<class I, class container>
template< class BinaryOp, class UnaryOp, class Grid>
container FieldAligned<I,container>::evaluate( BinaryOp binary, UnaryOp unary, unsigned p0, unsigned rounds, Grid g_) const
{
    //idea: simply apply I+/I- enough times on the init2d vector to get the result in each plane
    //unary function is always such that the p0 plane is at x=0
    assert( g_.Nz() > 1);
    assert( p0 < g_.Nz());
    const typename G::perpendicular_grid g2d = g_.perp_grid();
    assert( g2d.size() == perp_size_ && Nz_== g_.Nz());
    container init2d = dg::pullback( binary, g2d), temp(init2d), tempP(init2d), tempM(init2d);
    container vec3d = dg::evaluate( dg::zero, g_);
    std::vector<container>  plus2d( g_.Nz(), (container)dg::evaluate(dg::zero, g2d) ), minus2d( plus2d), result( plus2d);
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
            dg::blas1::scal( tempP, unary(  (double)rep*g_.hz() ) );
            dg::blas1::scal( tempM, unary( -(double)rep*g_.hz() ) );
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


template< class I, class container>
void FieldAligned<I, container >::operator()(enum whichMatrix which, const container& f, container& fe)
{
    if(which == einsPlus || which == einsMinusT) einsPlus( which, f, fe);
    if(which == einsMinus || which == einsPlusT) einsMinus( which, f, fe);
}

template< class I, class container>
void FieldAligned<I, container>::einsPlus( enum whichMatrix which, const container& f, container& fpe)
{
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView rightV( right_.begin(), right_.end());
    for( unsigned i0=0; i0<Nz_; i0++)
    {
        unsigned ip = (i0==Nz_-1) ? 0:i0+1;

        cView fp( f.cbegin() + ip*perp_size_, f.cbegin() + (ip+1)*perp_size_);
        cView f0( f.cbegin() + i0*perp_size_, f.cbegin() + (i0+1)*perp_size_);
        View fP( fpe.begin() + i0*perp_size_, fpe.begin() + (i0+1)*perp_size_);
        if(which == einsPlus) cusp::multiply( plus, fp, fP);
        else if(which == einsMinusT) cusp::multiply( minusT, fp, fP );
        //make ghostcells i.e. modify fpe in the limiter region
        if( i0==Nz_-1 && bcz_ != dg::PER)
        {
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

template< class I, class container>
void FieldAligned<I, container>::einsMinus( enum whichMatrix which, const container& f, container& fme)
{
    //note that thrust functions don't work on views
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView leftV( left_.begin(), left_.end());
    for( unsigned i0=0; i0<Nz_; i0++)
    {
        unsigned im = (i0==0) ? Nz_-1:i0-1;
        cView fm( f.cbegin() + im*perp_size_, f.cbegin() + (im+1)*perp_size_);
        cView f0( f.cbegin() + i0*perp_size_, f.cbegin() + (i0+1)*perp_size_);
        View fM( fme.begin() + i0*perp_size_, fme.begin() + (i0+1)*perp_size_);
        if(which == einsPlusT) cusp::multiply( plusT, fm, fM );
        else if (which == einsMinus) cusp::multiply( minus, fm, fM );
        //make ghostcells
        if( i0==0 && bcz_ != dg::PER)
        {
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

