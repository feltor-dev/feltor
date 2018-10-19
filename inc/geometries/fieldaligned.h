#pragma once
#include <cmath>
#include <array>
#include <cusp/csr_matrix.h>

#include "dg/backend/transpose.h"
#include "dg/blas.h"
#include "dg/geometry/grid.h"
#include "dg/geometry/interpolation.h"
#include "dg/geometry/projection.h"
#include "dg/geometry/functions.h"
#include "dg/geometry/split_and_join.h"

#include "dg/geometry/geometry.h"
#include "dg/functors.h"
#include "dg/nullstelle.h"
#include "dg/adaptive.h"
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



struct DSFieldCylindrical
{
    DSFieldCylindrical( const dg::geo::BinaryVectorLvl0& v, Grid2d boundary):v_(v), m_b(boundary) { }
    void operator()( double t, const std::array<double,3>& y, std::array<double,3>& yp) const {
        double R = y[0], Z = y[1];
        m_b.shift_topologic( y[0], y[1], R, Z); //shift R,Z onto domain
        double vz = v_.z()(R, Z);
        yp[0] = v_.x()(R, Z)/vz;
        yp[1] = v_.y()(R, Z)/vz;
        yp[2] = 1./vz;
    }

    private:
    dg::geo::BinaryVectorLvl0 v_;
    dg::Grid2d m_b;
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
        dg::blas1::pointwiseDivide(1.,    v_phi, v_phi);
        dzetadphi_  = dg::create::forward_transform( v_zeta, g );
        detadphi_   = dg::create::forward_transform( v_eta, g );
        dsdphi_     = dg::create::forward_transform( v_phi, g );
    }
    //interpolate the vectors given in the constructor on the given point
    //if point lies outside of grid boundaries zero is returned
    void operator()(double t, const std::array<double,3>& y, std::array<double,3>& yp) const
    {
        double R = y[0], Z = y[1];
        g_.get().shift_topologic( y[0], y[1], R, Z); //shift R,Z onto domain
        if( !g_.get().contains( R, Z))
        {
            yp[0] = yp[1] = 0; //Let's hope this never happens?
        }
        else
        {
            //else interpolate
            yp[0] = interpolate( R, Z, dzetadphi_, g_.get());
            yp[1] = interpolate( R, Z, detadphi_,  g_.get());
            yp[2] = interpolate( R, Z, dsdphi_,    g_.get());
        }
    }
    private:
    thrust::host_vector<double> dzetadphi_, detadphi_, dsdphi_;
    dg::ClonePtr<dg::aGeometry2d> g_;
};

double ds_norm( const std::array<double,3>& x0){
    return sqrt( x0[0]*x0[0] +x0[1]*x0[1] + x0[2]*x0[2]);
}

//used in constructor of Fieldaligned
template<class real_type>
void integrate_all_fieldlines2d( const dg::geo::BinaryVectorLvl0& vec,
    const dg::aRealGeometry2d<real_type>& grid_field,
    const dg::aRealTopology2d<real_type>& grid_evaluate,
    std::array<thrust::host_vector<real_type>,3>& yp,
    std::array<thrust::host_vector<real_type>,3>& ym,
    real_type deltaPhi, real_type eps)
{
    //grid_field contains the global geometry for the field and the boundaries
    //grid_evaluate contains the points to actually integrate
    thrust::host_vector<real_type> tmp( dg::evaluate( dg::cooX2d, grid_evaluate));
    std::array<thrust::host_vector<real_type>,3> y{tmp,tmp,tmp};; //x
    y[1] = dg::evaluate( dg::cooY2d, grid_evaluate); //y
    y[2] = dg::evaluate( dg::zero, grid_evaluate); //s
    yp.fill(tmp); ym.fill(tmp); //allocate memory for output
    //construct field on high polynomial grid, then integrate it
    dg::geo::detail::DSField field( vec, grid_field);
    //field in case of cartesian grid
    dg::geo::detail::DSFieldCylindrical cyl_field(vec, (dg::Grid2d)grid_field);
    unsigned size = grid_evaluate.size();
    for( unsigned i=0; i<size; i++)
    {
        std::array<real_type,3> coords{y[0][i],y[1][i],y[2][i]}, coordsP, coordsM;
        //x,y,s
        real_type phi1 = deltaPhi;
        if( dynamic_cast<const dg::CartesianGrid2d*>( &grid_field))
            dg::integrateERK( "Dormand-Prince-7-4-5", cyl_field, 0., coords, phi1, coordsP, 0., dg::pid_control, ds_norm, eps,1e-10); //integration
        else
            dg::integrateERK( "Dormand-Prince-7-4-5", field, 0., coords, phi1, coordsP, 0., dg::pid_control, ds_norm, eps,1e-10); //integration
        phi1 =  - deltaPhi;
        if( dynamic_cast<const dg::CartesianGrid2d*>( &grid_field))
            dg::integrateERK( "Dormand-Prince-7-4-5", cyl_field, 0., coords, phi1, coordsM, 0., dg::pid_control, ds_norm, eps,1e-10); //integration
        else
            dg::integrateERK( "Dormand-Prince-7-4-5", field, 0., coords, phi1, coordsM, 0., dg::pid_control, ds_norm, eps,1e-10); //integration
        yp[0][i] = coordsP[0], yp[1][i] = coordsP[1], yp[2][i] = coordsP[2];
        ym[0][i] = coordsM[0], ym[1][i] = coordsM[1], ym[2][i] = coordsM[2];
    }
}


}//namespace detail
///@endcond


    /*!@class hide_fieldaligned_physics_parameters
    * @tparam Limiter Class that can be evaluated on a 2d grid, returns 1 if there
        is a limiter and 0 if there isn't.
        If a field line crosses the limiter in the plane \f$ \phi=0\f$ then the limiter boundary conditions apply.
    * @param vec The vector field to integrate
    * @param grid The grid on which to integrate fieldlines.
    * @param bcx This parameter is passed on to \c dg::create::interpolation(const thrust::host_vector<real_type>&,const thrust::host_vector<real_type>&,const aRealTopology2d<real_type>&,dg::bc,dg::bc) (see there for more details)
    * function and deterimens what happens when the endpoint of the fieldline integration leaves the domain boundaries of \c grid. Note that \c bcx and \c grid.bcx() have to be either both periodic or both not periodic.
    * @param bcy analogous to \c bcx, applies to y direction
    * @param limit Instance of the limiter class
        (Note that if \c grid.bcz()==dg::PER this parameter is ignored, Default is a limiter everywhere)
    */
    /*!@class hide_fieldaligned_numerics_parameters
    * @param eps Desired accuracy of the fieldline integrator
    * @param multiplyX defines the resolution in X of the fine grid relative to grid (Set to 1, if the x-component of \c vec vanishes, else as
    * high as possible in given amount of time)
    * @param multiplyY analogous in Y
    * @param deltaPhi Is either <0 (then it's ignored), or may differ from \c grid.hz() if \c grid.Nz()==1, then \c deltaPhi is taken instead of \c grid.hz()
    * @note If there is a limiter, the boundary condition on the first/last plane is set
        by the \c grid.bcz() variable and can be changed by the set_boundaries function.
        If there is no limiter, the boundary condition is periodic.
    */
//////////////////////////////FieldalignedCLASS////////////////////////////////////////////
/**
* @brief Create and manage interpolation matrices from fieldline integration
*
* @ingroup fieldaligned
* @snippet ds_t.cu doxygen
* @tparam ProductGeometry must be either \c dg::aProductGeometry3d or \c dg::aProductMPIGeometry3d or any derivative
* @tparam IMatrix The type of the interpolation matrix
    - \c dg::IHMatrix, or \c dg::IDMatrix, \c dg::MIHMatrix, or \c dg::MIDMatrix
* @tparam container The container-class on which the interpolation matrix operates on
    - \c dg::HVec, or \c dg::DVec, \c dg::MHVec, or \c dg::MDVec
* @sa The pdf <a href="./parallel.pdf" target="_blank">parallel derivative</a> writeup
*/
template<class ProductGeometry, class IMatrix, class container >
struct Fieldaligned
{

    ///@brief do not allocate memory; no member call except construct is valid
    Fieldaligned(){}
   ///@brief Construct from a magnetic field and a grid
   ///@copydoc hide_fieldaligned_physics_parameters
   ///@copydoc hide_fieldaligned_numerics_parameters
    template <class Limiter>
    Fieldaligned(const dg::geo::TokamakMagneticField& vec,
        const ProductGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned multiplyX=10, unsigned multiplyY=10,
        double deltaPhi = -1)
    {
        dg::geo::BinaryVectorLvl0 bhat( (dg::geo::BHatR)(vec), (dg::geo::BHatZ)(vec), (dg::geo::BHatP)(vec));
        construct( bhat, grid, bcx, bcy, limit, eps, multiplyX, multiplyY, deltaPhi);
    }

    ///@brief Construct from a vector field and a grid
    ///@copydoc hide_fieldaligned_physics_parameters
    ///@copydoc hide_fieldaligned_numerics_parameters
    template <class Limiter>
    Fieldaligned(const dg::geo::BinaryVectorLvl0& vec,
        const ProductGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned multiplyX=10, unsigned multiplyY=10,
        double deltaPhi = -1)
    {
        construct( vec, grid, bcx, bcy, limit, eps, multiplyX, multiplyY, deltaPhi);
    }
    ///@brief Construct from a field and a grid
    ///@copydoc hide_fieldaligned_physics_parameters
    ///@copydoc hide_fieldaligned_numerics_parameters
    template <class Limiter>
    void construct(const dg::geo::BinaryVectorLvl0& vec,
        const ProductGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned multiplyX=10, unsigned multiplyY=10,
        double deltaPhi = -1);

    dg::bc bcx()const{
        return m_bcx;
    }
    dg::bc bcy()const{
        return m_bcy;
    }


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
    * @param left spatially variable left boundary value (2d size)
    * @param right spatially variable right boundary value (2d size)
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
        dg::split( global, m_temp, m_g.get());
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
     * @param p0 The index of the plane to start
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
     * The algorithm does the equivalent of the following:
     *  - Evaluate the given \c BinaryOp on a 2d plane
     *  - Apply the plus and minus transformation each \f$ r N_z\f$ times where \f$ N_z\f$ is the number of planes in the global 3d grid and \f$ r\f$ is the number of rounds.
     *  - Scale the transformations with \f$ u ( \pm (iN_z + j)h_z) \f$, where \c u is the given \c UnarayOp, \c i is the round index and \c j is the plane index.
     *  - Sum all transformations with the same plane index \c j , where the minus transformations get the inverted index \f$ N_z - j\f$.
     *  - Shift the index by \f$ p_0\f$
     *  .
     * @tparam BinaryOp Binary Functor
     * @tparam UnaryOp Unary Functor
     * @param binary Functor to evaluate in x-y
     * @param unary Functor to evaluate in z
     * @param p0 The index of the plane to start
     * @param rounds The number of rounds \c r to follow a fieldline; can be zero, then the fieldlines are only followed within the current box ( no periodicity)
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

    ///@brief Inverse distance between the planes \f$ (s^{k}-s^{k-1})^{-1} \f$
    ///@return three-dimensional vector
    const container& hm_inv()const {
        return m_hm;
    }
    ///@brief Inverse distance between the planes \f$ (s^{k+1}-s^{k})^{-1} \f$
    ///@return three-dimensional vector
    const container& hp_inv()const {
        return m_hp;
    }
    ///@brief Inverse distance between the planes \f$ (s^{k+1}-s^{k-1})^{-1} \f$
    ///@return three-dimensional vector
    const container& h0_inv()const {
        return m_h0;
    }
    ///Grid used for construction
    const ProductGeometry& grid()const{return m_g.get();}
    private:
    void ePlus( enum whichMatrix which, const container& in, container& out);
    void eMinus(enum whichMatrix which, const container& in, container& out);
    IMatrix m_plus, m_minus, m_plusT, m_minusT; //2d interpolation matrices
    container m_h0, m_hm, m_hp; //3d size
    container m_h; //2d size
    container m_left, m_right;      //perp_size
    container m_limiter;            //perp_size
    container m_ghostM, m_ghostP;   //perp_size
    unsigned m_Nz, m_perp_size;
    dg::bc m_bcx, m_bcy, m_bcz;
    std::vector<dg::View<const container>> m_f;
    std::vector<dg::View< container>> m_temp;
    dg::ClonePtr<ProductGeometry> m_g;
};

///@cond
////////////////////////////////////DEFINITIONS////////////////////////////////////////
//


template<class Geometry, class IMatrix, class container>
template <class Limiter>
void Fieldaligned<Geometry, IMatrix, container>::construct(
    const dg::geo::BinaryVectorLvl0& vec, const Geometry& grid,
    dg::bc bcx, dg::bc bcy, Limiter limit, double eps,
    unsigned mx, unsigned my, double deltaPhi)
{
    ///Let us check boundary conditions:
    if( (grid.bcx() == PER && bcx != PER) || (grid.bcx() != PER && bcx == PER) )
        throw( dg::Error(dg::Message(_ping_)<<"Fieldaligned: Got conflicting periodicity in x. The grid says "<<bc2str(grid.bcx())<<" while the parameter says "<<bc2str(bcx)));
    if( (grid.bcy() == PER && bcy != PER) || (grid.bcy() != PER && bcy == PER) )
        throw( dg::Error(dg::Message(_ping_)<<"Fieldaligned: Got conflicting boundary conditions in y. The grid says "<<bc2str(grid.bcy())<<" while the parameter says "<<bc2str(bcy)));
    m_Nz=grid.Nz(), m_bcx = bcx, m_bcy = bcy, m_bcz=grid.bcz();
    m_g.reset(grid);
    if( deltaPhi <=0) deltaPhi = grid.hz();
    else assert( grid.Nz() == 1 || grid.hz()==deltaPhi);
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    dg::ClonePtr<dg::aGeometry2d> grid_coarse( grid.perp_grid()) ;
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    m_perp_size = grid_coarse.get().size();
    dg::assign( dg::pullback(limit, grid_coarse.get()), m_limiter);
    dg::assign( dg::evaluate(zero, grid_coarse.get()), m_left);
    m_ghostM = m_ghostP = m_right = m_left;
    ///%%%%%%%%%%Set starting points and integrate field lines%%%%%%%%%%%//
#ifdef DG_BENCHMARK
    dg::Timer t;
    t.tic();
#endif //DG_BENCHMARK
    std::array<thrust::host_vector<double>,3> yp_coarse, ym_coarse, yp, ym;
    dg::ClonePtr<dg::aGeometry2d> grid_magnetic = grid_coarse;//INTEGRATE HIGH ORDER GRID
    grid_magnetic.get().set( 7, grid_magnetic.get().Nx(), grid_magnetic.get().Ny());
    dg::Grid2d grid_fine( grid_coarse.get() );//FINE GRID
    grid_fine.multiplyCellNumbers((double)mx, (double)my);
#ifdef DG_BENCHMARK
    t.toc();
    std::cout << "# DS: High order grid gen  took: "<<t.diff()<<"\n";
    t.tic();
#endif //DG_BENCHMARK
    detail::integrate_all_fieldlines2d( vec, grid_magnetic.get(), grid_coarse.get(), yp_coarse, ym_coarse, deltaPhi, eps);
    dg::IHMatrix interpolate = dg::create::interpolation( grid_fine, grid_coarse.get());  //INTERPOLATE TO FINE GRID
    yp.fill(dg::evaluate( dg::zero, grid_fine));
    ym = yp;
    for( int i=0; i<2; i++) //only R and Z get interpolated
    {
        dg::blas2::symv( interpolate, yp_coarse[i], yp[i]);
        dg::blas2::symv( interpolate, ym_coarse[i], ym[i]);
    }
#ifdef DG_BENCHMARK
    t.toc();
    std::cout << "# DS: Computing all points took: "<<t.diff()<<"\n";
    t.tic();
#endif //DG_BENCHMARK
    ///%%%%%%%%%%%%%%%%Create interpolation and projection%%%%%%%%%%%%%%//
    dg::IHMatrix plusFine  = dg::create::interpolation( yp[0], yp[1], grid_coarse.get(), bcx, bcy), plus, plusT;
    dg::IHMatrix minusFine = dg::create::interpolation( ym[0], ym[1], grid_coarse.get(), bcx, bcy), minus, minusT;
    dg::IHMatrix projection = dg::create::projection( grid_coarse.get(), grid_fine);
    cusp::multiply( projection, plusFine, plus);
    cusp::multiply( projection, minusFine, minus);
#ifdef DG_BENCHMARK
    t.toc();
    std::cout << "# DS: Multiplication PI    took: "<<t.diff()<<"\n";
#endif //DG_BENCHMARK
    plusT = dg::transpose( plus);
    minusT = dg::transpose( minus);
    dg::blas2::transfer( plus, m_plus);
    dg::blas2::transfer( plusT, m_plusT);
    dg::blas2::transfer( minus, m_minus);
    dg::blas2::transfer( minusT, m_minusT);
    ///%%%%%%%%%%%%%%%%%%%%copy into h vectors %%%%%%%%%%%%%%%%%%%//
    dg::assign( dg::evaluate( dg::zero, grid), m_h0);
    m_hp = m_hm = m_h0;
    container temp;
    dg::assign( yp_coarse[2], temp); //2d vector
    dg::split( m_hp, m_temp, grid); //3d vector
    for( unsigned i=0; i<m_Nz; i++)
        dg::blas1::copy( temp, m_temp[i]);
    dg::assign( ym_coarse[2], temp); //2d vector
    dg::split( m_hm, m_temp, grid); //3d vector
    for( unsigned i=0; i<m_Nz; i++)
        dg::blas1::copy( temp, m_temp[i]);
    dg::blas1::axpby( 1., m_hp, -1., m_hm, m_h0);//hm is negative
    dg::blas1::pointwiseDivide( -1., m_hm, m_hm);
    dg::blas1::pointwiseDivide( 1., m_hp, m_hp);
    dg::blas1::pointwiseDivide( 1., m_h0, m_h0);
}

template<class G, class I, class container>
template< class BinaryOp, class UnaryOp>
container Fieldaligned<G, I,container>::evaluate( const BinaryOp& binary, const UnaryOp& unary, unsigned p0, unsigned rounds) const
{
    //idea: simply apply I+/I- enough times on the init2d vector to get the result in each plane
    //unary function is always such that the p0 plane is at x=0
    assert( p0 < m_g.get().Nz());
    const dg::ClonePtr<aGeometry2d> g2d = m_g.get().perp_grid();
    container init2d = dg::pullback( binary, g2d.get());
    container zero2d = dg::evaluate( dg::zero, g2d.get());

    container temp(init2d), tempP(init2d), tempM(init2d);
    container vec3d = dg::evaluate( dg::zero, m_g.get());
    std::vector<container>  plus2d(m_Nz, zero2d), minus2d(plus2d), result(plus2d);
    unsigned turns = rounds;
    if( turns ==0) turns++;
    //first apply Interpolation many times, scale and store results
    for( unsigned r=0; r<turns; r++)
        for( unsigned i0=0; i0<m_Nz; i0++)
        {
            dg::blas1::copy( init2d, tempP);
            dg::blas1::copy( init2d, tempM);
            unsigned rep = r*m_Nz + i0;
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
    dg::split( fpe, m_temp, m_g.get());
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
            dg::blas1::pointwiseDot( m_right, m_h, m_ghostP);
            dg::blas1::axpby( 1., m_ghostP, 1., m_f[i0], m_ghostP);
        }
        //interlay ghostcells with periodic cells: L*g + (1-L)*fpe
        dg::blas1::axpby( 1., m_ghostP, -1., m_temp[i0], m_ghostP);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostP, 1., m_temp[i0]);
    }
}

template< class G, class I, class container>
void Fieldaligned<G, I, container>::eMinus( enum whichMatrix which, const container& f, container& fme)
{
    dg::split( f, m_f, m_g.get());
    dg::split( fme, m_temp, m_g.get());
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
            dg::blas1::pointwiseDot( m_left, m_h, m_ghostM);
            dg::blas1::axpby( -1., m_ghostM, 1., m_f[i0], m_ghostM);
        }
        //interlay ghostcells with periodic cells: L*g + (1-L)*fme
        dg::blas1::axpby( 1., m_ghostM, -1., m_temp[i0], m_ghostM);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostM, 1., m_temp[i0]);
    }
}


///@endcond


}//namespace geo
}//namespace dg
