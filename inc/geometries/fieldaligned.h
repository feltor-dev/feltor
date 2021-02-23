#pragma once
#include <cmath>
#include <array>
#include <cusp/csr_matrix.h>

#include "dg/backend/transpose.h"
#include "dg/blas.h"
#include "dg/topology/grid.h"
#include "dg/topology/interpolation.h"
#include "dg/topology/projection.h"
#include "dg/topology/functions.h"
#include "dg/topology/split_and_join.h"

#include "dg/topology/geometry.h"
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
    einsPlus = 0,   //!< plus interpolation in next plane
    einsPlusT = 1,  //!< transposed plus interpolation in previous plane
    einsMinus = 2,  //!< minus interpolation in previous plane
    einsMinusT = 3, //!< transposed minus interpolation in next plane
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
    DSFieldCylindrical( const dg::geo::CylindricalVectorLvl0& v,
            const dg::aGeometry2d& g, bool stay_in_box):
        m_v(v), m_g(g), m_in_box( stay_in_box){}
    void operator()( double t, const std::array<double,3>& y, std::array<double,3>& yp) const {
        double R = y[0], Z = y[1];
        if( m_in_box && !m_g->contains( R,Z) )
        {
            yp[0] = yp[1] = yp[2] = 0.;
            return;
        }
        double vz = m_v.z()(R, Z);
        yp[0] = m_v.x()(R, Z)/vz;
        yp[1] = m_v.y()(R, Z)/vz;
        yp[2] = 1./vz;
    }

    private:
    dg::geo::CylindricalVectorLvl0 m_v;
    dg::ClonePtr<dg::aGeometry2d> m_g;
    bool m_in_box;
};

struct DSField
{
    //z component of v may not vanish
    DSField( const dg::geo::CylindricalVectorLvl0& v, const dg::aGeometry2d& g, bool stay_in_box ): m_g(g), m_in_box( stay_in_box)
    {
        thrust::host_vector<double> v_zeta, v_eta;
        dg::pushForwardPerp( v.x(), v.y(), v_zeta, v_eta, g);
        thrust::host_vector<double> v_phi = dg::pullback( v.z(), g);
        dg::blas1::pointwiseDivide(v_zeta, v_phi, v_zeta);
        dg::blas1::pointwiseDivide(v_eta, v_phi, v_eta);
        dg::blas1::pointwiseDivide(1.,    v_phi, v_phi);
        dzetadphi_  = dg::forward_transform( v_zeta, g );
        detadphi_   = dg::forward_transform( v_eta, g );
        dsdphi_     = dg::forward_transform( v_phi, g );
    }
    //interpolate the vectors given in the constructor on the given point
    void operator()(double t, const std::array<double,3>& y, std::array<double,3>& yp) const
    {
        if( m_in_box && !m_g->contains( y[0],y[1]) )
        {
            yp[0] = yp[1] = yp[2] = 0.;
            return;
        }
        // else shift point into domain
        yp[0] = interpolate(dg::lspace, dzetadphi_, y[0], y[1], *m_g);
        yp[1] = interpolate(dg::lspace, detadphi_,  y[0], y[1], *m_g);
        yp[2] = interpolate(dg::lspace, dsdphi_,    y[0], y[1], *m_g);
    }
    private:
    thrust::host_vector<double> dzetadphi_, detadphi_, dsdphi_;
    dg::ClonePtr<dg::aGeometry2d> m_g;
    bool m_in_box;
};

template<class real_type>
real_type ds_norm( const std::array<real_type,3>& x0){
    return sqrt( x0[0]*x0[0] +x0[1]*x0[1] + x0[2]*x0[2]);
}

//used in constructor of Fieldaligned
template<class real_type>
void integrate_all_fieldlines2d( const dg::geo::CylindricalVectorLvl0& vec,
    const dg::aRealGeometry2d<real_type>& grid_field,
    const dg::aRealTopology2d<real_type>& grid_evaluate,
    std::array<thrust::host_vector<real_type>,3>& yp,
    std::array<thrust::host_vector<real_type>,3>& ym,
    thrust::host_vector<real_type>& yp2b,
    thrust::host_vector<real_type>& ym2b,
    thrust::host_vector<bool>& in_boxp,
    thrust::host_vector<bool>& in_boxm,
    real_type deltaPhi, real_type eps)
{
    //grid_field contains the global geometry for the field and the boundaries
    //grid_evaluate contains the points to actually integrate
    thrust::host_vector<real_type> tmp( dg::evaluate( dg::cooX2d, grid_evaluate));
    std::array<thrust::host_vector<real_type>,3> y{tmp,tmp,tmp};; //x
    y[1] = dg::evaluate( dg::cooY2d, grid_evaluate); //y
    y[2] = dg::evaluate( dg::zero, grid_evaluate); //s
    yp.fill(tmp); ym.fill(tmp); yp2b = ym2b = tmp; //allocate memory for output
    in_boxp.resize( tmp.size()), in_boxm.resize( tmp.size() );
    //construct field on high polynomial grid, then integrate it
    dg::geo::detail::DSField field( vec, grid_field, false);
    //field in case of cartesian grid
    dg::geo::detail::DSFieldCylindrical cyl_field(vec, grid_field, false);
    const unsigned size = grid_evaluate.size();
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
        in_boxp[i] = grid_field.contains( yp[0][i], yp[1][i]) ? true : false;
        in_boxm[i] = grid_field.contains( ym[0][i], ym[1][i]) ? true : false;
    }
    yp2b = yp[2], ym2b = ym[2];
    //Now integrate again but this time find the boundary distance
    field = dg::geo::detail::DSField( vec, grid_field,true);
    cyl_field = dg::geo::detail::DSFieldCylindrical(vec, grid_field, true);
    for( unsigned i=0; i<size; i++)
    {
        std::array<real_type,3> coords{y[0][i],y[1][i],y[2][i]}, coordsP, coordsM;
        if( false == in_boxp[i])
        {
            //x,y,s
            real_type phi1 = deltaPhi;
            if( dynamic_cast<const dg::CartesianGrid2d*>( &grid_field))
                dg::integrateERK( "Dormand-Prince-7-4-5", cyl_field, 0., coords, phi1, coordsP, 0., dg::pid_control, ds_norm, eps,1e-10); //integration
            else
                dg::integrateERK( "Dormand-Prince-7-4-5", field, 0., coords, phi1, coordsP, 0., dg::pid_control, ds_norm, eps,1e-10); //integration
            yp2b[i] = coordsP[2];
        }
        if( false == in_boxm[i])
        {
            real_type phi1 =  - deltaPhi;
            if( dynamic_cast<const dg::CartesianGrid2d*>( &grid_field))
                dg::integrateERK( "Dormand-Prince-7-4-5", cyl_field, 0., coords, phi1, coordsM, 0., dg::pid_control, ds_norm, eps,1e-10); //integration
            else
                dg::integrateERK( "Dormand-Prince-7-4-5", field, 0., coords, phi1, coordsM, 0., dg::pid_control, ds_norm, eps,1e-10); //integration
            ym2b[i] = coordsM[2];
        }
    }
}


}//namespace detail
///@endcond


    /*!@class hide_fieldaligned_physics_parameters
    * @tparam Limiter Class that can be evaluated on a 2d grid, returns 1 if there
        is a limiter and 0 if there isn't.
        If a field line crosses the limiter in the plane \f$ \phi=0\f$ then the limiter boundary conditions apply.
    * @param vec The vector field to integrate. Note that you can control how the boundary conditions are represented by changing vec outside the grid domain using e.g. the \c periodify function.
    * @param grid The grid on which to integrate fieldlines.
    * @param bcx This parameter is passed on to \c dg::create::interpolation(const thrust::host_vector<real_type>&,const thrust::host_vector<real_type>&,const aRealTopology2d<real_type>&,dg::bc,dg::bc) (see there for more details)
    * function and deterimens what happens when the endpoint of the fieldline integration leaves the domain boundaries of \c grid. Note that \c bcx and \c grid.bcx() have to be either both periodic or both not periodic.
    * @param bcy analogous to \c bcx, applies to y direction
    * @param limit Instance of the limiter class
        (Note that if \c grid.bcz()==dg::PER this parameter is ignored, Default is a limiter everywhere)
    */
    /*!@class hide_fieldaligned_numerics_parameters
    * @param eps Desired accuracy of the fieldline integrator
    * @param mx refinement factor in X of the fine grid relative to grid (Set to 1, if the x-component of \c vec vanishes, else as
    * high as possible, 10 is a good start)
    * @param my analogous to \c mx, applies to y direction
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
        unsigned mx=10, unsigned my=10,
        double deltaPhi = -1):
            Fieldaligned( dg::geo::createBHat(vec),
                grid, bcx, bcy, limit, eps, mx, my, deltaPhi)
    {
    }

    ///@brief Construct from a vector field and a grid
    ///@copydoc hide_fieldaligned_physics_parameters
    ///@copydoc hide_fieldaligned_numerics_parameters
    template <class Limiter>
    Fieldaligned(const dg::geo::CylindricalVectorLvl0& vec,
        const ProductGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned mx=10, unsigned my=10,
        double deltaPhi = -1);
    /**
    * @brief Perfect forward parameters to one of the constructors
    * @tparam Params deduced by the compiler
    * @param ps parameters forwarded to constructors
    */
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = Fieldaligned( std::forward<Params>( ps)...);
    }

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
        dg::split( global, m_f, *m_g);
        dg::blas1::axpby( scal_left,  m_f[0],      0, m_left);
        dg::blas1::axpby( scal_right, m_f[m_Nz-1], 0, m_right);
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
     * @attention It is recommended to use  \c mx>1 and \c my>1 when this function is used, else there might occur some unfavourable summation effects due to the repeated use of transformations especially for low perpendicular resolution.
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
     *  - Scale the transformations with \f$ u ( \pm (iN_z + j)h_z) \f$, where \c u is the given \c UnarayOp, \c i in [0..r] is the round index and \c j in [0..Nz] is the plane index.
     *  - %Sum all transformations with the same plane index \c j , where the minus transformations get the inverted index \f$ N_z - j\f$.
     *  - Shift the index by \f$ p_0\f$
     *  .
     * @tparam BinaryOp Binary Functor
     * @tparam UnaryOp Unary Functor
     * @param binary Functor to evaluate in x-y
     * @param unary Functor to evaluate in z
     * @param p0 The index of the plane to start
     * @param rounds The number of rounds \c r to follow a fieldline; can be zero, then the fieldlines are only followed within the current box ( no periodicity)
     * @note \c unary is evaluated such that \c p0 corresponds to z=0, p0+1 corresponds to z=hz, p0-1 to z=-hz, ...
     * @attention It is recommended to use  \c mx>1 and \c my>1 when this function is used, else there might occur some unfavourable summation effects due to the repeated use of transformations especially for low perpendicular resolution.
     *
     * @return Returns an instance of container
     */
    template< class BinaryOp, class UnaryOp>
    container evaluate( const BinaryOp& binary, const UnaryOp& unary, unsigned p0, unsigned rounds) const;

    /**
    * @brief Apply the interpolation to three-dimensional vectors
    *
    * computes \f$  y = 1^\pm \otimes \mathcal T x\f$
    * @param which specify what interpolation should be applied
    * @param in input
    * @param out output may not equal input
    */
    void operator()(enum whichMatrix which, const container& in, container& out);

    ///@brief Distance between the planes \f$ (s_{k}-s_{k-1}) \f$
    ///@return three-dimensional vector
    const container& hm()const {
        return m_hm;
    }
    ///@brief Distance between the planes \f$ (s_{k+1}-s_{k}) \f$
    ///@return three-dimensional vector
    const container& hp()const {
        return m_hp;
    }
    ///@brief Distance between the planes and the boundary \f$ (s_{k}-s_{b}^-) \f$
    ///@return three-dimensional vector
    const container& hbm()const {
        return m_hbm;
    }
    ///@brief Distance between the planes \f$ (s_b^+-s_{k}) \f$
    ///@return three-dimensional vector
    const container& hbp()const {
        return m_hbp;
    }
    ///@brief Mask minus, 1 if fieldline intersects wall in minus direction but not in plus direction, 0 else
    ///@return three-dimensional vector
    const container& bbm()const {
        return m_bbm;
    }
    ///@brief Mask both, 1 if fieldline intersects wall in plus direction and in minus direction, 0 else
    ///@return three-dimensional vector
    const container& bbo()const {
        return m_bbo;
    }
    ///@brief Mask plus, 1 if fieldline intersects wall in plus direction but not in minus direction, 0 else
    ///@return three-dimensional vector
    const container& bbp()const {
        return m_bbp;
    }
    ///Grid used for construction
    const ProductGeometry& grid()const{return *m_g;}

    /**
    * @brief Interpolate along fieldlines from a coarse to a fine grid in phi
    *
    * In this function we assume that the Fieldaligned object lives on the fine
    * grid and we now want to interpolate values from a vector living on a coarse grid along the fieldlines onto the fine grid.
    * Here, coarse and fine are with respect to the phi direction. The perpendicular directions need to have the same resolution in both input and output, i.e. there
    * is no interpolation in those directions.
    * @param grid_coarse The coarse grid (\c coarse_grid.Nz() must integer divide \c Nz from input grid) The x and y dimensions must be equal
    * @param coarse the coarse input vector
    *
    * @return the input interpolated onto the grid given in the constructor
    * @note the interpolation weights are taken in the phi distance not the s-distance, which makes the interpolation linear in phi
    */
    container interpolate_from_coarse_grid( const ProductGeometry& grid_coarse, const container& coarse);
    /**
    * @brief Integrate a 2d function on the fine grid \f[ \frac{1}{\Delta\varphi} \int_{-\Delta\varphi}^{\Delta\varphi}d \varphi w(\varphi) f(R(\varphi), Z(\varphi) \f]
    *
    * @param grid_coarse The coarse grid (\c coarse_grid.Nz() must integer divide \c Nz from input grid) The x and y dimensions must be equal
    * @param coarse the 2d input vector
    * @param out the integral (2d vector)
    */
    void integrate_between_coarse_grid( const ProductGeometry& grid_coarse, const container& coarse, container& out );
    private:
    void ePlus( enum whichMatrix which, const container& in, container& out);
    void eMinus(enum whichMatrix which, const container& in, container& out);
    IMatrix m_plus, m_minus, m_plusT, m_minusT; //2d interpolation matrices
    container m_hm, m_hp, m_hbm, m_hbp;         //3d size
    container m_bbm, m_bbp, m_bbo;  //3d size masks
    container m_hm2d, m_hp2d;       //2d size
    container m_left, m_right;      //perp_size
    container m_limiter;            //perp_size
    container m_ghostM, m_ghostP;   //perp_size
    unsigned m_Nz, m_perp_size;
    dg::bc m_bcx, m_bcy, m_bcz;
    std::vector<dg::View<const container>> m_f;
    std::vector<dg::View< container>> m_temp;
    dg::ClonePtr<ProductGeometry> m_g;
    template<class Geometry>
    void assign3dfrom2d( const thrust::host_vector<double>& in2d, container& out, const Geometry& grid)
    {
        dg::split( out, m_temp, grid); //3d vector
        container tmp2d;
        dg::assign( in2d, tmp2d);
        for( unsigned i=0; i<m_Nz; i++)
            dg::blas1::copy( tmp2d, m_temp[i]);
    }
};

///@cond

////////////////////////////////////DEFINITIONS///////////////////////////////////////
template<class Geometry, class IMatrix, class container>
template <class Limiter>
Fieldaligned<Geometry, IMatrix, container>::Fieldaligned(
    const dg::geo::CylindricalVectorLvl0& vec, const Geometry& grid,
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
    m_perp_size = grid_coarse->size();
    dg::assign( dg::pullback(limit, *grid_coarse), m_limiter);
    dg::assign( dg::evaluate(zero, *grid_coarse), m_left);
    m_ghostM = m_ghostP = m_right = m_left;
    ///%%%%%%%%%%Set starting points and integrate field lines%%%%%%%%%%%//
#ifdef DG_BENCHMARK
    dg::Timer t;
    t.tic();
#endif //DG_BENCHMARK
    std::array<thrust::host_vector<double>,3> yp_coarse, ym_coarse, yp, ym;
    dg::ClonePtr<dg::aGeometry2d> grid_magnetic = grid_coarse;//INTEGRATE HIGH ORDER GRID
    grid_magnetic->set( 7, grid_magnetic->Nx(), grid_magnetic->Ny());
    dg::Grid2d grid_fine( *grid_coarse );//FINE GRID
    grid_fine.multiplyCellNumbers((double)mx, (double)my);
#ifdef DG_BENCHMARK
    t.toc();
    std::cout << "# DS: High order grid gen  took: "<<t.diff()<<"\n";
    t.tic();
#endif //DG_BENCHMARK
    thrust::host_vector<bool> in_boxp, in_boxm;
    thrust::host_vector<double> hbp, hbm;
    detail::integrate_all_fieldlines2d( vec, *grid_magnetic, *grid_coarse,
            yp_coarse, ym_coarse, hbp, hbm, in_boxp, in_boxm, deltaPhi, eps);
    dg::IHMatrix interpolate = dg::create::interpolation( grid_fine, *grid_coarse);  //INTERPOLATE TO FINE GRID
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
    dg::IHMatrix plusFine  = dg::create::interpolation( yp[0], yp[1], *grid_coarse, bcx, bcy), plus, plusT;
    dg::IHMatrix minusFine = dg::create::interpolation( ym[0], ym[1], *grid_coarse, bcx, bcy), minus, minusT;
    if( mx == my && mx == 1)
    {
        plus = plusFine;
        minus = minusFine;
    }
    else
    {
        dg::IHMatrix projection = dg::create::projection( *grid_coarse, grid_fine);
        cusp::multiply( projection, plusFine, plus);
        cusp::multiply( projection, minusFine, minus);
    }
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
    dg::assign( dg::evaluate( dg::zero, grid), m_hm);
    m_temp  = dg::split( m_hm, grid); //3d vector
    m_f     = dg::split( (const container&)m_hm, grid);
    m_hbp = m_hbm = m_hp = m_hm;
    dg::assign( yp_coarse[2], m_hp2d); //2d vector
    dg::assign( ym_coarse[2], m_hm2d); //2d vector
    assign3dfrom2d( hbp, m_hbp, grid);
    assign3dfrom2d( hbm, m_hbm, grid);
    assign3dfrom2d( yp_coarse[2], m_hp, grid);
    assign3dfrom2d( ym_coarse[2], m_hm, grid);
    dg::blas1::scal( m_hm2d, -1.);
    dg::blas1::scal( m_hbm, -1.);
    dg::blas1::scal( m_hm, -1.);

    ///%%%%%%%%%%%%%%%%%%%%create mask vectors %%%%%%%%%%%%%%%%%%%//
    thrust::host_vector<double> bbm( in_boxp.size(),0.), bbo(bbm), bbp(bbm);
    for( unsigned i=0; i<in_boxp.size(); i++)
    {
        if( !in_boxp[i] && !in_boxm[i])
            bbo[i] = 1.;
        else if( !in_boxp[i] && in_boxm[i])
            bbp[i] = 1.;
        else if( in_boxp[i] && !in_boxm[i])
            bbm[i] = 1.;
        // else all are 0
    }
    m_bbm = m_bbo = m_bbp = m_hm;
    assign3dfrom2d( bbm, m_bbm, grid);
    assign3dfrom2d( bbo, m_bbo, grid);
    assign3dfrom2d( bbp, m_bbp, grid);
}

template<class G, class I, class container>
template< class BinaryOp, class UnaryOp>
container Fieldaligned<G, I,container>::evaluate( const BinaryOp& binary, const UnaryOp& unary, unsigned p0, unsigned rounds) const
{
    //idea: simply apply I+/I- enough times on the init2d vector to get the result in each plane
    //unary function is always such that the p0 plane is at x=0
    assert( p0 < m_g->Nz());
    const dg::ClonePtr<aGeometry2d> g2d = m_g->perp_grid();
    container init2d = dg::pullback( binary, *g2d);
    container zero2d = dg::evaluate( dg::zero, *g2d);

    container temp(init2d), tempP(init2d), tempM(init2d);
    container vec3d = dg::evaluate( dg::zero, *m_g);
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
                //!!! The value of f at the plus plane is I^- of the current plane
                dg::blas2::symv( m_minus, tempP, temp);
                temp.swap( tempP);
                //!!! The value of f at the minus plane is I^+ of the current plane
                dg::blas2::symv( m_plus, tempM, temp);
                temp.swap( tempM);
            }
            dg::blas1::scal( tempP, unary(  (double)rep*m_g->hz() ) );
            dg::blas1::scal( tempM, unary( -(double)rep*m_g->hz() ) );
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
container Fieldaligned<G, I,container>::interpolate_from_coarse_grid( const G& grid, const container& in)
{
    //I think we need grid as input to split input vector and we need to interpret
    //the grid nodes as node centered not cell-centered!
    //idea: apply I+/I- cphi - 1 times in each direction and then apply interpolation formula
    assert( m_g->Nz() % grid.Nz() == 0);
    unsigned Nz_coarse = grid.Nz(), Nz = m_g->Nz();
    unsigned cphi = Nz / Nz_coarse;

    container out = dg::evaluate( dg::zero, *m_g);
    container helper = dg::evaluate( dg::zero, *m_g);
    dg::split( helper, m_temp, *m_g);
    std::vector<dg::View< container>> out_split = dg::split( out, *m_g);
    std::vector<dg::View< const container>> in_split = dg::split( in, grid);
    for ( int i=0; i<(int)Nz_coarse; i++)
    {
        //1. copy input vector to appropriate place in output
        dg::blas1::copy( in_split[i], out_split[i*cphi]);
        dg::blas1::copy( in_split[i], m_temp[i*cphi]);
    }
    //Step 1 needs to finish so that m_temp contains values everywhere
    //2. Now apply plus and minus T to fill in the rest
    for ( int i=0; i<(int)Nz_coarse; i++)
    {
        for( int j=1; j<(int)cphi; j++)
        {
            //!!! The value of f at the plus plane is I^- of the current plane
            dg::blas2::symv( m_minus, out_split[i*cphi+j-1], out_split[i*cphi+j]);
            //!!! The value of f at the minus plane is I^+ of the current plane
            dg::blas2::symv( m_plus, m_temp[(i*cphi+cphi+1-j)%Nz], m_temp[i*cphi+cphi-j]);
        }
    }
    //3. Now add up with appropriate weights
    for( int i=0; i<(int)Nz_coarse; i++)
        for( int j=1; j<(int)cphi; j++)
        {
            double alpha = (double)(cphi-j)/(double)cphi;
            double beta = (double)j/(double)cphi;
            dg::blas1::axpby( alpha, out_split[i*cphi+j], beta, m_temp[i*cphi+j], out_split[i*cphi+j]);
        }
    return out;
}
template<class G, class I, class container>
void Fieldaligned<G, I,container>::integrate_between_coarse_grid( const G& grid, const container& in, container& out)
{
    assert( m_g->Nz() % grid.Nz() == 0);
    unsigned Nz_coarse = grid.Nz(), Nz = m_g->Nz();
    unsigned cphi = Nz / Nz_coarse;

    out = in;
    container helperP( in), helperM(in), tempP(in), tempM(in);

    //1. Apply plus and minus T and sum up
    for( int j=1; j<(int)cphi; j++)
    {
        //!!! The value of f at the plus plane is I^- of the current plane
        dg::blas2::symv( m_minus, helperP, tempP);
        dg::blas1::axpby( (double)(cphi-j)/(double)cphi, tempP, 1., out  );
        helperP.swap(tempP);
        //!!! The value of f at the minus plane is I^+ of the current plane
        dg::blas2::symv( m_plus, helperM, tempM);
        dg::blas1::axpby( (double)(cphi-j)/(double)cphi, tempM, 1., out  );
        helperM.swap(tempM);
    }
    dg::blas1::scal( out, 1./(double)cphi);
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
    dg::split( f, m_f, *m_g);
    dg::split( fpe, m_temp, *m_g);
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
            dg::blas1::pointwiseDot( m_right, m_hp2d, m_ghostP);
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
    dg::split( f, m_f, *m_g);
    dg::split( fme, m_temp, *m_g);
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
            dg::blas1::pointwiseDot( m_left, m_hm2d, m_ghostM);
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
