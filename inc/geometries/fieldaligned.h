#pragma once
#include <cmath>
#include <array>

#include "dg/algorithm.h"
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
    einsPlusT, //!< transposed plus interpolation in previous plane
    einsMinus, //!< minus interpolation in previous plane
    einsMinusT,//!< transposed minus interpolation in next plane
    zeroPlus,  //!< plus interpolation in the current plane
    zeroMinus, //!< minus interpolation in the current plane
    zeroPlusT, //!< transposed plus interpolation in the current plane
    zeroMinusT, //!< transposed minus interpolation in the current plane
    zeroForw  //!< from dg to transformed coordinates
};

///@brief Full Limiter means there is a limiter everywhere
///@ingroup fieldaligned
typedef ONE FullLimiter;

///@brief No Limiter
///@ingroup fieldaligned
typedef ZERO NoLimiter;
///@cond
namespace detail{

static inline void parse_method( std::string method, std::string& i, std::string& p, std::string& f)
{
    f = "dg";
    if( method == "dg")                 i = "dg",       p = "dg";
    else if( method == "linear")        i = "linear",   p = "dg";
    else if( method == "cubic")         i = "cubic",    p = "dg";
    else if( method == "nearest")       i = "nearest",  p = "dg";
    else if( method == "dg-nearest")      i = "dg",      p = "nearest";
    else if( method == "linear-nearest")  i = "linear",  p = "nearest";
    else if( method == "cubic-nearest")   i = "cubic",   p = "nearest";
    else if( method == "nearest-nearest") i = "nearest",  p = "nearest";
    else if( method == "dg-linear")      i = "dg",       p = "linear";
    else if( method == "linear-linear")  i = "linear",   p = "linear";
    else if( method == "cubic-linear")   i = "cubic",    p = "linear";
    else if( method == "nearest-linear") i = "nearest",  p = "linear";
    else if( method == "dg-equi")       i = "dg",       p = "dg", f = "equi";
    else if( method == "linear-equi")   i = "linear",   p = "dg", f = "equi";
    else if( method == "cubic-equi")    i = "cubic",    p = "dg", f = "equi";
    else if( method == "nearest-equi")  i = "nearest",  p = "dg", f = "equi";
    else if( method == "dg-equi-nearest")      i = "dg",      p = "nearest", f = "equi";
    else if( method == "linear-equi-nearest")  i = "linear",  p = "nearest", f = "equi";
    else if( method == "cubic-equi-nearest")   i = "cubic",   p = "nearest", f = "equi";
    else if( method == "nearest-equi-nearest") i = "nearest", p = "nearest", f = "equi";
    else if( method == "dg-equi-linear")      i = "dg",       p = "linear", f = "equi";
    else if( method == "linear-equi-linear")  i = "linear",   p = "linear", f = "equi";
    else if( method == "cubic-equi-linear")   i = "cubic",    p = "linear", f = "equi";
    else if( method == "nearest-equi-linear") i = "nearest",  p = "linear", f = "equi";
    else
        throw Error( Message(_ping_) << "The method "<< method << " is not recognized\n");
}

struct DSFieldCylindrical3
{
    DSFieldCylindrical3( const dg::geo::CylindricalVectorLvl0& v): m_v(v){}
    void operator()( double, const std::array<double,3>& y,
            std::array<double,3>& yp) const {
        double R = y[0], Z = y[1];
        double vz = m_v.z()(R, Z);
        yp[0] = m_v.x()(R, Z)/vz;
        yp[1] = m_v.y()(R, Z)/vz;
        yp[2] = 1./vz;
    }
    private:
    dg::geo::CylindricalVectorLvl0 m_v;
};

struct DSFieldCylindrical4
{
    DSFieldCylindrical4( const dg::geo::CylindricalVectorLvl1& v): m_v(v){}
    void operator()( double, const std::array<double,3>& y,
            std::array<double,3>& yp) const {
        double R = y[0], Z = y[1];
        double vx = m_v.x()(R,Z);
        double vy = m_v.y()(R,Z);
        double vz = m_v.z()(R,Z);
        double divvvz = m_v.divvvz()(R,Z);
        yp[0] = vx/vz;
        yp[1] = vy/vz;
        yp[2] = divvvz*y[2];
    }

    private:
    dg::geo::CylindricalVectorLvl1 m_v;
};

struct DSField
{
    DSField() = default;
    //z component of v may not vanish
    DSField( const dg::geo::CylindricalVectorLvl1& v,
            const dg::aGeometry2d& g ):
        m_g(g)
    {
        dg::HVec v_zeta, v_eta;
        dg::pushForwardPerp( v.x(), v.y(), v_zeta, v_eta, g);
        dg::HVec vx = dg::pullback( v.x(), g);
        dg::HVec vy = dg::pullback( v.y(), g);
        dg::HVec vz = dg::pullback( v.z(), g);
        dg::HVec divvvz = dg::pullback( v.divvvz(), g);
        dg::blas1::pointwiseDivide(v_zeta,  vz, v_zeta);
        dg::blas1::pointwiseDivide(v_eta,   vz, v_eta);
        dzetadphi_  = dg::forward_transform( v_zeta, g );
        detadphi_   = dg::forward_transform( v_eta, g );
        dvdphi_     = dg::forward_transform( divvvz, g );
    }
    //interpolate the vectors given in the constructor on the given point
    void operator()(double, const std::array<double,3>& y, std::array<double,3>& yp) const
    {
        // shift point into domain
        yp[0] = interpolate(dg::lspace, dzetadphi_, y[0], y[1], *m_g);
        yp[1] = interpolate(dg::lspace, detadphi_,  y[0], y[1], *m_g);
        yp[2] = interpolate(dg::lspace, dvdphi_,    y[0], y[1], *m_g)*y[2];
    }
    private:
    thrust::host_vector<double> dzetadphi_, detadphi_, dvdphi_;
    dg::ClonePtr<dg::aGeometry2d> m_g;
};

//used in constructor of Fieldaligned
template<class real_type>
void integrate_all_fieldlines2d( const dg::geo::CylindricalVectorLvl1& vec,
    const dg::aRealGeometry2d<real_type>& grid_field,
    const dg::aRealTopology2d<real_type>& grid_evaluate,
    std::array<thrust::host_vector<real_type>,3>& yp,
    const thrust::host_vector<double>& vol0,
    thrust::host_vector<real_type>& yp2b,
    thrust::host_vector<bool>& in_boxp,
    real_type deltaPhi, real_type eps)
{
    //grid_field contains the global geometry for the field and the boundaries
    //grid_evaluate contains the points to actually integrate
    std::array<thrust::host_vector<real_type>,3> y{
        dg::evaluate( dg::cooX2d, grid_evaluate),
        dg::evaluate( dg::cooY2d, grid_evaluate),
        vol0
    };
    yp.fill(dg::evaluate( dg::zero, grid_evaluate));
    //construct field on high polynomial grid, then integrate it
    dg::geo::detail::DSField field;
    if( !dynamic_cast<const dg::CartesianGrid2d*>( &grid_field))
        field = dg::geo::detail::DSField( vec, grid_field);

    //field in case of cartesian grid
    dg::geo::detail::DSFieldCylindrical4 cyl_field(vec);
    const unsigned size = grid_evaluate.size();
    dg::Adaptive<dg::ERKStep<std::array<real_type,3>>> adapt(
            "Dormand-Prince-7-4-5", std::array<real_type,3>{0,0,0});
    dg::AdaptiveTimeloop< std::array<real_type,3>> odeint;
    if( dynamic_cast<const dg::CartesianGrid2d*>( &grid_field))
        odeint = dg::AdaptiveTimeloop<std::array<real_type,3>>( adapt,
                cyl_field, dg::pid_control, dg::fast_l2norm, eps, 1e-10);
    else
        odeint = dg::AdaptiveTimeloop<std::array<real_type,3>>( adapt,
                field, dg::pid_control, dg::fast_l2norm, eps, 1e-10);

    for( unsigned i=0; i<size; i++)
    {
        std::array<real_type,3> coords{y[0][i],y[1][i],y[2][i]}, coordsP;
        //x,y,s
        real_type phi1 = deltaPhi;
        odeint.set_dt( deltaPhi/2.);
        odeint.integrate( 0, coords, phi1, coordsP);
        yp[0][i] = coordsP[0], yp[1][i] = coordsP[1], yp[2][i] = coordsP[2];
    }
    yp2b.assign( grid_evaluate.size(), deltaPhi); //allocate memory for output
    in_boxp.resize( yp2b.size());
    //Now integrate again but this time find the boundary distance
    for( unsigned i=0; i<size; i++)
    {
        std::array<real_type,3> coords{y[0][i],y[1][i],y[2][i]}, coordsP;
        in_boxp[i] = grid_field.contains( std::array{yp[0][i], yp[1][i]}) ? true : false;
        if( false == in_boxp[i])
        {
            //x,y,s
            real_type phi1 = deltaPhi;
            odeint.integrate_in_domain( 0., coords, phi1, coordsP, 0., (const
                        dg::aRealTopology2d<real_type>&)grid_field, eps);
            yp2b[i] = phi1;
        }
    }
}


}//namespace detail
///@endcond

    /*!@class hide_fieldaligned_physics_parameters
    * @tparam Limiter Class that can be evaluated on a 2d grid, returns 1 if there
        is a limiter and 0 if there isn't.
        If a field line crosses the limiter in the plane \f$ \phi=0\f$ then the limiter boundary conditions apply.
    * @param vec The vector field to integrate (contravariant components). In
    * case the type is a \c dg::geo::TokamakMagneticField then \c
    * dg::geo::createBHat is called. Note that you can control how the boundary
    * conditions are represented by changing vec outside the grid domain using
    * e.g. the \c periodify function.
    * @param grid The grid on which to integrate fieldlines.
    * @param bcx This parameter is passed on to \c dg::create::interpolation(const thrust::host_vector<real_type>&,const thrust::host_vector<real_type>&,const aRealTopology2d<real_type>&,dg::bc,dg::bc,std::string) (see there for more details)
    * function and deterimens what happens when the endpoint of the fieldline integration leaves the domain boundaries of \c grid. Note that \c bcx and \c grid.bcx() have to be either both periodic or both not periodic.
    * @param bcy analogous to \c bcx, applies to y direction
    * @param limit Instance of the limiter class
        (Note that if \c grid.bcz()==dg::PER this parameter is ignored, Default is a limiter everywhere)
    */
    /*!@class hide_fieldaligned_numerics_parameters
    * @param eps Desired accuracy of the fieldline integrator
    * @param mx refinement factor in X of the fine grid relative to grid (Set to 1, if the x-component of \c vec vanishes, else as
    * high as possible, 12 is a good start) If the projection method (parsed from \c interpolation_method) is not "dg" then \c mx must be a multiple of \c nx
    * @param my analogous to \c mx, applies to y direction
    * @param deltaPhi The angular distance that the fieldline-integrator will
    * integrate. Per default this is the distance between planes, which is
    * chosen automatically if you set it <=0, i.e. if deltaPhi <=0 then it will
    * be overwritten to deltaPhi = grid.hz().  Sometimes however, you may want
    * to set it to a different value from \c grid.hz() for example for 2d problems
    * or for a staggered grid.
    * @note  deltaPhi influences the interpolation matrices and the parallel
    * modulation in the evaluate() member function.
    * @note If there is a limiter, the boundary condition on the first/last
    * plane is set
        by the \c grid.bcz() variable and can be changed by the set_boundaries
        function.  If there is no limiter, the boundary condition is periodic.
     * @param interpolation_method parsed according to
     *
     * |interpolation_method | interpolation | projection|
     * |---------------------|---------------|-----------|
     * |"dg"                 |    "dg"       | "dg"      |
     * |"linear"             |    "linear"   | "dg"      |
     * |"cubic"              |    "cubic"    | "dg"      |
     * |"nearest"            |    "nearest"  | "dg"      |
     * |"linear-nearest" (##)   | "linear"      | "nearest" |
     * |"cubic-nearest"      | "cubic"       | "nearest" |
     * |"nearest-nearest"    | "nearest"     | "nearest" |
     * |"dg-linear"          | "dg"          | "linear"  |
     * |"linear-linear"      | "linear"      | "linear"  |
     * |"cubic-linear"       | "cubic"       | "linear"  |
     * |"nearest-linear"     | "nearest"     | "linear"  |
     *
     * (##) Use "linear-nearest" if in doubt.
     * The table yields one parameter passed to \c create::interpolation
     * (from the given grid to the fine grid) and one parameter
     * to \c create::projection (from the fine grid to the given grid)
     *
     * @param benchmark If true write construction timings to std::cout
    */
//////////////////////////////FieldalignedCLASS////////////////////////////////////////////
/**
* @brief Create and manage interpolation matrices from fieldline integration
*
* @ingroup fieldaligned
* @snippet ds_t.cpp doxygen
* @tparam ProductGeometry must be either \c dg::aProductGeometry3d or \c dg::aProductMPIGeometry3d or any derivative
* @tparam IMatrix The type of the interpolation matrix
    - \c dg::IHMatrix, or \c dg::IDMatrix, \c dg::MIHMatrix, or \c dg::MIDMatrix
* @tparam container The container-class on which the interpolation matrix operates on
    - \c dg::HVec, or \c dg::DVec, \c dg::MHVec, or \c dg::MDVec
* @sa The pdf <a href="https://www.overleaf.com/read/jjvstccqzcjv" target="_blank">parallel derivative</a> writeup
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
        unsigned mx=12, unsigned my=12,
        double deltaPhi = -1,
        std::string interpolation_method = "linear-nearest",
        bool benchmark=true
        ):
            Fieldaligned( dg::geo::createBHat(vec),
                grid, bcx, bcy, limit, eps, mx, my, deltaPhi, interpolation_method, benchmark)
    {
    }

    ///@brief Construct from a vector field and a grid
    ///@copydoc hide_fieldaligned_physics_parameters
    ///@copydoc hide_fieldaligned_numerics_parameters
    template <class Limiter>
    Fieldaligned(const dg::geo::CylindricalVectorLvl1& vec,
        const ProductGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned mx=12, unsigned my=12,
        double deltaPhi = -1,
        std::string interpolation_method = "linear-nearest",
        bool benchmark=true
        );
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
    * @brief Apply the interpolation to three-dimensional vectors
    *
    * computes \f$  y = 1^\pm \otimes \mathcal T x\f$
    * @param which specify what interpolation should be applied
    * @param in input
    * @param out output may not equal input
    */
    void operator()(enum whichMatrix which, const container& in, container& out);

    double deltaPhi() const{return m_deltaPhi;}
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
    ///@brief Volume form (including weights) \f$ \sqrt{G}_{k} \f$
    ///@return three-dimensional vector
    const container& sqrtG()const {
        return m_G;
    }
    ///@brief Volume form on minus plane (including weights) \f$ \sqrt{G}_{k-1} \f$
    ///@return three-dimensional vector
    const container& sqrtGm()const {
        return m_Gm;
    }
    ///@brief Volume form on plus plane (including weights) \f$ \sqrt{G}_{k+1} \f$
    ///@return three-dimensional vector
    const container& sqrtGp()const {
        return m_Gp;
    }
    ///@brief The contravariant phi component (3rd component) of the vector field \f$ v^\varphi\f$
    ///@return three-dimensional vector
    const container& bphi()const {
        return m_bphi;
    }
    ///@brief bphi on minus plane
    ///@return three-dimensional vector
    const container& bphiM()const {
        return m_bphiM;
    }
    ///@brief bphi on plus plane
    ///@return three-dimensional vector
    const container& bphiP()const {
        return m_bphiP;
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
    * @param grid_coarse The coarse grid (\c coarse_grid.Nz() must integer
    * divide \c Nz from input grid). The x and y dimensions must be equal to
    * the input grid.
    * @param coarse the 2d input vector
    * @param out the integral (2d vector, may alias coarse)
    */
    void integrate_between_coarse_grid( const ProductGeometry& grid_coarse, const container& coarse, container& out );


    /**
     * @brief %Evaluate a 2d functor and transform to all planes along the fieldline

     *
     * The algorithm does the equivalent of the following:
     *  - %Evaluate the given \c BinaryOp on a 2d plane
     *  - Apply the plus and minus transformation each \f$ r N_z\f$ times where
     *  \f$ N_z\f$ is the number of planes in the global 3d grid and \f$ r\f$ is the number of rounds.
     *  - Scale the transformations with \f$ u ( \pm (iN_z + j)\Delta\varphi) \f$,
     *  where \c u is the given \c UnarayOp, \c i in [0..r] is the round index and \c j in [0.  Nz] is the plane index
     *  and \f$\Delta\varphi\f$ is the angular distance given in the constructor
     *  (can be different from the actual grid distance hz!).
     *  - %Sum all transformations with the same plane index \c j , where the minus transformations get the inverted index \f$ N_z - j\f$.
     *  - Shift the index by \f$ p_0\f$
     *  .
     * @attention This version of the algorithm is less exact but much faster than
     * dg::geo::fieldaligned_evaluate
     * @tparam BinaryOp Binary Functor
     * @tparam UnaryOp Unary Functor
     * @param binary Functor to evaluate in x-y
     * @param unary Functor to evaluate in z
     * @note \c unary is evaluated such that \c p0 corresponds to z=0, p0+1 corresponds to z=hz, p0-1 to z=-hz, ...
     * @param p0 The index of the plane to start
     * @param rounds The number of rounds \c r to follow a fieldline; can be zero, then the fieldlines are only followed within the current box ( no periodicity)
     * @attention It is recommended to use  \c mx>1 and \c my>1 when this function is used, else there might occur some unfavourable summation effects due to the repeated use of transformations especially for low perpendicular resolution.
     *
     * @return 3d vector
     */
    template< class BinaryOp, class UnaryOp>
    container evaluate( BinaryOp binary, UnaryOp unary,
            unsigned p0, unsigned rounds) const;

    ///Return the \c interpolation_method string given in the constructor
    std::string method() const{return m_interpolation_method;}

    private:
    void ePlus( enum whichMatrix which, const container& in, container& out);
    void eMinus(enum whichMatrix which, const container& in, container& out);
    void zero( enum whichMatrix which, const container& in, container& out);
    IMatrix m_plus, m_zero, m_minus, m_plusT, m_minusT; //2d interpolation matrices
    container m_hbm, m_hbp;         //3d size
    container m_G, m_Gm, m_Gp; // 3d size
    container m_bphi, m_bphiM, m_bphiP; // 3d size
    container m_bbm, m_bbp, m_bbo;  //3d size masks

    container m_left, m_right;      //perp_size
    container m_limiter;            //perp_size
    container m_ghostM, m_ghostP;   //perp_size
    unsigned m_Nz, m_perp_size;
    dg::bc m_bcx, m_bcy, m_bcz;
    std::vector<dg::View<const container>> m_f;
    std::vector<dg::View< container>> m_temp;
    dg::ClonePtr<ProductGeometry> m_g;
    dg::InverseKroneckerTriDiagonal2d<container> m_inv_linear;
    double m_deltaPhi;
    std::string m_interpolation_method;

    bool m_have_adjoint = false;
    void updateAdjoint( )
    {
        m_plusT = m_plus.transpose();
        m_minusT = m_minus.transpose();
        m_have_adjoint = true;
    }
};

///@cond

////////////////////////////////////DEFINITIONS///////////////////////////////////////
template<class Geometry, class IMatrix, class container>
template <class Limiter>
Fieldaligned<Geometry, IMatrix, container>::Fieldaligned(
    const dg::geo::CylindricalVectorLvl1& vec,
    const Geometry& grid,
    dg::bc bcx, dg::bc bcy, Limiter limit, double eps,
    unsigned mx, unsigned my, double deltaPhi, std::string interpolation_method, bool benchmark) :
        m_g(grid),
        m_interpolation_method(interpolation_method)
{

    std::string inter_m, project_m, fine_m;
    detail::parse_method( interpolation_method, inter_m, project_m, fine_m);
    if( benchmark) std::cout << "# Interpolation method: \""<<inter_m << "\" projection method: \""<<project_m<<"\" fine grid \""<<fine_m<<"\"\n";
    ///Let us check boundary conditions:
    if( (grid.bcx() == PER && bcx != PER) || (grid.bcx() != PER && bcx == PER) )
        throw( dg::Error(dg::Message(_ping_)<<"Fieldaligned: Got conflicting periodicity in x. The grid says "<<bc2str(grid.bcx())<<" while the parameter says "<<bc2str(bcx)));
    if( (grid.bcy() == PER && bcy != PER) || (grid.bcy() != PER && bcy == PER) )
        throw( dg::Error(dg::Message(_ping_)<<"Fieldaligned: Got conflicting boundary conditions in y. The grid says "<<bc2str(grid.bcy())<<" while the parameter says "<<bc2str(bcy)));
    m_Nz=grid.Nz(), m_bcx = bcx, m_bcy = bcy, m_bcz=grid.bcz();
    if( deltaPhi <=0) deltaPhi = grid.hz();
    ///%%%%%%%%%%%%%%%%%%%%%Setup grids%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    //  grid_trafo -> grid_equi -> grid_fine -> grid_equi -> grid_trafo
    dg::Timer t;
    if( benchmark) t.tic();
    dg::ClonePtr<dg::aGeometry2d> grid_transform( grid.perp_grid()) ;
    // We do not need metric of grid_equidist or or grid_fine
    dg::RealGrid2d<double> grid_equidist( *grid_transform) ;
    dg::RealGrid2d<double> grid_fine( *grid_transform);
    grid_equidist.set( 1, grid.gx().size(), grid.gy().size());
    dg::ClonePtr<dg::aGeometry2d> grid_magnetic = grid_transform;//INTEGRATE HIGH ORDER GRID
    grid_magnetic->set( grid_transform->n() < 3 ? 4 : 7, grid_magnetic->Nx(), grid_magnetic->Ny());
    // For project method "const" we round up to the nearest multiple of n
    if( project_m != "dg" && fine_m == "dg")
    {
        unsigned rx = mx % grid.nx(), ry = my % grid.ny();
        if( 0 != rx || 0 != ry)
        {
            std::cerr << "#Warning: for projection method \"const\" mx and my must be multiples of nx and ny! Rounding up for you ...\n";
            mx = mx + grid.nx() - rx;
            my = my + grid.ny() - ry;
        }
    }
    if( fine_m == "equi")
        grid_fine = grid_equidist;
    grid_fine.multiplyCellNumbers((double)mx, (double)my);
    if( benchmark)
    {
        t.toc();
        std::cout << "# DS: High order grid gen  took: "<<t.diff()<<"\n";
        t.tic();
    }
    ///%%%%%%%%%%Set starting points and integrate field lines%%%%%%%%%%%//
    std::array<thrust::host_vector<double>,3> yp_trafo, ym_trafo, yp, ym;
    thrust::host_vector<bool> in_boxp, in_boxm;
    thrust::host_vector<double> hbp, hbm;
    thrust::host_vector<double> vol = dg::tensor::volume(grid.metric()), vol2d0;
    auto vol2d = dg::split( vol, grid);
    dg::assign( vol2d[0], vol2d0);
    detail::integrate_all_fieldlines2d( vec, *grid_magnetic, *grid_transform,
            yp_trafo, vol2d0, hbp, in_boxp, deltaPhi, eps);
    detail::integrate_all_fieldlines2d( vec, *grid_magnetic, *grid_transform,
            ym_trafo, vol2d0, hbm, in_boxm, -deltaPhi, eps);
    dg::HVec Xf = dg::evaluate(  dg::cooX2d, grid_fine);
    dg::HVec Yf = dg::evaluate(  dg::cooY2d, grid_fine);
    {
    dg::IHMatrix interpolate = dg::create::interpolation( Xf, Yf,
            *grid_transform, dg::NEU, dg::NEU, grid_transform->n() < 3 ? "cubic" : "dg");
    yp.fill(dg::evaluate( dg::zero, grid_fine));
    ym = yp;
    for( int i=0; i<2; i++) //only R and Z get interpolated
    {
        dg::blas2::symv( interpolate, yp_trafo[i], yp[i]);
        dg::blas2::symv( interpolate, ym_trafo[i], ym[i]);
    }
    } // release memory for interpolate matrix
    if( benchmark)
    {
        t.toc();
        std::cout << "# DS: Computing all points took: "<<t.diff()<<"\n";
        t.tic();
    }
    ///%%%%%%%%%%%%%%%%Create interpolation and projection%%%%%%%%%%%%%%//
    { // free memory after use
    dg::IHMatrix fine, projection, multi, temp;
    if( project_m == "dg")
        projection = dg::create::projection( *grid_transform, grid_fine);
    else
    {
        projection = dg::create::inv_backproject( *grid_transform)*
            dg::create::projection( grid_equidist, grid_fine, project_m);
    }
    std::array<dg::HVec*,3> xcomp{ &yp[0], &Xf, &ym[0]};
    std::array<dg::HVec*,3> ycomp{ &yp[1], &Yf, &ym[1]};
    std::array<IMatrix*,3> result{ &m_plus, &m_zero, &m_minus};

    for( unsigned u=0; u<3; u++)
    {
        if( inter_m == "dg")
        {
            *result[u] = projection*dg::create::interpolation( *xcomp[u], *ycomp[u],
                *grid_transform, bcx, bcy, "dg");
        }
        else
        {
            *result[u] = projection *  dg::create::interpolation( *xcomp[u], *ycomp[u],
                grid_equidist, bcx, bcy, inter_m) *  dg::create::backproject(
                *grid_transform); // from dg to equidist
        }
    }
    }

    if( benchmark)
    {
        t.toc();
        std::cout << "# DS: Multiplication PI    took: "<<t.diff()<<"\n";
    }
    ///%%%%%%%%%%%%%%%%%%%%copy into h vectors %%%%%%%%%%%%%%%%%%%//
    dg::HVec hbphi( yp_trafo[2]), hbphiP(hbphi), hbphiM(hbphi);
    hbphi = dg::pullback( vec.z(), *grid_transform);
    //this is a pullback bphi( R(zeta, eta), Z(zeta, eta)):
    if( dynamic_cast<const dg::CartesianGrid2d*>( grid_transform.get()))
    {
        for( unsigned i=0; i<hbphiP.size(); i++)
        {
            hbphiP[i] = vec.z()(yp_trafo[0][i], yp_trafo[1][i]);
            hbphiM[i] = vec.z()(ym_trafo[0][i], ym_trafo[1][i]);
        }
    }
    else
    {
        dg::HVec Ihbphi = dg::pullback( vec.z(), *grid_magnetic);
        dg::HVec Lhbphi = dg::forward_transform( Ihbphi, *grid_magnetic);
        for( unsigned i=0; i<yp_trafo[0].size(); i++)
        {
            hbphiP[i] = dg::interpolate( dg::lspace, Lhbphi, yp_trafo[0][i],
                    yp_trafo[1][i], *grid_magnetic);
            hbphiM[i] = dg::interpolate( dg::lspace, Lhbphi, ym_trafo[0][i],
                    ym_trafo[1][i], *grid_magnetic);
        }
    }
    dg::assign3dfrom2d( hbphi,  m_bphi,  grid);
    dg::assign3dfrom2d( hbphiM, m_bphiM, grid);
    dg::assign3dfrom2d( hbphiP, m_bphiP, grid);

    dg::assign3dfrom2d( yp_trafo[2], m_Gp, grid);
    dg::assign3dfrom2d( ym_trafo[2], m_Gm, grid);
    // The weights don't matter since they fall out in Div and Lap anyway
    // But they are good for testing
    m_G = vol;
    container weights = dg::create::weights( grid);
    dg::blas1::pointwiseDot( m_G, weights, m_G);
    dg::blas1::pointwiseDot( m_Gp, weights, m_Gp);
    dg::blas1::pointwiseDot( m_Gm, weights, m_Gm);

    dg::assign( dg::evaluate( dg::zero, grid), m_hbm);
    m_f     = dg::split( (const container&)m_hbm, grid);
    m_temp  = dg::split( m_hbm, grid);
    dg::assign3dfrom2d( hbp, m_hbp, grid);
    dg::assign3dfrom2d( hbm, m_hbm, grid);
    dg::blas1::scal( m_hbm, -1.);

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
    dg::assign3dfrom2d( bbm, m_bbm, grid);
    dg::assign3dfrom2d( bbo, m_bbo, grid);
    dg::assign3dfrom2d( bbp, m_bbp, grid);

    m_deltaPhi = deltaPhi; // store for evaluate

    ///%%%%%%%%%%%%%%%%%%%%%Assign Limiter%%%%%%%%%%%%%%%%%%%%%%%%%//
    m_perp_size = grid_transform->size();
    dg::assign( dg::pullback(limit, *grid_transform), m_limiter);
    dg::assign( dg::evaluate(dg::zero, *grid_transform), m_left);
    m_ghostM = m_ghostP = m_right = m_left;
}


template<class G, class I, class container>
container Fieldaligned<G, I,container>::interpolate_from_coarse_grid(
        const G& grid, const container& in)
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
    if(     which == einsPlus  || which == einsMinusT ) ePlus(  which, f, fe);
    else if(which == einsMinus || which == einsPlusT  ) eMinus( which, f, fe);
    else if(which == zeroMinus || which == zeroPlus ||
            which == zeroMinusT|| which == zeroPlusT ||
            which == zeroForw  ) zero(   which, f, fe);
}

template< class G, class I, class container>
void Fieldaligned<G, I, container>::zero( enum whichMatrix which,
        const container& f, container& f0)
{
    dg::split( f, m_f, *m_g);
    dg::split( f0, m_temp, *m_g);
    //1. compute 2d interpolation in every plane and store in m_temp
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        if(which == zeroPlus)
            dg::blas2::symv( m_plus,   m_f[i0], m_temp[i0]);
        else if(which == zeroMinus)
            dg::blas2::symv( m_minus,  m_f[i0], m_temp[i0]);
        else if(which == zeroPlusT)
        {
            if( ! m_have_adjoint) updateAdjoint( );
            dg::blas2::symv( m_plusT,  m_f[i0], m_temp[i0]);
        }
        else if(which == zeroMinusT)
        {
            if( ! m_have_adjoint) updateAdjoint( );
            dg::blas2::symv( m_minusT, m_f[i0], m_temp[i0]);
        }
        else if( which == zeroForw)
        {
            if ( m_interpolation_method != "dg" )
            {
                dg::blas2::symv( m_zero, m_f[i0], m_temp[i0]);
            }
            else
                dg::blas1::copy( m_f[i0], m_temp[i0]);
        }
    }
}
template< class G, class I, class container>
void Fieldaligned<G, I, container>::ePlus( enum whichMatrix which,
        const container& f, container& fpe)
{
    dg::split( f, m_f, *m_g);
    dg::split( fpe, m_temp, *m_g);
    //1. compute 2d interpolation in every plane and store in m_temp
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        unsigned ip = (i0==m_Nz-1) ? 0:i0+1;
        if(which == einsPlus)
            dg::blas2::symv( m_plus,   m_f[ip], m_temp[i0]);
        else if(which == einsMinusT)
        {
            if( ! m_have_adjoint) updateAdjoint( );
            dg::blas2::symv( m_minusT, m_f[ip], m_temp[i0]);
        }
    }
    //2. apply right boundary conditions in last plane
    unsigned i0=m_Nz-1;
    if( m_bcz != dg::PER)
    {
        if( m_bcz == dg::DIR || m_bcz == dg::NEU_DIR)
            dg::blas1::axpby( 2, m_right, -1., m_f[i0], m_ghostP);
        if( m_bcz == dg::NEU || m_bcz == dg::DIR_NEU)
            dg::blas1::axpby( m_deltaPhi, m_right, 1., m_f[i0], m_ghostP);
        //interlay ghostcells with periodic cells: L*g + (1-L)*fpe
        dg::blas1::axpby( 1., m_ghostP, -1., m_temp[i0], m_ghostP);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostP, 1., m_temp[i0]);
    }
}

template< class G, class I, class container>
void Fieldaligned<G, I, container>::eMinus( enum whichMatrix which,
        const container& f, container& fme)
{
    dg::split( f, m_f, *m_g);
    dg::split( fme, m_temp, *m_g);
    //1. compute 2d interpolation in every plane and store in m_temp
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        unsigned im = (i0==0) ? m_Nz-1:i0-1;
        if(which == einsPlusT)
        {
            if( ! m_have_adjoint) updateAdjoint( );
            dg::blas2::symv( m_plusT, m_f[im], m_temp[i0]);
        }
        else if (which == einsMinus)
            dg::blas2::symv( m_minus, m_f[im], m_temp[i0]);
    }
    //2. apply left boundary conditions in first plane
    unsigned i0=0;
    if( m_bcz != dg::PER)
    {
        if( m_bcz == dg::DIR || m_bcz == dg::DIR_NEU)
            dg::blas1::axpby( 2., m_left,  -1., m_f[i0], m_ghostM);
        if( m_bcz == dg::NEU || m_bcz == dg::NEU_DIR)
            dg::blas1::axpby( -m_deltaPhi, m_left, 1., m_f[i0], m_ghostM);
        //interlay ghostcells with periodic cells: L*g + (1-L)*fme
        dg::blas1::axpby( 1., m_ghostM, -1., m_temp[i0], m_ghostM);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostM, 1., m_temp[i0]);
    }
}

template<class G, class I, class container>
template< class BinaryOp, class UnaryOp>
container Fieldaligned<G, I,container>::evaluate( BinaryOp binary,
        UnaryOp unary, unsigned p0, unsigned rounds) const
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
            dg::blas1::scal( tempP, unary(  (double)rep*m_deltaPhi ) );
            dg::blas1::scal( tempM, unary( -(double)rep*m_deltaPhi ) );
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


///@endcond

/**
 * @brief %Evaluate a 2d functor and transform to all planes along the fieldlines
 *
 * The algorithm does the equivalent of the following:
 *  - %Evaluate the given \c BinaryOp on a 2d plane
 *  - integrate fieldlines \f$ r N_z \Delta\varphi\f$ in both directions where
 *  \f$ N_z\f$ is the number of planes in the global 3d grid and \f$ r\f$ is
 *  the number of rounds. Then evaluate BinaryOp at the resulting points.
 *  - Scale the transformations with \f$ u ( \pm (iN_z + j)\Delta\varphi) \f$, where \c u is the given \c UnarayOp, \c i in [0..r] is the round index and \c j in [0..Nz] is the plane index and \f$\Delta\varphi\f$ is the angular distance given in the constructor (can be different from the actual grid distance hz!).
 *  - %Sum all transformations with the same plane index \c j , where the minus transformations get the inverted index \f$ N_z - j\f$.
 *  - Shift the index by \f$ p_0\f$
 *  .
 * @attention This version of the algorithm is more exact but much slower than the
 * %evaluate member contained in dg::geo::Fieldaligned
 * @tparam BinaryOp Binary Functor
 * @tparam UnaryOp Unary Functor
 * @param grid The grid on which to integrate fieldlines.
 * @param vec The vector field to integrate. Note that you can control how the
 * boundary conditions are represented by changing vec outside the grid domain
 * using e.g. the \c periodify function.
 * @param binary Functor to evaluate in x-y
 * @param unary Functor to evaluate in z
 * @note \c unary is evaluated such that \c p0 corresponds to z=0, p0+1 corresponds to z=hz, p0-1 to z=-hz, ...
 * @param p0 The index of the plane to start
 * @param rounds The number of rounds \c r to follow a fieldline; can be zero,
 * then the fieldlines are only followed within the current box ( no periodicity)
 * @param eps The accuracy of the fieldline integrator
 *
 * @return 3d vector
 * @ingroup fieldaligned
 */
template<class BinaryOp, class UnaryOp>
thrust::host_vector<double> fieldaligned_evaluate(
        const aProductGeometry3d& grid,
        const CylindricalVectorLvl0& vec,
        const BinaryOp& binary,
        const UnaryOp& unary,
        unsigned p0,
        unsigned rounds,
        double eps = 1e-5)
{
    unsigned Nz = grid.Nz();
    const dg::ClonePtr<aGeometry2d> g2d = grid.perp_grid();
    // Construct for field-aligned output
    dg::HVec tempP = dg::evaluate( dg::zero, *g2d), tempM( tempP);
    std::vector<dg::HVec>  plus2d(Nz, tempP), minus2d(plus2d), result(plus2d);
    dg::HVec vec3d = dg::evaluate( dg::zero, grid);
    dg::HVec init2d = dg::pullback( binary, *g2d);
    std::array<dg::HVec,3> yy0{
        dg::pullback( dg::cooX2d, *g2d),
        dg::pullback( dg::cooY2d, *g2d),
        dg::evaluate( dg::zero, *g2d)}, yy1(yy0), xx0( yy0), xx1(yy0); //s
    dg::geo::detail::DSFieldCylindrical3 cyl_field(vec);
    double deltaPhi = grid.hz();
    double phiM0 = 0., phiP0 = 0.;
    unsigned turns = rounds;
    if( turns == 0) turns++;
    for( unsigned r=0; r<turns; r++)
        for( unsigned  i0=0; i0<Nz; i0++)
        {
            unsigned rep = r*Nz + i0;
            if( rep == 0)
                tempM = tempP = init2d;
            else
            {
                dg::Adaptive<dg::ERKStep<std::array<double,3>>> adapt(
                        "Dormand-Prince-7-4-5", std::array<double,3>{0,0,0});
                dg::AdaptiveTimeloop<std::array<double,3>> odeint( adapt,
                        cyl_field, dg::pid_control, dg::fast_l2norm, eps,
                        1e-10);
                for( unsigned i=0; i<g2d->size(); i++)
                {
                    // minus direction needs positive integration!
                    double phiM1 = phiM0 + deltaPhi;
                    std::array<double,3>
                        coords0{yy0[0][i],yy0[1][i],yy0[2][i]}, coords1;
                    odeint.integrate_in_domain( phiM0, coords0, phiM1, coords1,
                            deltaPhi, *g2d, eps);
                    yy1[0][i] = coords1[0], yy1[1][i] = coords1[1], yy1[2][i] =
                        coords1[2];
                    tempM[i] = binary( yy1[0][i], yy1[1][i]);

                    // plus direction needs negative integration!
                    double phiP1 = phiP0 - deltaPhi;
                    coords0 = std::array<double,3>{xx0[0][i],xx0[1][i],xx0[2][i]};
                    odeint.integrate_in_domain( phiP0, coords0, phiP1, coords1,
                            -deltaPhi, *g2d, eps);
                    xx1[0][i] = coords1[0], xx1[1][i] = coords1[1], xx1[2][i] =
                        coords1[2];
                    tempP[i] = binary( xx1[0][i], xx1[1][i]);
                }
                std::swap( yy0, yy1);
                std::swap( xx0, xx1);
                phiM0 += deltaPhi;
                phiP0 -= deltaPhi;
            }
            dg::blas1::scal( tempM, unary( -(double)rep*deltaPhi ) );
            dg::blas1::scal( tempP, unary(  (double)rep*deltaPhi ) );
            dg::blas1::axpby( 1., tempM, 1., minus2d[i0]);
            dg::blas1::axpby( 1., tempP, 1., plus2d[i0]);
        }
    //now we have the plus and the minus filaments
    if( rounds == 0) //there is a limiter
    {
        for( unsigned i0=0; i0<Nz; i0++)
        {
            int idx = (int)i0 - (int)p0;
            if(idx>=0)
                result[i0] = plus2d[idx];
            else
                result[i0] = minus2d[abs(idx)];
            thrust::copy( result[i0].begin(), result[i0].end(), vec3d.begin() +
                    i0*g2d->size());
        }
    }
    else //sum up plus2d and minus2d
    {
        for( unsigned i0=0; i0<Nz; i0++)
        {
            unsigned revi0 = (Nz - i0)%Nz; //reverted index
            dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
            dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
        }
        dg::blas1::axpby( -1., init2d, 1., result[0]);
        for(unsigned i0=0; i0<Nz; i0++)
        {
            int idx = ((int)i0 -(int)p0 + Nz)%Nz; //shift index
            thrust::copy( result[idx].begin(), result[idx].end(), vec3d.begin()
                    + i0*g2d->size());
        }
    }
    return vec3d;
}

}//namespace geo
}//namespace dg
