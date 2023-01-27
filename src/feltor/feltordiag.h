#include <string>
#include <vector>
#include <functional>

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"

#include "feltor/feltor.h"
#include "feltor/parameters.h"

#include "feltor/init.h"

namespace feltor{

// This file constitutes the diagnostics module for feltor
// The way it works is that it allocates global lists of Records that describe what goes into the file
// You can register you own diagnostics in one of three diagnostics lists (static 3d, dynamic 3d and
// dynamic 2d) further down
// which will then be applied during a simulation

namespace routines{

struct RadialParticleFlux{
    RadialParticleFlux( double tau, double mu):
        m_tau(tau), m_mu(mu){
    }
    //jsNC
    DG_DEVICE double operator()( double ne, double ue,
        double d0S, double d1S, double d2S, //Psip
        double curv0,       double curv1,       double curv2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        double curvKappaS = curvKappa0*d0S+curvKappa1*d1S+curvKappa2*d2S;
        double curvS = curv0*d0S+curv1*d1S+curv2*d2S;
        double JPsi =
            + ne * m_mu*ue*ue*curvKappaS
            + ne * m_tau*curvS;
        return JPsi;
    }
    //jsNA
    DG_DEVICE double operator()( double ne, double ue, double A,
        double d0A, double d1A, double d2A,
        double d0S, double d1S, double d2S, //Psip
        double b_0,         double b_1,         double b_2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        double curvKappaS = curvKappa0*d0S+curvKappa1*d1S+curvKappa2*d2S;
        double SA = b_0*( d1S*d2A-d2S*d1A)+
                    b_1*( d2S*d0A-d0S*d2A)+
                    b_2*( d0S*d1A-d1S*d0A);
        double JPsi =
            ne*ue* (A*curvKappaS + SA );
        return JPsi;
    }
    private:
    double m_tau, m_mu;
};
struct RadialEnergyFlux{
    RadialEnergyFlux( double tau, double mu, double z):
        m_tau(tau), m_mu(mu), m_z(z){
    }

    DG_DEVICE double operator()( double ne, double ue, double P,
        double d0P, double d1P, double d2P, //Phi
        double d0S, double d1S, double d2S, //Psip
        double b_0,         double b_1,         double b_2,
        double curv0,  double curv1,  double curv2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        double curvKappaS = curvKappa0*d0S+curvKappa1*d1S+curvKappa2*d2S;
        double curvS = curv0*d0S+curv1*d1S+curv2*d2S;
        double PS = b_0 * ( d1P * d2S - d2P * d1S )+
                    b_1 * ( d2P * d0S - d0P * d2S )+
                    b_2 * ( d0P * d1S - d1P * d0S );
        double JN =
            + ne * PS
            + ne * m_mu*ue*ue*curvKappaS
            + ne * m_tau*curvS;
        double Je = m_z*(m_tau * log(ne <=0 ? 1e-16 : ne) + 0.5*m_mu*ue*ue + P)*JN
            + m_z*m_mu*m_tau*ne*ue*ue*curvKappaS;
        return Je;
    }
    DG_DEVICE double operator()( double ne, double ue, double P, double A,
        double d0A, double d1A, double d2A,
        double d0S, double d1S, double d2S, //Psip
        double b_0,         double b_1,         double b_2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        double curvKappaS = curvKappa0*d0S+curvKappa1*d1S+curvKappa2*d2S;
        double SA = b_0 * ( d1S * d2A - d2S * d1A )+
                    b_1 * ( d2S * d0A - d0S * d2A )+
                    b_2 * ( d0S * d1A - d1S * d0A );
        double JN = m_z*ne*ue* (A*curvKappaS + SA );
        double Je = m_z*( m_tau * log(ne <= 0 ? 1e-16 : ne) + 0.5*m_mu*ue*ue + P )*JN
                    + m_z*m_tau*ne*ue* (A*curvKappaS + SA );
        return Je;
    }
    //energy dissipation
    DG_DEVICE double operator()( double ne, double ue, double P,
        double lambdaN, double lambdaU){
        return m_z*(m_tau*(1+log(ne <= 0 ? 1e-16 : ne))+P+0.5*m_mu*ue*ue)*lambdaN
                + m_z*m_mu*ne*ue*lambdaU;
    }
    //energy source
    DG_DEVICE double operator()( double ne, double ue, double P,
        double source){
        return m_z*(m_tau*(1+log(ne <= 0 ? 1e-16 : ne))+P+0.5*m_mu*ue*ue)*source;
    }
    private:
    double m_tau, m_mu, m_z;
};
struct PositiveLN
{
    DG_DEVICE double operator()(double x)
    {
        if( x > 0) return log(x);
        return log(1e-16); // avoid nans in output
    }
};

template<class Container>
void dot( const std::array<Container, 3>& v,
          const std::array<Container, 3>& w,
          Container& result)
{
    dg::blas1::evaluate( result, dg::equals(), dg::PairSum(),
        v[0], w[0], v[1], w[1], v[2], w[2]);
}
template<class Container>
void dot( double alpha, const std::array<Container, 3>& v,
          const std::array<Container, 3>& w, double beta,
          Container& result)
{
    dg::blas1::evaluate( result, dg::Axpby<double>(alpha,beta), dg::PairSum(),
        v[0], w[0], v[1], w[1], v[2], w[2]);
}


struct Dot{
    DG_DEVICE void operator()(
            double lambda,
        double d0P, double d1P, double d2P,
        double& c_0, double& c_1, double& c_2)
    {
        c_0 = lambda*(d0P);
        c_1 = lambda*(d1P);
        c_2 = lambda*(d2P);
    }
};
template<class Container>
void scal( const Container& lambda,
          const std::array<Container, 3>& a,
          std::array<Container, 3>& c)
{
    dg::blas1::subroutine( Dot(), lambda,
        a[0], a[1], a[2], c[0], c[1], c[2]);
}

struct Times{
    DG_DEVICE void operator()(
            double lambda,
        double d0P, double d1P, double d2P, //any three vectors
        double d0S, double d1S, double d2S,
        double& c_0, double& c_1, double& c_2)
    {
        c_0 = lambda*(d1P*d2S-d2P*d1S);
        c_1 = lambda*(d2P*d0S-d0P*d2S);
        c_2 = lambda*(d0P*d1S-d1P*d0S);
    }
};
template<class Container>
void times(
          const std::array<Container, 3>& a,
          const std::array<Container, 3>& b,
          std::array<Container, 3>& c)
{
    dg::blas1::subroutine( Times(), 1.,
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}
template<class Container>
void times(
          const Container& lambda,
          const std::array<Container, 3>& a,
          const std::array<Container, 3>& b,
          std::array<Container, 3>& c)
{
    dg::blas1::subroutine( Times(), lambda,
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}
struct Jacobian{
    DG_DEVICE double operator()(
        double d0P, double d1P, double d2P, //any three vectors
        double d0S, double d1S, double d2S,
        double b_0, double b_1, double b_2)
    {
        return      b_0*( d1P*d2S-d2P*d1S)+
                    b_1*( d2P*d0S-d0P*d2S)+
                    b_2*( d0P*d1S-d1P*d0S);
    }
};
template<class Container>
void jacobian(
          const std::array<Container, 3>& a,
          const std::array<Container, 3>& b,
          const std::array<Container, 3>& c,
          Container& result)
{
    dg::blas1::evaluate( result, dg::equals(), Jacobian(),
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}
template<class Container>
void jacobian(
        double alpha,
          const std::array<Container, 3>& a,
          const std::array<Container, 3>& b,
          const std::array<Container, 3>& c,
          double beta,
          Container& result)
{
    dg::blas1::evaluate( result, dg::Axpby(alpha,beta), Jacobian(),
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}
}//namespace routines

//From here on, we use the typedefs to ease the notation

struct Variables{
    feltor::Explicit<dg::x::CylindricalGrid3d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec>& f;
    std::array<std::array<dg::x::DVec,2>,2>& y0;
    feltor::Parameters p;
    dg::geo::TokamakMagneticField mag;
    dg::geo::Nablas<dg::x::CylindricalGrid3d, dg::x::DMatrix, dg::x::DVec> nabla;
    const std::array<dg::x::DVec, 3>& gradPsip;
    std::array<dg::x::DVec, 3> tmp;
    std::array<dg::x::DVec, 3> tmp2;
    std::array<dg::x::DVec, 3> tmp3;
    dg::x::DVec hoo; //keep hoo there to avoid pullback
    double duration;
    const unsigned* nfailed;
};

struct Record{
    std::string name;
    std::string long_name;
    bool integral; //indicates whether the function should be time-integrated
    std::function<void( dg::x::DVec&, Variables&)> function;
};

struct Record1d{
    std::string name;
    std::string long_name;
    std::function<double( Variables&)> function;
};

struct Record_static{
    std::string name;
    std::string long_name;
    std::function<void( dg::x::HVec&, Variables&, dg::x::CylindricalGrid3d& grid)> function;
};

///%%%%%%%%%%%%%%%%%%%%EXTEND LISTS WITH YOUR DIAGNOSTICS HERE%%%%%%%%%%%%%%%%%%%%%%
///%%%%%%%%%%%%%%%%%%%%EXTEND LISTS WITH YOUR DIAGNOSTICS HERE%%%%%%%%%%%%%%%%%%%%%%
///%%%%%%%%%%%%%%%%%%%%EXTEND LISTS WITH YOUR DIAGNOSTICS HERE%%%%%%%%%%%%%%%%%%%%%%
//Here is a list of static (time-independent) 3d variables that go into the output
//Cannot be feltor internal variables
std::vector<Record_static> diagnostics3d_static_list = {
    { "BR", "R-component of magnetic field in cylindrical coordinates",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            dg::geo::BFieldR fieldR(v.mag);
            result = dg::pullback( fieldR, grid);
        }
    },
    { "BZ", "Z-component of magnetic field in cylindrical coordinates",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            dg::geo::BFieldZ fieldZ(v.mag);
            result = dg::pullback( fieldZ, grid);
        }
    },
    { "BP", "Contravariant P-component of magnetic field in cylindrical coordinates",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            dg::geo::BFieldP fieldP(v.mag);
            result = dg::pullback( fieldP, grid);
        }
    },
    { "Psip", "Flux-function psi",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
             result = dg::pullback( v.mag.psip(), grid);
        }
    },
    { "vol3d", "Volume form in 3d",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
             result = dg::create::volume( grid);
        }
    },
    { "xc", "x-coordinate in Cartesian coordinate system",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::evaluate( dg::cooRZP2X, grid);
        }
    },
    { "yc", "y-coordinate in Cartesian coordinate system",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::evaluate( dg::cooRZP2Y, grid);
        }
    },
    { "zc", "z-coordinate in Cartesian coordinate system",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::evaluate( dg::cooRZP2Z, grid);
        }
    },
};

std::array<std::tuple<std::string, std::string, dg::x::HVec>, 3> generate_cyl2cart( dg::x::CylindricalGrid3d& grid)
{
    dg::x::HVec xc = dg::evaluate( dg::cooRZP2X, grid);
    dg::x::HVec yc = dg::evaluate( dg::cooRZP2Y, grid);
    dg::x::HVec zc = dg::evaluate( dg::cooRZP2Z, grid);
    std::array<std::tuple<std::string, std::string, dg::x::HVec>, 3> list = {{
        { "xc", "x-coordinate in Cartesian coordinate system", xc },
        { "yc", "y-coordinate in Cartesian coordinate system", yc },
        { "zc", "z-coordinate in Cartesian coordinate system", zc }
    }};
    return list;
}

// Here are all 3d outputs we want to have
std::vector<Record> diagnostics3d_list = {
    {"electrons", "electron density", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(0), result);
        }
    },
    {"ions", "ion density", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(1), result);
        }
    },
    {"Ue", "parallel electron velocity", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(0), result);
        }
    },
    {"Ui", "parallel ion velocity", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(1), result);
        }
    },
    {"potential", "electric potential", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.potential(0), result);
        }
    },
    {"aparallel", "parallel magnetic potential", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.aparallel(), result);
        }
    }
};

//Here is a list of static (time-independent) 2d variables that go into the output
//MW: if they stay they should be documented in feltor.tex
//MW: we should add initialization and source terms here
//( we make 3d variables here but only the first 2d slice is output)
std::vector<Record_static> diagnostics2d_static_list = {
    { "Psip2d", "Flux-function psi",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.psip(), grid);
        }
    },
    { "PsipR2d", "Flux-function psi R-derivative",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.psipR(), grid);
        }
    },
    { "PsipZ2d", "Flux-function psi Z-derivative",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.psipZ(), grid);
        }
    },
    { "PsipRR2d", "Flux-function psi RR-derivative",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.psipRR(), grid);
        }
    },
    { "PsipRZ2d", "Flux-function psi RZ-derivative",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.psipRZ(), grid);
        }
    },
    { "PsipZZ2d", "Flux-function psi ZZ-derivative",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.psipZZ(), grid);
        }
    },
    { "Ipol", "Poloidal current",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.ipol(), grid);
        }
    },
    { "IpolR", "Poloidal current R-derivative",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.ipolR(), grid);
        }
    },
    { "IpolZ", "Poloidal current Z-derivative",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.ipolZ(), grid);
        }
    },
    { "Rho_p", "Normalized Poloidal flux label",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( dg::geo::RhoP( v.mag), grid);
        }
    },
    { "Bmodule", "Magnetic field strength",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( dg::geo::Bmodule(v.mag), grid);
        }
    },
    { "Divb", "The divergence of the magnetic unit vector",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            dg::assign( v.f.divb(), result);
        }
    },
    { "InvB", "Inverse of Bmodule",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            dg::assign( v.f.binv(), result);
        }
    },
    { "CurvatureKappaR", "R-component of the Kappa B curvature vector",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            dg::assign( v.f.curvKappa()[0], result);
        }
    },
    { "CurvatureKappaZ", "Z-component of the Kappa B curvature vector",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            dg::assign( v.f.curvKappa()[1], result);
        }
    },
    { "CurvatureKappaP", "Contravariant Phi-component of the Kappa B curvature vector",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            dg::assign( v.f.curvKappa()[2], result);
        }
    },
    { "DivCurvatureKappa", "Divergence of the Kappa B curvature vector",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            dg::assign( v.f.divCurvKappa(), result);
        }
    },
    { "CurvatureR", "R-component of the curvature vector",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            dg::assign( v.f.curv()[0], result);
        }
    },
    { "CurvatureZ", "Z-component of the full curvature vector",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            dg::assign( v.f.curv()[1], result);
        }
    },
    { "CurvatureP", "Contravariant Phi-component of the full curvature vector",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            dg::assign( v.f.curv()[2], result);
        }
    },
    { "bphi", "Covariant Phi-component of the magnetic unit vector",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            dg::assign( v.f.bphi(), result);
        }
    },
    { "BHatR", "R-component of the magnetic field unit vector",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            result = dg::pullback( dg::geo::BHatR(v.mag), grid);
        }
    },
    { "BHatZ", "Z-component of the magnetic field unit vector",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            result = dg::pullback( dg::geo::BHatZ(v.mag), grid);
        }
    },
    { "BHatP", "P-component of the magnetic field unit vector",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            result = dg::pullback( dg::geo::BHatP(v.mag), grid);
        }
    },
    { "NormGradPsip", "Norm of gradient of Psip",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid){
            result = dg::pullback(
                dg::geo::SquareNorm( dg::geo::createGradPsip(v.mag),
                    dg::geo::createGradPsip(v.mag)), grid);
        }
    },
    { "Wall", "Wall Region",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            dg::assign( v.f.get_wall(), result);
        }
    },
    { "Sheath", "Sheath Region",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            dg::assign( v.f.get_sheath(), result);
        }
    },
    { "SheathCoordinate", "Sheath Coordinate of field lines",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            dg::assign( v.f.get_sheath_coordinate(), result);
        }
    },
    { "Nprof", "Density profile (that the source may force)",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            dg::assign( v.f.get_source_prof(), result);
        }
    },
    { "Source", "Source region",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            dg::assign( v.f.get_source(), result);
        }
    },
    { "neinit", "Initial condition for electrons",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            dg::assign( v.y0[0][0], result);
        }
    },
    { "niinit", "Initial condition for ions",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            dg::assign( v.y0[0][1], result);
        }
    },
    { "weinit", "Initial condition for electron canonical velocity",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            dg::assign( v.y0[1][0], result);
        }
    },
    { "wiinit", "Initial condition for ion canonical velocity",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            dg::assign( v.y0[1][1], result);
        }
    },
    { "vol2d", "Volume form (including R) in 2d",
        []( dg::x::HVec& result, Variables& v, dg::x::CylindricalGrid3d& grid ){
            result = dg::create::volume(grid);
        }
    },
};
// and here are all the 2d outputs we want to produce (currently ~ 100)
std::vector<Record> basicDiagnostics2d_list = {
    {"electrons", "Electron density", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(0), result);
        }
    },
    {"ions", "Ion gyro-centre density", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(1), result);
        }
    },
    {"Ue", "Electron parallel velocity", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(0), result);
        }
    },
    {"Ui", "Ion parallel velocity", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(1), result);
        }
    },
    {"potential", "Electric potential", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.potential(0), result);
        }
    },
    {"psi", "Ion potential psi", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.potential(1), result);
        }
    },
    {"aparallel", "Magnetic potential", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.aparallel(), result);
        }
    },
    /// -----------------Miscellaneous additions --------------------//
    {"vorticity", "Minus Lap_perp of electric potential", false,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_lapMperpP(0, result);
        }
    },
    {"vorticity_i", "Minus Lap_perp of ion potential", false,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_lapMperpP(1, result);
        }
    },
    // Does not work due to direct application of Laplace
    // The Laplacian of Aparallel looks smooth in paraview
    //{"apar_vorticity", "Minus Lap_perp of magnetic potential", false,
    //    []( dg::x::DVec& result, Variables& v ) {
    //        v.f.compute_lapMperpA( result);
    //    }
    //},
    {"dssue", "2nd parallel derivative of electron velocity", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::copy( v.f.dssU( 0), result);
        }
    },
    {"lperpinv", "Perpendicular density gradient length scale", false,
        []( dg::x::DVec& result, Variables& v ) {
            const std::array<dg::x::DVec, 3>& dN = v.f.gradN(0);
            dg::tensor::multiply3d( v.f.projection(), //grad_perp
                dN[0], dN[1], dN[2], v.tmp[0], v.tmp[1], v.tmp[2]);
            routines::dot(dN, v.tmp, result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
            dg::blas1::transform( result, result, dg::SQRT<double>());
        }
    },
    {"perpaligned", "Perpendicular density alignement", false,
        []( dg::x::DVec& result, Variables& v ) {
            const std::array<dg::x::DVec, 3>& dN = v.f.gradN(0);
            dg::tensor::multiply3d( v.f.projection(), //grad_perp
                dN[0], dN[1], dN[2], v.tmp[0], v.tmp[1], v.tmp[2]);
            routines::dot(dN, v.tmp, result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
        }
    },
    {"lparallelinv", "Parallel density gradient length scale", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDivide( v.f.dsN(0), v.f.density(0), result);
            dg::blas1::pointwiseDot ( result, result, result);
            dg::blas1::transform( result, result, dg::SQRT<double>());
        }
    },
    {"aligned", "Parallel density alignement", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::copy( v.f.dsN( 0), result);
            dg::blas1::pointwiseDot ( result, result, result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
        }
    },
    /// ------------------ Correlation terms --------------------//
    {"ne2", "Square of electron density", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.density(0), v.f.density(0), result);
        }
    },
    {"phi2", "Square of electron potential", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.potential(0), v.f.potential(0), result);
        }
    },
    {"nephi", "Product of electron potential and electron density", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.potential(0), v.f.density(0), result);
        }
    }
};

std::vector<Record> MassConsDiagnostics2d_list = {
    /// ------------------ Density terms ------------------------//
    {"jsneE_tt", "Radial electron particle flux: ExB contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( result, v.f.density(0), result);
        }
    },
    {"jsneC_tt", "Radial electron particle flux: curvature contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialParticleFlux( v.p.tau[0], v.p.mu[0]),
                v.f.density(0), v.f.velocity(0),
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"divneuE_tt", "Divergence of ExB electron particle flux (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.f.curv(), v.f.gradP(0), result);
            dg::blas1::pointwiseDot( result, v.f.density(0), result)
            routines::jacobian( 1., v.f.bhatgB(), v.f.gradP(0), v.gradN(0), 1., result);
        }
    },
    {"divniuE_tt", "Divergence of ExB ion particle flux (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.f.curv(), v.f.gradP(1), result);
            dg::blas1::pointwiseDot( result, v.f.density(1), result)
            routines::jacobian( 1., v.f.bhatgB(), v.f.gradP(1), v.gradN(1), 1., result);
        }
    },
    {"divneuA_tt", "Divergence of magnetic flutter electron particle flux (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.f.curv(), v.f.aparallel(), result);
            routines::dot( -1., v.f.curvKappa(), v.f.aparallel(), 1., result);
            dg::blas1::pointwiseDot( 1., v.f.aparallel(), v.f.divCurvKappa(),
                    -1., result); //div bp
            dg::blas1::pointwiseDot( 1., v.f.density(0), v.f.velocity(0),
                    result, 0., result); // NU div bp
            dg::blas1::evaluate( result, dg::Axpby(1., 1.),
                routines::RadialParticleFlux( v.p.tau[0], v.p.mu[0]),
                v.f.density(0), 1., v.f.aparallel(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.f.gradU(0)[0], v.gradU(0)[1], v.gradU(0)[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            ); // + N bp grad U
            dg::blas1::evaluate( result, dg::Axpby(1., 1.),
                routines::RadialParticleFlux( v.p.tau[0], v.p.mu[0]),
                1., v.f.velocity(0), v.f.aparallel(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.f.gradN(0)[0], v.gradN(0)[1], v.gradN(0)[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            ); // + U bp grad N
        }
    },
    {"divniuA_tt", "Divergence of magnetic flutter ion particle flux (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.f.curv(), v.f.aparallel(), result);
            routines::dot( -1., v.f.curvKappa(), v.f.aparallel(), 1., result);
            dg::blas1::pointwiseDot( 1., v.f.aparallel(), v.f.divCurvKappa(),
                    -1., result); //div bp
            dg::blas1::pointwiseDot( 1., v.f.density(1), v.f.velocity(1),
                    result, 0., result); // NU div bp
            dg::blas1::evaluate( result, dg::Axpby(1., 1.),
                routines::RadialParticleFlux( v.p.tau[1], v.p.mu[1]),
                v.f.density(1), 1., v.f.aparallel(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.f.gradU(1)[0], v.gradU(1)[1], v.gradU(1)[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            ); // + N bp grad U
            dg::blas1::evaluate( result, dg::Axpby(1., 1.),
                routines::RadialParticleFlux( v.p.tau[1], v.p.mu[1]),
                1., v.f.velocity(1), v.f.aparallel(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.f.gradN(1)[0], v.gradN(1)[1], v.gradN(1)[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            ); // + U bp grad N
        }
    },
    {"divcurvne_tt", "Divergence of curvature term (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.p.tau[0], v.f.curv(), v.f.gradN(0), 0., result);
        }
    },
    {"divcurvni_tt", "Divergence of curvature term (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.p.tau[1], v.f.curv(), v.f.gradN(1), 0., result);
        }
    },
    {"divcurvkappane_tt", "Divergence of curvature term (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            dg::blas1::pointwiseDot( 1., v.f.density(0), v.f.velocity(0),
                    v.f.velocity(0), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[0], result, v.f.divCurvKappa(),
                    0., result]);
            routines::dot( v.f.curvKappa(), v.f.gradN(0), v.tmp[1]);
            dg::blas1::pointwiseDot( v.p.mu[0], v.f.velocity(0),
                    v.f.velocity(0), v.tmp[1], 1., result);
            routines::dot( v.f.curvKappa(), v.f.gradU(0), v.tmp[2]);
            dg::blas1::pointwiseDot( 2.*v.p.mu[0], v.f.density(0),
                    v.f.velocity(0), v.tmp[2], 1., result);
        }
    },
    {"divcurvkappani_tt", "Divergence of curvature term (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            dg::blas1::pointwiseDot( 1., v.f.density(1), v.f.velocity(1),
                    v.f.velocity(1), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.divCurvKappa(),
                    0., result]);
            routines::dot( v.f.curvKappa(), v.f.gradN(1), v.tmp[1]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.velocity(1),
                    v.f.velocity(1), v.tmp[1], 1., result);
            routines::dot( v.f.curvKappa(), v.f.gradU(1), v.tmp[2]);
            dg::blas1::pointwiseDot( 2.*v.p.mu[1], v.f.density(1),
                    v.f.velocity(1), v.tmp[2], 1., result);
        }
    },
    {"jsdiae_tt", "Radial electron particle flux: diamagnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            // u_D Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradN(0), v.gradPsip, result);
            dg::blas1::scal( result, v.p.tau[0]);
        }
    },
    {"jsneA_tt", "Radial electron particle flux: magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialParticleFlux( v.p.tau[0], v.p.mu[0]),
                v.f.density(0), v.f.velocity(0), v.f.aparallel(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"lneperp_tt", "Perpendicular electron diffusion (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_perp_diffusiveN( 1., v.f.density(0), v.tmp[0],
                    v.tmp[1], 0., result);
        }
    },
    {"lneparallel_tt", "Parallel electron diffusion (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::axpby( v.p.nu_parallel_n, v.f.lapParN(0), 0., result);
        }
    },
    {"sne_tt", "Source term for electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::copy( v.f.density_source(0), result);
        }
    },
    {"divjnepar_tt", "Divergence of Parallel velocity term for electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::copy( v.f.divNUb(0), result);
        }
    },
    {"jsniE_tt", "Radial ion particle flux: ExB contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(1), v.gradPsip, result);
            dg::blas1::pointwiseDot( result, v.f.density(1), result);
        }
    },
    {"jsniC_tt", "Radial ion particle flux: curvature contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialParticleFlux( v.p.tau[1], v.p.mu[1]),
                v.f.density(1), v.f.velocity(1),
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"jsdiai_tt", "Radial ion particle flux: diamagnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            // u_D Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradN(1), v.gradPsip, result);
            dg::blas1::scal( result, v.p.tau[1]);
        }
    },
    {"jsniA_tt", "Radial ion particle flux: magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialParticleFlux( v.p.tau[1], v.p.mu[1]),
                v.f.density(1), v.f.velocity(1), v.f.aparallel(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"lniperp_tt", "Perpendicular ion diffusion (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_perp_diffusiveN( 1., v.f.density(1), v.tmp[0],
                    v.tmp[1], 1., result);
        }
    },
    {"lniparallel_tt", "Parallel ion diffusion (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::axpby( v.p.nu_parallel_n, v.f.lapParN(1), 0., result);
        }
    },
    {"sni_tt", "Source term for ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::copy( v.f.density_source(1), result);
        }
    },
    {"divjnipar_tt", "Divergence of Parallel velocity term in ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::copy( v.f.divNUb(1), result);
        }
    }
};

std::vector<Record> EnergyDiagnostics2d_list = {
    /// ------------------- Energy terms ------------------------//
    {"nelnne", "Entropy electrons", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(0), result, routines::PositiveLN());
            dg::blas1::pointwiseDot( result, v.f.density(0), result);
        }
    },
    {"nilnni", "Entropy ions", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(1), result, routines::PositiveLN());
            dg::blas1::pointwiseDot( v.p.tau[1], result, v.f.density(1), 0., result);
        }
    },
    {"aperp2", "Magnetic energy", false,
        []( dg::x::DVec& result, Variables& v ) {
            if( v.p.beta == 0)
            {
                dg::blas1::scal( result, 0.);
            }
            else
            {
                dg::tensor::multiply3d( v.f.projection(), //grad_perp
                    v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                    v.tmp[0], v.tmp[1], v.tmp[2]);
                routines::dot( v.tmp, v.f.gradA(), result);
                dg::blas1::scal( result, 1./2./v.p.beta);
            }
        }
    },
    {"ue2", "ExB energy", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::tensor::multiply3d( v.f.projection(), //grad_perp
                v.f.gradP(0)[0], v.f.gradP(0)[1], v.f.gradP(0)[2],
                v.tmp[0], v.tmp[1], v.tmp[2]);
            routines::dot( v.tmp, v.f.gradP(0), result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( 0.5, v.f.density(1), result, 0., result);
        }
    },
    {"neue2", "Parallel electron energy", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( -0.5*v.p.mu[0], v.f.density(0),
                v.f.velocity(0), v.f.velocity(0), 0., result);
        }
    },
    {"niui2", "Parallel ion energy", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 0.5*v.p.mu[1], v.f.density(1),
                v.f.velocity(1), v.f.velocity(1), 0., result);
        }
    },
    /// ------------------- Energy dissipation ----------------------//
    {"resistivity_tt", "Energy dissipation through resistivity (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 1., v.f.velocity(1), v.f.density(0), -1.,
                    v.f.velocity(0), v.f.density(0), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( 1., v.f.velocity(1), v.f.density(1), -1.,
                    v.f.velocity(0), v.f.density(0), 0., v.tmp[1]);
            dg::blas1::pointwiseDot( -v.p.eta, v.tmp[0], v.tmp[1], 0., result);
        }
    },
    {"see_tt", "Energy sink/source for electrons", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.f.density_source(0)
            );
        }
    },
    {"sei_tt", "Energy sink/source for ions", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.f.density_source(1)
            );
        }
    },
    /// ------------------ Energy flux terms ------------------------//
    {"jsee_tt", "Radial electron energy flux without magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.f.gradP(0)[0], v.f.gradP(0)[1], v.f.gradP(0)[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"jseea_tt", "Radial electron energy flux: magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0), v.f.aparallel(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"jsei_tt", "Radial ion energy flux without magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.f.gradP(1)[0], v.f.gradP(1)[1], v.f.gradP(1)[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    {"jseia_tt", "Radial ion energy flux: magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1), v.f.aparallel(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
        }
    },
    /// ------------------------ Energy dissipation terms ------------------//
    {"leeperp_tt", "Perpendicular electron energy dissipation (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_perp_diffusiveN( 1., v.f.density(0), result, v.tmp[2],
                    0., v.tmp[0]);
            v.f.compute_perp_diffusiveU( 1., v.f.velocity(0), v.f.density(0), result, v.tmp[2],
                    0., v.tmp[1]);
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.tmp[0], v.tmp[1]
            );
        }
    },
    {"leiperp_tt", "Perpendicular ion energy dissipation (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_perp_diffusiveN( 1., v.f.density(1), result, v.tmp[2],
                    0., v.tmp[0]);
            v.f.compute_perp_diffusiveU( 1., v.f.velocity(1), v.f.density(1), result, v.tmp[2],
                    0., v.tmp[1]);
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.tmp[0], v.tmp[1]
            );
        }
    },
    {"leeparallel_tt", "Parallel electron energy dissipation (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::axpby( v.p.nu_parallel_n, v.f.lapParN(0), 0., v.tmp[0]);
            dg::blas1::pointwiseDivide( v.p.nu_parallel_u[0], v.f.lapParU(0),
                v.f.density(0), 0., v.tmp[1]);
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.tmp[0], v.tmp[1]
            );
        }
    },
    {"leiparallel_tt", "Parallel ion energy dissipation (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::axpby( v.p.nu_parallel_n, v.f.lapParN(1), 0., v.tmp[0]);
            dg::blas1::pointwiseDivide( v.p.nu_parallel_u[1], v.f.lapParU(1),
                v.f.density(1), 0., v.tmp[1]);
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.tmp[0], v.tmp[1]
            );
        }
    },
    {"divjeepar_tt", "Divergence of Parallel electron energy flux (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            // Multiply out divNUb to get implementable form
            double z = -1.;
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[0], v.p.mu[0], z),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.f.divNUb(0)
            );
            dg::blas1::pointwiseDot( z*v.p.tau[0], v.f.velocity(0), v.f.dsN(0),
                    1., result);
            dg::blas1::pointwiseDot( z, v.f.velocity(0), v.f.density(0),
                    v.f.dsP(0), 1., result);
            dg::blas1::pointwiseDot( 1., v.f.density(0), v.f.velocity(0),
                    v.f.velocity(0), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( z*v.p.mu[0], v.tmp[0], v.f.dsU(0), 1., result);
        }
    },
    {"divjeipar_tt", "Divergence of Parallel ion energy flux (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            // Multiply out divNUb to get implementable form
            double z = +1.;
            dg::blas1::evaluate( result, dg::equals(),
                routines::RadialEnergyFlux( v.p.tau[1], v.p.mu[1], z),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.f.divNUb(1)
            );
            dg::blas1::pointwiseDot( z*v.p.tau[1], v.f.velocity(1), v.f.dsN(1),
                    1., result);
            dg::blas1::pointwiseDot( z, v.f.velocity(1), v.f.density(1),
                    v.f.dsP(1), 1., result);
            dg::blas1::pointwiseDot( 1., v.f.density(1), v.f.velocity(1),
                    v.f.velocity(1), 0., v.tmp[1]);
            dg::blas1::pointwiseDot( z*v.p.mu[1], v.tmp[1], v.f.dsU(1), 1., result);
        }
    }
};

std::vector<Record> ToroidalExBDiagnostics2d_list = {
    /// ------------------------ Vorticity terms ---------------------------//
    {"oexbi", "ExB vorticity term with ion density", false,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density(1), 0., result);
        }
    },
    {"oexbe", "ExB vorticity term with electron density", false,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density(0), 0., result);
        }
    },
    {"odiai", "Diamagnetic vorticity term with ion density", false,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.f.gradN(1), v.gradPsip, result);
            dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    {"odiae", "Diamagnetic vorticity term with electron density", false,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.f.gradN(0), v.gradPsip, result);
            dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    /// --------------------- Vorticity flux terms ---------------------------//
    {"jsoexbi_tt", "ExB vorticity flux term with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // Omega_E
            routines::dot( v.f.gradP(0), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.binv(), v.f.binv(), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], v.f.density(1), 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"jsoexbe_tt", "ExB vorticity flux term with electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // Omega_E
            routines::dot( v.f.gradP(0), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.binv(), v.f.binv(), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], v.f.density(0), 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"jsodiaiUE_tt", "Diamagnetic vorticity flux by ExB veloctiy term with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // Omega_D,phi
            routines::dot( v.f.gradN(1), v.gradPsip, v.tmp[0]);
            dg::blas1::scal( v.tmp[0], v.p.mu[1]*v.p.tau[1]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"jsodiaeUE_tt", "Diamagnetic vorticity flux by ExB velocity term with electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // Omega_D,phi
            routines::dot( v.f.gradN(0), v.gradPsip, v.tmp[0]);
            dg::blas1::scal( v.tmp[0], v.p.mu[1]*v.p.tau[1]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"jsoexbiUD_tt", "ExB vorticity flux term by diamagnetic velocity with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // bxGradN/B Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradN(1), v.gradPsip, result);
            dg::blas1::scal( result, v.p.tau[1]);

            // m Omega_E,phi
            routines::dot( v.f.gradP(0), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], v.f.binv(), v.f.binv(), 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"jsoexbeUD_tt", "ExB vorticity flux term by diamagnetic velocity with electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // bxGradN/B Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradN(0), v.gradPsip, result);
            dg::blas1::scal( result, v.p.tau[1]);

            // m Omega_E,phi
            routines::dot( v.f.gradP(0), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], v.f.binv(), v.f.binv(), 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"jsoapar_tt", "A parallel vorticity flux term (Maxwell stress) (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            if( v.p.beta == 0)
                dg::blas1::scal( result, 0.);
            else
            {
                routines::jacobian( v.f.bhatgB(), v.f.gradA(), v.gradPsip, result);
                routines::dot( v.f.gradA(), v.gradPsip, v.tmp[0]);
                dg::blas1::pointwiseDot( -1./v.p.beta, result, v.tmp[0], 0., result);
            }
        }
    },
    {"jsodiaApar_tt", "A parallel diamagnetic vorticity flux term (magnetization stress) (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            if( v.p.beta == 0)
                dg::blas1::scal( result, 0.);
            else
            {
                routines::dot( v.gradPsip, v.f.gradU(1), v.tmp[0]);
                routines::dot( v.gradPsip, v.f.gradN(1), v.tmp[1]);
                dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.density(1), 1., v.tmp[0], v.f.velocity(1), 0., result);

                routines::jacobian( v.f.bhatgB(), v.f.gradA(), v.gradPsip, result);
                dg::blas1::pointwiseDot( -1./2.*v.p.tau[1], result, v.tmp[0], 0., result);
            }
        }
    },
    {"jsoexbApar_tt", "A parallel ExB vorticity flux term (magnetization stress) (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            if( v.p.beta == 0)
                dg::blas1::scal( result, 0.);
            else
            {
                routines::jacobian( v.f.bhatgB(), v.f.gradU(1), v.gradPsip, v.tmp[0]);
                routines::jacobian( v.f.bhatgB(), v.f.gradN(1), v.gradPsip, v.tmp[1]);
                dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.density(1), 1., v.tmp[1], v.f.velocity(1), 0., result);
                routines::dot( v.f.gradA(), v.gradPsip, v.tmp[2]);
                dg::blas1::pointwiseDot( -1./2.*v.p.tau[1], result, v.tmp[2], 0., result);
            }
        }
    },
    {"sosne_tt", "ExB vorticity source term with electron source", true,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density_source(0), 0., result);
        }
    },
    {"sospi_tt", "Diamagnetic vorticity source term with electron source", true,
        []( dg::x::DVec& result, Variables& v){
            v.f.compute_gradSN( 0, v.tmp);
            routines::dot( v.tmp, v.gradPsip, result);
            dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    {"loexbe_tt", "Vorticity dissipation term with electron Lambda", true,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);

            v.f.compute_perp_diffusiveN( 1., v.f.density(0), v.tmp[0],
                    v.tmp[1], 0., v.tmp[2]);
            dg::blas1::axpby( v.p.nu_parallel_n, v.f.lapParN(0), 0., v.tmp[1]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[1], result,
                v.p.mu[1], v.tmp[2], result, 0., result);
        }
    }
};

std::vector<Record> ParallelMomDiagnostics2d_list = {
    ///-----------------------Parallel momentum terms ------------------------//
    {"neue", "Product of electron density and velocity", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.density(0), v.f.velocity(0), result);
        }
    },
    {"niui", "Product of ion gyrocentre density and velocity", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.density(1), v.f.velocity(1), result);
        }
    },
    {"niuibphi", "Product of NiUi and covariant phi component of magnetic field unit vector", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 1.,
                v.f.density(1), v.f.velocity(1), v.f.bphi(), 0., result);
        }
    },
    /// --------------------- Parallel momentum flux terms ---------------------//
    {"jsparexbi_tt", "Parallel momentum radial flux by ExB velocity with electron potential (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // parallel momentum mu_iN_iU_i
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), result, 0., result);
        }
    },
    {"jsparbphiexbi_tt", "Parallel angular momentum radial flux by ExB velocity with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(1), v.gradPsip, result);

            // parallel momentum mu_iN_iU_i
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0],v.f.bphi(), 0., result);
        }
    },
    {"jspardiai_tt", "Parallel momentum radial flux by Diamagnetic velocity with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // DiaN Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradN(1), v.gradPsip, v.tmp[0]);
            // DiaU Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradU(1), v.gradPsip, v.tmp[1]);

            // Multiply everything
            dg::blas1::pointwiseDot( v.p.mu[1]*v.p.tau[1], v.tmp[0], v.f.velocity(1), v.p.mu[1]*v.p.tau[1], v.tmp[1], v.f.density(1), 0., result);
        }
    },
    {"jsparkappai_tt", "Parallel momentum radial flux by curvature velocity (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.f.curvKappa(), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), v.tmp[0], 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.velocity(1), v.f.velocity(1), v.tmp[0], 0., v.tmp[1]);
            dg::blas1::axpbypgz( 2.*v.p.tau[1], v.tmp[0], +1., v.tmp[1], 0., result);
        }
    },
    {"jsparbphidiai_tt", "Parallel angular momentum radial flux by Diamagnetic velocity with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // bphi K Dot GradPsi
            routines::dot( v.f.curv(), v.gradPsip, result);
            dg::blas1::pointwiseDot( result, v.f.bphi(), result);
            // Multiply everything
            dg::blas1::pointwiseDot( v.p.mu[1]*v.p.tau[1], result, v.f.velocity(1), v.f.density(1), 0., result);
        }
    },
    {"jsparbphikappai_tt", "Parallel angular momentum radial flux by curvature velocity (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.f.curvKappa(), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), v.tmp[0], 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.velocity(1), v.f.velocity(1), v.tmp[0], 0., v.tmp[1]);
            dg::blas1::axpbypgz( 2.*v.p.tau[1], v.tmp[0], +1., v.tmp[1], 0., result);
            dg::blas1::pointwiseDot( result, v.f.bphi(), result);
        }
    },
    {"jsparApar_tt", "Parallel momentum radial flux by magnetic flutter (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            if( v.p.beta == 0)
            {
                dg::blas1::scal( result, 0.);
            }
            else
            {
                //b_\perp^v
                routines::jacobian( v.f.gradA() , v.f.bhatgB(), v.gradPsip, v.tmp[2]);
                dg::blas1::pointwiseDot( -v.p.mu[0], v.f.velocity(0), v.f.velocity(0), v.f.density(0),  0., v.tmp[0]);
                dg::blas1::pointwiseDot( +v.p.mu[1], v.f.velocity(1), v.f.velocity(1), v.f.density(1),  0., v.tmp[1]);
                dg::blas1::pointwiseDot( -v.p.tau[0], v.f.density(0), v.tmp[2], 0., result);
                dg::blas1::pointwiseDot( +v.p.tau[1], v.f.density(1), v.tmp[2], 1., result);
                dg::blas1::pointwiseDot( 1., v.tmp[0], v.tmp[2], 1., result);
                dg::blas1::pointwiseDot( 1., v.tmp[1], v.tmp[2], 1., result);
            }
        }
    },
    {"jsparbphiApar_tt", "Parallel angular momentum radial flux by magnetic flutter (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            if( v.p.beta == 0)
            {
                dg::blas1::scal( result, 0.);
            }
            else
            {
                //b_\perp^v
                routines::jacobian( v.f.gradA() , v.f.bhatgB(), v.gradPsip, v.tmp[2]);
                dg::blas1::pointwiseDot( v.tmp[2], v.f.bphi(), v.tmp[2]);
                dg::blas1::pointwiseDot( -v.p.mu[0], v.f.velocity(0), v.f.velocity(0), v.f.density(0),  0., v.tmp[0]);
                dg::blas1::pointwiseDot( +v.p.mu[1], v.f.velocity(1), v.f.velocity(1), v.f.density(1),  0., v.tmp[1]);
                dg::blas1::pointwiseDot( -v.p.tau[0], v.f.density(0), v.tmp[2], 0., result);
                dg::blas1::pointwiseDot( +v.p.tau[1], v.f.density(1), v.tmp[2], 1., result);
                dg::blas1::pointwiseDot( 1., v.tmp[0], v.tmp[2], 1., result);
                dg::blas1::pointwiseDot( 1., v.tmp[1], v.tmp[2], 1., result);
            }
        }
    },
    /// --------------------- Parallel momentum source terms ---------------------//
    {"divjpare_tt", "Divergence of parallel electron momentum flux", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( -v.p.mu[0], v.f.divNUb(0), v.f.velocity(0),
                    0., result);
            dg::blas1::pointwiseDot( -v.p.mu[0], v.f.density(0),
                    v.f.velocity(0), v.f.dsU(0), 1., result);
        }
    },
    {"divjpari_tt", "Divergence of parallel ion momentum flux", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.divNUb(1), v.f.velocity(1),
                    0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1),
                    v.f.velocity(1), v.f.dsU(1), 1., result);
        }
    },
    //not so important
    {"spardivKappa_tt", "Divergence Kappa Source for parallel momentum", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( -v.p.mu[1]*v.p.tau[1], v.f.density(1),
                    v.f.velocity(1), v.f.divCurvKappa(), 0., result);
        }
    },
    //not so important
    {"sparKappaphi_tt", "Kappa Phi Source for parallel momentum", true,
        []( dg::x::DVec& result, Variables& v ) {
            routines::dot( v.f.curvKappa(), v.f.gradP(1), result);
            dg::blas1::pointwiseDot( -v.p.mu[1], v.f.density(1), v.f.velocity(1), result, 0., result);
        }
    },
    // should be zero in new implementation
    {"sparsni_tt", "Parallel momentum source by density and velocity sources", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.p.mu[1],
                v.f.density_source(1), v.f.velocity(1),
                v.p.mu[1], v.f.velocity_source(1), v.f.density(1), 0., result);
        }
    },
    {"sparsnibphi_tt", "Parallel angular momentum source by density and velocity sources", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.p.mu[1],
                v.f.density_source(1), v.f.velocity(1), v.f.bphi(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1],
                v.f.velocity_source(1), v.f.density(1), v.f.bphi(), 1., result);
        }
    },
    //should be zero
    {"lparpar_tt", "Parallel momentum dissipation by parallel diffusion", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::axpby( v.p.nu_parallel_u[1], v.f.lapParU(1), 0., result);
            dg::blas1::pointwiseDot( v.p.nu_parallel_n, v.f.velocity(1), v.f.lapParN(1), 1., result);
        }
    },
    {"lparperp_tt", "Parallel momentum dissipation by perp diffusion", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_perp_diffusiveN( 1., v.f.density(1), result, v.tmp[2],
                    0., v.tmp[0]);
            v.f.compute_perp_diffusiveU( 1., v.f.velocity(1), v.f.density(1), result, v.tmp[2],
                    0., v.tmp[1]);
            dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.velocity(1),
                    1., v.tmp[1], v.f.density(1), 0., result);
        }
    },
    /// --------------------- Mirror force term ---------------------------//
    {"sparmirrore_tt", "Parallel electron pressure (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::axpby( v.p.tau[0], v.f.dsN(0), 0., result);
        }
    },
    {"sparmirrorAe_tt", "Apar Mirror force term with electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            routines::jacobian( v.f.gradA() , v.f.bhatgB(), v.f.gradN(0), result);
            dg::blas1::scal( result, v.p.tau[0]);
        }
    },
    {"sparmirrori_tt", "Parallel ion pressure (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::axpby( -v.p.tau[1], v.f.dsN(1), 0., result);
        }
    },
    //electric force balance usually well-fulfilled
    {"sparphie_tt", "Electric force in electron momentum density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::pointwiseDot( 1., v.f.dsP(0), v.f.density(0), 0., result);
        }
    },
    {"sparphiAe_tt", "Apar Electric force in electron momentum density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            routines::jacobian( v.f.gradA() , v.f.bhatgB(), v.f.gradP(0), result);
            dg::blas1::pointwiseDot( v.f.density(0), result, result);
        }
    },
    {"spardotAe_tt", "Apar Electric force in electron momentum density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            v.f.compute_dot_aparallel( result);
            dg::blas1::pointwiseDot( v.f.density(0), result, result);
        }
    },
    //These two should be almost the same
    {"sparphii_tt", "Electric force term in ion momentum density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::pointwiseDot( -1., v.f.dsP(1), v.f.density(1), 0., result);
        }
    },
    {"friction_tt", "Friction force in momentum density (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 1., v.f.velocity(1), v.f.density(1), -1.,
                    v.f.velocity(0), v.f.density(0), 0., result);
            dg::blas1::pointwiseDot( v.p.eta, result, v.f.density(0), 0, result);
        }
    },
    /// --------------------- Lorentz force terms ---------------------------//
    {"socurve_tt", "Vorticity source term electron curvature (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.f.curv(), v.gradPsip, result);
            dg::blas1::pointwiseDot( -v.p.tau[0], v.f.density(0), result, 0., result);
        }
    },
    {"socurvi_tt", "Vorticity source term ion curvature (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.f.curv(), v.gradPsip, result);
            dg::blas1::pointwiseDot( v.p.tau[1], v.f.density(1), result, 0., result);
        }
    },
    {"socurvkappae_tt", "Vorticity source term electron kappa curvature (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.f.curvKappa(), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., v.f.density(0), v.f.velocity(0), v.f.velocity(0), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( -v.p.mu[0], v.tmp[0], result, 0., result);
        }
    },
    {"socurvkappai_tt", "Vorticity source term ion kappa curvature (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.f.curvKappa(), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., v.f.density(1), v.f.velocity(1), v.f.velocity(1), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], result, 0., result);
        }
    }
};

std::vector<Record> RSDiagnostics2d_list = {
    /// --------------------- Zonal flow energy terms------------------------//
    {"nei0", "inertial factor", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.f.density(0), v.hoo, result);
        }
    },
    {"snei0_tt", "inertial factor source", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.f.density_source(0), v.hoo, result);
        }
    }
};

std::vector<Record> COCEDiagnostics2d_list = {
    /// ----------------- COCE EQUATION ----------------//
    /// ---------- Polarization charge densities -----------///

    {"v_Omega_E", "Electric PCD", false, //CHECKED
        []( dg::x::DVec& result, Variables& v) {
        dg::blas1::pointwiseDot(v.f.binv(), v.f.binv(), v.tmp2[0]);
        dg::blas1::pointwiseDot(v.tmp2[0], v.f.density(0), v.tmp2[0]);
        routines::scal(v.tmp2[0], v.f.gradP(0), v.tmp); //ne grad(phi)/B^2
        v.nabla.div(v.tmp[0], v.tmp[1], result);
        dg::blas1::scal( result, v.p.mu[1]);
        }
    },
     {"v_Omega_E_gf", "Electric PCD GF", false, //CHECKED
        []( dg::x::DVec& result, Variables& v) {
        dg::blas1::pointwiseDot(v.f.binv(), v.f.binv(), v.tmp2[0]);
        dg::blas1::pointwiseDot(v.tmp2[0], v.f.density(1), v.tmp2[0]);
        routines::scal(v.tmp2[0], v.f.gradP(0), v.tmp); //Ni grad(phi)/B^2
        v.nabla.div(v.tmp[0], v.tmp[1], result);
        dg::blas1::scal( result, v.p.mu[1]);
        }
    },
    {"v_Omega_D", "Diamagnetic PCD", false, //CHECKED
        []( dg::x::DVec& result, Variables& v) {
         dg::blas1::pointwiseDot(v.f.binv(), v.f.binv(), v.tmp2[0]);
         routines::scal(v.tmp2[0], v.f.gradN(0), v.tmp); //grad(ni)/B^2= grad(ne)/B^2
         v.nabla.div(v.tmp[0], v.tmp[1], result);
         dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    {"v_Omega_D_gf", "Diamagnetic PCD GF", false, //CHECKED
        []( dg::x::DVec& result, Variables& v) {
         dg::blas1::pointwiseDot(v.f.binv(), v.f.binv(), v.tmp2[0]);
         routines::scal(v.tmp2[0], v.f.gradN(1), v.tmp); //grad(Ni)/B^2!= grad(ne)/B^2
         v.nabla.div(v.tmp[0], v.tmp[1], result);
         dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    
    /// ------------ Polarization advections ------------------//
    {"v_adv_E_tt", "Electric advective term (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {//CHECKED
             routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E
             dg::blas1::pointwiseDot(v.f.binv(), v.f.binv(), v.tmp2[0]);
             dg::blas1::pointwiseDot(v.tmp2[0], v.f.density(0), v.tmp2[0]);
             routines::scal(v.tmp2[0], v.f.gradP(0), v.tmp3); //ne Grad_phi/B^2
             v.nabla.div(v.tmp3[0], v.tmp3[1], result); //Div(n_e Grad_phi/B^2 )
             routines::scal(result, v.tmp, v.tmp2);//Div(n_e Grad_phi/B^2 )*u_E
             v.nabla.v_dot_nabla_f(v.tmp3[0], v.tmp3[1], v.tmp[0], v.tmp[0]);
             v.nabla.v_dot_nabla_f(v.tmp3[0], v.tmp3[1], v.tmp[1], v.tmp[1]); //ne grad(phi)/B^2*div(u_E)
             dg::blas1::axpby(1.0,v.tmp2, 1.0, v.tmp);//Div(n_e Grad_phi/B^2 )*u_E+  ne grad(phi)/B^2*div(u_E)
             v.nabla.div(v.tmp[0], v.tmp[1], result);//Div(Div(n_e Grad_phi/B^2 )*u_E+  ne grad(phi)/B^2*div(u_E)  )
             dg::blas1::scal( result, v.p.mu[1]);
        }
    },

    {"v_adv_E_gf_tt", "Electric advective term GF (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {//CHECKED
             routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E
             dg::blas1::pointwiseDot(v.f.binv(), v.f.binv(), v.tmp2[0]);
             dg::blas1::pointwiseDot(v.tmp2[0], v.f.density(1), v.tmp2[0]);
             routines::scal(v.tmp2[0], v.f.gradP(0), v.tmp3); //Ni Grad_phi/B^2
             v.nabla.div(v.tmp3[0], v.tmp3[1], result); //Div(Ni Grad_phi/B^2 )
             routines::scal(result, v.tmp, v.tmp2);//Div(Ni Grad_phi/B^2 )*u_E
             v.nabla.v_dot_nabla_f(v.tmp3[0], v.tmp3[1], v.tmp[0], v.tmp[0]);
             v.nabla.v_dot_nabla_f(v.tmp3[0], v.tmp3[1], v.tmp[1], v.tmp[1]); //Ni grad(phi)/B^2*div(u_E)
             dg::blas1::axpby(1.0,v.tmp2, 1.0, v.tmp);//Div(Ni Grad_phi/B^2 )*u_E+  Ni grad(phi)/B^2*div(u_E)
             v.nabla.div(v.tmp[0], v.tmp[1], result);//Div(Div(Ni Grad_phi/B^2 )*u_E+  Ni grad(phi)/B^2*div(u_E)  )
             dg::blas1::scal( result, v.p.mu[1]);
        }
    },
    {"v_adv_E_main_tt", "Main electric advective term (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
             routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E
             dg::blas1::pointwiseDot(v.f.density(0), v.f.binv(), v.tmp2[0]);
             dg::blas1::pointwiseDot(v.tmp2[0], v.f.binv(), v.tmp2[0]);
             routines::scal(v.tmp2[0], v.f.gradP(0), v.tmp3); //n_e Grad_phi/B^2
             v.nabla.div(v.tmp3[0], v.tmp3[1], result); //Div(n_e Grad_phi/B^2 )
             routines::scal(result, v.tmp, v.tmp);  //Div(n_e Grad_phi/B^2 )*u_E
             v.nabla.div(v.tmp[0], v.tmp[1], result);//Div(Div(n_e Grad_phi/B^2 )*u_E)
        }
    },
     
    {"v_adv_D_tt", "Diamagnetic advective term (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) { //CHECKED
             routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E
             dg::blas1::pointwiseDot(v.f.binv(), v.f.binv(), v.tmp2[0]);
             routines::scal(v.tmp2[0], v.f.gradN(0), v.tmp3); //Grad_n_e/B^2
             v.nabla.div(v.tmp3[0], v.tmp3[1], result);
             routines::scal(result, v.tmp, v.tmp2);
             v.nabla.v_dot_nabla_f(v.tmp3[0], v.tmp3[1], v.tmp[0], v.tmp[0]);
             v.nabla.v_dot_nabla_f(v.tmp3[0], v.tmp3[1], v.tmp[1], v.tmp[1]); //grad(n_e)/B^2*div u_E
             dg::blas1::axpby(1.0,v.tmp2, 1.0, v.tmp);
             v.nabla.div(v.tmp[0], v.tmp[1], result);
             dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },

    {"v_adv_D_gf_tt", "Diamagnetic advective term GF (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) { //CHECKED
             routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E
             dg::blas1::pointwiseDot(v.f.binv(), v.f.binv(), v.tmp2[0]);
             routines::scal(v.tmp2[0], v.f.gradN(1), v.tmp3); //Grad_N_i/B^2
             v.nabla.div(v.tmp3[0], v.tmp3[1], result);
             routines::scal(result, v.tmp, v.tmp2);
             v.nabla.v_dot_nabla_f(v.tmp3[0], v.tmp3[1], v.tmp[0], v.tmp[0]);
             v.nabla.v_dot_nabla_f(v.tmp3[0], v.tmp3[1], v.tmp[1], v.tmp[1]); //grad(N_i)/B^2*div u_E
             dg::blas1::axpby(1.0,v.tmp2, 1.0, v.tmp);
             v.nabla.div(v.tmp[0], v.tmp[1], result);
             dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    {"v_adv_D_main_tt", "Main diamagnetic term (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
             routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E
             dg::blas1::pointwiseDot(v.f.binv(), v.f.binv(), v.tmp2[0]);
             routines::scal(v.tmp2[0], v.f.gradN(0), v.tmp3); //Grad_N_i/B^2
             v.nabla.div(v.tmp3[0], v.tmp3[1], result);
             routines::scal(result, v.tmp, v.tmp3);
             v.nabla.div(v.tmp3[0], v.tmp3[1], result);
             dg::blas1::scal(result, v.p.tau[1]);
        }
    },

    ///---------------- J_b_perp components --------//
    {"v_M_em_tt", "Magnetization term (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) { //CHECKED
             routines::scal(v.f.velocity(1), v.f.gradN(0), v.tmp);
             routines::scal(v.f.density(0), v.f.gradU(1), v.tmp2);
             dg::blas1::axpby(1, v.tmp, 1, v.tmp2);
             routines::scal(v.f.binv(), v.tmp2, v.tmp3);
             routines::scal(v.f.binv(), v.tmp3, v.tmp2); //M^em
             routines::times(v.f.bhatgB(), v.f.gradA(), v.tmp); //b_perp
             v.nabla.div(v.tmp2[0], v.tmp2[1], result);
             routines::scal(result, v.tmp, v.tmp3);
             v.nabla.v_dot_nabla_f(v.tmp2[0], v.tmp2[1], v.tmp[0], v.tmp[0]);
             v.nabla.v_dot_nabla_f(v.tmp2[0], v.tmp2[1], v.tmp[1], v.tmp[1]); ////N grad(phi)/B^2*div u_E
             dg::blas1::axpby(1.0,v.tmp3, 1.0, v.tmp);
             v.nabla.div(v.tmp[0], v.tmp[1], result);
             dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },

    {"v_M_em_gf_tt", "Magnetization term GF(time integrated)", true,
        []( dg::x::DVec& result, Variables& v) { //CHECKED
             routines::scal(v.f.velocity(1), v.f.gradN(1), v.tmp);
             routines::scal(v.f.density(1), v.f.gradU(1), v.tmp2);
             dg::blas1::axpby(1, v.tmp, 1, v.tmp2);
             routines::scal(v.f.binv(), v.tmp2, v.tmp3);
             routines::scal(v.f.binv(), v.tmp3, v.tmp2); //M^em  GF
             routines::times(v.f.bhatgB(), v.f.gradA(), v.tmp); //b_perp
             v.nabla.div(v.tmp2[0], v.tmp2[1], result);
             routines::scal(result, v.tmp, v.tmp3);
             v.nabla.v_dot_nabla_f(v.tmp2[0], v.tmp2[1], v.tmp[0], v.tmp[0]);
             v.nabla.v_dot_nabla_f(v.tmp2[0], v.tmp2[1], v.tmp[1], v.tmp[1]); ////N grad(phi)/B^2*div u_E
             dg::blas1::axpby(1.0,v.tmp3, 1.0, v.tmp);
             v.nabla.div(v.tmp[0], v.tmp[1], result);
             dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    {"v_J_mag_tt", "Magnetization current term (time integrated)", true, //FINAL
        []( dg::x::DVec& result, Variables& v) {
             routines::scal(v.f.velocity(1), v.f.gradN(0), v.tmp);
             routines::scal(v.f.density(0), v.f.gradU(1), v.tmp2);
             dg::blas1::axpby(1.0, v.tmp, 1.0, v.tmp2);
             routines::scal(v.f.binv(), v.tmp2, v.tmp3);
             routines::scal(v.f.binv(), v.tmp3, v.tmp2); //M^em
             routines::times(v.f.bhatgB(), v.f.gradA(), v.tmp); //b_perp
             v.nabla.div(v.tmp2[0], v.tmp2[1], result);
             routines::scal(result, v.tmp, v.tmp3);
             v.nabla.div(v.tmp3[0], v.tmp3[1], result);
             dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]*0.5);
        }
    },
    {"v_J_mag_gf_tt", "Magnetization current term GF (time integrated)", true, //FINAL
        []( dg::x::DVec& result, Variables& v) {
             routines::scal(v.f.velocity(1), v.f.gradN(1), v.tmp);
             routines::scal(v.f.density(1), v.f.gradU(1), v.tmp2);
             dg::blas1::axpby(1.0, v.tmp, 1.0, v.tmp2);
             routines::scal(v.f.binv(), v.tmp2, v.tmp3);
             routines::scal(v.f.binv(), v.tmp3, v.tmp2); //M^em
             routines::times(v.f.bhatgB(), v.f.gradA(), v.tmp); //b_perp
             v.nabla.div(v.tmp2[0], v.tmp2[1], result);
             routines::scal(result, v.tmp, v.tmp3);
             v.nabla.div(v.tmp3[0], v.tmp3[1], result);
             dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]*0.5);
        }
    },
    {"v_J_bperp_tt", "Div J_par times b_perp term (time integrated)", true, //FINAL
        []( dg::x::DVec& result, Variables& v) {
             routines::times(v.f.bhatgB(), v.f.gradA(), v.tmp); //b_perp
             dg::blas1::pointwiseDot(v.f.density(0), v.f.velocity(1), v.tmp2[0]);
             dg::blas1::pointwiseDot(-1., v.f.density(0), v.f.velocity(0), 1., v.tmp2[0]); //J_||
             routines::scal(v.tmp2[0], v.tmp, v.tmp3);
             v.nabla.div(v.tmp3[0], v.tmp3[1], result);
                    
        }
    },
    
    ///-------------- Parallel divergence of parallel current -------//
    {"v_J_par_tt", "Parallel current term (time integrated)", true, //FINAL
        []( dg::x::DVec& result, Variables& v) {
             dg::blas1::copy( v.f.divNUb(0), result);
             dg::blas1::copy( v.f.divNUb(1), v.tmp[0]);
             dg::blas1::axpby(1, v.tmp[0], -1, result);
        }
    },
    
    ///------------- Curvature current components ------------///
    {"v_J_D_tt", "Diamagnetic current (time integrated)", true, //FINAL
        []( dg::x::DVec& result, Variables& v) {
        routines::dot(v.f.gradN(0), v.f.curv(), result);
        dg::blas1::scal(result, v.p.tau[1]-v.p.tau[0]);
        }
    },
    {"v_J_D_gf_tt", "Diamagnetic current (time integrated)", true, //FINAL
        []( dg::x::DVec& result, Variables& v) {
        routines::dot(v.f.gradN(1), v.f.curv(), result);
        dg::blas1::scal(result, v.p.tau[1]);
        routines::dot(v.f.gradN(0), v.f.curv(), v.tmp[0]);
        dg::blas1::scal(v.tmp[0], v.p.tau[0]);
        dg::blas1::axpby(-1.0, v.tmp[0], 1.0, result);
        }
    },
    {"v_J_JAK_tt", "JAK Parallel current with curvature Kappa term(time integrated)", true, //FINAL
        []( dg::x::DVec& result, Variables& v) {
        dg::blas1::pointwiseDot(v.f.density(0), v.f.velocity(1), v.tmp[0]);
            dg::blas1::pointwiseDot(-1., v.f.density(0), v.f.velocity(0), 1., v.tmp[0]); //J_||
            dg::blas1::pointwiseDot(v.f.aparallel(), v.tmp[0], v.tmp[0]);
            routines::scal(v.tmp[0], v.f.curvKappa(), v.tmp2);
            v.nabla.div(v.tmp2[0], v.tmp2[1], result);
        }
    },

    {"v_J_JAK_gf_tt", "JAK Parallel current with curvature Kappa term GF(time integrated)", true, //FINAL
        []( dg::x::DVec& result, Variables& v) {
        dg::blas1::pointwiseDot(v.f.density(1), v.f.velocity(1), v.tmp[0]);
            dg::blas1::pointwiseDot(-1., v.f.density(0), v.f.velocity(0), 1., v.tmp[0]); //J_||
            dg::blas1::pointwiseDot(v.f.aparallel(), v.tmp[0], v.tmp[0]);
            routines::scal(v.tmp[0], v.f.curvKappa(), v.tmp2);
            v.nabla.div(v.tmp2[0], v.tmp2[1], result);
        }
    },
  {"v_J_NUK_tt", "NUK Curvature current term (time integrated)", true, //FINAL
        []( dg::x::DVec& result, Variables& v) {
        dg::blas1::pointwiseDot(v.f.velocity(0), v.f.velocity(0), v.tmp[0]);
        dg::blas1::pointwiseDot(v.f.velocity(1), v.f.velocity(1), v.tmp[1]);
        dg::blas1::pointwiseDot(v.f.density(0), v.tmp[0], v.tmp[0]);
        dg::blas1::pointwiseDot(v.f.density(0), v.tmp[1], v.tmp[1]);
        dg::blas1::scal(v.tmp[0], v.p.mu[0]);
        dg::blas1::scal(v.tmp[1], v.p.mu[1]);
        dg::blas1::axpby(1.0, v.tmp[1], -1.0, v.tmp[0]);
        routines::scal(v.tmp[0], v.f.curvKappa(), v.tmp2);
        v.nabla.div(v.tmp2[0], v.tmp2[1], result);
        }
    },

    {"v_J_NUK_gf_tt", "NUK Curvature current term (time integrated)", true, //FINAL
        []( dg::x::DVec& result, Variables& v) {
        dg::blas1::pointwiseDot(v.f.velocity(0), v.f.velocity(0), v.tmp[0]);
        dg::blas1::pointwiseDot(v.f.velocity(1), v.f.velocity(1), v.tmp[1]);
        dg::blas1::pointwiseDot(v.f.density(0), v.tmp[0], v.tmp[0]);
        dg::blas1::pointwiseDot(v.f.density(1), v.tmp[1], v.tmp[1]);
        dg::blas1::scal(v.tmp[0], v.p.mu[0]);
        dg::blas1::scal(v.tmp[1], v.p.mu[1]);
        dg::blas1::axpby(1.0, v.tmp[1], -1.0, v.tmp[0]);
        routines::scal(v.tmp[0], v.f.curvKappa(), v.tmp2);
        v.nabla.div(v.tmp2[0], v.tmp2[1], result);
        }
    },

    ///-------------- Sources term----------/// (MISSING GYRO-FLUID DEFINITIONS)
    {"v_S_E_tt", "Electric source vorticity (time integrated)", true, //FINAL
        []( dg::x::DVec& result, Variables& v) {
            routines::scal(v.f.density_source(0), v.f.gradP(0), v.tmp);
            routines::scal(v.f.binv(), v.tmp, v.tmp2);
            routines::scal(v.f.binv(), v.tmp2, v.tmp);
            v.nabla.div(v.tmp[0], v.tmp[1], result);
            dg::blas1::scal( result, v.p.mu[1]);
        }
    },
    {"v_S_D_tt", "Diamagnetic source vorticity (time integrated)", true, //FINAL
        []( dg::x::DVec& result, Variables& v) {
             v.f.compute_gradSN(0,  v.tmp);
             routines::scal(v.f.binv(), v.tmp, v.tmp2);
             routines::scal(v.f.binv(), v.tmp2, v.tmp);
             v.nabla.div(v.tmp[0], v.tmp[1], result);
             dg::blas1::scal(result, v.p.tau[1]*v.p.mu[1]);
        }
    },

    /// ------------- Radial Force Balance terms ---------///
    {"RFB_E_r_GradPsip_tt", "Radial electric field in RFB (time integrated)", true,
         []( dg::x::DVec& result, Variables& v){
             routines::dot(v.f.gradP(0), v.gradPsip, result);
             dg::blas1::scal( result, -1.);
         }
     },
    {"RFB_GradPi_GradPsip_tt", "Radial pressure gradient component of RFB (time integrated)", true,
         []( dg::x::DVec& result, Variables& v){
             routines::dot(v.f.gradN(0), v.gradPsip, result);
             dg::blas1::pointwiseDivide(result, v.f.density(0), result);
             dg::blas1::scal( result, v.p.tau[1]);
         }
     },

    
};

std::vector<Record> probe_list = {
     {"electrons_probe", "probe measurement of electron density", false,
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.density(0), result);
         }
     },
     {"ions_probe", "probe measurement of ion density", false,
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.density(1), result);
         }
     },
     {"Ue_probe", "probe measurement of parallel electron velocity", false,
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.velocity(0), result);
         }
     },
     {"Ui_probe", "probe measurement of parallel ion velocity", false,
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.velocity(1), result);
         }
     },
     {"potential_probe", "probe measurement of electric potential", false,
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.potential(0), result);
         }
     },
     {"aparallel_probe", "probe measurement of parallel magnetic potential", false,
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.aparallel(), result);
         }
     }
 };

// Here is a list of useful 1d variables of general interest
std::vector<Record1d> diagnostics1d_list = {
    {"failed", "Accumulated Number of failed steps",
        []( Variables& v ) {
            return *v.nfailed;
        }
    },
    {"duration", "Computation time between the latest 3d outputs (without the output time itself)",
        []( Variables& v ) {
            return v.duration;
        }
    },
    {"nsteps", "Accumulated Number of calls to the right-hand-side (including failed steps)",
        [](Variables& v) {
            return v.f.called();
        }
    }
};

///%%%%%%%%%%%%%%%%%%%%%%%%%%END DIAGNOSTICS LIST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
///%%%%%%%%%%%%%%%%%%%%%%%%%%END DIAGNOSTICS LIST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
///%%%%%%%%%%%%%%%%%%%%%%%%%%END DIAGNOSTICS LIST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
std::vector<Record> restart3d_list = {
    {"restart_electrons", "electron density", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.restart_density(0), result);
        }
    },
    {"restart_ions", "ion density", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.restart_density(1), result);
        }
    },
    {"restart_Ue", "parallel electron velocity", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.restart_velocity(0), result);
        }
    },
    {"restart_Ui", "parallel ion velocity", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.restart_velocity(1), result);
        }
    },
    {"restart_aparallel", "parallel magnetic potential", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.restart_aparallel(), result);
        }
    }
};

template<class Container>
void slice_vector3d( const Container& transfer, Container& transfer2d, size_t local_size2d)
{
#ifdef WITH_MPI
    thrust::copy(
        transfer.data().begin(),
        transfer.data().begin() + local_size2d,
        transfer2d.data().begin()
    );
#else
    thrust::copy(
        transfer.begin(),
        transfer.begin() + local_size2d,
        transfer2d.begin()
    );
#endif
}
}//namespace feltor
