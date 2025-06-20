#pragma once

#include <string>
#include <vector>
#include <functional>

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"

#include "feltor.h"
#include "parameters.h"

#include "init.h"
#include "common.h"

namespace feltor{

// This file constitutes the diagnostics module for feltor
// The way it works is that it allocates global lists of Records that describe what goes into the file
// You can register you own diagnostics in one of three diagnostics lists (static 3d, dynamic 3d and
// dynamic 2d) further down
// which will then be applied during a simulation
struct RadialEnergyFlux{
    RadialEnergyFlux( double tau, double mu, double z):
        m_tau(tau), m_mu(mu), m_z(z){
    }

    DG_DEVICE void operator()( double ne, double ue, double P,
        double d0P, double d1P, double d2P, //Phi
        double& jE0, double& jE1, double& jE2, //j_E
        double b_0,         double b_1,         double b_2,
        double curv0,  double curv1,  double curv2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        // J_N
        jE0 = ne*(b_1*d2P - b_2*d1P + m_mu*ue*ue*curvKappa0 + m_tau*curv0);
        jE1 = ne*(b_2*d0P - b_0*d2P + m_mu*ue*ue*curvKappa1 + m_tau*curv1);
        jE2 = ne*(b_0*d1P - b_1*d0P + m_mu*ue*ue*curvKappa2 + m_tau*curv2);
        jE0 = m_z*(m_tau * log(ne <=0 ? 1e-16 : ne) + 0.5*m_mu*ue*ue + P)*jE0
            + m_z*m_mu*m_tau*ne*ue*ue*curvKappa0;
        jE1 = m_z*(m_tau * log(ne <=0 ? 1e-16 : ne) + 0.5*m_mu*ue*ue + P)*jE1
            + m_z*m_mu*m_tau*ne*ue*ue*curvKappa1;
        jE2 = m_z*(m_tau * log(ne <=0 ? 1e-16 : ne) + 0.5*m_mu*ue*ue + P)*jE2
            + m_z*m_mu*m_tau*ne*ue*ue*curvKappa2;
    }
    DG_DEVICE void operator()( double ne, double ue, double P, double A,
        double d0A, double d1A, double d2A,
        double& jE0, double& jE1, double& jE2, //jE
        double b_0,         double b_1,         double b_2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        // NU bperp
        jE0 = ne*ue*(b_2*d1A - b_1*d2A + A*curvKappa0);
        jE1 = ne*ue*(b_0*d2A - b_2*d0A + A*curvKappa1);
        jE2 = ne*ue*(b_1*d0A - b_0*d1A + A*curvKappa2);
        jE0 = m_z*(m_tau * log(ne <=0 ? 1e-16 : ne) + 0.5*m_mu*ue*ue + P)*jE0
            + m_z*m_tau*jE0;
        jE1 = m_z*(m_tau * log(ne <=0 ? 1e-16 : ne) + 0.5*m_mu*ue*ue + P)*jE1
            + m_z*m_tau*jE1;
        jE2 = m_z*(m_tau * log(ne <=0 ? 1e-16 : ne) + 0.5*m_mu*ue*ue + P)*jE2
            + m_z*m_tau*jE2;
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
        return m_z*(m_tau*(1+log(ne <= 0 ? 1e-16 : ne))+P-0.5*m_mu*ue*ue)*source;
    }
    private:
    double m_tau, m_mu, m_z;
};
struct PositiveLN
{
    DG_DEVICE double operator()(double ne)
    {
        return log(ne <= 0 ? 1e-16 : ne); // avoid nans in output
    }
};
struct Positive
{
    DG_DEVICE double operator()(double ne)
    {
        return ne <= 0 ? 1e-16 : ne; // avoid nans in output
    }
};


//From here on, we use the typedefs to ease the notation

struct Variables{
    feltor::Explicit<dg::x::CylindricalGrid3d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec>& f;
    std::array<std::array<dg::x::DVec,2>,2>& y0;
    feltor::Parameters p;
    dg::geo::TokamakMagneticField mag;
    const std::array<dg::x::DVec, 3>& gradPsip;
    std::array<dg::x::DVec, 3> tmp;
    std::array<dg::x::DVec, 3> tmp2;
    std::array<dg::x::DVec, 3> tmp3;
    double duration;
    const unsigned* nfailed;
};

struct Record{
    std::string name;
    std::string long_name;
    bool integral; //indicates whether the function should be time-integrated
    std::function<void( dg::x::DVec&, Variables&)> function;
};

///%%%%%%%%%%%%%%%%%%%%EXTEND LISTS WITH YOUR DIAGNOSTICS HERE%%%%%%%%%%%%%%%%%%%%%%
///%%%%%%%%%%%%%%%%%%%%EXTEND LISTS WITH YOUR DIAGNOSTICS HERE%%%%%%%%%%%%%%%%%%%%%%
///%%%%%%%%%%%%%%%%%%%%EXTEND LISTS WITH YOUR DIAGNOSTICS HERE%%%%%%%%%%%%%%%%%%%%%%
//Here is a list of static (time-independent) 3d variables that go into the output
//Cannot be feltor internal variables
std::vector<dg::file::Record<void( dg::x::HVec&, const dg::geo::TokamakMagneticField&, const dg::x::CylindricalGrid3d&), dg::file::LongNameAttribute>> diagnostics3d_static_list = {
    { "BR", "R-component of magnetic field in cylindrical coordinates",
        []( dg::x::HVec& result, const dg::geo::TokamakMagneticField& mag, const dg::x::CylindricalGrid3d& grid){
            dg::geo::BFieldR fieldR(mag);
            result = dg::pullback( fieldR, grid);
        }
    },
    { "BZ", "Z-component of magnetic field in cylindrical coordinates",
        []( dg::x::HVec& result, const dg::geo::TokamakMagneticField& mag, const dg::x::CylindricalGrid3d& grid){
            dg::geo::BFieldZ fieldZ(mag);
            result = dg::pullback( fieldZ, grid);
        }
    },
    { "BP", "Contravariant P-component of magnetic field in cylindrical coordinates",
        []( dg::x::HVec& result, const dg::geo::TokamakMagneticField& mag, const dg::x::CylindricalGrid3d& grid){
            dg::geo::BFieldP fieldP(mag);
            result = dg::pullback( fieldP, grid);
        }
    },
    { "Psip", "Flux-function psi",
        []( dg::x::HVec& result, const dg::geo::TokamakMagneticField& mag, const dg::x::CylindricalGrid3d& grid){
             result = dg::pullback( mag.psip(), grid);
        }
    },
    { "vol3d", "Volume form in 3d",
        []( dg::x::HVec& result, const dg::geo::TokamakMagneticField&, const dg::x::CylindricalGrid3d& grid){
             result = dg::create::volume( grid);
        }
    },
    { "xc", "x-coordinate in Cartesian coordinate system",
        []( dg::x::HVec& result, const dg::geo::TokamakMagneticField&, const dg::x::CylindricalGrid3d& grid ){
            result = dg::evaluate( dg::cooRZP2X, grid);
        }
    },
    { "yc", "y-coordinate in Cartesian coordinate system",
        []( dg::x::HVec& result, const dg::geo::TokamakMagneticField&, const dg::x::CylindricalGrid3d& grid ){
            result = dg::evaluate( dg::cooRZP2Y, grid);
        }
    },
    { "zc", "z-coordinate in Cartesian coordinate system",
        []( dg::x::HVec& result, const dg::geo::TokamakMagneticField&, const dg::x::CylindricalGrid3d& grid ){
            result = dg::evaluate( dg::cooRZP2Z, grid);
        }
    },
};

// Here are all 3d outputs we want to have
std::vector<dg::file::Record<void(dg::x::DVec&, Variables&), dg::file::LongNameAttribute>> diagnostics3d_list = { // 6
    {"electrons", "electron density",
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(0), result);
        }
    },
    {"ions", "ion density",
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(1), result);
        }
    },
    {"Ue", "parallel electron velocity",
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(0), result);
        }
    },
    {"Ui", "parallel ion velocity",
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(1), result);
        }
    },
    {"potential", "electric potential",
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.potential(0), result);
        }
    },
    {"aparallel", "parallel magnetic potential",
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.aparallel(), result);
        }
    }
};

//Here is a list of static (time-independent) 2d variables that go into the output
//MW: if they stay they should be documented in feltor.tex
//MW: we should add initialization and source terms here
//( we make 3d variables here but only the first 2d slice is output)
std::vector<dg::file::Record<void(dg::x::HVec&, Variables&, const dg::x::CylindricalGrid3d&), dg::file::LongNameAttribute>> diagnostics2d_static_list = {
    { "Psip2d", "Flux-function psi",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.psip(), grid);
        }
    },
    { "PsipR2d", "Flux-function psi R-derivative",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.psipR(), grid);
        }
    },
    { "PsipZ2d", "Flux-function psi Z-derivative",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.psipZ(), grid);
        }
    },
    { "PsipRR2d", "Flux-function psi RR-derivative",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.psipRR(), grid);
        }
    },
    { "PsipRZ2d", "Flux-function psi RZ-derivative",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.psipRZ(), grid);
        }
    },
    { "PsipZZ2d", "Flux-function psi ZZ-derivative",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.psipZZ(), grid);
        }
    },
    { "Ipol", "Poloidal current",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.ipol(), grid);
        }
    },
    { "IpolR", "Poloidal current R-derivative",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.ipolR(), grid);
        }
    },
    { "IpolZ", "Poloidal current Z-derivative",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( v.mag.ipolZ(), grid);
        }
    },
    { "Rho_p", "Normalized Poloidal flux label",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( dg::geo::RhoP( v.mag), grid);
        }
    },
    { "Bmodule", "Magnetic field strength",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid ){
            result = dg::pullback( dg::geo::Bmodule(v.mag), grid);
        }
    },
    { "Divb", "The divergence of the magnetic unit vector",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& ){
            dg::assign( v.f.divb(), result);
        }
    },
    { "InvB", "Inverse of Bmodule",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& ){
            dg::assign( v.f.binv(), result);
        }
    },
    { "CurvatureKappaR", "R-component of the Kappa B curvature vector",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& ){
            dg::assign( v.f.curvKappa()[0], result);
        }
    },
    { "CurvatureKappaZ", "Z-component of the Kappa B curvature vector",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d&){
            dg::assign( v.f.curvKappa()[1], result);
        }
    },
    { "CurvatureKappaP", "Contravariant Phi-component of the Kappa B curvature vector",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d&){
            dg::assign( v.f.curvKappa()[2], result);
        }
    },
    { "DivCurvatureKappa", "Divergence of the Kappa B curvature vector",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d&){
            dg::assign( v.f.divCurvKappa(), result);
        }
    },
    { "CurvatureR", "R-component of the curvature vector",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d&){
            dg::assign( v.f.curv()[0], result);
        }
    },
    { "CurvatureZ", "Z-component of the full curvature vector",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d&){
            dg::assign( v.f.curv()[1], result);
        }
    },
    { "CurvatureP", "Contravariant Phi-component of the full curvature vector",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d&){
            dg::assign( v.f.curv()[2], result);
        }
    },
    { "bphi", "Covariant Phi-component of the magnetic unit vector",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d&){
            dg::assign( v.f.bphi(), result);
        }
    },
    { "BHatR", "R-component of the magnetic field unit vector",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid){
            result = dg::pullback( dg::geo::BHatR(v.mag), grid);
        }
    },
    { "BHatZ", "Z-component of the magnetic field unit vector",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid){
            result = dg::pullback( dg::geo::BHatZ(v.mag), grid);
        }
    },
    { "BHatP", "P-component of the magnetic field unit vector",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid){
            result = dg::pullback( dg::geo::BHatP(v.mag), grid);
        }
    },
    { "NormGradPsip", "Norm of gradient of Psip",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& grid){
            result = dg::pullback(
                dg::geo::SquareNorm( dg::geo::createGradPsip(v.mag),
                    dg::geo::createGradPsip(v.mag)), grid);
        }
    },
    { "Wall", "Wall Region",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& ){
            dg::assign( v.f.get_wall(), result);
        }
    },
    { "Sheath", "Sheath Region",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& ){
            dg::assign( v.f.get_sheath(), result);
        }
    },
    { "SheathCoordinate", "Sheath Coordinate of field lines",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& ){
            dg::assign( v.f.get_sheath_coordinate(), result);
        }
    },
    { "Nprof", "Density profile (that the source may force)",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& ){
            dg::assign( v.f.get_source_prof(), result);
        }
    },
    { "Source", "Source region",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& ){
            dg::assign( v.f.get_source(), result);
        }
    },
    { "neinit", "Initial condition for electrons",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& ){
            dg::assign( v.y0[0][0], result);
        }
    },
    { "niinit", "Initial condition for ions",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& ){
            dg::assign( v.y0[0][1], result);
        }
    },
    { "weinit", "Initial condition for electron canonical velocity",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& ){
            dg::assign( v.y0[1][0], result);
        }
    },
    { "wiinit", "Initial condition for ion canonical velocity",
        []( dg::x::HVec& result, Variables& v, const dg::x::CylindricalGrid3d& ){
            dg::assign( v.y0[1][1], result);
        }
    },
    { "vol2d", "Volume form (including R) in 2d",
        []( dg::x::HVec& result, Variables&, const dg::x::CylindricalGrid3d& grid ){
            result = dg::create::volume(grid);
        }
    },
};

// and here are all the 2d outputs we want to produce (currently ~ 150)
std::vector<Record> basicDiagnostics2d_list = { // 22
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
    {"gammaNi", "Gamma Ni", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.gammaNi(), result);
        }
    },
    {"gammaPhi", "Gamma Phi", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.gammaPhi(), result);
        }
    },
    /// -----------------Miscellaneous additions --------------------//
    {"vorticity", "Minus Lap_perp of electric potential", false,
        []( dg::x::DVec& result, Variables& v ) {
            // has no jump terms
            v.f.compute_lapMperpP(0, result);
        }
    },
    {"vorticity_i", "Minus Lap_perp of ion potential", false,
        []( dg::x::DVec& result, Variables& v ) {
            // has no jump terms
            v.f.compute_lapMperpP(1, result);
        }
    },
    {"laplace_ne", "Positive Lap_perp of electron density", false,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_lapMperpN(-1.0, v.f.density(0), v.tmp[0], 0., result);
        }
    },
    {"laplace_ni", "Positive Lap_perp of ion density", false,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_lapMperpN(-1.0, v.f.density(1), v.tmp[0], 0., result);
        }
    },
    // Does not work due to direct application of Laplace
    // The Laplacian of Aparallel looks smooth in paraview
    {"apar_vorticity", "Minus Lap_perp of magnetic potential", false,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_lapMperpA( result);
        }
    },
    {"dssue", "2nd parallel derivative of electron velocity", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::copy( v.f.dssU( 0), result);
        }
    },
    {"lperpinv", "Perpendicular density gradient length scale", false,
        []( dg::x::DVec& result, Variables& v ) {
            const std::array<dg::x::DVec, 3>& dN = v.f.gradN(0);
            dg::blas1::pointwiseDivide( 1., v.f.density(0), v.tmp[0]);
            dg::tensor::scalar_product3d( 1., v.tmp[0],
                dN[0], dN[1], dN[2], v.f.projection(), v.tmp[0], //grad_perp
                dN[0], dN[1], dN[2], 0., result); // ((grad N)/N)**2
            dg::blas1::transform( result, result, dg::SQRT<double>());
        }
    },
    {"perpaligned", "Perpendicular density alignement", false,
        []( dg::x::DVec& result, Variables& v ) {
            const std::array<dg::x::DVec, 3>& dN = v.f.gradN(0);
            dg::tensor::scalar_product3d( 1., 1.,
                dN[0], dN[1], dN[2], v.f.projection(), 1., //grad_perp
                dN[0], dN[1], dN[2], 0., result); // (grad N)**2
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
            dg::blas1::pointwiseDot ( v.f.dsN(0), v.f.dsN(0), result);
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

std::vector<Record> MassConsDiagnostics2d_list = { // 26
    /// ------------------ Density terms ------------------------//
    ////////////////// electron particle flux /////////////////////
    {"jsneE_tt", "Radial electron particle flux: ExB contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( result, v.f.density(0), result);
        }
    },
    {"divneE_tt", "Divergence of ExB electron particle flux (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.f.curv(), v.f.gradP(0), result);
            dg::blas1::pointwiseDot( result, v.f.density(0), result);
            routines::jacobian( 1., v.f.bhatgB(), v.f.gradP(0), v.f.gradN(0), 1., result);
        }
    },
    {"jscurvne_tt", "Radial electron particle flux: curvature contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            routines::dot( v.f.curv(), v.gradPsip, result);
            dg::blas1::pointwiseDot( v.p.tau[0], v.f.density(0), result, 0., result);
        }
    },
    {"divcurvne_tt", "Divergence of curvature term (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.p.tau[0], v.f.curv(), v.f.gradN(0), 0., result);
        }
    },
    {"jscurvkappane_tt", "Radial electron particle flux: curvature contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.p.mu[0], v.f.density(0), v.f.velocity(0),
                    v.f.velocity(0), 0., result);
            routines::dot( v.f.curvKappa(), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( v.tmp[0], result, result);
        }
    },
    {"divcurvkappane_tt", "Divergence of curvature term (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            dg::blas1::pointwiseDot( v.p.mu[0], v.f.density(0), v.f.velocity(0),
                    v.f.velocity(0), 0., v.tmp3[0]);
            v.f.centered_div( v.tmp3[0], v.f.curvKappa(), v.tmp2[0], result);
        }
    },
    {"jsneA_tt", "Radial electron particle flux: magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_bperp(v.tmp);
            routines::dot( v.tmp, v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., v.f.density(0), v.f.velocity(0), result, 0., result);
        }
    },
    {"divneA_tt", "Divergence of magnetic flutter electron particle flux (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            dg::blas1::pointwiseDot( v.f.density(0), v.f.velocity(0), v.tmp3[0]);
            v.f.compute_bperp(v.tmp);
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"jsdiae_tt", "Radial electron particle flux: diamagnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            // u_D Dot GradPsi
            routines::jacobian( v.p.tau[0], v.f.bhatgB(), v.f.gradN(0), v.gradPsip, 0., result);
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
    /// ------------------ Density terms ------------------------//
    //////////////////// ion particle flux ////////////////////////
    {"jsniE_tt", "Radial ion particle flux: ExB contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(1), v.gradPsip, result);
            dg::blas1::pointwiseDot( result, v.f.density(1), result);
        }
    },
    {"divniE_tt", "Divergence of ExB ion particle flux (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.f.curv(), v.f.gradP(1), result);
            dg::blas1::pointwiseDot( result, v.f.density(1), result);
            routines::jacobian( 1., v.f.bhatgB(), v.f.gradP(1), v.f.gradN(1), 1., result);
        }
    },
    {"jscurvni_tt", "Radial ion particle flux: curvature contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            routines::dot( v.f.curv(), v.gradPsip, result);
            dg::blas1::pointwiseDot( v.p.tau[1], v.f.density(1), result, 0., result);
        }
    },
    {"divcurvni_tt", "Divergence of curvature term (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            routines::dot( v.p.tau[1], v.f.curv(), v.f.gradN(1), 0., result);
        }
    },
    {"jscurvkappani_tt", "Radial ion particle flux: curvature contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1),
                    v.f.velocity(1), 0., result);
            routines::dot( v.f.curvKappa(), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( v.tmp[0], result, result);
        }
    },
    {"divcurvkappani_tt", "Divergence of curvature term (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1),
                    v.f.velocity(1), 0., v.tmp3[0]);
            v.f.centered_div( v.tmp3[0], v.f.curvKappa(), v.tmp2[0], result);
        }
    },
    {"jsniA_tt", "Radial ion particle flux: magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_bperp(v.tmp);
            routines::dot( v.tmp, v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., v.f.density(1), v.f.velocity(1), result, 0., result);
        }
    },
    {"divniA_tt", "Divergence of magnetic flutter ion particle flux (Time average)", true,
        []( dg::x::DVec& result, Variables& v) {
            dg::blas1::pointwiseDot( v.f.density(1), v.f.velocity(1), v.tmp3[0]);
            v.f.compute_bperp(v.tmp);
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"jsdiai_tt", "Radial ion particle flux: diamagnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            // u_D Dot GradPsi
            routines::jacobian( v.p.tau[1], v.f.bhatgB(), v.f.gradN(1), v.gradPsip, 0., result);
        }
    },
    {"lniperp_tt", "Perpendicular ion diffusion (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_perp_diffusiveN( 1., v.f.density(1), v.tmp[0],
                    v.tmp[1], 0., result);
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

std::vector<Record> EnergyDiagnostics2d_list = { // 23
    /// ------------------- Energy terms ------------------------//
    {"nelnne", "Entropy electrons", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(0), result, PositiveLN());
            dg::blas1::pointwiseDot( result, v.f.density(0), result);
        }
    },
    {"nilnni", "Entropy ions", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(1), result, PositiveLN());
            dg::blas1::pointwiseDot( v.p.tau[1], result, v.f.density(1), 0., result);
        }
    },
    {"aperp2", "Magnetic energy", false,
        []( dg::x::DVec& result, Variables& v ) {
            if( v.p.beta == 0) // avoid divison by zero
            {
                dg::blas1::scal( result, 0.);
            }
            else
            {
                dg::tensor::scalar_product3d( 1./2./v.p.beta, 1.,
                    v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                    v.f.projection(), 1., //grad_perp
                    v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2], 0., result);
            }
        }
    },
    {"ue2", "ExB energy", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 0.5, v.f.density(1), v.f.uE2(), 0., result);
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
                RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.f.density_source(0)
            );
        }
    },
    {"sei_tt", "Energy sink/source for ions", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.f.density_source(1)
            );
        }
    },
    /// ------------------ Energy flux terms ------------------------//
    {"jsee_tt", "Radial electron energy flux without magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::subroutine(
                RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.f.gradP(0)[0], v.f.gradP(0)[1], v.f.gradP(0)[2],
                v.tmp[0], v.tmp[1], v.tmp[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            routines::dot( v.tmp, v.gradPsip, result);
        }
    },
    {"jseea_tt", "Radial electron energy flux: magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::subroutine(
                RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0), v.f.aparallel(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.tmp[0], v.tmp[1], v.tmp[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            routines::dot( v.tmp, v.gradPsip, result);
        }
    },
    {"jsei_tt", "Radial ion energy flux without magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::subroutine(
                RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.f.gradP(1)[0], v.f.gradP(1)[1], v.f.gradP(1)[2],
                v.tmp[0], v.tmp[1], v.tmp[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            routines::dot( v.tmp, v.gradPsip, result);
        }
    },
    {"jseia_tt", "Radial ion energy flux: magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::subroutine(
                RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1), v.f.aparallel(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.tmp[0], v.tmp[1], v.tmp[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            routines::dot( v.tmp, v.gradPsip, result);
        }
    },
    {"divee_tt", "Radial electron energy flux without magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::subroutine(
                RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.f.gradP(0)[0], v.f.gradP(0)[1], v.f.gradP(0)[2],
                v.tmp[0], v.tmp[1], v.tmp[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            v.f.centered_div( 1., v.tmp, v.tmp2[0], result);
        }
    },
    {"diveea_tt", "Radial electron energy flux: magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::subroutine(
                RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0), v.f.aparallel(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.tmp[0], v.tmp[1], v.tmp[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            v.f.centered_div( 1., v.tmp, v.tmp2[0], result);
        }
    },
    {"divei_tt", "Radial ion energy flux without magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::subroutine(
                RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.f.gradP(1)[0], v.f.gradP(1)[1], v.f.gradP(1)[2],
                v.tmp[0], v.tmp[1], v.tmp[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            v.f.centered_div( 1., v.tmp, v.tmp2[0], result);
        }
    },
    {"diveia_tt", "Radial ion energy flux: magnetic contribution (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::subroutine(
                RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1), v.f.aparallel(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.tmp[0], v.tmp[1], v.tmp[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            v.f.centered_div( 1., v.tmp, v.tmp2[0], result);
        }
    },
    /// ------------------------ Energy dissipation terms ------------------//
    {"leeperp_tt", "Perpendicular electron energy dissipation (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_perp_diffusiveN( 1., v.f.density(0), v.tmp3[0], v.tmp3[1],
                    0., v.tmp[0]);
            v.f.compute_perp_diffusiveU( 1., v.f.velocity(0), v.f.density(0),
                    v.tmp3[0], v.tmp3[1], v.tmp3[2], v.tmp2[0], 0., v.tmp[1]);
            dg::blas1::evaluate( result, dg::equals(),
                RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.tmp[0], v.tmp[1]
            );
        }
    },
    {"leiperp_tt", "Perpendicular ion energy dissipation (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_perp_diffusiveN( 1., v.f.density(1), v.tmp3[0], v.tmp3[1],
                    0., v.tmp[0]);
            v.f.compute_perp_diffusiveU( 1., v.f.velocity(1), v.f.density(1),
                    v.tmp3[0], v.tmp3[1], v.tmp3[2], v.tmp2[0], 0., v.tmp[1]);
            dg::blas1::evaluate( result, dg::equals(),
                RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.tmp[0], v.tmp[1]
            );
        }
    },
    {"leeparallel_tt", "Parallel electron energy dissipation (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_parallel_diffusiveN( 0, v.tmp[0]);
            v.f.compute_parallel_diffusiveU( 0, v.tmp[1]);
            dg::blas1::evaluate( result, dg::equals(),
                RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.tmp[0], v.tmp[1]
            );
        }
    },
    {"leiparallel_tt", "Parallel ion energy dissipation (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_parallel_diffusiveN( 1, v.tmp[0]);
            v.f.compute_parallel_diffusiveU( 1, v.tmp[1]);
            dg::blas1::evaluate( result, dg::equals(),
                RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
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
                RadialEnergyFlux( v.p.tau[0], v.p.mu[0], z),
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.f.divNUb(0), 0.
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
                RadialEnergyFlux( v.p.tau[1], v.p.mu[1], z),
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.f.divNUb(1), 0.
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

std::vector<Record> ToroidalExBDiagnostics2d_list = { //27
    /// ------------------------ Vorticity terms ---------------------------//
    /// ----------------------with ion density -------------------------///
    {"oexbi", "ExB vorticity term with ion density", false,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density(1), 0., result);
        }
    },
    {"odiai", "Diamagnetic vorticity term with ion density", false,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.f.gradN(1), v.gradPsip, result);
            dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    {"jsoexbi_tt", "ExB vorticity flux term with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, v.tmp[0]);

            // Omega_E
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density(1), 0., result);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"divoexbi_tt", "ExB vorticity flux term with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E

            // Omega_E
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density(1), 0., v.tmp3[0]);

            // Divergence
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"jsoexbiUD_tt", "ExB vorticity flux term by diamagnetic velocity with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // bxGradN/B Dot GradPsi
            routines::jacobian( v.p.tau[1], v.f.bhatgB(), v.f.gradN(1), v.gradPsip, 0., result);

            // m Omega_E,phi
            routines::dot( v.f.gradP(0), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], v.f.binv(), v.f.binv(), 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"divoexbiUD_tt", "ExB vorticity flux term by diamagnetic velocity with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            routines::times( v.f.bhatgB(), v.f.gradN(1), v.tmp);

            // Omega_E
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( v.p.tau[1]*v.p.mu[1], result, v.f.binv(), v.f.binv(), 0., v.tmp3[0]);

            // Divergence
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"jsodiaiUE_tt", "Diamagnetic vorticity flux by ExB veloctiy term with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // Omega_D,phi
            routines::dot( v.p.mu[1]*v.p.tau[1], v.f.gradN(1), v.gradPsip, 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"divodiaiUE_tt", "Diamagnetic vorticity flux by ExB veloctiy term with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB
            routines::times( v.f.bhatgB(), v.f.gradP(0), v.tmp);

            // Omega_D,phi
            routines::dot( v.p.mu[1]*v.p.tau[1], v.f.gradN(1), v.gradPsip, 0., v.tmp3[0]);

            // Divergence
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    /// ----------------------with electron density --------------------///
    {"oexbe", "ExB vorticity term with electron density", false,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density(0), 0., result);
        }
    },
    {"odiae", "Diamagnetic vorticity term with electron density", false,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.f.gradN(0), v.gradPsip, result);
            dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    {"jsoexbe_tt", "ExB vorticity flux term with electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, v.tmp[0]);

            // Omega_E
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density(0), 0., result);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"divoexbe_tt", "ExB vorticity flux term with electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E

            // Omega_E
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density(0), 0., v.tmp3[0]);

            // Divergence
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"jsoexbeUD_tt", "ExB vorticity flux term by diamagnetic velocity with electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // bxGradN/B Dot GradPsi
            routines::jacobian( v.p.tau[1], v.f.bhatgB(), v.f.gradN(0), v.gradPsip, 0., result);

            // m Omega_E,phi
            routines::dot( v.f.gradP(0), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], v.f.binv(), v.f.binv(), 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"divoexbeUD_tt", "ExB vorticity flux term by diamagnetic velocity with electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            routines::times( v.f.bhatgB(), v.f.gradN(0), v.tmp);

            // Omega_E
            routines::dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( v.p.tau[1]*v.p.mu[1], result, v.f.binv(), v.f.binv(), 0., v.tmp3[0]);

            // Divergence
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"jsodiaeUE_tt", "Diamagnetic vorticity flux by ExB velocity term with electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // Omega_D,phi
            routines::dot( v.p.mu[1]*v.p.tau[1], v.f.gradN(0), v.gradPsip, 0., v.tmp[0]);

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], 0., result);
        }
    },
    {"divodiaeUE_tt", "Diamagnetic vorticity flux by ExB veloctiy term with electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB
            routines::times( v.f.bhatgB(), v.f.gradP(0), v.tmp);

            // Omega_D,phi
            routines::dot( v.p.mu[1]*v.p.tau[1], v.f.gradN(0), v.gradPsip, 0., v.tmp2[1]);

            // Divergence
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    /// ----------------------Remainders--------------------------------///
    {"jsoApar_tt", "A parallel vorticity flux term (Maxwell stress) (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            if( v.p.beta == 0) // avoid division by zero
            {
                dg::blas1::scal( result, 0.);
            }
            else
            {
                routines::jacobian( v.f.bhatgB(), v.f.gradA(), v.gradPsip, result);
                routines::dot( -1./v.p.beta, v.f.gradA(), v.gradPsip, 0., v.tmp[0]);
                dg::blas1::pointwiseDot( result, v.tmp[0], result);
            }
        }
    },
    {"divoApar_tt", "A parallel vorticity flux term (Maxwell stress) (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            if( v.p.beta == 0) // avoid division by zero
            {
                dg::blas1::scal( result, 0.);
            }
            else
            {
                routines::times( v.f.bhatgB(), v.f.gradA(), v.tmp);
                routines::dot( -1./v.p.beta, v.f.gradA(), v.gradPsip, 0., v.tmp3[0]);
                v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
            }
        }
    },
    {"jsodiaApar_tt", "A parallel diamagnetic vorticity flux term (magnetization stress) (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( v.gradPsip, v.f.gradU(1), v.tmp[0]);
            routines::dot( v.gradPsip, v.f.gradN(1), v.tmp[1]);
            dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.density(1), 1., v.tmp[0], v.f.velocity(1), 0., result);

            routines::jacobian( v.f.bhatgB(), v.f.gradA(), v.gradPsip, result);
            dg::blas1::pointwiseDot( -1./2.*v.p.tau[1], result, v.tmp[0], 0., result);
        }
    },
    {"jsoexbApar_tt", "A parallel ExB vorticity flux term (magnetization stress) (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            routines::jacobian( v.f.bhatgB(), v.f.gradU(1), v.gradPsip, v.tmp[0]);
            routines::jacobian( v.f.bhatgB(), v.f.gradN(1), v.gradPsip, v.tmp[1]);
            dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.density(1), 1., v.tmp[1], v.f.velocity(1), 0., result);
            routines::dot( v.f.gradA(), v.gradPsip, v.tmp[2]);
            dg::blas1::pointwiseDot( -1./2.*v.p.tau[1], result, v.tmp[2], 0., result);
        }
    },
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
            routines::dot( v.p.mu[1]*v.p.tau[1], v.tmp, v.gradPsip, 0., result);
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

std::vector<Record> ParallelMomDiagnostics2d_list = { //36
    ///-----------------------Parallel momentum terms ------------------------//
    {"niui", "Product of ion gyrocentre density and velocity", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.density(1), v.f.velocity(1), result);
        }
    },
    {"jsparexbi_tt", "Parallel momentum radial flux by ExB velocity with electron potential (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // parallel momentum mu_iN_iU_i
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), result, 0., result);
        }
    },
    {"divparexbi_tt", "Divergence of parallel momentum radial flux by ExB velocity with electron potential (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB
            routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E
            // parallel momentum mu_iN_iU_i
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), 0., v.tmp3[0]);
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"divparexbii_tt", "Divergence of parallel momentum radial flux by ExB velocity with ion potential (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB
            routines::times(v.f.bhatgB(), v.f.gradP(1), v.tmp); //u_E
            // parallel momentum mu_iN_iU_i
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), 0., v.tmp3[0]);
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"divpardiai_tt", "Parallel momentum radial flux by Diamagnetic velocity with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( 1., v.f.curv(), v.f.gradN(1), 0., v.tmp[0]);
            routines::dot( 1., v.f.curv(), v.f.gradU(1), 0., v.tmp[1]);
            // Multiply everything
            dg::blas1::pointwiseDot( v.p.mu[1]*v.p.tau[1], v.tmp[0], v.f.velocity(1),
                v.p.mu[1]*v.p.tau[1], v.tmp[1], v.f.density(1), 0., result);
        }
    },
    {"divparkappai_tt", "Parallel momentum radial flux by curvature velocity (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), 0., v.tmp[0]); // mu NU
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.velocity(1), v.f.velocity(1), v.tmp[0], 0., v.tmp[1]); //muNU mu U**2
            // mu NU(mu U^2 + 2tau)
            dg::blas1::axpbypgz( 2.*v.p.tau[1], v.tmp[0], +1., v.tmp[1], 0., v.tmp3[0]);
            v.f.centered_div( v.tmp3[0], v.f.curvKappa(), v.tmp2[0], result);
        }
    },
    {"divparmirrorAi_tt", "Divergence of parallel magnetic flutter force (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            //b_\perp
            v.f.compute_bperp( v.tmp);
            v.f.centered_div( v.f.density(1), v.tmp, v.tmp2[0], result);
            dg::blas1::scal( result, v.p.tau[1]);
        }
    },
    {"divparmirrorAe_tt", "Divergence of parallel magnetic flutter force (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            //b_\perp
            v.f.compute_bperp( v.tmp);
            v.f.centered_div( v.f.density(0), v.tmp, v.tmp2[0], result);
            dg::blas1::scal( result, -1.);
        }
    },
    {"divparApari_tt", "Parallel momentum radial flux by magnetic flutter (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.velocity(1), v.f.velocity(1), v.f.density(1),  0., v.tmp3[0]);
            //b_\perp^v
            v.f.compute_bperp( v.tmp);
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"divparApare_tt", "Divergence of parallel momentum radial flux by magnetic flutter (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::pointwiseDot( -v.p.mu[0], v.f.velocity(0), v.f.velocity(0), v.f.density(0),  0., v.tmp3[0]);
            //b_\perp^v
            v.f.compute_bperp( v.tmp);
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    /// --------------------- Parallel momentum source terms ---------------------//
    {"divjpari_tt", "Divergence of parallel ion momentum flux", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.divNUb(1), v.f.velocity(1),
                    0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1),
                    v.f.velocity(1), v.f.dsU(1), 1., result);
        }
    },
    {"divjpare_tt", "Divergence of parallel electron momentum flux", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( -v.p.mu[0], v.f.divNUb(0), v.f.velocity(0),
                    0., result);
            dg::blas1::pointwiseDot( -v.p.mu[0], v.f.density(0),
                    v.f.velocity(0), v.f.dsU(0), 1., result);
        }
    },
    {"lparpar_tt", "Parallel momentum dissipation by parallel diffusion", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_parallel_diffusiveN( 1, v.tmp[0]);
            v.f.compute_parallel_diffusiveU( 1, v.tmp[1]);
            dg::blas1::pointwiseDot( 1., v.f.density(1), v.tmp[1], 1.,
                v.f.velocity(1), v.tmp[0], 0., result);
        }
    },
    {"lparperp_tt", "Parallel momentum dissipation by perp diffusion", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_perp_diffusiveN( 1., v.f.density(1), v.tmp3[0], v.tmp3[1],
                    0., v.tmp[0]);
            v.f.compute_perp_diffusiveU( 1., v.f.velocity(1), v.f.density(1),
                    v.tmp3[0], v.tmp3[1], v.tmp3[2], v.tmp2[0], 0., v.tmp[1]);
            dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.velocity(1),
                    1., v.tmp[1], v.f.density(1), 0., result);
        }
    },
    {"lparparbphi_tt", "Parallel angular momentum dissipation by parallel diffusion", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_parallel_diffusiveN( 1, v.tmp[0]);
            v.f.compute_parallel_diffusiveU( 1, v.tmp[1]);
            dg::blas1::pointwiseDot( 1., v.f.density(1), v.tmp[1], 1.,
                v.f.velocity(1), v.tmp[0], 0., result);
            dg::blas1::pointwiseDot( result, v.f.bphi(), result);
        }
    },
    {"lparperpbphi_tt", "Parallel angular momentum dissipation by perp diffusion", true,
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_perp_diffusiveN( 1., v.f.density(1), v.tmp3[0], v.tmp3[1],
                    0., v.tmp[0]);
            v.f.compute_perp_diffusiveU( 1., v.f.velocity(1), v.f.density(1),
                    v.tmp3[0], v.tmp3[1], v.tmp3[2], v.tmp2[0], 0., v.tmp[1]);
            dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.velocity(1),
                    1., v.tmp[1], v.f.density(1), 0., result);
            dg::blas1::pointwiseDot( result, v.f.bphi(), result);
        }
    },
    //not so important
    {"sparKappaphii_tt", "Kappa Phi Source for parallel momentum", true,
        []( dg::x::DVec& result, Variables& v ) {
            routines::dot( v.f.curvKappa(), v.f.gradP(1), result);
            dg::blas1::pointwiseDot( -v.p.mu[1], v.f.density(1), v.f.velocity(1), result, 0., result);
        }
    },
    //not so important
    {"sparmirrorKappai_tt", "Generalized mirror force (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::pointwiseDot( v.p.mu[1]*v.p.tau[1], v.f.density(1), v.f.velocity(1),
                v.f.divCurvKappa(), 0., result);
        }
    },
    ///-----------------------Parallel anbular momentum terms-----------------//
    {"niuibphi", "Product of NiUi and covariant phi component of magnetic field unit vector", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 1.,
                v.f.density(1), v.f.velocity(1), v.f.bphi(), 0., result);
        }
    },
    {"jsparbphiexbi_tt", "Parallel angular momentum radial flux by ExB velocity with electron potential (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB Dot GradPsi
            routines::jacobian( v.f.bhatgB(), v.f.gradP(0), v.gradPsip, result);

            // parallel momentum mu_iN_iU_i
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), result, 0., result);

            // Multiply bphi
            dg::blas1::pointwiseDot( 1., result, v.f.bphi(), 0., result);
        }
    },
    {"divparbphiexbi_tt", "Divergence of parallel angular momentum radial flux by ExB velocity with electron potential (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB
            routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E
            // parallel momentum mu_iN_iU_i bphi
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), v.f.bphi(), 0., v.tmp3[0]);
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"divparbphiexbii_tt", "Divergence of parallel angular momentum radial flux by ExB velocity with ion potential (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            // ExB
            routines::times(v.f.bhatgB(), v.f.gradP(1), v.tmp); //u_E
            // parallel momentum mu_iN_iU_i bphi
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), v.f.bphi(), 0., v.tmp3[0]);
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"divparbphidiai_tt", "Parallel angular momentum radial flux by Diamagnetic velocity with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            routines::dot( 1., v.f.curv(), v.f.gradN(1), 0., v.tmp[0]);
            routines::dot( 1., v.f.curv(), v.f.gradU(1), 0., v.tmp[1]);
            dg::blas1::pointwiseDot( v.tmp[0], v.f.bphi(), v.tmp[0]);
            dg::blas1::pointwiseDot( v.tmp[1], v.f.bphi(), v.tmp[1]);
            // Multiply everything
            dg::blas1::pointwiseDot( v.p.mu[1]*v.p.tau[1], v.tmp[0], v.f.velocity(1),
                v.p.mu[1]*v.p.tau[1], v.tmp[1], v.f.density(1), 0., result);
        }
    },
    {"divparbphikappai_tt", "Parallel angular momentum radial flux by curvature velocity (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.density(1), v.f.velocity(1), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.f.velocity(1), v.f.velocity(1), v.tmp[0],  0., v.tmp[1]);
            // mu NU(mu U^2 + 2tau)bphi
            dg::blas1::pointwiseDot( 2.*v.p.tau[1], v.tmp[0], v.f.bphi(), +1., v.tmp[1], v.f.bphi(), 0., v.tmp3[0]);
            v.f.centered_div( v.tmp3[0], v.f.curvKappa(), v.tmp2[0], result);
        }
    },
    {"divparbphiApar_tt", "Parallel angular momentum radial flux by magnetic flutter (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::pointwiseDot( -v.p.mu[0], v.f.velocity(0), v.f.velocity(0), v.f.density(0),  0., result);
            dg::blas1::pointwiseDot( +v.p.mu[1], v.f.velocity(1), v.f.velocity(1), v.f.density(1),  1., result);
            dg::blas1::axpbypgz( -v.p.tau[0], v.f.density(0),
                                 +v.p.tau[1], v.f.density(1), 1., result);
            dg::blas1::pointwiseDot( v.f.bphi(), result, v.tmp3[0]);
            v.f.compute_bperp( v.tmp);
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    /// --------------------electron force balance usually well-fulfilled ----//
    {"sparphie_tt", "Electric force in electron momentum density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::pointwiseDot( 1., v.f.dsP(0), v.f.density(0), 0., result);
        }
    },
    {"sparmirrore_tt", "Parallel electron pressure (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::axpby( v.p.tau[0], v.f.dsN(0), 0., result);
        }
    },
    {"sparmirrorAe_tt", "Apar Mirror force term with electron density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            v.f.compute_bperp( v.tmp);
            routines::dot( v.p.tau[0], v.tmp, v.f.gradN(0), 0., result);
        }
    },

    {"sparphiAe_tt", "Apar Electric force in electron momentum density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            v.f.compute_bperp( v.tmp);
            routines::dot( v.tmp, v.f.gradP(0), result);
            dg::blas1::pointwiseDot( v.f.density(0), result, result);
        }
    },
    {"spardotAe_tt", "Apar Electric force in electron momentum density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            v.f.compute_dot_aparallel( result);
            dg::blas1::pointwiseDot( v.f.density(0), result, result);
        }
    },
    {"neue", "Product of electron density and velocity", false,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.density(0), v.f.velocity(0), result);
        }
    },
    /// -----------Ion force balance ----------------------///
    {"sparphii_tt", "Electric force term in ion momentum density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::pointwiseDot( -1., v.f.dsP(1), v.f.density(1), 0., result);
        }
    },
    {"sparmirrori_tt", "Parallel ion pressure (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            dg::blas1::axpby( -v.p.tau[1], v.f.dsN(1), 0., result);
        }
    },
    {"sparmirrorAi_tt", "Apar Mirror force term with ion density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            v.f.compute_bperp( v.tmp);
            routines::dot( -v.p.tau[1], v.tmp, v.f.gradN(1), 0., result);
        }
    },
    {"sparphiAi_tt", "Apar Electric force in ion momentum density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            v.f.compute_bperp( v.tmp);
            routines::dot( v.tmp, v.f.gradP(1), result);
            dg::blas1::pointwiseDot( -1., v.f.density(1), result, 0., result);
        }
    },
    {"spardotAi_tt", "Apar Electric force in ion momentum density (Time average)", true,
        []( dg::x::DVec& result, Variables& v){
            v.f.compute_dot_aparallel( result);
            dg::blas1::pointwiseDot( -1., v.f.density(1), result, 0., result);
        }
    },
    {"friction_tt", "Friction force in momentum density (Time average)", true,
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 1., v.f.velocity(1), v.f.density(1), -1.,
                    v.f.velocity(0), v.f.density(0), 0., result);
            dg::blas1::pointwiseDot( v.p.eta, result, v.f.density(0), 0, result);
        }
    },
};

std::vector<Record> RSDiagnostics2d_list = { //2
    /// --------------------- Zonal flow energy terms------------------------//
    {"nei0", "inertial factor", false,
        []( dg::x::DVec& result, Variables& v ) {
            routines::dot( v.gradPsip, v.gradPsip, result);
            dg::blas1::transform( result, result, Positive());
            dg::blas1::pointwiseDot( v.mag.R0()*v.mag.R0(),
                result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDivide( v.f.density(0), result, result);
        }
    },
    {"snei0_tt", "inertial factor source", true,
        []( dg::x::DVec& result, Variables& v ) {
            routines::dot( v.gradPsip, v.gradPsip, result);
            dg::blas1::transform( result, result, Positive());
            dg::blas1::pointwiseDot( v.mag.R0()*v.mag.R0(),
                result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDivide( v.f.density_source(0), result, result);
        }
    }
};

std::vector<Record> COCEDiagnostics2d_list = { // 16
    /// ----------------- COCE EQUATION ----------------//
    /// ---------- Polarization charge densities -----------///
    {"v_Omega_E", "Electron polarisation term", false,
        []( dg::x::DVec& result, Variables& v) {
            v.f.compute_pol( 1., v.f.density(0), v.tmp[0], 0., result);
        }
    },
    {"v_Omega_E_gf", "Ion polarisation term", false,
        []( dg::x::DVec& result, Variables& v) {
            v.f.compute_pol( 1., v.f.density(1), v.tmp[0], 0., result);
        }
    },
    /// ------------ Polarization advections ------------------//
    //The fsa of the main and rest terms is almost the same as the one of divoexbi
    {"v_adv_E_main_tt", "Main electric advective term (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
            v.f.compute_pol( 1., v.f.density(0), v.tmp[0], 0., v.tmp3[0]);
            routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"v_adv_E_main_gf_tt", "Main electric advective term (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
            v.f.compute_pol( 1., v.f.density(1), v.tmp[0], 0., v.tmp3[0]);
            routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
        }
    },
    {"v_adv_E_rest_tt", "Electric advective term (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
            // NOT implemented for true curvature mode
            routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp2); //u_E
            dg::blas1::pointwiseDot(1., v.f.binv(), v.f.binv(), v.f.density(0), 0., v.tmp[0]);
            routines::scal(v.tmp[0], v.f.gradP(0), v.tmp3); //ne Grad_phi/B^2
            v.f.centered_v_dot_nabla(v.tmp3, v.tmp2[0], v.tmp[2], v.tmp[0]); //t3*nabla(u_E^R)
            v.f.centered_v_dot_nabla(v.tmp3, v.tmp2[1], v.tmp[2], v.tmp[1]); //t3*nabla(u_E^Z)
            v.f.centered_div( v.p.mu[1], v.tmp, v.tmp2[0], result);
        }
    },
    {"v_adv_E_rest_gf_tt", "Electric advective term GF (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
            // NOT implemented for true curvature mode
            routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp2); //u_E
            dg::blas1::pointwiseDot(1., v.f.binv(), v.f.binv(), v.f.density(1), 0., v.tmp[0]);
            routines::scal(v.tmp[0], v.f.gradP(0), v.tmp3); //ne Grad_phi/B^2
            v.f.centered_v_dot_nabla(v.tmp3, v.tmp2[0], v.tmp[2], v.tmp[0]); //t3*nabla(u_E^R)
            v.f.centered_v_dot_nabla(v.tmp3, v.tmp2[1], v.tmp[2], v.tmp[1]); //t3*nabla(u_E^Z)
            v.f.centered_div( v.p.mu[1], v.tmp, v.tmp2[0], result);
        }
    },
    //The fsa of the main and rest terms is almost the same as the one of divodiaiUE
    {"v_adv_D_main_tt", "Main diamagnetic term (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
            v.f.compute_lapMperpN(-1.0, v.f.density(0), v.tmp[0], 0., v.tmp3[0]);
            routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
            dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    {"v_adv_D_main_gf_tt", "Main diamagnetic term (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
            v.f.compute_lapMperpN(-1.0, v.f.density(1), v.tmp[0], 0., v.tmp3[0]);
            routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp); //u_E
            v.f.centered_div( v.tmp3[0], v.tmp, v.tmp2[0], result);
            dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    {"v_adv_D_rest_tt", "Diamagnetic advective term (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
            // NOT implemented for true curvature mode
            routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp2); //u_E
            v.f.centered_v_dot_nabla(v.f.gradN(0), v.tmp2[0], v.tmp[2], v.tmp[0]); //t3*nabla(u_E^R)
            v.f.centered_v_dot_nabla(v.f.gradN(0), v.tmp2[1], v.tmp[2], v.tmp[1]); //t3*nabla(u_E^Z)
            v.f.centered_div( v.p.tau[1]*v.p.mu[1], v.tmp, v.tmp2[0], result);
        }
    },

    {"v_adv_D_rest_gf_tt", "Diamagnetic advective term GF (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
            // NOT implemented for true curvature mode
            routines::times(v.f.bhatgB(), v.f.gradP(0), v.tmp2); //u_E
            v.f.centered_v_dot_nabla(v.f.gradN(1), v.tmp2[0], v.tmp[2], v.tmp[0]); //t3*nabla(u_E^R)
            v.f.centered_v_dot_nabla(v.f.gradN(1), v.tmp2[1], v.tmp[2], v.tmp[1]); //t3*nabla(u_E^Z)
            v.f.centered_div( v.p.tau[1]*v.p.mu[1], v.tmp, v.tmp2[0], result);
        }
    },
    ///---------------- J_b_perp components --------//
    {"v_J_mag_tt", "Magnetization current term GF (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
            // take care to use correct derivatives...
            v.f.compute_lapMperpN( -1., v.f.density(1), v.tmp[0], 0., v.tmp2[0]);
            dg::blas1::pointwiseDot( v.f.velocity(1), v.tmp2[0], v.tmp2[0]);
            v.f.compute_lapMperpU( 1, v.tmp2[1]);
            dg::blas1::pointwiseDot( v.f.density(1), v.tmp2[1], v.tmp2[1]);
            dg::tensor::scalar_product3d( 1., 1.,
                    v.f.gradN(1)[0], v.f.gradN(1)[1], v.f.gradN(1)[2],
                    v.f.projection(), 1.,
                    v.f.gradU(1)[0], v.f.gradU(1)[1], v.f.gradU(1)[2],
                    0., v.tmp2[2]);
            v.f.compute_bperp(v.tmp);
            dg::blas1::axpbypgz( 1., v.tmp2[0], 1., v.tmp2[1], 1., v.tmp2[2]);
            v.f.centered_div( v.tmp2[2], v.tmp, v.tmp2[0], result);
            dg::blas1::scal ( result, v.p.tau[1]/2.);
        }
    },
    {"v_J_bperp_tt", "Div J_par times b_perp term (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
            v.f.compute_bperp(v.tmp);
            dg::blas1::pointwiseDot(1., v.f.density(1), v.f.velocity(1), -1.,
                    v.f.density(0), v.f.velocity(0), 0, v.tmp2[0]);
            v.f.centered_div( v.tmp2[0], v.tmp, v.tmp3[0], result);
        }
    },
    ///-------------- Sources term----------///
    {"v_S_E_tt", "Electric source vorticity (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
            v.f.compute_source_pol( 1., v.f.density_source(0), v.tmp[0], 0., result);
        }
    },
    {"v_S_D_tt", "Diamagnetic source vorticity (time integrated)", true,
        []( dg::x::DVec& result, Variables& v) {
            v.f.compute_lapMperpN(-v.p.tau[1]*v.p.mu[1], v.f.density_source(0), v.tmp[0], 0., result);
        }
    },
    /// ------------- Radial Force Balance terms ---------///
    {"RFB_E_r_GradPsip_tt", "Radial electric field in RFB (time integrated)", true,
         []( dg::x::DVec& result, Variables& v){
             routines::dot(-1., v.f.gradP(0), v.gradPsip, 0., result);
         }
     },
    {"RFB_GradPi_GradPsip_tt", "Radial pressure gradient component of RFB (time integrated)", true,
         []( dg::x::DVec& result, Variables& v){
             routines::dot(v.f.gradN(0), v.gradPsip, result);
             dg::blas1::pointwiseDivide(v.p.tau[1], result, v.f.density(0), 0.,
                     result);
         }
     },

};

// probes list
std::vector<dg::file::Record<void(dg::x::DVec&,Variables&), dg::file::LongNameAttribute>> probe_list = {
     {"ne", "probe measurement of electron density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.density(0), result);
         }
     },
     {"ni", "probe measurement of ion density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.density(1), result);
         }
     },
     {"ue", "probe measurement of parallel electron velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.velocity(0), result);
         }
     },
     {"ui", "probe measurement of parallel ion velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.velocity(1), result);
         }
     },
     {"phi", "probe measurement of electric potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.potential(0), result);
         }
     },
     {"apar", "probe measurement of parallel magnetic potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.aparallel(), result);
         }
     },
     {"neR", "probe measurement of d/dR electron density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradN(0)[0], result);
         }
     },
     {"niR", "probe measurement of d/dR ion density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradN(1)[0], result);
         }
     },
     {"ueR", "probe measurement of d/dR parallel electron velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradU(0)[0], result);
         }
     },
     {"uiR", "probe measurement of d/dR parallel ion velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradU(1)[0], result);
         }
     },
     {"phiR", "probe measurement of d/dR electric potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradP(0)[0], result);
         }
     },
     {"aparR", "probe measurement of d/dR parallel magnetic potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradA()[0], result);
         }
     },
     {"neZ", "probe measurement of d/dZ electron density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradN(0)[1], result);
         }
     },
     {"niZ", "probe measurement of d/dZ ion density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradN(1)[1], result);
         }
     },
     {"ueZ", "probe measurement of d/dZ parallel electron velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradU(0)[1], result);
         }
     },
     {"uiZ", "probe measurement of d/dZ parallel ion velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradU(1)[1], result);
         }
     },
     {"phiZ", "probe measurement of d/dZ electric potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradP(0)[1], result);
         }
     },
     {"aparZ", "probe measurement of d/dZ parallel magnetic potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.gradA()[1], result);
         }
     },
     {"nePar", "probe measurement of d/dPar electron density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.dsN(0), result);
         }
     },
     {"niPar", "probe measurement of d/dPar ion density",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.dsN(1), result);
         }
     },
     {"uePar", "probe measurement of d/dPar parallel electron velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.dsU(0), result);
         }
     },
     {"uiPar", "probe measurement of d/dPar parallel ion velocity",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.dsU(1), result);
         }
     },
     {"phiPar", "probe measurement of d/dPar electric potential",
         []( dg::x::DVec& result, Variables& v ) {
              dg::blas1::copy(v.f.dsP(0), result);
         }
     }
 };

// Here is a list of useful 1d variables of general interest
std::vector<dg::file::Record<double(Variables&),dg::file::LongNameAttribute>> diagnostics1d_list = {
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
        [](Variables& v ) {
            return v.f.called();
        }
    },
};
///%%%%%%%%%%%%%%%%%%%%%%%%%%END DIAGNOSTICS LIST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
///%%%%%%%%%%%%%%%%%%%%%%%%%%END DIAGNOSTICS LIST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
///%%%%%%%%%%%%%%%%%%%%%%%%%%END DIAGNOSTICS LIST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
template<class NcFile>
void write_global_attributes( NcFile& file, int argc, char* argv[],
        std::string inputfile)
{
    std::map<std::string, dg::file::nc_att_t> att;
    att["title"] = "Output file of feltor/src/feltor/feltor.cpp";
    att["Conventions"] = "CF-1.8";
    att["history"] = dg::file::timestamp( argc, argv);
    att["comment"] = "Find more info in feltor/src/feltor/feltor.tex";
    att["source"] = "FELTOR";
    att["references"] = "https://github.com/feltor-dev/feltor";
    att["inputfile"] = inputfile;
    file.put_atts( att);

    file.put_atts( dg::file::version_flags);

}

template<class NcFile, class HostList>
void write_static_list( NcFile& file, const HostList& records, Variables& var,
    const dg::x::CylindricalGrid3d& grid, const dg::x::CylindricalGrid3d& g3d_out,
    dg::geo::CylindricalFunctor transition )
{
    // the unique thing here is that we evaluate 3d but only write 2d
    std::unique_ptr<dg::x::aGeometry2d> g2d_out_ptr( g3d_out.perp_grid());

    dg::x::HVec resultH = dg::evaluate( dg::zero, grid);
    dg::x::HVec transferH( dg::evaluate(dg::zero, g3d_out));
    dg::MultiMatrix<dg::x::HMatrix,dg::x::HVec> projectH =
            dg::create::fast_projection( grid, 1, var.p.cx, var.p.cy);
    for ( auto& record : records)
    {
        record.function( resultH, var, grid);
        dg::blas2::symv( projectH, resultH, transferH);
        file.defput_var( record.name, {"y", "x"}, record.atts, {*g2d_out_ptr},
                transferH);
    }
    resultH = dg::pullback( transition, grid);
    dg::blas2::symv( projectH, resultH, transferH);
    file.defput_var( "MagneticTransition", {"y", "x"}, {{"long_name",
        "The region where the magnetic field is modified"}},
        {*g2d_out_ptr}, transferH);
}

void append_equations( std::vector<feltor::Record>& list, const std::vector<feltor::Record>& b)
{
    list.insert( list.begin(), b.begin(), b.end());
}
std::vector<feltor::Record> generate_equation_list( const dg::file::WrappedJsonValue& js)
{
    std::vector<feltor::Record> list;
    bool equation_list_exists = js["output"].isMember("equations");
    if( equation_list_exists)
    {
        for( unsigned i=0; i<js["output"]["equations"].size(); i++)
        {
            std::string eqn = js["output"]["equations"][i].asString();
            if( eqn == "Basic")
                append_equations( list, feltor::basicDiagnostics2d_list);
            else if( eqn == "Mass-conserv")
                append_equations( list, feltor::MassConsDiagnostics2d_list);
            else if( eqn == "Energy-theorem")
                append_equations( list, feltor::EnergyDiagnostics2d_list);
            else if( eqn == "Toroidal-momentum")
                append_equations( list, feltor::ToroidalExBDiagnostics2d_list);
            else if( eqn == "Parallel-momentum")
                append_equations( list, feltor::ParallelMomDiagnostics2d_list);
            else if( eqn == "Zonal-Flow-Energy")
                append_equations( list, feltor::RSDiagnostics2d_list);
            else if( eqn == "COCE")
                append_equations( list, feltor::COCEDiagnostics2d_list);
            else
                throw std::runtime_error( "output: equations: "+eqn+" not recognized!\n");
        }
    }
    else // default diagnostics
    {
        append_equations(list, feltor::basicDiagnostics2d_list);
        append_equations(list, feltor::MassConsDiagnostics2d_list);
        append_equations(list, feltor::EnergyDiagnostics2d_list);
        append_equations(list, feltor::ToroidalExBDiagnostics2d_list);
        append_equations(list, feltor::ParallelMomDiagnostics2d_list);
        append_equations(list, feltor::RSDiagnostics2d_list);
    }
    return list;
}

template< class NcFile>
struct WriteIntegrateDiagnostics2dList
{
    WriteIntegrateDiagnostics2dList( NcFile& file,
        const dg::x::CylindricalGrid3d& grid,
        const dg::x::CylindricalGrid3d& g3d_out,
        const std::vector<feltor::Record>& equation_list) :
            m_file(&file), m_slab(grid), m_grid(grid), m_g3d_out(g3d_out),
            m_equation_list(equation_list)
    {
#ifdef WITH_MPI
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI
        std::unique_ptr<dg::x::aGeometry2d> g2d_out_ptr( g3d_out.perp_grid());
        for( auto& record : m_equation_list)
        {
            std::string name = record.name + "_ta2d";
            std::string long_name = record.long_name + " (Toroidal average)";
            m_file->template def_var_as<double>( name, {"time", "y", "x"},
                    {{"long_name", long_name}});
            name = record.name + "_2d";
            long_name = record.long_name+ " (Evaluated on phi = 0 plane)";
            m_file->template def_var_as<double>( name, {"time", "y", "x"},
                    {{"long_name", long_name}});
        }
        m_slab = {*g2d_out_ptr};
        #ifdef WITH_MPI // only root group needs to track
        if( dg::file::detail::mpi_comm_global2local_rank( g2d_out_ptr->communicator()) == MPI_UNDEFINED)
            m_track  = false;
        #endif
        m_resultD = dg::evaluate( dg::zero, grid);
        m_transferD = dg::evaluate(dg::zero, g3d_out);
        m_transferH = dg::evaluate(dg::zero, g3d_out);
        m_projectD = dg::create::fast_projection( grid, 1, grid.Nx()/g3d_out.Nx(), grid.Ny()/g3d_out.Ny());
        m_transferH2d = dg::evaluate( dg::zero, *g2d_out_ptr);
        m_toroidal_average = { g3d_out, dg::coo3d::z};
    }
    // same as buffer and flush
    void write( double time, Variables& var)
    {
        buffer( time, var);
        flush(var);
    }

    void buffer( double time, Variables& var)
    {
        // evaluates function and updates time integrals for all integrals
        auto transferD2d_view = dg::split( m_transferD, m_g3d_out);
        for( auto& record : m_equation_list)
        {
            if( record.integral)
            {
                record.function( m_resultD, var);
                dg::blas2::symv( m_projectD, m_resultD, m_transferD);
                //toroidal average and add to time integral
                std::string name = record.name+"_ta2d";
                dg::assign( m_transferD, m_transferH);
                m_toroidal_average( m_transferH, m_transferH2d, false);
                if(m_track && m_first_buffer) m_time_integrals[name].init(
                    time, m_transferH2d);
                if(m_track && !m_first_buffer) m_time_integrals.at(name).add( time,
                    m_transferH2d);

                // 2d data of plane varphi = 0
                name = record.name + "_2d";
                dg::split( m_transferD, transferD2d_view, m_g3d_out);
                dg::assign( transferD2d_view[0], m_transferH2d);
                if(m_track && m_first_buffer) m_time_integrals[name].init(
                    time, m_transferH2d);
                if(m_track && !m_first_buffer) m_time_integrals.at(name).add( time,
                    m_transferH2d);
            }
        }
        if( m_first_buffer)
            m_first_buffer = false;
    }
    void flush( Variables& var )
    {
        // write time integrals for
        auto transferD2d_view = dg::split( m_transferD, m_g3d_out);
        for( auto& record : m_equation_list)
        {
            if(record.integral) // we already computed the output...
            {
                std::string name = record.name+"_ta2d";
                if(m_track) m_transferH2d = m_time_integrals.at(name).get_integral();
                m_file->put_var( name, {m_start, m_slab}, m_transferH2d);
                if(m_track) m_time_integrals.at(name).flush();

                name = record.name + "_2d";
                if(m_track) m_transferH2d = m_time_integrals.at(name).get_integral( );
                m_file->put_var( name, {m_start, m_slab}, m_transferH2d);
                if(m_track) m_time_integrals.at(name).flush();
            }
            else // compute from scratch
            {
                record.function( m_resultD, var);
                dg::blas2::symv( m_projectD, m_resultD, m_transferD);

                dg::assign( m_transferD, m_transferH);
                m_toroidal_average( m_transferH, m_transferH2d, false);
                m_file->put_var( record.name+"_ta2d", {m_start, m_slab}, m_transferH2d);

                // 2d data of plane varphi = 0
                dg::split( m_transferD, transferD2d_view, m_g3d_out);
                dg::assign( transferD2d_view[0], m_transferH2d);
                m_file->put_var( record.name +"_2d", {m_start, m_slab}, m_transferH2d);
            }
        }
        m_start++;
    }
    private:
    NcFile * m_file;
    typename NcFile::Hyperslab m_slab;
    bool m_first_buffer = true;
    bool m_track = true;
    size_t m_start = 0;
    dg::x::DVec m_resultD;
    dg::x::DVec m_transferD;
    dg::x::HVec m_transferH;
    dg::MultiMatrix<dg::x::DMatrix,dg::x::DVec> m_projectD;
    dg::x::HVec m_transferH2d;
    dg::Average<dg::x::IHMatrix, dg::x::HVec> m_toroidal_average;
    std::map<std::string, dg::Simpsons<dg::x::HVec>> m_time_integrals;
    const dg::x::CylindricalGrid3d m_grid, m_g3d_out;
    std::vector<feltor::Record> m_equation_list;

};


}//namespace feltor
