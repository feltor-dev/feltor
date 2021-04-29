#pragma once

#include "dg/file/json_utilities.h"
#include "init_from_file.h"

namespace feltor
{

namespace detail
{


struct TorpexSource
{
    TorpexSource( double R0, double Z0, double a, double b, double c):m_R0(R0), m_Z0(Z0), m_a(a), m_b(b), m_c(c){}
    DG_DEVICE
    double operator()( double R, double Z) const{
        if( R > m_R0)
            return exp( - (R-m_R0)*(R-m_R0)/m_a/m_a - (Z-m_Z0)*(Z-m_Z0)/m_b/m_b);
        return 0.5*exp( - (R-m_R0)*(R-m_R0)/m_a/m_a -2.*m_c*(R-m_R0)*(Z-m_Z0)- (Z-m_Z0)*(Z-m_Z0)/m_b/m_b )
              +0.5*exp( - (R-m_R0)*(R-m_R0)/m_a/m_a +2.*m_c*(R-m_R0)*(Z-m_Z0)- (Z-m_Z0)*(Z-m_Z0)/m_b/m_b );
    }
    DG_DEVICE
    double operator()( double R, double Z, double p) const{
        return this->operator()(R,Z);
    }
    private:
    double m_R0, m_Z0, m_a, m_b, m_c;
};

struct Radius : public dg::geo::aCylindricalFunctor<Radius>
{
    Radius ( double R0, double Z0): m_R0(R0), m_Z0(Z0) {}
    DG_DEVICE
    double do_compute( double R, double Z) const{
        return sqrt( (R-m_R0)*(R-m_R0) + (Z-m_Z0)*(Z-m_Z0));
    }
    private:
    double m_R0, m_Z0;
};


dg::x::HVec xpoint_damping(
    const dg::x::CylindricalGrid3d& grid,
    const dg::geo::TokamakMagneticField& mag )
{
    dg::x::HVec xpoint_damping = dg::evaluate( dg::one, grid);
    if( mag.params().getDescription() == dg::geo::description::standardX)
    {
        double RX = mag.R0() - 1.1*mag.params().triangularity()*mag.params().a();
        double ZX = -1.1*mag.params().elongation()*mag.params().a();
        dg::geo::findXpoint( mag.get_psip(), RX, ZX);
        xpoint_damping = dg::pullback(
            dg::geo::ZCutter(ZX), grid);
    }
    return xpoint_damping;
}

dg::x::HVec make_profile(
    const dg::x::CylindricalGrid3d& grid,
    const dg::geo::TokamakMagneticField& mag,
    dg::file::WrappedJsonValue js )
{
    //js = input["profile"]
    std::string type = js.get("type","zero").asString();
    dg::x::HVec profile = dg::evaluate( dg::zero, grid);
    if( "zero" == type)
    {
    }
    else if( "aligned" == type)
    {
        double RO=mag.R0(), ZO=0.;
        dg::geo::findOpoint( mag.get_psip(), RO, ZO);
        double psipO = mag.psip()( RO, ZO);
        profile = dg::pullback( dg::compose(dg::LinearX(
                    1./psipO, 0.), mag.psip()), grid);
        double nprofamp = js.get( "amplitude", 1.0).asDouble();
        dg::blas1::scal( profile, nprofamp );
    }
    else if ( "gaussian" == type )
    {
        double posX     = js.get( "posX", 1.0).asDouble();
        double posY     = js.get( "posY", 1.0).asDouble();
        double sigma    = js.get( "sigma", 1.0).asDouble();
        double nprofamp = js.get( "amplitude", 1.0).asDouble();
        if( sigma == 0)
            throw dg::Error(dg::Message()<< "Invalid parameter: sigma must not be 0 in turbulence on gaussian\n");
        dg::Gaussian prof( mag.R0()+posX*mag.params().a(), posY*mag.params().a(), sigma,
            sigma, nprofamp);
        profile = dg::pullback( prof, grid);
    }
    else
        throw dg::Error(dg::Message()<< "Invalid profile type "<<type<<"\n");
    return profile;
}

dg::x::HVec make_damping(
    const dg::x::CylindricalGrid3d& grid,
    const dg::geo::TokamakMagneticField& mag,
    dg::file::WrappedJsonValue js )
{
    //js = input["damping"]
    std::string type = js.get("type","none").asString();
    dg::x::HVec damping = dg::evaluate( dg::one, grid);
    if( "none" == type)
    {
    }
    else if( "aligned" == type)
    {

        double damping_alpha = js.get("alpha", 0.2).asDouble();
        if( damping_alpha == 0)
            throw dg::Error(dg::Message()<< "Invalid parameter: damping alpha must not be 0\n");
        double damping_boundary = js.get("boundary", 0.2).asDouble();
        //we also need to avoid being too far in the PFR where psi can become very negative
        damping = dg::pullback(
            dg::compose(dg::PolynomialHeaviside(
                damping_boundary-damping_alpha/2., damping_alpha/2., -1),
                    dg::geo::RhoP(mag)), grid);
        dg::blas1::pointwiseDot( xpoint_damping(grid,mag),
            damping, damping);
    }
    else if ( "circular" == type)
    {
        double damping_alpha = js.get("alpha", 0.2).asDouble();
        double radius = js.get("boundary", 1.0).asDouble();
        if( damping_alpha == 0)
            throw dg::Error(dg::Message()<< "Invalid parameter: damping alpha must not be 0\n");
        damping =  dg::pullback( dg::compose(
                dg::PolynomialHeaviside( radius*mag.params().a(),
                    mag.params().a()*damping_alpha/2., -1),
                Radius( mag.R0(), 0.)), grid);
    }
    else
        throw dg::Error(dg::Message()<< "Invalid damping type "<<type<<"\n");
    return damping;
}
dg::x::HVec make_ntilde(
    const dg::x::CylindricalGrid3d& grid,
    const dg::geo::TokamakMagneticField& mag,
    dg::file::WrappedJsonValue js )
{
    //js = input["ntilde"]
    std::string type = js.get("type","zero").asString();
    dg::x::HVec ntilde = dg::evaluate( dg::zero, grid);
    if( "zero" == type)
    {
    }
    else if( "blob" == type)
    {
        double amp   = js.get( "amplitude", 0.).asDouble();
        double sigma = js.get( "sigma", 0.).asDouble();
        double posX  = js.get( "posX", 0.).asDouble();
        double posY  = js.get( "posY", 0.).asDouble();
        if( sigma == 0)
            throw dg::Error(dg::Message()<< "Invalid parameter: sigma must not be 0 in straight blob initial condition\n");
        dg::Gaussian init0( mag.R0()+posX*mag.params().a(), posY*mag.params().a(), sigma, sigma, amp);
        if( grid.Nz() == 1 )
            ntilde = dg::pullback( init0, grid);
        else
        {
            double rk4eps = js.get("rk4eps", 1e-6).asDouble();
            unsigned mx = js["refine"].get( 0u, 5).asUInt();
            unsigned my = js["refine"].get( 1u, 5).asUInt();
            dg::geo::Fieldaligned<dg::x::CylindricalGrid3d, dg::x::IHMatrix, dg::x::HVec>
                fieldaligned( mag, grid, grid.bcx(), grid.bcy(),
                dg::geo::NoLimiter(), rk4eps, mx, my);
            //evaluate should always be used with mx,my > 1 (but this takes a lot of memory)
            unsigned revolutions = js.get( "revolutions", 1).asUInt();
            double sigma_z = js.get( "sigma_z", 0.).asDouble();
            if( sigma_z == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma_z must not be 0 in blob initial condition\n");
            dg::GaussianZ gaussianZ( 0., sigma_z*M_PI, 1.0);
            ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, revolutions);
        }
    }
    else if ( "turbulence" == type)
    {
        double amp = js.get("amplitude", 0.).asDouble();
        dg::BathRZ init0(16,16,grid.x0(),grid.y0(), 30.,2.,amp);
        if( grid.Nz() == 1)
            ntilde = dg::pullback( init0, grid);
        else
        {
            double rk4eps = js.get("rk4eps", 1e-6).asDouble();
            unsigned mx = js["refine"].get( 0u, 2).asUInt();
            unsigned my = js["refine"].get( 1u, 2).asUInt();
            dg::geo::Fieldaligned<dg::x::CylindricalGrid3d, dg::x::IHMatrix, dg::x::HVec>
                fieldaligned( mag, grid, grid.bcx(), grid.bcy(),
                dg::geo::NoLimiter(), rk4eps, mx, my);
            //evaluate should always be used with mx,my > 1 (but this takes more memory)
            unsigned revolutions = js.get( "revolutions", 1).asUInt();
            double sigma_z = js.get( "sigma_z", 0.).asDouble();
            if( sigma_z == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma_z must not be 0 in turbulence initial condition\n");
            dg::GaussianZ gaussianZ( 0., sigma_z*M_PI, 1.0);
            ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, revolutions);
        }
    }
    else if(  "zonal" == type)
    {
        dg::x::HVec ntilde = dg::evaluate(dg::zero, grid);
        double k_psi  = js.get( "k_psi", 0.).asDouble();
        double amp    = js.get( "amplitude", 0.).asDouble();
        dg::SinX sinX( amp, 0., 2.*M_PI*k_psi);
        ntilde = dg::pullback( dg::compose( sinX, mag.psip()), grid);
    }
    else
        throw dg::Error(dg::Message()<< "Invalid ntilde type "<<type<<"\n");
    return ntilde;
}

}//namespace detail

/* The purpose of this file is to provide an interface for custom initial conditions and
 * source profiles.  Just add your own to the relevant map below.
 */

std::array<std::array<dg::x::DVec,2>,2> initial_conditions(
    Explicit<dg::x::CylindricalGrid3d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec>& feltor,
    const dg::x::CylindricalGrid3d& grid, const feltor::Parameters& p,
    const dg::geo::TokamakMagneticField& mag, dg::file::WrappedJsonValue js,
    double & time )
{
    time = 0;
    //js = input["init"]
    std::array<std::array<dg::x::DVec,2>,2> y0;
    y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<dg::x::DVec>(
        dg::evaluate( dg::zero, grid));
    std::string type = js.get("type", "zero").asString();
    if ( "zero" == type)
        return y0;
    else if( "ne" == type || "ni" == type)
    {
        dg::x::HVec ntilde  = detail::make_ntilde(  grid, mag, js["ntilde"]);
        dg::x::HVec profile = detail::make_profile( grid, mag, js["profile"]);
        dg::x::HVec damping = detail::make_damping( grid, mag, js["damping"]);
        dg::blas1::axpby( 1., ntilde, 1., profile, profile);
        dg::blas1::pointwiseDot( damping, profile, profile);
        //actually we should always invert according to Markus
        //because the dG direct application is supraconvergent
        if( "ne" == type)
        {
            dg::assign( profile, y0[0][0]);
            std::string initphi = js.get("potential", "zero_pol").asString();
#ifdef WITH_MPI
            int rank;
            MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
            DG_RANK0 std::cout << "initialize potential with "<<initphi << std::endl;
            feltor.initializeni( y0[0][0], y0[0][1], initphi);
            double minimalni = dg::blas1::reduce( y0[0][1], 1, thrust::minimum<double>());
            DG_RANK0 std::cerr << "Minimum Ni value "<<minimalni+1<<std::endl;
            if( minimalni <= -1)
            {
                throw dg::Error(dg::Message()<< "ERROR: invalid initial condition. Increase value for alpha since now the ion gyrocentre density is negative!\n"
                    << "Minimum Ni value "<<minimalni+1);
            }
        }
        else if( "ni" == type)
        {
            dg::assign( profile, y0[0][1]);
            std::string initphi = js.get("potential", "zero_pol").asString();
            feltor.initializene( y0[0][1], y0[0][0], initphi);
        }
    }
    else if( "restart" == type)
    {
        std::string file = js["init"]["file"].asString();
        return init_from_file( file, grid, p, time);
    }
    else
        throw dg::Error(dg::Message()<< "Invalid initial condition "<<type<<"\n");
    //we = 0
    //Wi = 0
    return y0;
};

dg::x::HVec source_profiles(
    bool& fixed_profile, //indicate whether a profile should be forced (yes or no)
    dg::x::HVec& ne_profile,    // if fixed_profile is yes you need to construct something here, if no then you can ignore the parameter; if you construct something it will show in the output file
    const dg::x::CylindricalGrid3d& grid,
    const dg::geo::TokamakMagneticField& mag,
    const dg::file::WrappedJsonValue& js )
{
    //js = input["source"]

    std::string type  = js.get( "type", "zero").asString();
    dg::x::HVec source = dg::evaluate( dg::zero, grid);
    ne_profile = source;
    if( "zero" == type)
    {
    }
    else if( "fixed_profile" == type)
    {
        fixed_profile = true;
        ne_profile = detail::make_profile(grid,mag, js["profile"]);
        source = detail::make_damping( grid, mag, js["damping"]);
    }
    else if("influx" == type)
    {
        fixed_profile = false;
        source  = detail::make_ntilde(  grid, mag, js["ntilde"]);
        ne_profile = detail::make_profile( grid, mag, js["profile"]);
        dg::x::HVec damping = detail::make_damping( grid, mag, js["damping"]);
        dg::blas1::axpby( 1., source, 1., ne_profile, source);
        dg::blas1::pointwiseDot( damping, ne_profile, ne_profile);
        dg::blas1::pointwiseDot( damping, source, source);
    }
    else if( "torpex" == type)
    {
        fixed_profile = false;
        double rhosinm = 0.98 / mag.R0();
        double rhosinm2 = rhosinm*rhosinm;
        source = dg::pullback( detail::TorpexSource(
        0.98/rhosinm, -0.02/rhosinm, 0.0335/rhosinm, 0.05/rhosinm, 565*rhosinm2),
            grid);
    }
    else if( "tcv" == type)
    {
        fixed_profile = false;
        const double R_0 = 1075., Z_0 = -10.;
        const double psip0 = mag.psip()( R_0, Z_0);
        const double sigma = 9.3e-3*psip0/0.4;
        source = dg::pullback(
            dg::compose( dg::GaussianX( psip0, sigma, 1.),  mag.psip() ), grid);
        dg::blas1::pointwiseDot( detail::xpoint_damping(grid,mag),
               source, source);
    }
    else
        throw dg::Error(dg::Message()<< "Invalid source type "<<type<<"\n");
    return source;
};

} //namespace feltor
