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
    double operator()( double R, double Z, double) const{
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
    const dg::geo::TokamakMagneticField& mag ) // unmodified field
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
    if( mag.params().getDescription() == dg::geo::description::doubleX)
    {
        double RX1 = mag.R0() - 1.1*mag.params().triangularity()*mag.params().a();
        double ZX1 = -1.1*mag.params().elongation()*mag.params().a();
        dg::geo::findXpoint( mag.get_psip(), RX1, ZX1);
        double RX2 = mag.R0() - 1.1*mag.params().triangularity()*mag.params().a();
        double ZX2 = +1.1*mag.params().elongation()*mag.params().a();
        dg::geo::findXpoint( mag.get_psip(), RX2, ZX2);
        auto mod1 = dg::geo::ZCutter(ZX1);
        auto mod2 = dg::geo::ZCutter(ZX2, -1);
        auto mod = dg::geo::mod::SetIntersection( mod1, mod2);
        xpoint_damping = dg::pullback( mod, grid);
    }
    return xpoint_damping;
}
dg::x::HVec pfr_damping(
    const dg::x::CylindricalGrid3d& grid,
    const dg::geo::TokamakMagneticField& mag, // unmodified field
    dg::file::WrappedJsonValue js
    )
{
    dg::x::HVec xpoint = xpoint_damping( grid, mag); // zero outside
    dg::x::HVec damping = dg::evaluate( dg::one, grid);
    if( mag.params().getDescription() == dg::geo::description::standardX
        || mag.params().getDescription() == dg::geo::description::doubleX)
    {
        double alpha0 = js["alpha"].get(0, 0.2).asDouble();
        double alpha1 = js["alpha"].get(1, 0.1).asDouble();
        if( alpha0 == 0)
            throw dg::Error(dg::Message()<< "Invalid parameter: damping alpha must not be 0\n");
        double boundary0 = js["boundary"].get(0, 1.1).asDouble();
        double boundary1 = js["boundary"].get(1, 1.0).asDouble();
        dg::x::HVec damping0 = damping, damping1 = damping;
        damping0 = dg::pullback(
            dg::compose(dg::PolynomialHeaviside(
                boundary0-alpha0/2., alpha0/2., -1),
                    dg::geo::RhoP(mag)), grid);
        damping1 = dg::pullback(
            dg::compose(dg::PolynomialHeaviside(
                boundary1+alpha1/2., alpha1/2., +1),
                    dg::geo::RhoP(mag)), grid);
        // Set Union
        dg::blas1::evaluate( damping1, dg::equals(), []DG_DEVICE(double x, double y)
                { return x+y-x*y;}, xpoint, damping1);
        // Set Intersection
        dg::blas1::pointwiseDot( damping0, damping1, damping);
    }
    return damping;
}

dg::x::HVec make_profile(
    const dg::x::CylindricalGrid3d& grid,
    const dg::geo::TokamakMagneticField& mag,
    dg::file::WrappedJsonValue js, double& nbg )
{
    //js = input["profile"]
    std::string type = js.get("type","const").asString();
    nbg      = js.get( "background", 1.0).asDouble();
    dg::x::HVec profile = dg::evaluate( dg::zero, grid);
    if( "const" == type)
        dg::blas1::plus( profile, nbg);
    else if( "aligned" == type)
    {
        double RO=mag.R0(), ZO=0.;
        dg::geo::findOpoint( mag.get_psip(), RO, ZO);
        double psipO = mag.psip()( RO, ZO);
        double npeak = js.get( "npeak", 1.0).asDouble();
        double nsep = js.get( "nsep", 1.0).asDouble();
        profile = dg::pullback( dg::compose(
                [npeak,nsep,nbg, psipO]DG_DEVICE ( double psip){
                    if( psip/psipO  > 0)
                        return npeak*psip/psipO + nsep*(psipO-psip)/psipO;
                    else
                        return nbg + exp( (npeak-nsep)/psipO/(nsep-nbg)* psip) *( nsep-nbg);
                }, mag.psip()), grid);
    }
    else if ( "gaussian" == type )
    {
        double x0  = mag.R0() + js.get( "posX", 0.).asDouble() *mag.params().a();
        double y0  = mag.params().a()*js.get( "posY", 0.).asDouble();
        double sigma    = js.get( "sigma", 1.0).asDouble();
        double nprofamp = js.get( "amplitude", 1.0).asDouble();
        if( sigma == 0)
            throw dg::Error(dg::Message()<< "Invalid parameter: sigma must not be 0 in turbulence on gaussian\n");
        dg::Gaussian prof( x0, y0, sigma, sigma, nprofamp);
        profile = dg::pullback( prof, grid);
        dg::blas1::plus( profile, nbg);
    }
    else if( "Compass_L_mode" == type)
        {
        profile = dg::pullback( dg::compose([]DG_DEVICE ( double rho_p){
                        return -2.4 + (8.5 +2.4)/(1 + pow(rho_p/0.784,2.88));
                }, dg::geo::RhoP( mag)), grid);
        }
    else
        throw dg::Error(dg::Message()<< "Invalid profile type "<<type<<"\n");
    return profile;
}


dg::x::HVec make_damping(
    const dg::x::CylindricalGrid3d& grid,
    const dg::geo::TokamakMagneticField& mag, //unmodified field
    dg::file::WrappedJsonValue js)
    //js = input["damping"]
{
    dg::x::HVec damping = dg::evaluate( dg::one, grid);
    //Q: I think damping is the same as " 1 - createWallRegion"?
    //A: No the wall uses outside alpha but alignedPFR creates inside alpha!
    //This is not a problem in parameter setup
    //Also there may be a difference whether psip or rho_p is used to evaluate
    //For now we keep original init for backward compatibility
    //and add sol_pfr and sol_pfr_2X for future use
    std::string type = js.get("type","none").asString();
    try{
        dg::geo::str2modifier.at( type);
        auto wall_f = dg::geo::createWallRegion( mag, js);
        auto wall  = dg::pullback(wall_f, grid);
        // damping = 1 - wall
        dg::blas1::axpby( 1., damping, -1., wall, damping);
        return damping;
    }
    catch ( std::out_of_range& err)
    {
    }
    // This part is kept for backwards compatibility but is deprecated
    if( "none" == type)
        ;
    else if( "aligned" == type || "alignedX" == type)
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
        if( "alignedX" == type)
            dg::blas1::pointwiseDot( xpoint_damping(grid,mag),
                damping, damping); // SetIntersection
    }
    else if( "alignedPFR" == type)
    {
        damping = pfr_damping( grid, mag, js);
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
template<class Explicit>
dg::x::HVec make_ntilde(
    Explicit& feltor,
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
    else if( "blob" == type || "circle" == type)
    {
        double amp   = js.get( "amplitude", 0.).asDouble();
        double sigma = js.get( "sigma", 0.).asDouble() * mag.params().a();
        double x0  = mag.R0() + js.get( "posX", 0.).asDouble() *mag.params().a();
        double y0  = mag.params().a()*js.get( "posY", 0.).asDouble();
        if( sigma == 0)
            throw dg::Error(dg::Message()<< "Invalid parameter: sigma must not be 0 in straight blob initial condition\n");
        dg::geo::CylindricalFunctor init0 = dg::Gaussian(
                x0, y0, sigma, sigma, amp);
        if( type == "circle")
            init0 = [amp, sigma, x0, y0]( double x, double y) {
                if( (x-x0)*(x-x0) + (y-y0)*(y-y0) < sigma*sigma)
                    return amp;
                return 0.;
            };
        if( grid.Nz() == 1 )
            ntilde = dg::pullback( init0, grid);
        else
        {
            std::string parallel = js.get( "parallel", "gaussian").asString();
            unsigned revolutions = js.get( "revolutions", 1).asUInt();
            double sigma_z = js.get( "sigma_z", 0.).asDouble()*M_PI;
            auto bhat = dg::geo::createBHat(mag);
            if( sigma_z == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma_z must not be 0 in blob initial condition\n");
            if( parallel == "gaussian")
            {
                dg::GaussianZ gaussianZ( 0., sigma_z, 1.0);
                ntilde = feltor.fieldaligned().evaluate( init0,
                        gaussianZ, 0, revolutions);
            }
            else if( parallel == "exact-gaussian")
            {
                double rk4eps = js.get("rk4eps", 1e-6).asDouble();
                dg::GaussianZ gaussianZ( 0., sigma_z, 1.0);
                ntilde = dg::geo::fieldaligned_evaluate( grid, bhat, init0,
                        gaussianZ, 0, revolutions, rk4eps);
            }
            else if( parallel == "toroidal-gaussian")
            {
                std::function<double(double,double,double)> initT = dg::Gaussian3d(
                        x0, y0, M_PI, sigma, sigma, sigma_z, amp);
                if( type == "circle")
                    initT = [amp, sigma, sigma_z, x0, y0]( double x, double y, double z) {
                        if( (x-x0)*(x-x0) + (y-y0)*(y-y0) < sigma*sigma)
                            return amp*exp( -(z-M_PI)*(z-M_PI)/2/sigma_z/sigma_z);
                        return 0.;
                    };
                ntilde = dg::pullback( initT, grid);
            }
            else if( parallel == "toroidal")
            {
                std::function<double(double,double,double)> initT = dg::Gaussian(
                        x0, y0, sigma, sigma, amp);
                if( type == "circle")
                    initT = [amp, sigma, x0, y0]( double x, double y, double) {
                        if( (x-x0)*(x-x0) + (y-y0)*(y-y0) < sigma*sigma)
                            return amp;
                        return 0.;
                    };
                ntilde = dg::pullback( initT, grid);
            }
            else if( parallel == "step")
            {
                double rk4eps = js.get("rk4eps", 1e-6).asDouble();
                dg::Iris gaussianZ( -sigma_z, +sigma_z);
                ntilde = dg::geo::fieldaligned_evaluate( grid, bhat, init0,
                        gaussianZ, 0, revolutions, rk4eps);
            }
            else if( parallel == "double-step")
            {
                double rk4eps = js.get("rk4eps", 1e-6).asDouble();
                ntilde = dg::geo::fieldaligned_evaluate( grid, bhat, init0,
                        [sigma_z](double s) {
                        if( (s <  0) && (s > -sigma_z)) return 0.5;
                        if( (s >= 0) && (s < +sigma_z)) return 1.0;
                        return 0.;}, 0, revolutions, rk4eps);
            }
            else
                throw dg::Error(dg::Message()<< "Invalid parallel initial condition: "<<parallel<<"\n");
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
            std::string parallel = js.get( "parallel", "gaussian").asString();
            unsigned revolutions = js.get( "revolutions", 1).asUInt();
            double sigma_z = js.get( "sigma_z", 0.).asDouble()*M_PI;
            auto bhat = dg::geo::createBHat(mag);
            if( sigma_z == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma_z must not be 0 in turbulence initial condition\n");
            if( parallel == "gaussian")
            {
                dg::GaussianZ gaussianZ( 0., sigma_z, 1.0);
                ntilde = feltor.fieldaligned().evaluate( init0,
                        gaussianZ, 0, revolutions);
            }
            else if( parallel == "exact-gaussian")
            {
                double rk4eps = js.get("rk4eps", 1e-6).asDouble();
                dg::GaussianZ gaussianZ( 0., sigma_z, 1.0);
                ntilde = dg::geo::fieldaligned_evaluate( grid, bhat, init0,
                        gaussianZ, 0, revolutions, rk4eps);
            }
            else if( parallel == "step")
            {
                double rk4eps = js.get("rk4eps", 1e-6).asDouble();
                dg::Iris gaussianZ( -sigma_z, +sigma_z);
                ntilde = dg::geo::fieldaligned_evaluate( grid, bhat, init0,
                        gaussianZ, 0, revolutions, rk4eps);

            }
            else
                throw dg::Error(dg::Message()<< "Invalid parallel initial condition for turbulence: "<<parallel<<"\n");
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
 * @note y0[1] has to be the staggered velocity
 */

std::array<std::array<dg::x::DVec,2>,2> initial_conditions(
    Explicit<dg::x::CylindricalGrid3d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec>& feltor,
    const dg::x::CylindricalGrid3d& grid, const feltor::Parameters& p,
    const dg::geo::TokamakMagneticField& mag,
    const dg::geo::TokamakMagneticField& unmod_mag,
    dg::file::WrappedJsonValue js,
    double & time, dg::geo::CylindricalFunctor& sheath_coordinate )
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    time = 0;
    //js = input["init"]
    std::array<std::array<dg::x::DVec,2>,2> y0;
    y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<dg::x::DVec>(
        dg::evaluate( dg::zero, grid));
    std::string type = js.get("type", "zero").asString();
    if( "fields" == type)
    {
        std::string ntype = js["density"].get("type", "zero").asString();
        DG_RANK0 std::cout << "# Initialize density with "<<ntype << std::endl;
        if ( "const" == ntype)
        {
            double nbg = js["density"].get("background", 0.1).asDouble();
            y0[0][0] = y0[0][1] = dg::construct<dg::x::DVec>(
                    dg::evaluate( dg::CONSTANT( nbg), grid));
        }
        else if( "ne" == ntype || "ni" == ntype)
        {
            double nbg = 0.;
            dg::x::HVec ntilde  = detail::make_ntilde(  feltor, grid, mag,
                    js["density"]["ntilde"]);
            dg::x::HVec profile = detail::make_profile( grid, mag,
                    js["density"]["profile"], nbg);
            dg::x::HVec damping = detail::make_damping( grid, unmod_mag,
                    js["density"]["damping"]);
            dg::x::HVec density = profile;
            dg::blas1::subroutine( [nbg]( double profile, double ntilde, double
                        damping, double& density)
                    { density = (profile+ntilde-nbg)*damping+nbg;},
                    profile, ntilde, damping, density);
            //actually we should always invert according to Markus
            //because the dG direct application is supraconvergent
            std::string ptype = js["potential"].get("type", "zero_pol").asString();
            DG_RANK0 std::cout << "# Initialize potential with "<<ptype << std::endl;
            if( "ne" == ntype)
            {
                dg::assign( density, y0[0][0]);
                feltor.initializeni( y0[0][0], y0[0][1], ptype);
                double minimalni = dg::blas1::reduce( y0[0][1], 1e10,
                        thrust::minimum<double>());
                DG_RANK0 std::cerr << "# Minimum Ni value "<<minimalni<<std::endl;
                if( minimalni <= 0.0)
                {
                    throw dg::Error(dg::Message()<< "ERROR: invalid initial condition. Increase value for alpha since now the ion gyrocentre density is negative!\n"
                        << "Minimum Ni value "<<minimalni);
                }
            }
            else if( "ni" == ntype)
            {
                dg::assign( density, y0[0][1]);
                feltor.initializene( y0[0][1], y0[0][0], ptype);
            }
        }
        else
            throw dg::Error(dg::Message()<< "Invalid density initial condition "<<ntype<<"\n");
        // init (staggered) velocity and thus canonical W
        std::string utype = js["velocity"].get("type", "zero").asString();
        DG_RANK0 std::cout << "# Initialize velocity with "<<utype << std::endl;
        if( "zero" == utype)
            ; // velocity already is zero
        else if( "ui" == utype )
        {
            std::string uprofile = js["velocity"].get("profile", "linear_cs").asString();
            if( !(uprofile == "linear_cs"))
                throw dg::Error(dg::Message(_ping_)<<"Warning! Unkown velocity profile '"<<uprofile<<"'! I don't know what to do! I exit!\n");

            std::unique_ptr<dg::x::aGeometry2d> perp_grid_ptr( grid.perp_grid());
            dg::x::HVec coord2d = dg::pullback( sheath_coordinate,
                    *perp_grid_ptr);
            dg::x::HVec ui;
            dg::assign3dfrom2d( coord2d, ui, grid);
            dg::blas1::scal( ui, sqrt( 1.0+p.tau[1]));
            dg::assign( ui, y0[1][1]); // Wi = Ui

            std::string atype = js["aparallel"].get("type", "zero").asString();
            DG_RANK0 std::cout << "# Initialize aparallel with "<<atype << std::endl;
            // ue = Ui
            if( atype == "zero")
            {
                dg::blas1::subroutine( [] DG_DEVICE( double ne, double ni,
                            double& ue, double ui)
                        { ue = ni*ui/ne;}, y0[0][0], y0[0][1],
                        y0[1][0], y0[1][1]);
            }
            else if( !(atype == "zero"))
            {
                throw dg::Error(dg::Message(_ping_)<<"Warning! aparallel type '"<<atype<<"' not recognized. I have beta = "<<p.beta<<" ! I don't know what to do! I exit!\n");
            }

        }
        else
            throw dg::Error(dg::Message()<< "Invalid velocity initial condition "<<utype<<"\n");
    }
    else if( "restart" == type)
    {
        std::string file = js["init"]["file"].asString();
        return init_from_file( file, grid, p, time);
    }
    else
        throw dg::Error(dg::Message()<< "Invalid initial condition "<<type<<"\n");
    return y0;
};

dg::x::HVec source_profiles(
    Explicit<dg::x::CylindricalGrid3d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec>& feltor,
    bool& fixed_profile, //indicate whether a profile should be forced (yes or no)
    dg::x::HVec& ne_profile,    // if fixed_profile is yes you need to construct something here, if no then you can ignore the parameter; if you construct something it will show in the output file
    const dg::x::CylindricalGrid3d& grid,
    const dg::geo::TokamakMagneticField& mag,
    const dg::geo::TokamakMagneticField& unmod_mag,
    const dg::file::WrappedJsonValue& js,
    double& minne, double& minrate, double& minalpha
    )
{
    //js = input["source"]
    minne = js.get("minne", 0.).asDouble();
    minrate =  minalpha = 0;
    if( minne != 0)
    {
        minrate = js.get("minrate", 1.).asDouble();
        minalpha = js.get("minalpha", 0.05).asDouble();
    }

    std::string type  = js.get( "type", "zero").asString();
    dg::x::HVec source = dg::evaluate( dg::zero, grid);
    ne_profile = source;
    if( "zero" == type)
    {
    }
    else if( "fixed_profile" == type)
    {
        fixed_profile = true;
        double nbg = 0;
        ne_profile = detail::make_profile(grid, mag, js["profile"], nbg);
        source = detail::make_damping( grid, unmod_mag, js["damping"]);
    }
    else if("influx" == type)
    {
        fixed_profile = false;
        double nbg = 0.;
        source  = detail::make_ntilde( feltor, grid, mag, js["ntilde"]);
        ne_profile = detail::make_profile( grid, mag, js["profile"], nbg);
        dg::x::HVec damping = detail::make_damping( grid, unmod_mag, js["damping"]);
        dg::blas1::subroutine( [nbg]( double& profile, double& ntilde, double
                    damping) {
                    ntilde  = (profile+ntilde-nbg)*damping+nbg;
                    profile = (profile-nbg)*damping +nbg;
                },
                ne_profile, source, damping);
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
