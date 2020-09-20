#pragma once
#include "dg/file/nc_utilities.h"

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
HVec circular_damping( const Geometry& grid,
    const feltor::Parameters& p,
    const dg::geo::TokamakMagneticField& mag )
{
    if( p.profile_alpha == 0)
        throw dg::Error(dg::Message()<< "Invalid parameter: profile alpha must not be 0\n");
    HVec circular = dg::pullback( dg::compose(
                dg::PolynomialHeaviside( mag.params().a(), mag.params().a()*p.profile_alpha/2., -1),
                Radius( mag.R0(), 0.)), grid);
    return circular;
}


HVec xpoint_damping(const Geometry& grid,
    const feltor::Parameters& p,
    const dg::geo::TokamakMagneticField& mag )
{
    HVec xpoint_damping = dg::evaluate( dg::one, grid);
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
HVec profile_damping(const Geometry& grid,
    const feltor::Parameters& p,
    const dg::geo::TokamakMagneticField& mag )
{
    if( p.profile_alpha == 0)
        throw dg::Error(dg::Message()<< "Invalid parameter: profile alpha must not be 0\n");
    HVec profile_damping = dg::pullback( dg::compose(dg::PolynomialHeaviside(
        1.-p.profile_alpha/2., p.profile_alpha/2., -1), dg::geo::RhoP(mag)), grid);
    dg::blas1::pointwiseDot( xpoint_damping(grid,p,mag),
        profile_damping, profile_damping);
    return profile_damping;
}
HVec profile(const Geometry& grid,
    const feltor::Parameters& p,
    const dg::geo::TokamakMagneticField& mag ){
    double RO=mag.R0(), ZO=0.;
    dg::geo::findOpoint( mag.get_psip(), RO, ZO);
    double psipO = mag.psip()( RO, ZO);
    //First the profile and the source (on the host since we want to output those)
    HVec profile = dg::pullback( dg::compose(dg::LinearX(
        p.nprofamp/psipO, 0.), mag.psip()), grid);
    dg::blas1::pointwiseDot( profile_damping(grid,p,mag), profile, profile);
    return profile;
}
HVec source_damping(const Geometry& grid,
    const feltor::Parameters& p,
    const dg::geo::TokamakMagneticField& mag )
{
    if( p.source_alpha == 0)
        throw dg::Error(dg::Message()<< "Invalid parameter: source alpha must not be 0\n");
    HVec source_damping = dg::pullback(
        dg::compose(dg::PolynomialHeaviside(
            p.source_boundary-p.source_alpha/2.,
        p.source_alpha/2., -1 ), dg::geo::RhoP(mag)), grid);
    dg::blas1::pointwiseDot( xpoint_damping(grid,p,mag),
           source_damping, source_damping);
    return source_damping;
}


void init_ni(
    std::array<std::array<DVec,2>,2>& y0,
    Explicit<Geometry, IDMatrix, DMatrix, DVec>& feltor,
    const Geometry& grid, const feltor::Parameters& p,
    dg::geo::TokamakMagneticField& mag )
{
#ifdef FELTOR_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    MPI_OUT std::cout << "initialize ni with "<<p.initphi << std::endl;
    feltor.initializeni( y0[0][0], y0[0][1], p.initphi);
    double minimalni = dg::blas1::reduce( y0[0][1], 1, thrust::minimum<double>());
    MPI_OUT std::cerr << "Minimum Ni value "<<minimalni+1<<std::endl;
    if( minimalni <= -1)
    {
        throw dg::Error(dg::Message()<< "ERROR: invalid initial condition. Increase value for alpha since now the ion gyrocentre density is negative!\n"
            << "Minimum Ni value "<<minimalni+1);
    }
};
}//namespace detail

//for wall shadow
HVec wall_damping(const Geometry& grid,
    const feltor::Parameters& p,
    const dg::geo::TokamakMagneticField& mag )
{
    if( p.source_alpha == 0)
        throw dg::Error(dg::Message()<< "Invalid parameter: damping alpha must not be 0\n");
    HVec wall_damping = dg::pullback(dg::compose( dg::PolynomialHeaviside(
        p.damping_boundary+p.damping_alpha/2., p.damping_alpha/2., +1),
                dg::geo::RhoP(mag)), grid);
    return wall_damping;
}

/* The purpose of this file is to provide an interface for custom initial conditions and
 * source profiles.  Just add your own to the relevant map below.
 */

std::map<std::string, std::function< std::array<std::array<DVec,2>,2>(
    Explicit<Geometry, IDMatrix, DMatrix, DVec>& f,
    const Geometry& grid, const feltor::Parameters& p,
    dg::geo::TokamakMagneticField& mag )
> > initial_conditions =
{
    { "blob",
        []( Explicit<Geometry, IDMatrix, DMatrix, DVec>& f,
            const Geometry& grid, const feltor::Parameters& p,
            dg::geo::TokamakMagneticField& mag )
        {
            std::array<std::array<DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<DVec>(detail::profile(grid,p,mag));
            HVec ntilde = dg::evaluate(dg::zero,grid);
            if( p.sigma_z == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma_z must not be 0 in straight blob initial condition\n");
            dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
            if( p.sigma == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma must not be 0 in straight blob initial condition\n");
            dg::Gaussian init0( mag.R0()+p.posX*mag.params().a(), p.posY*mag.params().a(), p.sigma, p.sigma, p.amp);
            if( p.symmetric)
                ntilde = dg::pullback( init0, grid);
            else
            {
                dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                    fieldaligned( mag, grid, p.bcxN, p.bcyN,
                    dg::geo::NoLimiter(), p.rk4eps, 5, 5);
                //evaluate should always be used with mx,my > 1 (but this takes a lot of memory)
                ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, 3);
            }
            dg::blas1::axpby( 1., dg::construct<DVec>(ntilde), 1., y0[0][0]);
            detail::init_ni( y0, f,grid,p,mag);

            dg::blas1::copy( 0., y0[1][0]); //set we = 0
            dg::blas1::copy( 0., y0[1][1]); //set Wi = 0
            return y0;
        }
    },
    { "straight blob",
        []( Explicit<Geometry, IDMatrix, DMatrix, DVec>& f,
            const Geometry& grid, const feltor::Parameters& p,
            dg::geo::TokamakMagneticField& mag )
        {
            std::array<std::array<DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<DVec>(detail::profile(grid,p,mag));
            HVec ntilde = dg::evaluate(dg::zero,grid);
            if( p.sigma_z == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma_z must not be 0 in straight blob initial condition\n");
            dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
            if( p.sigma == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma must not be 0 in straight blob initial condition\n");
            dg::Gaussian init0( mag.R0()+p.posX*mag.params().a(), p.posY*mag.params().a(), p.sigma, p.sigma, p.amp);
            if( p.symmetric)
                ntilde = dg::pullback( init0, grid);
            else
            {
                dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                    fieldaligned( mag, grid, p.bcxN, p.bcyN,
                    dg::geo::NoLimiter(), p.rk4eps, 5, 5);
                //evaluate should always be used with mx,my > 1 (but this takes a lot of memory)
                ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, 1);
            }
            dg::blas1::axpby( 1., dg::construct<DVec>(ntilde), 1., y0[0][0]);
            detail::init_ni( y0, f,grid,p,mag);

            dg::blas1::copy( 0., y0[1][0]); //set we = 0
            dg::blas1::copy( 0., y0[1][1]); //set Wi = 0
            return y0;
        }
    },
    { "turbulence",
        []( Explicit<Geometry, IDMatrix, DMatrix, DVec>& f,
            const Geometry& grid, const feltor::Parameters& p,
            dg::geo::TokamakMagneticField& mag )
        {
            std::array<std::array<DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<DVec>(detail::profile(grid,p,mag));
            HVec ntilde = dg::evaluate(dg::zero,grid);
            if( p.sigma_z == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma_z must not be 0 in turbulence initial condition\n");
            dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
            dg::BathRZ init0(16,16,grid.x0(),grid.y0(), 30.,2.,p.amp);
            if( p.symmetric)
                ntilde = dg::pullback( init0, grid);
            else
            {
                dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                    fieldaligned( mag, grid, p.bcxN, p.bcyN,
                    dg::geo::NoLimiter(), p.rk4eps, 1, 1);
                //For turbulence the exact evaluate is maybe not so important (thus takes less memory)
                ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, 1);
            }
            dg::blas1::pointwiseDot( detail::profile_damping(grid,p,mag), ntilde, ntilde);
            dg::blas1::axpby( 1., dg::construct<DVec>(ntilde), 1., y0[0][0]);
            detail::init_ni( y0, f,grid,p,mag);

            dg::blas1::copy( 0., y0[1][0]); //set we = 0
            dg::blas1::copy( 0., y0[1][1]); //set Wi = 0
            return y0;
        }
    },
    { "zonal",
        []( Explicit<Geometry, IDMatrix, DMatrix, DVec>& f,
            const Geometry& grid, const feltor::Parameters& p,
            dg::geo::TokamakMagneticField& mag )
        {
            std::array<std::array<DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<DVec>(detail::profile(grid,p,mag));
            HVec ntilde = dg::evaluate(dg::zero,grid);
            dg::SinX sinX( p.amp, 0., p.k_psi);
            ntilde = dg::pullback( dg::compose( sinX, mag.psip()), grid);
            dg::blas1::pointwiseDot( detail::profile_damping(grid,p,mag), ntilde, ntilde);
            dg::blas1::axpby( 1., dg::construct<DVec>(ntilde), 1., y0[0][0]);
            detail::init_ni( y0, f,grid,p,mag);

            dg::blas1::copy( 0., y0[1][0]); //set we = 0
            dg::blas1::copy( 0., y0[1][1]); //set Wi = 0
            return y0;
        }
    },
    { "turbulence on gaussian",
        []( Explicit<Geometry, IDMatrix, DMatrix, DVec>& f,
            const Geometry& grid, const feltor::Parameters& p,
            dg::geo::TokamakMagneticField& mag )
        {
            if( p.sigma == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma must not be 0 in turbulence on gaussian\n");
            dg::Gaussian prof( mag.R0()+p.posX*mag.params().a(), p.posY*mag.params().a(), p.sigma,
                p.sigma, p.nprofamp);
            std::array<std::array<DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<DVec>(
                dg::pullback( prof, grid) );

            HVec ntilde = dg::evaluate(dg::zero,grid);
            if( p.sigma_z == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma_z must not be 0 in turbulence on gaussian\n");
            dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
            dg::BathRZ init0(16,16,grid.x0(),grid.y0(), 30.,2.,p.amp);
            if( p.symmetric)
                ntilde = dg::pullback( init0, grid);
            else
            {
                dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                    fieldaligned( mag, grid, p.bcxN, p.bcyN,
                    dg::geo::NoLimiter(), p.rk4eps, 1, 1);
                //For turbulence the exact evaluate is maybe not so important (thus takes less memory)
                ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, 1);
            }
            dg::blas1::axpby( 1., dg::construct<DVec>(ntilde), 1., y0[0][0] );
            dg::blas1::pointwiseDot( dg::construct<DVec>(detail::circular_damping(grid,p,mag)),
                y0[0][0], y0[0][0] );
            detail::init_ni( y0, f,grid,p,mag);

            dg::blas1::copy( 0., y0[1][0]); //set we = 0
            dg::blas1::copy( 0., y0[1][1]); //set Wi = 0
            return y0;
        }
    }
};

std::map<std::string, std::function< HVec(
    bool& fixed_profile, //indicate whether a profile should be forced (yes or no)
    HVec& ne_profile,    // if fixed_profile is yes you need to construct something here, if no then you can ignore the parameter; if you construct something it will show in the output file
    Geometry& grid, const feltor::Parameters& p,
    dg::geo::TokamakMagneticField& mag )
> > source_profiles =
{
    {"profile",
        []( bool& fixed_profile, HVec& ne_profile,
        Geometry& grid, const feltor::Parameters& p,
        dg::geo::TokamakMagneticField& mag )
        {
            fixed_profile = true;
            ne_profile = dg::construct<HVec>( detail::profile(grid, p,mag));
            HVec source_profile = dg::construct<HVec> ( detail::source_damping( grid, p,mag));
            return source_profile;
        }
    },
    {"influx",
        []( bool& fixed_profile, HVec& ne_profile,
        Geometry& grid, const feltor::Parameters& p,
        dg::geo::TokamakMagneticField& mag )
        {
            fixed_profile = false;
            ne_profile = dg::construct<HVec>( detail::profile(grid, p,mag));
            HVec source_profile = dg::construct<HVec> ( detail::source_damping( grid, p,mag));
            return source_profile;
        }
    },
    {"torpex",
        []( bool& fixed_profile, HVec& ne_profile,
        Geometry& grid, const feltor::Parameters& p,
        dg::geo::TokamakMagneticField& mag )
        {
            if( p.sigma == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma must not be 0 for torpex source profile\n");
            dg::Gaussian prof( mag.R0()+p.posX*mag.params().a(), p.posY*mag.params().a(), p.sigma,
                p.sigma, p.nprofamp);
            ne_profile = dg::pullback( prof, grid);
            fixed_profile = false;
            double rhosinm = 0.98 / mag.R0();
            double rhosinm2 = rhosinm*rhosinm;
            HVec source_profile = dg::construct<HVec> ( dg::pullback(
                detail::TorpexSource(0.98/rhosinm, -0.02/rhosinm, 0.0335/rhosinm, 0.05/rhosinm, 565*rhosinm2 ), grid) );
            return source_profile;
        }
    },
    {"tcv",
        []( bool& fixed_profile, HVec& ne_profile,
        Geometry& grid, const feltor::Parameters& p,
        dg::geo::TokamakMagneticField& mag )
        {
            const double R_0 = 1075., Z_0 = -10.;
            const double psip0 = mag.psip()( R_0, Z_0);
            const double sigma = 9.3e-3*psip0/0.4;

            fixed_profile = false;
            ne_profile = dg::construct<HVec>( detail::profile(grid, p,mag));
            HVec source_profile = dg::pullback(
                dg::compose( dg::GaussianX( psip0, sigma, 1.),  mag.psip() ), grid);
            dg::blas1::pointwiseDot( detail::xpoint_damping(grid,p,mag),
                   source_profile, source_profile);
            return source_profile;
        }
    },
    {"gaussian",
        []( bool& fixed_profile, HVec& ne_profile,
        Geometry& grid, const feltor::Parameters& p,
        dg::geo::TokamakMagneticField& mag )
        {
            fixed_profile = false;
            if( p.sigma == 0)
                throw dg::Error(dg::Message()<< "Invalid parameter: sigma must not be 0 for gaussian source profile\n");
            dg::Gaussian prof( mag.R0()+p.posX*mag.params().a(), p.posY*mag.params().a(), p.sigma,
                p.sigma, 1.);
            return dg::pullback( prof, grid);
        }
    },
};

} //namespace feltor
