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

struct Radius
{
    Radius ( double R0, double Z0): m_R0(R0), m_Z0(Z0) {}
    DG_DEVICE
    double operator()( double R, double Z) const{
        return sqrt( (R-m_R0)*(R-m_R0) + (Z-m_Z0)*(Z-m_Z0));
    }
    private:
    double m_R0, m_Z0;
};
HVec circular_damping( const Geometry& grid,
    const feltor::Parameters& p,
    const dg::geo::solovev::Parameters& gp, const dg::geo::TokamakMagneticField& mag )
{
    HVec circular = dg::pullback(dg::geo::Compose<dg::PolynomialHeaviside>(
        Radius( mag.R0(), 0.),
        gp.a, gp.a*p.alpha, -1), grid);
    return circular;
}


HVec xpoint_damping(const Geometry& grid,
    const feltor::Parameters& p,
    const dg::geo::solovev::Parameters& gp, const dg::geo::TokamakMagneticField& mag )
{
    HVec xpoint_damping = dg::evaluate( dg::one, grid);
    if( gp.hasXpoint() )
    {
        double RX = gp.R_0 - 1.1*gp.triangularity*gp.a;
        double ZX = -1.1*gp.elongation*gp.a;
        dg::geo::findXpoint( mag.get_psip(), RX, ZX);
        xpoint_damping = dg::pullback(
            dg::geo::ZCutter(ZX), grid);
    }
    return xpoint_damping;
}
HVec damping_damping(const Geometry& grid,
    const feltor::Parameters& p,
    const dg::geo::solovev::Parameters& gp, const dg::geo::TokamakMagneticField& mag )
{
    HVec damping_damping = dg::pullback(dg::geo::Compose<dg::PolynomialHeaviside>(
        //first change coordinate from psi to (psi_0 - psip)/psi_0
        dg::geo::Compose<dg::LinearX>( mag.psip(), -1./mag.psip()(mag.R0(), 0.),1.),
        //then shift
        p.rho_damping, p.alpha, +1), grid);
    return damping_damping;
}
HVec profile_damping(const Geometry& grid,
    const feltor::Parameters& p,
    const dg::geo::solovev::Parameters& gp, const dg::geo::TokamakMagneticField& mag )
{
    HVec profile_damping = dg::pullback( dg::geo::Compose<dg::PolynomialHeaviside>(
        mag.psip(), -p.alpha, p.alpha, -1), grid);
    dg::blas1::pointwiseDot( xpoint_damping(grid,p,gp,mag), profile_damping, profile_damping);
    return profile_damping;
}
HVec profile(const Geometry& grid,
    const feltor::Parameters& p,
    const dg::geo::solovev::Parameters& gp, const dg::geo::TokamakMagneticField& mag ){
    //First the profile and the source (on the host since we want to output those)
    HVec profile = dg::pullback( dg::geo::Compose<dg::LinearX>( mag.psip(),
        p.nprofamp/mag.psip()(mag.R0(), 0.), 0.), grid);
    dg::blas1::pointwiseDot( profile_damping(grid,p,gp,mag), profile, profile);
    return profile;
}
HVec source_damping(const Geometry& grid,
    const feltor::Parameters& p,
    const dg::geo::solovev::Parameters& gp, const dg::geo::TokamakMagneticField& mag )
{
    HVec source_damping = dg::pullback(dg::geo::Compose<dg::PolynomialHeaviside>(
        //first change coordinate from psi to (psi_0 - psip)/psi_0
        dg::geo::Compose<dg::LinearX>( mag.psip(), -1./mag.psip()(mag.R0(), 0.),1.),
        //then shift
        p.rho_source, p.alpha, -1), grid);
    dg::blas1::pointwiseDot( xpoint_damping(grid,p,gp,mag), source_damping, source_damping);
    return source_damping;
}


void init_ni(
    std::array<std::array<DVec,2>,2>& y0,
    Explicit<Geometry, IDMatrix, DMatrix, DVec>& feltor,
    const Geometry& grid, const feltor::Parameters& p,
    const dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag )
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

/* The purpose of this file is to provide an interface for custom initial conditions and
 * source profiles.  Just add your own to the relevant map below.
 */

std::map<std::string, std::function< std::array<std::array<DVec,2>,2>(
    Explicit<Geometry, IDMatrix, DMatrix, DVec>& f,
    const Geometry& grid, const feltor::Parameters& p,
    const dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag )
> > initial_conditions =
{
    { "blob",
        []( Explicit<Geometry, IDMatrix, DMatrix, DVec>& f,
            const Geometry& grid, const feltor::Parameters& p,
            const dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag )
        {
            std::array<std::array<DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<DVec>(detail::profile(grid,p,gp,mag));
            HVec ntilde = dg::evaluate(dg::zero,grid);
            dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
            dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
            if( p.symmetric)
                ntilde = dg::pullback( init0, grid);
            else
            {
                dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                    fieldaligned( mag, grid, p.bcxN, p.bcyN,
                    dg::geo::NoLimiter(), p.rk4eps, 5, 5);
                //evaluate should always be used with mx,my > 1
                ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, 3);
            }
            dg::blas1::axpby( 1., dg::construct<DVec>(ntilde), 1., y0[0][0]);
            detail::init_ni( y0, f,grid,p,gp,mag);

            dg::blas1::copy( 0., y0[1][0]); //set we = 0
            dg::blas1::copy( 0., y0[1][1]); //set Wi = 0
            return y0;
        }
    },
    { "straight blob",
        []( Explicit<Geometry, IDMatrix, DMatrix, DVec>& f,
            const Geometry& grid, const feltor::Parameters& p,
            const dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag )
        {
            std::array<std::array<DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<DVec>(detail::profile(grid,p,gp,mag));
            HVec ntilde = dg::evaluate(dg::zero,grid);
            dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
            dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
            if( p.symmetric)
                ntilde = dg::pullback( init0, grid);
            else
            {
                dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                    fieldaligned( mag, grid, p.bcxN, p.bcyN,
                    dg::geo::NoLimiter(), p.rk4eps, 5, 5);
                //evaluate should always be used with mx,my > 1
                ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, 1);
            }
            dg::blas1::axpby( 1., dg::construct<DVec>(ntilde), 1., y0[0][0]);
            detail::init_ni( y0, f,grid,p,gp,mag);

            dg::blas1::copy( 0., y0[1][0]); //set we = 0
            dg::blas1::copy( 0., y0[1][1]); //set Wi = 0
            return y0;
        }
    },
    { "turbulence",
        []( Explicit<Geometry, IDMatrix, DMatrix, DVec>& f,
            const Geometry& grid, const feltor::Parameters& p,
            const dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag )
        {
            std::array<std::array<DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<DVec>(detail::profile(grid,p,gp,mag));
            HVec ntilde = dg::evaluate(dg::zero,grid);
            dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
            dg::BathRZ init0(16,16,grid.x0(),grid.y0(), 30.,2.,p.amp);
            if( p.symmetric)
                ntilde = dg::pullback( init0, grid);
            else
            {
                dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                    fieldaligned( mag, grid, p.bcxN, p.bcyN,
                    dg::geo::NoLimiter(), p.rk4eps, 5, 5);
                //evaluate should always be used with mx,my > 1
                ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, 1);
            }
            dg::blas1::pointwiseDot( detail::profile_damping(grid,p,gp,mag), ntilde, ntilde);
            dg::blas1::axpby( 1., dg::construct<DVec>(ntilde), 1., y0[0][0]);
            detail::init_ni( y0, f,grid,p,gp,mag);

            dg::blas1::copy( 0., y0[1][0]); //set we = 0
            dg::blas1::copy( 0., y0[1][1]); //set Wi = 0
            return y0;
        }
    },
    { "zonal",
        []( Explicit<Geometry, IDMatrix, DMatrix, DVec>& f,
            const Geometry& grid, const feltor::Parameters& p,
            const dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag )
        {
            std::array<std::array<DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<DVec>(detail::profile(grid,p,gp,mag));
            HVec ntilde = dg::evaluate(dg::zero,grid);
            dg::geo::ZonalFlow init0(mag.psip(), p.amp, 0., p.k_psi);
            ntilde = dg::pullback( init0, grid);
            dg::blas1::pointwiseDot( detail::profile_damping(grid,p,gp,mag), ntilde, ntilde);
            dg::blas1::axpby( 1., dg::construct<DVec>(ntilde), 1., y0[0][0]);
            detail::init_ni( y0, f,grid,p,gp,mag);

            dg::blas1::copy( 0., y0[1][0]); //set we = 0
            dg::blas1::copy( 0., y0[1][1]); //set Wi = 0
            return y0;
        }
    },
    { "turbulence on gaussian",
        []( Explicit<Geometry, IDMatrix, DMatrix, DVec>& f,
            const Geometry& grid, const feltor::Parameters& p,
            const dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag )
        {
            dg::Gaussian prof( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma,
                p.sigma, p.nprofamp);
            std::array<std::array<DVec,2>,2> y0;
            y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<DVec>(
                dg::pullback( prof, grid) );

            HVec ntilde = dg::evaluate(dg::zero,grid);
            dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
            dg::BathRZ init0(16,16,grid.x0(),grid.y0(), 30.,2.,p.amp);
            if( p.symmetric)
                ntilde = dg::pullback( init0, grid);
            else
            {
                dg::geo::Fieldaligned<Geometry, IHMatrix, HVec>
                    fieldaligned( mag, grid, p.bcxN, p.bcyN,
                    dg::geo::NoLimiter(), p.rk4eps, 5, 5);
                //evaluate should always be used with mx,my > 1
                ntilde = fieldaligned.evaluate( init0, gaussianZ, 0, 1);
            }
            dg::blas1::axpby( 1., dg::construct<DVec>(ntilde), 1., y0[0][0] );
            dg::blas1::pointwiseDot( dg::construct<DVec>(detail::circular_damping(grid,p,gp,mag)),
                y0[0][0], y0[0][0] );
            detail::init_ni( y0, f,grid,p,gp,mag);

            dg::blas1::copy( 0., y0[1][0]); //set we = 0
            dg::blas1::copy( 0., y0[1][1]); //set Wi = 0
            return y0;
        }
    }
};

std::map<std::string, std::function< HVec(
    bool& fixed_profile, //indicate whether a profile should be forced (yes or no)
    HVec& ne_profile, //construct profile if yes, do nothing or construct (determines what is written in output fiele) if no
    Geometry& grid, const feltor::Parameters& p,
    const dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag )
> > source_profiles =
{
    {"profile",
        []( bool& fixed_profile, HVec& ne_profile,
        Geometry& grid, const feltor::Parameters& p,
        const dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag )
        {
            fixed_profile = true;
            ne_profile = dg::construct<HVec>( detail::profile(grid, p,gp,mag));
            HVec source_profile = dg::construct<HVec> ( detail::source_damping( grid, p,gp,mag));
            return source_profile;
        }
    },
    {"influx",
        []( bool& fixed_profile, HVec& ne_profile,
        Geometry& grid, const feltor::Parameters& p,
        const dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag )
        {
            fixed_profile = false;
            ne_profile = dg::construct<HVec>( detail::profile(grid, p,gp,mag));
            HVec source_profile = dg::construct<HVec> ( detail::source_damping( grid, p,gp,mag));
            return source_profile;
        }
    },
    {"torpex",
        []( bool& fixed_profile, HVec& ne_profile,
        Geometry& grid, const feltor::Parameters& p,
        const dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag )
        {
            dg::Gaussian prof( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma,
                p.sigma, p.nprofamp);
            ne_profile = dg::pullback( prof, grid);
            fixed_profile = false;
            double rhosinm = 0.98 / gp.R_0;
            double rhosinm2 = rhosinm*rhosinm;
            HVec source_profile = dg::construct<HVec> ( dg::pullback(
                detail::TorpexSource(0.98/rhosinm, -0.02/rhosinm, 0.0335/rhosinm, 0.05/rhosinm, 565*rhosinm2 ), grid) );
            return source_profile;
        }
    },
};

} //namespace feltor
