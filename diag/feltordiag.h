#include <string>
#include <vector>
#include <functional>

#include "dg/algorithm.h"
#include "dg/geometries/geometries.h"

#include "feltor/feltor.cuh"
#include "feltor/parameters.h"
namespace feltor{
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

struct RadialParticleFlux{
    RadialParticleFlux( double tau, double mu):
        m_tau(tau), m_mu(mu){
    }
    DG_DEVICE double operator()( double ne, double ue,
        double d0P, double d1P, double d2P, //Phi
        double d0S, double d1S, double d2S, //Psip
        double b_0,         double b_1,         double b_2,
        double curv0,       double curv1,       double curv2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        double curvKappaS = curvKappa0*d0S+curvKappa1*d1S+curvKappa2*d2S;
        double curvS = curv0*d0S+curv1*d1S+curv2*d2S;
        double PS = b_0*( d1P*d2S-d2P*d1S)+
                    b_1*( d2P*d0S-d0P*d2S)+
                    b_2*( d0P*d1S-d1P*d0S);
        double JPsi =
            + ne * PS
            + ne * m_mu*ue*ue*curvKappaS
            + ne * m_tau*curvS;
        return JPsi;
    }
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
        double Je = m_z*(m_tau * log(ne) + 0.5*m_mu*ue*ue + P)*JN
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
        double Je = m_z*( m_tau * log(ne) + 0.5*m_mu*ue*ue + P )*JN
                    + m_z*m_tau*ne*ue* (A*curvKappaS + SA );
        return Je;
    }
    DG_DEVICE double operator()( double ne, double ue, double P,
        double lapMperpN, double lapMperpU){
        return m_z*(m_tau*(1+log(ne))+P+0.5*m_mu*ue*ue)*lapMperpN
                + m_z*m_mu*ne*ue*lapMperpU;
    }
    private:
    double m_tau, m_mu, m_z;
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
void jacobian(
          const std::array<Container, 3>& a,
          const std::array<Container, 3>& b,
          const std::array<Container, 3>& c,
          Container& result)
{
    dg::blas1::evaluate( result, dg::equals(), Jacobian(),
        a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
}

using Feltor = feltor::Explicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec>;

struct Variables{
    Feltor f;
    feltor::Parameters p;
    std::array<dg::DVec, 3> gradPsip;
    std::array<dg::DVec, 3> tmp;
    dg::DVec dvdpsip3d;
};

struct Record{
    std::string name;
    std::string long_name;
    std::function<void( dg::DVec&, Variables&)> function;
};


std::vector<Record> records_list = {
    {"electrons", "Electron density",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(0), result);
        }
    },
    {"ions", "Ion gyro-centre density",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(1), result);
        }
    },
    {"Ue", "Electron parallel velocity",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(0), result);
        }
    },
    {"Ui", "Ion parallel velocity",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.velocity(1), result);
        }
    },
    {"potential", "Electric potential",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.potential()[0], result);
        }
    },
    {"psi", "Ion potential psi",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.potential()[1], result);
        }
    },
    {"induction", "Magnetic potential",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.induction(), result);
        }
    },
    {"vorticity", "Minus Lap_perp of electric potential",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.lapMperpP(0), result);
        }
    },
    {"apar_vorticity", "Minus Lap_perp of magnetic potential",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.lapMperpA(), result);
        }
    },
    {"neue", "Product of electron density and velocity",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.density(0), v.f.velocity(0), result);
        }
    },
    {"niui", "Product of ion gyrocentre density and velocity",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(
                v.f.density(1), v.f.velocity(1), result);
        }
    },
    {"neuebphi", "Product of neue and covariant phi component of magnetic field unit vector",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 1.,
                v.f.density(0), v.f.velocity(0), v.f.bphi(), 0., result);
        }
    },
    {"niuibphi", "Product of NiUi and covariant phi component of magnetic field unit vector",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 1.,
                v.f.density(1), v.f.velocity(1), v.f.bphi(), 0., result);
        }
    },
    {"lperpinv", "Perpendicular density gradient length scale",
        []( dg::DVec& result, Variables& v ) {
            const std::array<dg::DVec, 3>& dN = v.f.gradN(0);
            dg::tensor::multiply3d( v.f.projection(), //grad_perp
                dN[0], dN[1], dN[2], v.tmp[0], v.tmp[1], v.tmp[2]);
            dot(dN, v.tmp, result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
            dg::blas1::transform( result, result, dg::SQRT<double>());
        }
    },
    {"perpaligned", "Perpendicular density alignement",
        []( dg::DVec& result, Variables& v ) {
            const std::array<dg::DVec, 3>& dN = v.f.gradN(0);
            dg::tensor::multiply3d( v.f.projection(), //grad_perp
                dN[0], dN[1], dN[2], v.tmp[0], v.tmp[1], v.tmp[2]);
            dot(dN, v.tmp, result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
        }
    },
    {"lparallelinv", "Parallel density gradient length scale",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot ( v.f.dsN(0), v.f.dsN(0), result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
            dg::blas1::transform( result, result, dg::SQRT<double>());
        }
    },
    {"aligned", "Parallel density alignement",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot ( v.f.dsN(0), v.f.dsN(0), result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
        }
    },
    /// ------------------ Density terms ------------------------//
    {"jvne", "Radial electron particle flux without induction contribution",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                RadialParticleFlux( v.p.tau[0], v.p.mu[0]),
                v.f.density(0), v.f.velocity(0),
                v.f.gradP(0)[0], v.f.gradP(0)[1], v.f.gradP(0)[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            dg::blas1::pointwiseDot( result, v.dvdpsip3d, result);
        }
    },
    {"jvneA", "Radial electron particle flux: induction contribution",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                RadialParticleFlux( v.p.tau[0], v.p.mu[0]),
                v.f.density(0), v.f.velocity(0), v.f.induction(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            dg::blas1::pointwiseDot( result, v.dvdpsip3d, result);
        }
    },
    {"lneperp", "Perpendicular electron diffusion",
        []( dg::DVec& result, Variables& v ) {
            v.f.compute_diffusive_lapMperpN( v.f.density(0), v.tmp[0], result);
            dg::blas1::scal( result, -v.p.nu_perp);
        }
    },
    {"lneparallel", "Parallel electron diffusion",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::copy( v.f.lapParallelN(0), result);
            dg::blas1::scal( result, v.p.nu_parallel);
        }
    },
    /// ------------------- Energy terms ------------------------//
    {"nelnne", "Entropy electrons",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(0), result, dg::LN<double>());
            dg::blas1::pointwiseDot( result, v.f.density(0), result);
        }
    },
    {"nilnni", "Entropy ions",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(1), result, dg::LN<double>());
            dg::blas1::pointwiseDot( v.p.tau[1], result, v.f.density(1), 0., result);
        }
    },
    {"aperp2", "Magnetic energy",
        []( dg::DVec& result, Variables& v ) {
            if( v.p.beta == 0)
            {
                dg::blas1::scal( result, 0.);
            }
            else
            {
                dg::tensor::multiply3d( v.f.projection(), //grad_perp
                    v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                    v.tmp[0], v.tmp[1], v.tmp[2]);
                dot( v.tmp, v.f.gradA(), result);
                dg::blas1::scal( result, 1./2./v.p.beta);
            }
        }
    },
    {"ue2", "ExB energy",
        []( dg::DVec& result, Variables& v ) {
            dg::tensor::multiply3d( v.f.projection(), //grad_perp
                v.f.gradP(0)[0], v.f.gradP(0)[1], v.f.gradP(0)[2],
                v.tmp[0], v.tmp[1], v.tmp[2]);
            dot( v.tmp, v.f.gradP(0), result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( 0.5, v.f.density(1), result, 0., result);
        }
    },
    {"neue2", "Parallel electron energy",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( -0.5*v.p.mu[0], v.f.density(0),
                v.f.velocity(0), v.f.velocity(0), 0., result);
        }
    },
    {"niui2", "Parallel ion energy",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 0.5*v.p.mu[1], v.f.density(1),
                v.f.velocity(1), v.f.velocity(1), 0., result);
        }
    },
    {"resistivity", "Energy dissipation through resistivity",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::axpby( 1., v.f.velocity(1), -1., v.f.velocity(0), result);
            dg::blas1::pointwiseDot( result, v.f.density(0), result);
            dg::blas1::pointwiseDot( v.p.eta, result, result, 0., result);
        }
    },
    /// ------------------ Energy flux terms ------------------------//
    {"jvee", "Radial electron energy flux without induction contribution",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential()[0],
                v.f.gradP(0)[0], v.f.gradP(0)[1], v.f.gradP(0)[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            dg::blas1::pointwiseDot( result, v.dvdpsip3d, result);
        }
    },
    {"jveea", "Radial electron energy flux: induction contribution",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential()[0], v.f.induction(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            dg::blas1::pointwiseDot( result, v.dvdpsip3d, result);
        }
    },
    {"jvei", "Radial ion energy flux without induction contribution",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential()[1],
                v.f.gradP(1)[0], v.f.gradP(1)[1], v.f.gradP(1)[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curv()[0], v.f.curv()[1], v.f.curv()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            dg::blas1::pointwiseDot( result, v.dvdpsip3d, result);
        }
    },
    {"jveia", "Radial ion energy flux: induction contribution",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential()[1], v.f.induction(),
                v.f.gradA()[0], v.f.gradA()[1], v.f.gradA()[2],
                v.gradPsip[0], v.gradPsip[1], v.gradPsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            dg::blas1::pointwiseDot( result, v.dvdpsip3d, result);
        }
    },
    /// ------------------------ Energy dissipation terms ------------------//
    {"leeperp", "Perpendicular electron energy dissipation",
        []( dg::DVec& result, Variables& v ) {
            v.f.compute_diffusive_lapMperpN( v.f.density(0), result, v.tmp[0]);
            v.f.compute_diffusive_lapMperpU( v.f.velocity(0), result, v.tmp[1]);
            dg::blas1::evaluate( result, dg::times_equals(),
                RadialEnergyFlux( v.p.tau[0], v.p.mu[0], 1.),
                v.f.density(0), v.f.velocity(0), v.f.potential()[0],
                v.tmp[0], v.tmp[1]
            );
            dg::blas1::scal( result, -v.p.nu_perp);
        }
    },
    {"leiperp", "Perpendicular ion energy dissipation",
        []( dg::DVec& result, Variables& v ) {
            v.f.compute_diffusive_lapMperpN( v.f.density(1), result, v.tmp[0]);
            v.f.compute_diffusive_lapMperpU( v.f.velocity(1), result, v.tmp[1]);
            dg::blas1::evaluate( result, dg::times_equals(),
                RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential()[1],
                v.tmp[0], v.tmp[1]
            );
            dg::blas1::scal( result, -v.p.nu_perp);
        }
    },
    {"leeparallel", "Parallel electron energy dissipation",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::copy(v.f.lapParallelN(0), v.tmp[0]);
            dg::blas1::copy(v.f.lapParallelU(0), v.tmp[1]);
            dg::blas1::evaluate( result, dg::times_equals(),
                RadialEnergyFlux( v.p.tau[0], v.p.mu[0], -1.),
                v.f.density(0), v.f.velocity(0), v.f.potential()[0],
                v.tmp[0], v.tmp[1]
            );
            dg::blas1::scal( result, v.p.nu_parallel);
        }
    },
    {"leiparallel", "Parallel ion energy dissipation",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::copy(v.f.lapParallelN(0), v.tmp[0]);
            dg::blas1::copy(v.f.lapParallelU(0), v.tmp[1]);
            dg::blas1::evaluate( result, dg::times_equals(),
                RadialEnergyFlux( v.p.tau[1], v.p.mu[1], 1.),
                v.f.density(1), v.f.velocity(1), v.f.potential()[1],
                v.tmp[0], v.tmp[1]
            );
            dg::blas1::scal( result, v.p.nu_parallel);
        }
    },
    /// ------------------------ Vorticity terms ---------------------------//
    {"oexbi", "ExB vorticity term with ion density",
        []( dg::DVec& result, Variables& v){
            dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density(1), 0., result);
        }
    },
    {"oexbe", "ExB vorticity term with electron density",
        []( dg::DVec& result, Variables& v){
            dot( v.f.gradP(0), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., result, v.f.binv(), v.f.binv(), 0., result);
            dg::blas1::pointwiseDot( v.p.mu[1], result, v.f.density(0), 0., result);
        }
    },
    {"odiai", "Diamagnetic vorticity term with ion density",
        []( dg::DVec& result, Variables& v){
            dot( v.f.gradN(1), v.gradPsip, result);
            dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    {"odiae", "Diamagnetic vorticity term with electron density",
        []( dg::DVec& result, Variables& v){
            dot( v.f.gradN(0), v.gradPsip, result);
            dg::blas1::scal( result, v.p.mu[1]*v.p.tau[1]);
        }
    },
    /// --------------------- Vorticity flux terms ---------------------------//
    {"jvoexbi", "ExB vorticity flux term with ion density",
        []( dg::DVec& result, Variables& v){
            // - ExB Dot GradPsi
            jacobian( v.f.bhatgB(), v.gradPsip, v.f.gradP(0), result);

            // Omega
            dot( v.f.gradP(0), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.binv(), v.f.binv(), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], v.f.density(1), 0., v.tmp[0]);

            dot( v.f.gradN(1), v.gradPsip, v.tmp[1]);
            dg::blas1::axpby( v.p.mu[1]*v.p.tau[1], v.tmp[1], 1., v.tmp[0] );

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], v.dvdpsip3d, 0., result);
        }
    },
    {"jvoexbe", "ExB vorticity flux term with electron density",
        []( dg::DVec& result, Variables& v){
            // - ExB Dot GradPsi
            jacobian( v.f.bhatgB(), v.gradPsip, v.f.gradP(0), result);

            // Omega
            dot( v.f.gradP(0), v.gradPsip, v.tmp[0]);
            dg::blas1::pointwiseDot( 1., v.tmp[0], v.f.binv(), v.f.binv(), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], v.f.density(0), 0., v.tmp[0]);

            dot( v.f.gradN(0), v.gradPsip, v.tmp[1]);
            dg::blas1::axpby( v.p.mu[1]*v.p.tau[1], v.tmp[1], 1., v.tmp[0] );

            // Multiply everything
            dg::blas1::pointwiseDot( 1., result, v.tmp[0], v.dvdpsip3d, 0., result);
        }
    },
    {"jvoapar", "A parallel vorticity flux term (Maxwell stress)",
        []( dg::DVec& result, Variables& v){
            if( v.p.beta == 0)
                dg::blas1::scal( result, 0.);
            else
            {
                // - AxB Dot GradPsi
                jacobian( v.f.bhatgB(), v.gradPsip, v.f.gradA(), result);
                dot( v.f.gradA(), v.gradPsip, v.tmp[0]);
                dg::blas1::pointwiseDot( 1./v.p.beta, result, v.tmp[0], v.dvdpsip3d, 0., result);
            }
        }
    },
    /// --------------------- Vorticity source terms ---------------------------//
    {"socurve", "Vorticity source term electron curvature",
        []( dg::DVec& result, Variables& v) {
            dot( v.f.curv(), v.gradPsip, result);
            dg::blas1::pointwiseDot( -v.p.tau[0], v.f.density(0), result, 0., result);
        }
    },
    {"socurvi", "Vorticity source term ion curvature",
        []( dg::DVec& result, Variables& v) {
            dot( v.f.curv(), v.gradPsip, result);
            dg::blas1::pointwiseDot( v.p.tau[1], v.f.density(1), result, 0., result);
        }
    },
    {"socurvkappae", "Vorticity source term electron kappa curvature",
        []( dg::DVec& result, Variables& v) {
            dot( v.f.curvKappa(), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., v.f.density(0), v.f.velocity(0), v.f.velocity(0), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( -v.p.mu[0], v.tmp[0], result, 0., result);
        }
    },
    {"socurvkappai", "Vorticity source term ion kappa curvature",
        []( dg::DVec& result, Variables& v) {
            dot( v.f.curvKappa(), v.gradPsip, result);
            dg::blas1::pointwiseDot( 1., v.f.density(1), v.f.velocity(1), v.f.velocity(1), 0., v.tmp[0]);
            dg::blas1::pointwiseDot( v.p.mu[1], v.tmp[0], result, 0., result);
        }
    },


};

}//namespace feltor
