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
    RadialEnergyFlux( double tau, double mu):
        m_tau(tau), m_mu(mu){
    }

    DG_DEVICE double operator()( double ne, double ue, double A, double P,
        double d0A, double d1A, double d2A,
        double d0P, double d1P, double d2P, //Phi
        double d0S, double d1S, double d2S, //Psip
        double b_0,         double b_1,         double b_2,
        double curv0,  double curv1,  double curv2,
        double curvKappa0,  double curvKappa1,  double curvKappa2
        ){
        double curvKappaS = curvKappa0*d0S+curvKappa1*d1S+curvKappa2*d2S;
        double curvS = curv0*d0S+curv1*d1S+curv2*d2S;
        double SA = b_0*( d1S*d2A-d2S*d1A)+
                    b_1*( d2S*d0A-d0S*d2A)+
                    b_2*( d0S*d1A-d1S*d0A);
        double PS = b_0*( d1P*d2S-d2P*d1S)+
                    b_1*( d2P*d0S-d0P*d2S)+
                    b_2*( d0P*d1S-d1P*d0S);
        double JN =
            ne*ue* (A*curvKappaS + SA )
            + ne * PS
            + ne * m_mu*ue*ue*curvKappaS
            + ne * m_tau*curvS;
        double Je = (m_tau * log(ne) + 0.5*m_mu*ue*ue + P)*JN
            + m_mu*m_tau*ne*ue*ue*curvKappaS
            + m_tau*ne*ue* (A*curvKappaS + SA );
        return Je;
    }
    private:
    double m_tau, m_mu;
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
    std::array<dg::DVec, 3> dpsip;
    std::array<dg::DVec, 3> tmp;
    dg::DVec dvdpsip3d;
};

struct Record{
    std::string name;
    std::string long_name;
    std::function<void( dg::DVec&, Variables&)> function;
};


std::vector<Record> records_list = {
    {"ne", "Electron density",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(0), result);
        }
    },
    {"Ni", "Ion gyro-centre density",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(1), result);
        }
    },
    {"ue", "Electron parallel velocity",
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
    {"apar", "Magnetic potential",
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
    {"NiUi", "Product of ion gyrocentre density and velocity",
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
    {"NiUibphi", "Product of NiUi and covariant phi component of magnetic field unit vector",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 1.,
                v.f.density(1), v.f.velocity(1), v.f.bphi(), 0., result);
        }
    },
    {"Lperpinv", "Perpendicular density gradient length scale",
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
    {"perp_aligned", "Perpendicular density alignement",
        []( dg::DVec& result, Variables& v ) {
            const std::array<dg::DVec, 3>& dN = v.f.gradN(0);
            dg::tensor::multiply3d( v.f.projection(), //grad_perp
                dN[0], dN[1], dN[2], v.tmp[0], v.tmp[1], v.tmp[2]);
            dot(dN, v.tmp, result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
        }
    },
    {"Lparallelinv", "Parallel density gradient length scale",
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
    {"jvne", "Radial electron particle flux without induction contribution",
        []( dg::DVec& result, Variables& v ) {
            dg::blas1::evaluate( result, dg::equals(),
                RadialParticleFlux( v.p.tau[0], v.p.mu[0]),
                v.f.density(0), v.f.velocity(0),
                v.f.gradP(0)[0], v.f.gradP(0)[1], v.f.gradP(0)[2],
                v.dpsip[0], v.dpsip[1], v.dpsip[2],
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
                v.dpsip[0], v.dpsip[1], v.dpsip[2],
                v.f.bhatgB()[0], v.f.bhatgB()[1], v.f.bhatgB()[2],
                v.f.curvKappa()[0], v.f.curvKappa()[1], v.f.curvKappa()[2]
            );
            dg::blas1::pointwiseDot( result, v.dvdpsip3d, result);
        }
    }
};

void create_records_in_file( int ncid,
    int dim_ids[3], int dim_ids1d[2],
    std::map<std::string, int>& id0d,
    std::map<std::string, int>& id1d,
    std::map<std::string, int>& id2d)
{
    file::NC_Error_Handle err;
    for( auto record : records_list)
    {
        std::string name = record.name + "_ta2d";
        std::string long_name = record.long_name + " (Toroidal average)";
        err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids,
            &id2d[name]);//creates a new id2d entry
        err = nc_put_att_text( ncid, id2d[name], "long_name", long_name.size(),
            long_name.data());

        name = record.name + "_2d";
        long_name = record.long_name + " (Evaluated on phi = 0 plane)";
        err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids,
            &id2d[name]);
        err = nc_put_att_text( ncid, id2d[name], "long_name", long_name.size(),
            long_name.data());

        name = record.name + "_fluc2d";
        long_name = record.long_name + " (Fluctuations wrt fsa on phi = 0 plane)";
        err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids,
            &id2d[name]);
        err = nc_put_att_text( ncid, id2d[name], "long_name", long_name.size(),
            long_name.data());

        name = record.name + "_fsa2d";
        long_name = record.long_name + " (Flux surface average interpolated to 2d plane)";
        err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids,
            &id2d[name]);
        err = nc_put_att_text( ncid, id2d[name], "long_name", long_name.size(),
            long_name.data());

        name = record.name + "_fsa";
        long_name = record.long_name + " (Flux surface average)";
        err = nc_def_var( ncid, name.data(), NC_DOUBLE, 2, dim_ids1d,
            &id1d[name]);
        err = nc_put_att_text( ncid, id1d[name], "long_name", long_name.size(),
            long_name.data());

        name = record.name + "_ifs";
        long_name = record.long_name + " (Integrated Flux surface average unless it is a current then it is the derived flux surface average)";
        err = nc_def_var( ncid, name.data(), NC_DOUBLE, 2, dim_ids1d,
            &id1d[name]);
        err = nc_put_att_text( ncid, id1d[name], "long_name", long_name.size(),
            long_name.data());

        name = record.name + "_ifs_lcfs";
        long_name = record.long_name + " (Integrated Flux surface average evaluated on last closed flux surface unless it is a current then it is the fsa evaluated)";
        err = nc_def_var( ncid, name.data(), NC_DOUBLE, 1, dim_ids,
            &id0d[name]);
        err = nc_put_att_text( ncid, id0d[name], "long_name", long_name.size(),
            long_name.data());
    }
}
}//namespace feltor
