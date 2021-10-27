#pragma once

#include "dg/algorithm.h"

namespace esol
{
namespace routines
{
//unfortunately we cannot use device lambdas inside a list of lambdas
struct RadialEnergyDiff
{
    RadialEnergyDiff(double tau, double z): m_tau(tau), m_z(z){}

    double DG_DEVICE operator() ( double n, double P,
        double diffN){
        return m_z*(m_tau*(1.+log(n))+P)*diffN;
    }
    private:
    double m_tau, m_z;
};
struct RadialEnergyDiffDeltaF //TODO recalculate for Boussi
{
    RadialEnergyDiffDeltaF(double tau, double z): m_tau(tau), m_z(z){}

    double DG_DEVICE operator() ( double dn, double P,
        double diffN){
        return m_z*(m_tau*dn+P)*diffN;
    }
    private:
    double m_tau, m_z;
};
}




struct Variables{
    esol::Esol<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>& f;
    esol::Parameters p;
    std::array<dg::x::DVec,2> tmp;
    double duration;
};

struct Record1d{
    std::string name;
    std::string long_name;
    std::function<double( Variables&)> function;
};

struct Record{
    std::string name;
    std::string long_name;
    std::function<void( dg::x::DVec&, Variables&)> function;
};

std::vector<Record> diagnostics2d_static_list = {
    { "xc", "x-coordinate in Cartesian coordinate system",
        []( dg::x::DVec& result, Variables& v ) {
            result = dg::evaluate( dg::cooX2d, v.f.grid());
        }
    },
    { "yc", "y-coordinate in Cartesian coordinate system",
        []( dg::x::DVec& result, Variables& v ) {
            result = dg::evaluate( dg::cooY2d, v.f.grid());
        }
    }
};

std::vector<Record> diagnostics2d_list = {
    {"electrons", "Electron density",
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(0), result);
        }
    },
    {"ions", "Ion gyro-centre density",
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(1), result);
        }
    },
    {"potential", "Electric potential",
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.potential(0), result);
        }
    },
    {"vorticity", "ExB vorticity potential",
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_vorticity( 1., v.f.potential(0), 0., result);
        }
    },
    {"lperpinv", "Perpendicular density gradient length scale", 
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(1.0, v.f.gradn(0), v.f.gradn(0), 1.0, v.f.gradn(1), v.f.gradn(1), 0.0, result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
            dg::blas1::pointwiseDivide( result, v.f.density(0), result);
            dg::blas1::transform( result, result, dg::SQRT<double>());
        }
    },    
    {"lperpinvphi", "Perpendicular electric potential gradient length scale", 
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot(1.0, v.f.gradphi(0), v.f.gradphi(0), 1.0, v.f.gradphi(1), v.f.gradphi(1), 0.0, result);
            dg::blas1::transform( result, result, dg::SQRT<double>());
        }
    },
};

std::vector<Record> restart2d_list = {
    {"restart_electrons", "Electron density",
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(0), result);
        }
    },
    {"restart_ions", "Ion gyro-centre density",
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.density(1), result);
        }
    }    
};

std::vector<Record1d> diagnostics1d_list = {
    {"time_per_step", "Computation time per step",
        []( Variables& v ) {
            return v.duration;
        }
    },
    {"mass", "volume integrated electron density",
        []( Variables& v ) {
          return dg::blas1::dot( v.f.volume(), v.f.density(0));
        }
    },
    {"potvol", "volume integrated potential",
        []( Variables& v ) {
          return dg::blas1::dot( v.f.volume(), v.f.potential(0));
        }
    },
    /// ------------------- Energy terms ------------------------//
    {"ene", "Entropy electrons", //nelnne or (delta ne)^2
        []( Variables& v ) {
            if (v.p.equations == "ff-lwl" || v.p.equations == "ff-O2" ) {
                dg::blas1::transform( v.f.density(0), v.tmp[1], dg::LN<double>());
                dg::blas1::pointwiseDot( v.tmp[1], v.f.density(0), v.tmp[1]);
            }
            {
                dg::blas1::transform( v.f.density(0), v.tmp[1], dg::PLUS<double>(-1.0*(v.p.bgprofamp + v.p.profamp)));
                dg::blas1::pointwiseDot( 0.5, v.tmp[1], v.tmp[1], 0., v.tmp[1]);
            }
            return dg::blas1::dot( v.f.volume(), v.tmp[1]);
        }
    },
    {"eni", "Entropy ions", //nilnni or (delta ni)^2
        [](Variables& v ) {
            if (v.p.equations == "ff-lwl" || v.p.equations == "ff-O2" ) {
                dg::blas1::transform( v.f.density(1), v.tmp[1], dg::LN<double>());
                dg::blas1::pointwiseDot( v.p.tau[1], v.tmp[1], v.f.density(1), 0., v.tmp[1]);
            }
            else
            {
                dg::blas1::transform( v.f.density(1), v.tmp[1], dg::PLUS<double>(-1.0*(v.p.bgprofamp + v.p.profamp)));
                dg::blas1::pointwiseDot( 0.5*v.p.tau[1], v.tmp[1], v.tmp[1], 0., v.tmp[1]);
            }
            return dg::blas1::dot( v.f.volume(), v.tmp[1]);
        }
    },
    {"eexb", "ExB energy",
        []( Variables& v ) {
            if (v.p.equations == "ff-lwl" || v.p.equations == "ff-O2") {
                dg::blas1::pointwiseDot( -1.0, v.f.density(1), v.f.psi2(), 0., v.tmp[1]);
            }
            else {
                dg::blas1::axpby( -1.0, v.f.psi2(), 0., v.tmp[1]);
            }
            return dg::blas1::dot( v.f.volume(), v.tmp[1]);
        }
    },
 /// ------------------------ Energy dissipation terms ------------------//
    {"leeperp", "Perpendicular electron energy dissipation",
        [](  Variables& v ) {
            dg::blas1::axpby( 1., v.f.density(0), -1., 1., v.tmp[0]);
            v.f.compute_diff( 1., v.tmp[0], 0., v.tmp[0]);
            if (v.p.equations == "ff-lwl" || v.p.equations == "ff-O2" ) {
                dg::blas1::evaluate( v.tmp[1], dg::equals(),
                    routines::RadialEnergyDiff( v.p.tau[0], -1),
                    v.f.density(0),  v.f.potential(0), v.tmp[0]
                );
            }
            else { //TODO recalculate
                dg::blas1::evaluate( v.tmp[1], dg::equals(),
                    routines::RadialEnergyDiffDeltaF( v.p.tau[0], -1),
                    v.f.density(0),  v.f.potential(0), v.tmp[0]
                );
            }
            return dg::blas1::dot( v.f.volume(), v.tmp[1]);
        }
    },
    {"leiperp", "Perpendicular ion energy dissipation",
        [](  Variables& v ) {
            dg::blas1::axpby( 1., v.f.density(1), -1., 1., v.tmp[0]);
            v.f.compute_diff( 1., v.tmp[0], 0., v.tmp[0]);
            if (v.p.equations == "ff-lwl" || v.p.equations == "ff-O2" ) {
                dg::blas1::evaluate( v.tmp[1], dg::equals(),
                    routines::RadialEnergyDiff(  v.p.tau[1], 1),
                    v.f.density(1),  v.f.potential(1), v.tmp[0]
                );
            }
            else { //TODO recalculate for Boussi
                 dg::blas1::evaluate( v.tmp[1], dg::equals(),
                    routines::RadialEnergyDiffDeltaF(  v.p.tau[1], 1),
                    v.f.density(1),  v.f.potential(1), v.tmp[0]
                );
            }
            return dg::blas1::dot( v.f.volume(), v.tmp[1]);
        }
    }
};
}//namespace esol
