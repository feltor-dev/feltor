#pragma once

#include "dg/algorithm.h"

namespace asela
{

struct Variables{
    asela::Asela<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>& f;
    asela::Parameters p;
    std::array<dg::x::DVec,2> tmp;
};

struct Record{
    std::string name;
    std::string long_name;
    std::function<void( dg::x::DVec&, Variables&)> function;
};

std::vector<Record> diagnostics2d_list = {
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
    {"induction", "Magnetic potential", false,
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.induction(), result);
        }
    },
    {"lneperp", "Perpendicular electron diffusion",
        []( DVec& result, Variables& v ) {
            v.f.compute_diff( v.f.density(0), 0., result);
        }
    },
    /// ------------------- Energy terms ------------------------//
    {"nelnne", "Entropy electrons",
        []( DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(0), result, dg::LN<double>());
            dg::blas1::pointwiseDot( result, v.f.density(0), result);
        }
    },
    {"nilnni", "Entropy ions",
        []( DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(1), result, dg::LN<double>());
            dg::blas1::pointwiseDot( v.p.tau[1], result, v.f.density(1), 0., result);
        }
    },
    {"aperp2", "Magnetic energy",
        []( DVec& result, Variables& v ) {
            if( v.p.beta == 0)
            {
                dg::blas1::scal( result, 0.);
            }
            else
            {
                dg::blas1::pointwiseDot( 1., v.f.gradA(0)[0], v.f.gradA(0)[0],
                        1., v.f.gradA(0)[1], v.f.gradA(0)[1], 0., result);
                dg::blas1::scal( result, 1./2./v.p.beta);
            }
        }
    },
    {"ue2", "ExB energy",
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 1., v.f.gradP(0)[0], v.f.gradP(0)[0],
                    1., v.f.gradP(0)[1], v.f.gradP(0)[1], 0., result);
            dg::blas1::pointwiseDot( 0.5, v.f.density(1), result, 0., result);
        }
    },
    {"neue2", "Parallel electron energy",
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( -0.5*v.p.mu[0], v.f.density(0),
                v.f.velocity(0), v.f.velocity(0), 0., result);
        }
    },
    {"niui2", "Parallel ion energy",
        []( DVec& result, Variables& v ) {
            dg::blas1::pointwiseDot( 0.5*v.p.mu[1], v.f.density(1),
                v.f.velocity(1), v.f.velocity(1), 0., result);
        }
    },
    /// ------------------------ Energy dissipation terms ------------------//
    {"leeperp", "Perpendicular electron energy dissipation",
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::axpby( 1., v.f.density(0), -1., 1., v.tmp[0]);
            v.f.compute_diff( v.tmp[0], 0., v.tmp[0]);
            v.f.compute_diff( v.f.velocity(0), 0., v.tmp[1]);
            double mue = v.p.mu[0];
            auto RadialEnergyDiff = [mue]DG_DEVICE ( double ne, double ue, double P,
                double diffN, double diffU){
                return (1.+log(ne)-P-0.5*mue*ue*ue)*diffN + m_z*m_mu*ne*ue*diffU;
            }
            dg::blas1::evaluate( result, dg::equals(),
                RadialEnergyDiff,
                v.f.density(0), v.f.velocity(0), v.f.potential(0),
                v.tmp[0], v.tmp[1]
            );
        }
    },
    {"leiperp", "Perpendicular ion energy dissipation",
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::scal( result, -v.p.nu_perp);
            dg::blas1::axpby( 1., v.f.density(1), -1., 1., v.tmp[0]);
            v.f.compute_diff( v.tmp[0], 0., v.tmp[0]);
            v.f.compute_diff( v.f.velocity(1), 0., v.tmp[1]);
            double taui = v.p.tau[1];
            auto RadialEnergyDiff = [taui]DG_DEVICE ( double Ni, double Ui, double P,
                double diffN, double diffU){
                return (taui*(1.+log(Ni))+P-0.5*Ui*Ui)*diffN + Ni*Ui*diffU;
            }
            dg::blas1::evaluate( result, dg::equals(),
                RadialEnergyDiff,
                v.f.density(1), v.f.velocity(1), v.f.potential(1),
                v.tmp[0], v.tmp[1]
            );
        }
    },


};
}//namespace asela
