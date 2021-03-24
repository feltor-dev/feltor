#pragma once

#include "dg/algorithm.h"

namespace poet
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

}

struct Variables{
    poet::Poet<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>& f;
    poet::Parameters p;
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
    {"vorticity", "Laplace potential",
        []( dg::x::DVec& result, Variables& v ) {
            v.f.compute_lapM( -1., v.f.potential(0), 0., result);
        }
    },
    {"psi", "Ion potential psi",
        []( dg::x::DVec& result, Variables& v ) {
             dg::blas1::copy(v.f.potential(1), result);
        }
    },
    /// ------------------- Mass   terms ------------------------//
    {"lneperp", "Perpendicular electron diffusion",
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(0), v.tmp[0], dg::PLUS<double>(-1));
            v.f.compute_diff( 1., v.tmp[0], 0., result);
        }
    },
    /// ------------------- Energy terms ------------------------//
    {"nelnne", "Entropy electrons",
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(0), result, dg::LN<double>());
            dg::blas1::pointwiseDot( result, v.f.density(0), result);
        }
    },
    {"nilnni", "Entropy ions",
        []( dg::x::DVec& result, Variables& v ) {
            dg::blas1::transform( v.f.density(1), result, dg::LN<double>());
            dg::blas1::pointwiseDot( v.p.tau[1], result, v.f.density(1), 0., result);
        }
    },
//     {"ue2", "ExB energy",
//         []( dg::x::DVec& result, Variables& v ) {
//             dg::blas1::pointwiseDot( 1., v.f.gradP(0)[0], v.f.gradP(0)[0],
//                     1., v.f.gradP(0)[1], v.f.gradP(0)[1], 0., result);
//             dg::blas1::pointwiseDot( 0.5, v.f.density(1), result, 0., result);
//         }
//     },
};

std::vector<Record1d> diagnostics1d_list = {
    {"time_per_step", "Computation time per step",
        []( Variables& v ) {
            return v.duration;
        }
    },
};
}//namespace poet
