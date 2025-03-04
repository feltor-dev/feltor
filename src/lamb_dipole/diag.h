#pragma once

#include "dg/algorithm.h"
#include "dg/file/file.h"
#include "shu.h"

namespace shu{

struct Variables{
    shu::Shu<dg::CartesianGrid2d, dg::DMatrix, dg::DVec>& shu;
    const dg::CartesianGrid2d& grid;
    const dg::DVec& y0;
    const double& time;
    const dg::DVec& weights;
    double duration;
    dg::file::WrappedJsonValue& js;
};

std::vector<dg::file::Record<void(dg::DVec&, Variables&), dg::file::LongNameAttribute>> diagnostics2d_list = {
    {"vorticity", "Vorticity in 2d",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.y0, result);
        }
    },
    {"potential", "stream function",
        []( dg::DVec& result, Variables& v ) {
             dg::blas1::copy(v.shu.potential(), result);
        }
    },
};

std::vector<dg::file::Record<void(dg::HVec&, Variables&), dg::file::LongNameAttribute>> diagnostics2d_static_list = {
    { "weights", "integration weights in Cartesian coordinate system",
        []( dg::HVec& result, Variables& v ) {
            result = dg::create::weights( v.grid);
        }
    },
    { "xc", "x-coordinate in Cartesian coordinate system",
        []( dg::HVec& result, Variables& v ) {
            result = dg::evaluate( dg::cooX2d, v.grid);
        }
    },
    { "yc", "y-coordinate in Cartesian coordinate system",
        []( dg::HVec& result, Variables& v ) {
            result = dg::evaluate( dg::cooY2d, v.grid);
        }
    }
};

std::vector<dg::file::Record<double(Variables&), dg::file::LongNameAttribute>> diagnostics1d_list = {
    {"vorticity_1d", "Integrated Vorticity",
        []( Variables& v ) {
            return dg::blas1::dot(v.y0, v.weights);
        }
    },
    {"enstrophy_1d", "Integrated enstrophy",
        []( Variables& v ) {
            return 0.5*dg::blas2::dot( v.y0, v.weights, v.y0);
        }
    },
    {"energy_1d", "Integrated energy",
        []( Variables& v ) {
            return 0.5*dg::blas2::dot( v.y0, v.weights, v.shu.potential()) ;
        }
    },
    {"time_per_step", "Computation time between outputs",
        []( Variables& v ) {
            return v.duration;
        }
    },
    {"time", "",
        []( Variables& v) {
            return v.time;
        }
    },
    {"error", "Relative error to analytical solution (not available for every intitial condition)",
        []( Variables& v ) {
            std::string initial = v.js[ "init"].get( "type", "lamb").asString();
            if( "mms" == initial)
            {
                double R = v.js[ "init"].get( "sigma", 0.1).asDouble();
                double U = v.js[ "init"].get( "velocity", 1).asDouble();
                shu::MMSVorticity vortex( R, U, v.grid.ly(), v.time);
                dg::DVec sol = dg::evaluate( vortex, v.grid);
                dg::blas1::axpby( 1., v.y0, -1., sol);
                return sqrt( dg::blas2::dot( sol, v.weights, sol)/dg::blas2::dot( v.y0 , v.weights, v.y0));
            }
            else if( "sine" == initial)
            {
                double nu = 0.;
                unsigned order = 1;
                std::string regularization = v.js[ "regularization"].get( "type", "moddal").asString();
                if( "viscosity" == regularization)
                {
                    nu = v.js[ "regularization"].get( "nu_perp", 1e-3).asDouble();
                    order = v.js[ "regularization"].get( "order", 1).asUInt();
                }
                double time = v.time;
                dg::DVec sol = dg::evaluate( [time,nu,order](double x, double y) {
                    return 2.*sin(x)*sin(y)*exp( -pow(2.*nu,order)*time); },
                    v.grid);
                dg::blas1::axpby( 1., v.y0, -1., sol);
                return sqrt( dg::blas2::dot( sol, v.weights, sol)/
                             dg::blas2::dot( v.y0 , v.weights, v.y0));
            }
            return 0.;
        }
    }
};

}//namespace shu
