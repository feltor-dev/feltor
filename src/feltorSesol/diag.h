#pragma once
// diag.h
#include "feltor.h"
#include "parameters.h"

namespace eule
{

struct Variables
{
    eule::Explicit<dg::x::CartesianGrid2d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec>& feltor;
    eule::Implicit<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>& rolkar;
    const std::vector<dg::x::DVec>& y0;
    dg::x::DMatrix dy;
    const double& time;
    double accuracy;
    double dEdt;
};
std::vector<dg::file::Record<void( dg::x::DVec& result, Variables&)>> records = {
    {"electrons", {},
        []( dg::x::DVec& result, Variables& v) {
            dg::blas1::copy( v.y0[0], result);
        }
    },
    {"ions", {},
        []( dg::x::DVec& result, Variables& v) {
            dg::blas1::copy( v.y0[1], result);
        }
    },
    {"potential", {},
        []( dg::x::DVec& result, Variables& v) {
            dg::blas1::copy( v.feltor.potential()[0], result);
        }
    },
    {"vor", {},
        []( dg::x::DVec& result, Variables& v) {
            dg::blas2::gemv(v.rolkar.laplacianM(), v.feltor.potential()[0], result);
        }
    }
};
std::vector<dg::file::Record<double(Variables&)>> records0d = {
    {"energy_time", {},
        []( Variables& v) {
            return v.time;
        }
    },
    {"energy", {},
        []( Variables& v) {
            return v.feltor.energy();
        }
    },
    {"mass", {},
        []( Variables& v) {
            return v.feltor.mass();
        }
    },
    {"diffusion", {},
        []( Variables& v) {
            return v.feltor.mass_diffusion();
        }
    },
    {"Se", {},
        []( Variables& v) {
            return v.feltor.energy_vector()[0];
        }
    },
    {"Si", {},
        []( Variables& v) {
            return v.feltor.energy_vector()[1];
        }
    },
    {"Uperp", {},
        []( Variables& v) {
            return v.feltor.energy_vector()[2];
        }
    },
    {"dissipation", {},
        []( Variables& v) {
            return v.feltor.energy_diffusion();
        }
    },
    {"G_nex", {},
        []( Variables& v) {
            return v.feltor.radial_transport();
        }
    },
    {"Coupling", {},
        []( Variables& v) {
            return v.feltor.coupling();
        }
    },
    {"dEdt", {},
        [](Variables& var){
            return var.dEdt;
        }
    },
    {"accuracy", {},
        [](Variables& var){
            return var.accuracy;
        }
    }
};

std::vector<dg::file::Record<void( dg::x::DVec&, Variables&), dg::file::LongNameAttribute>> probe_list =
{
    {"electrons", "",
        [](dg::x::DVec& result, Variables& var){
            dg::blas1::copy( var.y0[0], result);
        }
    },
    {"phi", "",
        [](dg::x::DVec& result, Variables& var){
            dg::blas1::copy( var.feltor.potential()[0], result);
        }
    },
    {"phi_y", "Derivative in y direction",
        [](dg::x::DVec& result, Variables& var){
            dg::blas2::gemv( var.dy, var.feltor.potential()[0], result);
        },
    },
    {"gamma_x", "radial particle flux",
        [](dg::x::DVec& result, Variables& var){
            dg::blas2::gemv( var.dy, var.feltor.potential()[0], result);
            dg::blas1::pointwiseDot( -1., result, var.y0[0], 0., result);
        }
    }
};


} //namespace eule

