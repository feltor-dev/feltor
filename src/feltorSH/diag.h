#pragma once
// diag.h
#include "feltor.h"
#include "parameters.h"

namespace eule
{

struct Variables
{
    eule::Explicit<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>& feltor;
    eule::Implicit<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>& rolkar;
    const std::vector<dg::x::DVec>& y0;
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
    {"Telectrons", {},
        []( dg::x::DVec& result, Variables& v) {
            dg::blas1::copy( v.y0[2], result);
        }
    },
    {"Tions", {},
        []( dg::x::DVec& result, Variables& v) {
            dg::blas1::copy( v.y0[3], result);
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



} //namespace eule

