#pragma once

#include "dg/algorithm.h"
#include "thrust/extrema.h"

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
struct RadialEnergyDiffDeltaF
{
    RadialEnergyDiffDeltaF(double tau, double z): m_tau(tau), m_z(z){}

    double DG_DEVICE operator() ( double dn, double P,
        double diffN){
        return m_z*(m_tau*dn+P)*diffN;
    }
    private:
    double m_tau, m_z;
};
struct Heaviside2d
{
    Heaviside2d( double sigma):m_sigma2(sigma*sigma), m_x0(0), m_y0(0){}
    void set_origin( double x0, double y0){ m_x0=x0, m_y0=y0;}
    double operator()(double x, double y)const
    {
        double r2 = (x-m_x0)*(x-m_x0)+(y-m_y0)*(y-m_y0);
        if( r2 >= m_sigma2)
            return 0.;
        return 1.;
    }
  private:
    const double m_sigma2;
    double m_x0,m_y0;
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
    {"mass", "volume integrated electron density minus ref density",
        []( Variables& v ) {
          dg::blas1::transform( v.f.density(0), v.tmp[0], dg::PLUS<double>(-1));
          return dg::blas1::dot( v.f.volume(), v.tmp[0]);
        }
    },
    {"masstimesxcom", "x-component of center of mass position",
        []( Variables& v ) {
          dg::blas1::transform( v.f.density(0), v.tmp[0], dg::PLUS<double>(-1));
          v.tmp[1] = dg::evaluate( dg::cooX2d, v.f.grid());
          dg::blas1::pointwiseDot(v.tmp[1],v.tmp[0],v.tmp[1]);
          return dg::blas1::dot( v.f.volume(), v.tmp[1]);
        }
    },
    {"masstimesycom", "y-component of center of mass position",
        []( Variables& v ) {
          dg::blas1::transform( v.f.density(0), v.tmp[0], dg::PLUS<double>(-1));
          v.tmp[1] = dg::evaluate( dg::cooY2d, v.f.grid());
          dg::blas1::pointwiseDot(v.tmp[1],v.tmp[0],v.tmp[1]);
          return dg::blas1::dot( v.f.volume(), v.tmp[1]);
        }
    },
    {"ic", "not normalized blob compactness",
        []( Variables& v ) {
          dg::blas1::transform( v.f.density(0), v.tmp[0], dg::PLUS<double>(-1));
          routines::Heaviside2d heavi(v.p.sigma);
#ifdef WITH_MPI
          unsigned position = thrust::distance( v.tmp[0].data().begin(), thrust::max_element( v.tmp[0].data().begin(), v.tmp[0].data().end()) );
          v.tmp[1] = dg::evaluate( dg::cooX2d, v.f.grid());
          double X_max = v.tmp[1].data()[position] ;
          v.tmp[1] = dg::evaluate( dg::cooY2d, v.f.grid());
          double Y_max = v.tmp[1].data()[position] ;
#else
          unsigned position = thrust::distance( v.tmp[0].begin(), thrust::max_element( v.tmp[0].begin(), v.tmp[0].end()) );
          v.tmp[1] = dg::evaluate( dg::cooX2d, v.f.grid());
          double X_max = v.tmp[1][position];
          v.tmp[1] = dg::evaluate( dg::cooY2d, v.f.grid());
          double Y_max = v.tmp[1][position];
#endif
          heavi.set_origin( X_max, Y_max );
          v.tmp[1] = dg::evaluate( heavi, v.f.grid());
          //Compute m_x0max
          dg::blas1::pointwiseDot(v.tmp[1],v.tmp[0],v.tmp[1]);
          return dg::blas1::dot( v.f.volume(), v.tmp[1]);
        }
    },   
    /// ------------------- Mass   terms ------------------------//
    {"lneperp", "Perpendicular electron diffusion",
        [](  Variables& v ) {
            dg::blas1::transform( v.f.density(0), v.tmp[0], dg::PLUS<double>(-1));
            v.f.compute_diff( 1., v.tmp[0], 0., v.tmp[1]);
            return dg::blas1::dot( v.f.volume(), v.tmp[1]);
        }
    },
    /// ------------------- Energy terms ------------------------//
    {"ene", "Entropy electrons", //nelnne or (delta ne)^2
        []( Variables& v ) {
            if (v.p.equations == "ff-lwl" || v.p.equations == "ff-O2" || v.p.equations == "ff-O4") {
                dg::blas1::transform( v.f.density(0), v.tmp[1], dg::LN<double>());
                dg::blas1::pointwiseDot( v.tmp[1], v.f.density(0), v.tmp[1]);
            }
            {
                dg::blas1::transform( v.f.density(0), v.tmp[1], dg::PLUS<double>(-1.));
                dg::blas1::pointwiseDot( 0.5, v.tmp[1], v.tmp[1], 0., v.tmp[1]);
            }
            return dg::blas1::dot( v.f.volume(), v.tmp[1]);
        }
    },
    {"eni", "Entropy ions", //nilnni or (delta ni)^2
        [](Variables& v ) {
            if (v.p.equations == "ff-lwl" || v.p.equations == "ff-O2" || v.p.equations == "ff-O4") {
                dg::blas1::transform( v.f.density(1), v.tmp[1], dg::LN<double>());
                dg::blas1::pointwiseDot( v.p.tau[1], v.tmp[1], v.f.density(1), 0., v.tmp[1]);
            }
            else
            {
                dg::blas1::transform( v.f.density(1), v.tmp[1], dg::PLUS<double>(-1.));
                dg::blas1::pointwiseDot( 0.5*v.p.tau[1], v.tmp[1], v.tmp[1], 0., v.tmp[1]);
            }
            return dg::blas1::dot( v.f.volume(), v.tmp[1]);
        }
    },
    {"eexb", "ExB energy",
        []( Variables& v ) {
            if (v.p.equations == "ff-lwl" || v.p.equations == "ff-O2" || v.p.equations == "ff-O4") {
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
            if (v.p.equations == "ff-lwl" || v.p.equations == "ff-O2" || v.p.equations == "ff-O4") {
                dg::blas1::evaluate( v.tmp[1], dg::equals(),
                    routines::RadialEnergyDiff( v.p.tau[0], -1),
                    v.f.density(0),  v.f.potential(0), v.tmp[0]
                );
            }
            else {
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
            if (v.p.equations == "ff-lwl" || v.p.equations == "ff-O2" || v.p.equations == "ff-O4") {
                dg::blas1::evaluate( v.tmp[1], dg::equals(),
                    routines::RadialEnergyDiff(  v.p.tau[1], 1),
                    v.f.density(1),  v.f.potential(1), v.tmp[0]
                );
            }
            else {
                 dg::blas1::evaluate( v.tmp[1], dg::equals(),
                    routines::RadialEnergyDiffDeltaF(  v.p.tau[1], 1),
                    v.f.density(1),  v.f.potential(1), v.tmp[0]
                );
            }
            return dg::blas1::dot( v.f.volume(), v.tmp[1]);
        }
    }
};
}//namespace poet
