#pragma once

#include <map>
#include <functional>
#include "json/json.h"
#include "dg/algorithm.h"
#include "file/json_utilities.h"

namespace shu{

dg::CartesianGrid2d createGrid( Json::Value& js, enum file::error mode)
{
    unsigned n = file::get( mode, js, "grid", "n",3).asUInt();
    unsigned Nx = file::get( mode, js, "grid", "Nx", 48).asUInt();
    unsigned Ny = file::get( mode, js, "grid", "Ny", 48).asUInt();
    double x0 = file::get_idx( mode, js, "grid", "x", 0u, 0.).asDouble();
    double x1 = file::get_idx( mode, js, "grid", "x", 1u, 1.).asDouble();
    double y0 = file::get_idx( mode, js, "grid", "y", 0u, 0.).asDouble();
    double y1 = file::get_idx( mode, js, "grid", "y", 1u, 1.).asDouble();
    dg::bc bcx = dg::str2bc ( file::get_idx( mode, js, "grid", "bc", 0u, "DIR").asString());
    dg::bc bcy = dg::str2bc ( file::get_idx( mode, js, "grid", "bc", 1u, "PER").asString());
    return dg::CartesianGrid2d( x0, x1, y0, y1, n, Nx, Ny, bcx, bcy);
}

struct ShearLayer{

    ShearLayer( double rho, double delta): m_rho(rho), m_delta(delta){}
    DG_DEVICE
    double operator()(double x, double y) const{
    if( y<= M_PI)
        return m_delta*cos(x) - 1./m_rho/cosh( (y-M_PI/2.)/m_rho)/cosh( (y-M_PI/2.)/m_rho);
    return m_delta*cos(x) + 1./m_rho/cosh( (3.*M_PI/2.-y)/m_rho)/cosh( (3.*M_PI/2.-y)/m_rho);
    }
    private:
    double m_rho;
    double m_delta;
};

struct MMSVorticity{
    MMSVorticity( double radius, double velocity, double time): m_s(radius), m_v(velocity), m_t(time) {}
    DG_DEVICE
    double operator()( double x, double y)const{
        return -4.*x*exp( - (x*x + (m_t*m_v + y)*(m_t*m_v+y))/m_s/m_s)*
            (-2*m_s*m_s + x*x+ (m_t*m_v + y)*(m_t*m_v+y))/m_s/m_s/m_s/m_s;
    }
    private:
    double m_s, m_v, m_t;
};

struct MMSPotential{
    MMSPotential( double radius, double velocity, double time): m_s(radius), m_v(velocity), m_t(time) {}
    DG_DEVICE
    double operator()( double x, double y, double t)const{
        return exp( - (x*x + (m_t*m_v + y)*(m_t*m_v+y))/m_s/m_s) * x;
    }
    private:
    double m_s, m_v, m_t;
};
struct MMSSource{
    MMSSource( ): m_s(1), m_v(0) {}
    MMSSource( double radius, double velocity): m_s(radius), m_v(velocity) {}
    DG_DEVICE
    double operator()( double x, double y, double t)const{
        double ss = m_s*m_s;
        double ad = y+m_v*t;
        return 8.*x/ss/ss/ss*ad*exp( - 2.*(x*x + ad*ad)/ss)*( -ss +exp( (x*x+ad*ad)/ss)*m_v*(-3.*ss + x*x + ad*ad));
    }
    private:
    double m_s, m_v;
};

std::map<std::string, std::function< dg::HVec(
    Json::Value& js, enum file::error mode,
    const dg::CartesianGrid2d& grid) >
> initial_conditions =
{
    {"lamb", [](
        Json::Value& js, enum file::error mode,
        const dg::CartesianGrid2d& grid)
        {
            dg::HVec omega;
            double posX = file::get( mode, js, "init", "posX", 0.5).asDouble();
            double posY = file::get( mode, js, "init", "posY", 0.8).asDouble();
            double R = file::get( mode, js, "init", "sigma", 0.1).asDouble();
            double U = file::get( mode, js, "init", "velocity", 1).asDouble();
            dg::Lamb lamb( posX*grid.lx(), posY*grid.ly(), R, U);
            omega = dg::evaluate ( lamb, grid);
            return omega;
        }
    },
    {"shear", [](
        Json::Value& js, enum file::error mode,
        const dg::CartesianGrid2d& grid)
        {
            dg::HVec omega;
            double rho = file::get( mode, js, "init", "rho", M_PI/15.).asDouble();
            double delta = file::get( mode, js, "init", "delta", 0.05).asDouble();
            omega = dg::evaluate ( ShearLayer( rho, delta), grid);
            return omega;
        }
    },
    {"mms", [](
        Json::Value& js, enum file::error mode,
        const dg::CartesianGrid2d& grid)
        {
            dg::HVec omega;
            double sigma = file::get( mode, js, "init", "sigma", 0.2).asDouble();
            double velocity = file::get( mode, js, "init", "velocity", 0.1).asDouble();
            std::cout << "Sigma "<<sigma<<" "<<velocity<<std::endl;;
            omega = dg::evaluate ( MMSVorticity( sigma, velocity, 0.), grid);
            return omega;
        }
    }
};

}//namespace shu
