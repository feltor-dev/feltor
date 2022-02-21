#pragma once

#include <map>
#include <functional>
#include "json/json.h"
#include "dg/algorithm.h"
#include "dg/file/json_utilities.h"

namespace shu{

dg::CartesianGrid2d createGrid( dg::file::WrappedJsonValue grid)
{
    unsigned n = grid.get( "n", 3).asUInt();
    unsigned Nx = grid.get( "Nx", 48).asUInt();
    unsigned Ny = grid.get( "Ny", 48).asUInt();
    double x0 = grid["x"].get( 0u, 0.).asDouble();
    double x1 = grid["x"].get( 1u, 1.).asDouble();
    double y0 = grid["y"].get( 0u, 0.).asDouble();
    double y1 = grid["y"].get( 1u, 1.).asDouble();
    dg::bc bcx = dg::str2bc ( grid["bc"].get( 0u, "DIR").asString());
    dg::bc bcy = dg::str2bc ( grid["bc"].get( 1u, "PER").asString());
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
    MMSVorticity( double radius, double velocity, double ly, double time): m_s(radius), m_v(velocity), m_ly( ly), m_t(time) {}
    DG_DEVICE
    double operator()( double x, double y)const{
        return evaluate( x, y, 0) + evaluate( x,y,m_ly) + evaluate( x,y,-m_ly)
               + evaluate( x,y,2*m_ly) + evaluate( x,y,-2*m_ly);
    }
    private:
    DG_DEVICE
    double evaluate( double x, double y, double y0) const
    {
        double ad = y - y0 +m_v*m_t;
        if( fabs(ad ) > m_ly) return 0;
        return -4.*x*exp( - (x*x + ad*ad)/m_s/m_s)*
            (-2*m_s*m_s + x*x+ ad*ad)/m_s/m_s/m_s/m_s;
    }
    double m_s, m_v, m_ly, m_t;
};

struct MMSPotential{
    MMSPotential( double radius, double velocity, double ly, double time): m_s(radius), m_v(velocity), m_ly( ly), m_t(time) {}
    DG_DEVICE
    double operator()( double x, double y)const{
        return evaluate( x, y, 0) + evaluate( x,y,m_ly) + evaluate( x,y,-m_ly)
               + evaluate( x,y,2*m_ly) + evaluate( x,y,-2*m_ly);
    }
    private:
    DG_DEVICE
    double evaluate( double x, double y, double y0) const
    {
        double ad = y - y0 +m_v*m_t;
        if( fabs(ad ) > m_ly) return 0;
        return exp( - (x*x + ad*ad)/m_s/m_s) * x;
    }
    double m_s, m_v, m_ly, m_t;
};

struct MMSSource{
    MMSSource( ): m_s(1), m_v(0) {}
    MMSSource( double radius, double velocity, double ly): m_s(radius), m_v(velocity), m_ly( ly){}
    DG_DEVICE
    double operator()( double x, double y, double t)const{
        return evaluate( x, y,t, 0) + evaluate( x,y,t,m_ly) + evaluate( x,y,t,-m_ly)
               + evaluate( x,y,t,2*m_ly) + evaluate( x,y,t,-2*m_ly);
    }
    private:
    DG_DEVICE
    double evaluate( double x, double y, double t, double y0) const
    {
        double ss = m_s*m_s;
        double ad = y - y0 +m_v*t;
        if( fabs(ad ) > m_ly) return 0;
        return 8.*x/ss/ss/ss*ad*exp( - 2.*(x*x + ad*ad)/ss)*( -ss +exp( (x*x+ad*ad)/ss)*m_v*(-3.*ss + x*x + ad*ad));
    }
    double m_s, m_v, m_ly;
};

dg::HVec initial_conditions(
    dg::file::WrappedJsonValue init,
    const dg::CartesianGrid2d& grid)
{
    dg::HVec omega;
    std::string initial = init.get( "type", "lamb").asString();
    if( "lamb" == initial)
    {
        double posX = init.get("posX", 0.5).asDouble();
        double posY = init.get("posY", 0.8).asDouble();
        double R =    init.get("sigma", 0.1).asDouble();
        double U =    init.get("velocity", 1).asDouble();
        dg::Lamb lamb( posX*grid.lx(), posY*grid.ly(), R, U);
        omega = dg::evaluate ( lamb, grid);
    }
    else if( "shear" == initial)
    {
        double rho   = init.get( "rho", M_PI/15.).asDouble();
        double delta = init.get( "delta", 0.05).asDouble();
        omega = dg::evaluate ( ShearLayer( rho, delta), grid);
    }
    else if( "sine" == initial)
    {
        omega = dg::evaluate ( [](double x, double y) { return 2*sin(x)*sin(y);}, grid);
    }
    else if( "mms" == initial)
    {
        double sigma    = init.get( "sigma", 0.2).asDouble();
        double velocity = init.get( "velocity", 0.1).asDouble();
        omega = dg::evaluate ( MMSVorticity( sigma, velocity,grid.ly(), 0.), grid);
    }
    else
        throw dg::Error( dg::Message() << "Initial condition "<<initial<<" not recognized!");
    return omega;
};

}//namespace shu
