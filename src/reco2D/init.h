#pragma once


namespace asela{
///The purpose of this file is to provide an interface for custom initial
///conditions and /source profiles.  Just add your own to the function
///below.
std::array<std::array<dg::x::DVec,2>,2> initial_conditions(
    Asela<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>&,
    const dg::x::CartesianGrid2d& grid, const asela::Parameters& p,
    dg::file::WrappedJsonValue js)
{
    auto init = js["init"];
    std::string initial = init[ "type"].asString();
    std::array<std::array<dg::x::DVec,2>,2> y0;
    y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] =
        dg::construct<dg::x::DVec>(dg::evaluate( dg::zero, grid));
    if( initial== "zero")
    {
    }
    else if( initial== "harris")
    {
        double A = init[ "amplitude0"].asDouble( 1e-5);
        double B = init[ "amplitude1"].asDouble( 1e-5);
        double mY  = init["my"].asDouble(1e-5);
        double kx = 2.*M_PI/p.lxhalf, ky = mY*M_PI/p.lyhalf;
        double kxp = M_PI/p.lxhalf/2.;
        dg::x::DVec apar = dg::evaluate( [=](double x, double y){ return
            cos(kxp*x)*( A/cosh(kx*x)/cosh(kx*x) + B*cos( ky*y))*p.beta;
            }, grid);
        //Analytical Laplace from Mathematica
        y0[1][0] = dg::evaluate( [=](double x, double y){ return
                -(B*(pow(kxp,2) + pow(ky,2))*cos(kxp*x)*cos(ky*y)) - 2*A*pow(kx,2)*cos(kxp*x)*pow(1./cosh(kx*x),4) +
A*pow(1./cosh(kx*x),2)*(-(pow(kxp,2)*cos(kxp*x)) + 4*kx*kxp*sin(kxp*x)*tanh(kx*x) + 4*pow(kx,2)*cos(kxp*x)*pow(tanh(kx*x),2));
           }, grid);
        dg::blas1::axpby(1., y0[1][0], 1./p.mu[0], apar, y0[1][0]);
        // if we include Gamma A again we must take care here
        dg::blas1::axpby(1./p.mu[1], apar, 0.0, y0[1][1]);
    }
    else if( initial== "island")
    {
        double A = init[ "amplitude0"].asDouble( 1e-5);
        double B = init[ "amplitude1"].asDouble( 1e-5);
        double mY  = init["my"].asDouble(1e-5);
        double kx = 2.*M_PI/p.lxhalf, ky = mY*M_PI/p.lyhalf;
        double kxp = M_PI/p.lxhalf/2., e = 0.2;
        dg::x::DVec apar = dg::evaluate( [=](double x, double y){ return
            cos(kxp*x)*( A*log( cosh(kx*x) + e*cos( kx*y) )/kx + B*cos( ky*y))*p.beta;
            }, grid);
        //Analytical Laplace from Mathematica
        y0[1][0] = dg::evaluate( [=](double x, double y){ return
                cos(kxp*x)*(-(B*(pow(kxp,2) + pow(ky,2))*cos(ky*y)) - (A*(-1 + pow(e,2))*kx)/pow(e*cos(kx*y) + cosh(kx*x),2) -
  (A*pow(kxp,2)*log(e*cos(kx*y) + cosh(kx*x)))/kx) - (2*A*kxp*sin(kxp*x)*sinh(kx*x))/(e*cos(kx*y) + cosh(kx*x));
           }, grid);
        dg::blas1::axpby(1., y0[1][0], 1./p.mu[0], apar, y0[1][0]);
        // if we include Gamma A again we must take care here
        dg::blas1::axpby(1./p.mu[1], apar, 0.0, y0[1][1]);
    }
    else
        throw dg::Error( dg::Message() << "Initial condition "<<initial<<" not recognized! Is there a spelling error? I assume you do not want to continue with the wrong initial condition so I exit! Bye Bye :)");
    return y0;
};
}//namespace asela
