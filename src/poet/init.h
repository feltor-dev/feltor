#pragma once


namespace poet{
///The purpose of this file is to provide an interface for custom initial
///conditions and /source profiles.  Just add your own to the function
///below.
std::array<dg::x::DVec,2> initial_conditions(
    Poet<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>& f,
    const dg::x::CartesianGrid2d& grid, const poet::Parameters& p,
    dg::file::WrappedJsonValue js)
{
    auto init = js["init"];
    std::string initial = init[ "type"].asString();
    std::array<dg::x::DVec,2> y0;
    y0[0] = y0[1] =
        dg::construct<dg::x::DVec>(dg::evaluate( dg::zero, grid));
    if( initial== "zero")
    {
    }
    else if( initial== "blob")
    {
       y0[0] = dg::evaluate( dg::Gaussian( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp), grid);
       f.gamma1inv_y(y0[0],y0[1]);
       //Alternative via inversion
//        y0[1] = dg::evaluate( dg::Gaussian( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp), grid);
//        f.gamma1_y(y0[1], y0[0]);
    }

    else
        throw dg::Error( dg::Message() << "Initial condition "<<initial<<" not recognized! Is there a spelling error? I assume you do not want to continue with the wrong initial condition so I exit! Bye Bye :)");
    return y0;
};
}//namespace poet
