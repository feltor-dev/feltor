#pragma once


namespace esol{
///The purpose of this file is to provide an interface for custom initial
///conditions and /source profiles.  Just add your own to the function
///below.
    
std::array<dg::x::DVec,2> initial_conditions(
    Esol<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec>& f,
    const dg::x::CartesianGrid2d& grid, const esol::Parameters& p,
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
       y0[0] = dg::evaluate( dg::Gaussian( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp) , grid);
       y0[1] = dg::evaluate( dg::PolynomialRectangle(p.lx*p.xfac_d, p.sigma_d,p.lx*(1-p.xfac_d), p.sigma_d), grid); 
       dg::blas1::pointwiseDot(y0[1],y0[0],y0[0]);
       dg::blas1::plus(y0[0],1.0);
       if(p.bgproftype == "tanh"){
           y0[1] = dg::evaluate( dg::TanhProfX(p.lx*p.xfac_sep, p.ln,-1.0, p.bgprofamp,p.profamp), grid);
       }
       else if(p.bgproftype == "exp"){
           y0[1] = dg::evaluate( dg::ExpProfX(p.profamp, p.bgprofamp, p.ln), grid);
       }
       dg::blas1::pointwiseDot(y0[1],y0[0],y0[0]);
       dg::blas1::plus(y0[0],-1.0*(p.bgprofamp + p.profamp));
       f.gamma1inv_y(y0[0], y0[1]);
    }
    else if( initial== "sin")
    {
       y0[0] = dg::evaluate( dg::SinY(p.amp, 0.0, (p.my*2*M_PI)/p.ly) , grid);
       y0[1] = dg::evaluate( dg::PolynomialRectangle(p.lx*p.xfac_d, p.sigma_d,p.lx*(1-p.xfac_d), p.sigma_d), grid); 
       dg::blas1::pointwiseDot(y0[1],y0[0],y0[0]);
       dg::blas1::plus(y0[0],1.0);
       if(p.bgproftype == "tanh"){
           y0[1] = dg::evaluate( dg::TanhProfX(p.lx*p.xfac_sep, p.ln,-1.0, p.bgprofamp,p.profamp), grid);
       }
       else if(p.bgproftype == "exp"){
           y0[1] = dg::evaluate( dg::ExpProfX(p.profamp,p.bgprofamp, p.ln), grid);
       }
       dg::blas1::pointwiseDot(y0[1],y0[0],y0[0]);
       dg::blas1::plus(y0[0],-1.0*(p.bgprofamp + p.profamp));
       f.gamma1inv_y(y0[0], y0[1]);
    }
    else if ( initial =="bath")
    { 
        y0[0] = dg::evaluate( dg::BathRZ( 32, 32, grid.x0(),grid.y0(), 30., 5., p.amp), grid);
        y0[1] = dg::evaluate( dg::PolynomialRectangle(p.lx*p.xfac_d, p.sigma_d,p.lx*(1.0-p.xfac_d), p.sigma_d), grid); 
        dg::blas1::pointwiseDot(y0[1],y0[0],y0[0]);
        dg::blas1::plus(y0[0],1.0);
        if(p.bgproftype == "tanh"){
           y0[1] = dg::evaluate( dg::TanhProfX(p.lx*p.xfac_sep, p.ln,-1.0, p.bgprofamp,p.profamp), grid);
       }
       else if(p.bgproftype == "exp"){
           y0[1] = dg::evaluate( dg::ExpProfX(p.profamp, p.bgprofamp, p.ln), grid);
       } 
        dg::blas1::pointwiseDot(y0[1],y0[0],y0[0]);
        dg::blas1::plus(y0[0],-1.0*(p.bgprofamp + p.profamp));
        f.gamma1inv_y(y0[0], y0[1]);
    }
    else
        throw dg::Error( dg::Message() << "Initial condition "<<initial<<" not recognized! Is there a spelling error? I assume you do not want to continue with the wrong initial condition so I exit! Bye Bye :)");
    return y0;
};
}//namespace esol
