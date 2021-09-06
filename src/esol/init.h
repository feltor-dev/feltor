#pragma once


namespace poet{
///The purpose of this file is to provide an interface for custom initial
///conditions and /source profiles.  Just add your own to the function
///below.
    
struct ShearLayer{
    ShearLayer( double rho, double delta, double lx, double ly): m_rho(rho), m_delta(delta), m_lx(lx), m_ly(ly) {}
    DG_DEVICE
    double operator()(double x, double y) const{
    if( x<= m_lx/2.)
        return m_delta*cos(2.*M_PI*y/m_ly) - 1./m_rho/cosh( (2.*M_PI*x/m_lx-M_PI/2.)/m_rho)/cosh( (2.*M_PI*x/m_lx-M_PI/2.)/m_rho);
    return m_delta*cos(2.*M_PI*y/m_ly) + 1./m_rho/cosh( (3.*M_PI/2.-2.*M_PI*x/m_lx)/m_rho)/cosh( (3.*M_PI/2.-2.*M_PI*x/m_lx)/m_rho);
    }
    private:
    double m_rho, m_delta, m_lx, m_ly;    
};
    
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
    else if( initial== "blob_zeropol")
    {
        y0[0] = y0[1] = dg::evaluate( dg::Gaussian( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp), grid);
    }
    else if (initial =="rot_blob")
    {
        dg::Gaussian g1( (0.5-p.posX)*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp);
        dg::Gaussian g2( (0.5+p.posX)*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp);
    // 
        y0[0] = dg::evaluate( g1, grid); 
        y0[1] = dg::evaluate( g2, grid); 
        dg::blas1::axpby(1.0,y0[0],1.0,y0[1],y0[0]);
        dg::x::DVec omegaE = y0[0];
        dg::blas1::scal(omegaE,10.); //10. could be replaced by amp_omegaE parameter
        dg::x::DVec potential = dg::evaluate( dg::zero, grid);
        f.invLap_y(omegaE, potential); //phi 
        if( p.equations == "df-O2" || p.equations == "df-lwl")
        {
            f.solve_Ni_lwl(y0[0], potential, y0[1]); //if df
        }
        else 
        {
           throw dg::Error( dg::Message() << "not implemented for " << p.equations);
        }
    }
    else if (initial =="shear_flow")
    {
        ShearLayer shearlayer(M_PI/15., 0.05, p.lx, p.ly); //shear layer
        dg::x::DVec omegaE = dg::evaluate( shearlayer, grid);
        dg::x::DVec potential = dg::evaluate( dg::zero, grid);
        dg::blas1::scal(omegaE, p.amp);
        f.invLap_y(omegaE, potential); //phi 
        dg::blas1::scal(y0[0], 0.);
        if( p.equations == "df-O2" || p.equations == "df-lwl")
        {
            f.solve_Ni_lwl(y0[0], potential, y0[1]); //if df
        }
        else 
        {
           throw dg::Error( dg::Message() << "not implemented for " << p.equations);
        }
    }
    else
        throw dg::Error( dg::Message() << "Initial condition "<<initial<<" not recognized! Is there a spelling error? I assume you do not want to continue with the wrong initial condition so I exit! Bye Bye :)");
    return y0;
};
}//namespace poet
