#pragma once
// #include <string>
#include "dg/enums.h"

// #include "dg/algorithm.h"
// #include "dg/file/json_utilities.h"
namespace poet{
/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, Nx, Ny;
    double dt;
    unsigned n_out, Nx_out, Ny_out;
    unsigned itstp;
    unsigned maxout;
    unsigned stages;
    unsigned maxiter_sqrt;
    unsigned maxiter_cauchy;

    std::vector<double> eps_pol;

    double eps_gamma0, eps_gamma1, eps_cauchy;
    double jfactor;
    double tau[2];
    double kappa,  nu;

    double amp, sigma, posX, posY;

    double lx, ly;
    dg::bc bc_x, bc_y;

    std::string init, equations, output, timestepper;

    Parameters( const dg::file::WrappedJsonValue& ws ) {
        n  = ws["grid"].get("n", 5).asUInt();
        Nx = ws["grid"].get("Nx", 64).asUInt();
        Ny = ws["grid"].get("Ny", 64).asUInt();
        lx = ws["grid"].get("lx", 100).asDouble();
        ly = ws["grid"].get("ly", 100).asDouble();
     
        timestepper = ws["timestepper"].get("type", "multistep").asString();
        dt = ws["timestepper"].get("dt",0.05).asDouble();

        output = ws[ "output"]["type"].asString("glfw");    
        n_out  = ws["output"].get("n",5).asUInt();
        Nx_out = ws["output"].get("Nx",64).asUInt();
        Ny_out = ws["output"].get("Ny",64).asUInt();
        itstp  = ws["output"].get("itstp",5).asUInt();
        maxout = ws["output"].get("maxout",20).asUInt();

        auto ell = ws["elliptic"];
        stages   = ell.get("stages", 3).asUInt();
        eps_pol.resize(stages);
        eps_pol[0] = ell["eps_pol"].get(0,1e-6).asDouble();
        for( unsigned i=1;i<stages; i++)
        {
            eps_pol[i] = ell[ "eps_pol"].get(i, 1.0).asDouble();
            eps_pol[i]*=eps_pol[0];
        }
        jfactor     = ell.get( "jumpfactor", 1.0).asDouble( );
         
        eps_gamma1  = ws["helmholtz"].get("eps_gamma1",1e-6).asDouble();
        eps_gamma0  = ws["helmholtz"].get("eps_gamma0",1e-6).asDouble();
        eps_cauchy = ws["helmholtz"].get("eps_cauchy",1e-10).asDouble();
        maxiter_sqrt = ws["helmholtz"].get("maxiter_sqrt",500).asUInt();
        maxiter_cauchy = ws["helmholtz"].get("maxiter_cauchy",40).asUInt();
        
        tau[0]   = -1.;
        tau[1] = ws["physical"].get("tau",1.0).asDouble();
        kappa = ws["physical"].get("curvature",0.00015).asDouble();
        equations = ws["physical"].get("equations", "ff-O2").asString();
        
        init = ws["init"].get("type", "blob").asString();
        amp = ws["init"].get("amplitude", 1.0).asDouble();
        sigma = ws["init"].get("sigma", 5.0).asDouble();
        posX = ws["init"].get("posX", 0.25).asDouble();
        posY = ws["init"].get("posY", 0.5).asDouble();        

        nu = ws["nu_perp"].asDouble();
        bc_x = dg::str2bc(ws["bc_x"].asString());
        bc_y = dg::str2bc(ws["bc_y"].asString());
    }

};
}
