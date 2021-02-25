#pragma once
#include <string>
#include "dg/algorithm.h"
#include "json/json.h"
#include "dg/file/json_utilities.h"

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

    double eps_gamma, eps_time;
    double jfactor;
    double tau, kappa,  nu;

    double amp, sigma, posX, posY;

    double lx, ly;
    dg::bc bc_x, bc_y;

    std::string init, equations;

    Parameters(const Json::Value& js, enum dg::file::error mode = dg::file::error::is_warning ) {
        n  = js["n"].asUInt();
        Nx = js["Nx"].asUInt();
        Ny = js["Ny"].asUInt();
        dt = js["dt"].asDouble();
        n_out  = js["n_out"].asUInt();
        Nx_out = js["Nx_out"].asUInt();
        Ny_out = js["Ny_out"].asUInt();
        itstp = js["itstp"].asUInt();
        maxout = js["maxout"].asUInt();

        stages      = dg::file::get( mode, js, "stages", 3).asUInt();
        eps_pol.resize(stages);
        eps_pol[0] = dg::file::get_idx( mode, js, "eps_pol", 0, 1e-6).asDouble();
        for( unsigned i=1;i<stages; i++)
        {
            eps_pol[i] = dg::file::get_idx( mode, js, "eps_pol", i, 1).asDouble();
            eps_pol[i]*=eps_pol[0];
        }
        eps_gamma = js["eps_gamma"].asDouble();
        eps_time = js["eps_time"].asDouble();
        maxiter_sqrt = js["maxiter_sqrt"].asUInt();
        maxiter_cauchy = js["maxiter_cauchy"].asUInt();
        tau = js["tau"].asDouble();
        kappa = js["curvature"].asDouble();
        nu = js["nu_perp"].asDouble();
        amp = js["amplitude"].asDouble();
        sigma = js["sigma"].asDouble();
        posX = js["posX"].asDouble();
        posY = js["posY"].asDouble();
        lx = js["lx"].asDouble();
        ly = js["ly"].asDouble();
        bc_x = dg::str2bc(js["bc_x"].asString());
        bc_y = dg::str2bc(js["bc_y"].asString());
        init = "blob";
        equations = js.get("equations", "global").asString();
        jfactor = js.get("jfactor", 1.).asDouble();
    }

    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"    Viscosity:       = "<<nu<<"\n"
            <<"    Curvature_y:     = "<<kappa<<"\n"
            <<"    Ion-temperature: = "<<tau<<"\n";
        os << "Equation parameters are: \n"
            <<"    "<<equations<<"\n";
        os << "Boundary parameters are: \n"
            <<"    lx = "<<lx<<"\n"
            <<"    ly = "<<ly<<"\n";
        os << "Boundary conditions in x are: \n"
            <<"    "<<bc2str(bc_x)<<"\n";  //Curious! dg:: is not needed due to ADL!
        os << "Boundary conditions in y are: \n"
            <<"    "<<bc2str(bc_y)<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx<<"\n"
            <<"    Ny = "<<Ny<<"\n"
            <<"    dt = "<<dt<<"\n"
            <<"    n_out  = "<<n_out<<"\n"
            <<"    Nx_out = "<<Nx_out<<"\n"
            <<"    Ny_out = "<<Ny_out<<"\n"
            <<"    maxiter_sqrt = "<<maxiter_sqrt<<"\n"
            <<"    maxiter_cauchy = "<<maxiter_cauchy<<"\n";
        os  <<"Blob parameters are: \n"
            << "    width:        "<<sigma<<"\n"
            << "    amplitude:    "<<amp<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n";
        for( unsigned i=1; i<stages; i++)
            os <<"     Factors for Multigrid "<<i<<" "<<eps_pol[i]<<"\n";
        os  <<"scale for jump terms:    "<<jfactor<<"\n"
            <<"Stopping for Gamma CG:   "<<eps_gamma<<"\n"
            <<"Steps between output:    "<<itstp<<"\n"
            <<"Number of outputs:       "<<maxout<<std::endl; //the endl is for the implicit flush
    }
};
