#pragma once
#include "dg/enums.h"

namespace heat{
struct Parameters
{
    unsigned n, Nx, Ny, Nz;
    unsigned n_out, Nx_out, Ny_out, Nz_out;
    unsigned cx, cy;
    double dt;
    unsigned itstp, maxout;
    unsigned mx, my;
    std::string interpolation_method;
    std::string p_diff;
    double nu_parallel, nu_perp;

    double amp, sigma, posX, posY, sigma_z;

    enum dg::bc bcx, bcy;
    double boxscaleRp,boxscaleRm,boxscaleZp,boxscaleZm;
    double eps_time, rtol, rk4eps;
    Parameters( dg::file::WrappedJsonValue js) {
        n  = js["n"].asUInt();
        Nx = js["Nx"].asUInt();
        Ny = js["Ny"].asUInt();
        Nz = js["Nz"].asUInt();
        dt = js["dt"].asDouble();
        cx = js.get("cx", 1).asUInt();
        cy = js.get("cy", 1).asUInt();
        n_out = n, Nx_out = Nx/cx, Ny_out = Ny/cy, Nz_out = Nz;

        itstp = js["itstp"].asUInt();
        maxout = js["maxout"].asUInt();
        mx = js["mx"].asUInt();
        my = js["my"].asUInt();
        interpolation_method = js["interpolation-method"].asString();
        rk4eps = js.get("rk4eps", 1e-5).asDouble();

        nu_parallel = js["nu_parallel"].asDouble();
        nu_perp = js["nu_perp"].asDouble();
        amp = js["amplitude"].asDouble();
        sigma = js["sigma"].asDouble();
        posX = js["posX"].asDouble();
        posY = js["posY"].asDouble();
        sigma_z = js["sigma_z"].asDouble();

        eps_time = js["eps_time"].asDouble();
        rtol = js["rtol"].asDouble();
        bcx = dg::str2bc(js["bcx"].asString());
        bcy = dg::str2bc(js["bcy"].asString());
        boxscaleRp = js["boxscaleRp"].asDouble();
        boxscaleRm = js["boxscaleRm"].asDouble();
        boxscaleZp = js["boxscaleZp"].asDouble();
        boxscaleZm = js["boxscaleZm"].asDouble();
        p_diff      =js.get("diff","non-adjoint").asString();
    }

    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"     par. Viscosity:   = "<<nu_parallel<<"\n"
            <<"     perp.Viscosity:   = "<<nu_perp<<"\n";
        os  <<"Blob parameters are: \n"
            << "    amplitude:    "<<amp<<"\n"
            << "    width:        "<<sigma<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n"
            << "    sigma_z:      "<<sigma_z<<"\n";
        os << "Box parameters are: \n"
            <<"     boxscale R+:                  "<<boxscaleRp<<"\n"
            <<"     boxscale R-:                  "<<boxscaleRm<<"\n"
            <<"     boxscale Z+:                  "<<boxscaleZp<<"\n"
            <<"     boxscale Z-:                  "<<boxscaleZm<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"     n  = "<<n<<"\n"
            <<"     Nx = "<<Nx<<"\n"
            <<"     Ny = "<<Ny<<"\n"
            <<"     Nz = "<<Nz<<"\n"
            <<"     dt = "<<dt<<"\n";
        os << "Output parameters are: \n"
            <<"     n_out  =              "<<n_out<<"\n"
            <<"     Nx_out =              "<<Nx_out<<"\n"
            <<"     Ny_out =              "<<Ny_out<<"\n"
            <<"     Nz_out =              "<<Nz_out<<"\n"
            <<"     Steps between output: "<<itstp<<"\n"
            <<"     Number of outputs:    "<<maxout<<"\n";
        os << "Operator parameters are: \n"
            <<"     mx =                  "<<mx<<"\n"
            <<"     my =                  "<<my<<"\n"
            <<"     interpolation-method ="<<interpolation_method<<"\n"
            <<"     p_diff =              "<<p_diff<<"\n";
        os << "Boundary condition is: \n"
            <<"     BC X       =              "<<bc2str(bcx)<<"\n"
            <<"     BC Y       =              "<<bc2str(bcy)<<"\n";
        os << "PCG epsilon for time stepper: \n"
            <<"     eps_time  =              "<<eps_time<<"\n";
        os << std::flush;//the endl is for the implicit flush
    }
};

}//namespace eule

