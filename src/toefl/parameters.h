#ifndef _DG_PARAMETERS_ 
#define _DG_PARAMETERS_
#include <string>
#include "dg/enums.h"
#include "json/json.h"

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

    double eps_pol, eps_gamma, eps_time;
    double tau, kappa, nu;

    double amp, sigma, posX, posY;

    double lx, ly; 
    enum dg::bc bc_x, bc_y;

    std::string init, equations;
    bool boussinesq;

    /**
     * @brief constructor to make a const object
     *
     * @param v Vector from read_input function
     */
    Parameters( const std::vector< double>& v) {
        n  = (unsigned)v[1]; 
        Nx = (unsigned)v[2];
        Ny = (unsigned)v[3];
        dt = v[4];
        n_out = v[5];
        Nx_out = v[6];
        Ny_out = v[7];
        itstp = v[8];
        maxout = v[9];

        eps_pol = v[10];
        eps_gamma = v[11];
        eps_time = v[12];
        tau = v[13];
        kappa = v[14];
        nu = v[15];
        amp = v[16];
        sigma = v[17];
        posX = v[18];
        posY = v[19];
        lx = v[20]; 
        ly = v[21];
        bc_x = map((int)v[22]); 
        bc_y = map((int)v[23]);
        init = "blob";
        if( v[25] == 1) 
            equations = "global";
        else equations = "local";
    }
    /**
     * @brief constructor to make a const object
     *
     * @param js json object
     */
    Parameters( const Json::Value& js) {
        n  = js["n"].asUInt();
        Nx = js["Nx"].asUInt();
        Ny = js["Ny"].asUInt();
        dt = js["dt"].asDouble();
        n_out  = js["n_out"].asUInt();
        Nx_out = js["Nx_out"].asUInt();
        Ny_out = js["Ny_out"].asUInt();
        itstp = js["itstp"].asUInt();
        maxout = js["maxout"].asUInt();

        eps_pol = js["eps_pol"].asDouble();
        eps_gamma = js["eps_gamma"].asDouble();
        eps_time = js["eps_time"].asDouble();
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
        boussinesq = js.get("boussinesq", "false").asBool();
    }
    
    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"    Viscosity:       = "<<nu<<"\n"
            <<"    Curvature_y:     = "<<kappa<<"\n"
            <<"    Ion-temperature: = "<<tau<<"\n";
        os << "Equation parameters are: \n"
            <<"    "<<equations<<"\n"
            <<"    boussinesq  "<<boussinesq<<"\n";
        os << "Boundary parameters are: \n"
            <<"    lx = "<<lx<<"\n"
            <<"    ly = "<<ly<<"\n";
        displayBC( os, bc_x, bc_y);
        os << "Algorithmic parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx<<"\n"
            <<"    Ny = "<<Ny<<"\n"
            <<"    dt = "<<dt<<"\n"
            <<"    n_out  = "<<n_out<<"\n"
            <<"    Nx_out = "<<Nx_out<<"\n"
            <<"    Ny_out = "<<Ny_out<<"\n";
        os  <<"Blob parameters are: \n"
            << "    width:        "<<sigma<<"\n"
            << "    amplitude:    "<<amp<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n";
        os << "Stopping for CG:         "<<eps_pol<<"\n"
            <<"Stopping for Gamma CG:   "<<eps_gamma<<"\n"
            <<"Steps between output:    "<<itstp<<"\n"
            <<"Number of outputs:       "<<maxout<<std::endl; //the endl is for the implicit flush 
    }
    private:
    dg::bc map( int i)
    {
        switch( i)
        {
            case(0): return dg::PER;
            case(1): return dg::DIR;
            case(2): return dg::DIR_NEU;
            case(3): return dg::NEU_DIR;
            case(4): return dg::NEU;
            default: return dg::PER;
        }
    }
    void displayBC( std::ostream& os, dg::bc bcx, dg::bc bcy) const
    {
        os << "Boundary conditions in x are: \n";
        switch( bcx)
        {
            case(0): os << "    PERIODIC";
                     break;
            case(1): os << "    DIRICHLET";
                     break;
            case(2): os << "    DIR_NEU";
                     break;
            case(3): os << "    NEU_DIR";
                     break;
            case(4): os << "    NEUMANN";
                     break;
        }
        os << "\nBoundary conditions in y are: \n";
        switch( bcy)
        {
            case(0): os << "    PERIODIC";
                     break;
            case(1): os << "    DIRICHLET";
                     break;
            case(2): os << "    DIR_NEU";
                     break;
            case(3): os << "    NEU_DIR";
                     break;
            case(4): os << "    NEUMANN";
                     break;
        }
        os <<"\n";
    }
};


    

#endif//_DG_PARAMETERS_
