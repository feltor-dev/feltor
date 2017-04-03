#ifndef _DG_PARAMETERS_ 
#define _DG_PARAMETERS_
#include <string>
#include "dg/enums.h"
#include "json/json.h"
namespace eule{
/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, Nx, Ny; 
    double dt; 
    unsigned n_out, Nx_out, Ny_out; 
    unsigned itstp, maxout;

    double eps_pol,  eps_gamma, eps_time;
    double eps_hat;

    double mu[2];
    double tau[2];
    double mcv;
    double lx,ly;
    double ln;
    double dlocal;
    double l_para;
    double nu_perp, d, c;
    
    double amp, sigma, posX, posY;
    
    double  nprofileamp, bgprofamp;
    unsigned zf,fluxmode;
    double solb;
    double omega_source,sourceb,source_dampw;
    double omega_sink,sinkb;
    double dampw;
    enum dg::bc bc_x,bc_y,bc_x_phi; 

    /**
     * @brief constructor to make a const object
     *
     * @param js json object
     */
    Parameters(const Json::Value& js)        {
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
        eps_hat = 1.;
        mu[0] = js["mu_e"].asDouble();
        mu[1] = 1.;
        tau[0] = -1.;
        tau[1] = js["tau"].asDouble();
        mcv    = js["curvature"].asDouble();
        nu_perp = js["nu_perp"].asDouble();
        d = js["D"].asDouble();
        c = js["C"].asDouble();            
        l_para=js["l_para"].asDouble();            
        amp     = js["amplitude"].asDouble();
        sigma   = js["sigma"].asDouble();
        posX    = js["posX"].asDouble();
        posY    = js["posY"].asDouble();
        nprofileamp = js["prof_amp"].asDouble();
        bgprofamp =   js["bg_prof_amp"].asDouble();
        lx =  js["lx"].asDouble();
        ly =  js["ly"].asDouble();
        bc_x     = dg::str2bc(js["bc_x"].asString());
	    bc_x_phi = dg::str2bc(js["bc_x_phi"].asString());
        bc_y     = dg::str2bc(js["bc_y"].asString());
        zf =  js["hwmode"].asUInt();
        ln =   js["ln"].asDouble();
        dlocal = (double)(lx*d/c);
        solb = js["SOL_b"].asDouble();
        omega_source = js["prof_source_rate"].asDouble();
        sourceb = js["source_b"].asDouble();
	source_dampw = js["source_damping_width"].asDouble();       
	omega_sink = js["prof_sink_rate"].asDouble();
        sinkb = js["sink_b"].asDouble();
        dampw = js["damping_width"].asDouble();       
	fluxmode =  js["fluxmode"].asUInt();

    }
    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"     mu_e              = "<<mu[0]<<"\n"
            <<"     mu_i              = "<<mu[1]<<"\n"
            <<"     mcv               = "<<mcv<<"\n"
            <<"     El.-temperature:  = "<<tau[0]<<"\n"
            <<"     Ion-temperature:  = "<<tau[1]<<"\n"
            <<"     perp. Viscosity:  = "<<nu_perp<<"\n"
            <<"     par. Resistivity: = "<<c<<"\n"
            <<"     D:                = "<<d<<"\n"
            <<"     dlocal:           = "<<dlocal<<"\n"
            <<"     L_parallel:       = "<<l_para<<"\n";
        os  <<"Blob parameters are: \n"
            << "    amplitude:    "<<amp<<"\n"
            << "    width:        "<<sigma<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n";
        os << "Profile parameters are: \n"
            <<"     density profile amplitude:    "<<nprofileamp<<"\n"
            <<"     background profile amplitude: "<<bgprofamp<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"     n  = "<<n<<"\n"
            <<"     Nx = "<<Nx<<"\n"
            <<"     Ny = "<<Ny<<"\n"
            <<"     dt = "<<dt<<"\n";
        os << "     Stopping for Polar CG:   "<<eps_pol<<"\n"
            <<"     Stopping for Gamma CG:   "<<eps_gamma<<"\n"
            <<"     Stopping for Time  CG:   "<<eps_time<<"\n";
        os << "Output parameters are: \n"
            <<"     n_out  =              "<<n_out<<"\n"
            <<"     Nx_out =              "<<Nx_out<<"\n"
            <<"     Ny_out =              "<<Ny_out<<"\n"
            <<"     Steps between output: "<<itstp<<"\n"
            <<"     Number of outputs:    "<<maxout<<"\n";
        os << "Box params: \n"
            <<"     lx  =              "<<lx<<"\n"
            <<"     ly  =              "<<ly<<"\n";
        os << "modified/ordinary \n"
            <<"     zf =              "<<zf<<"\n"
            <<"     ln =              "<<ln<<"\n";
        os << "SOL/EDGE/Source params \n"
            <<"     sol boundary =    "<<solb<<"\n"            
            <<"     source rate  =    "<<omega_source<<"\n" 
            <<"     source boundary = "<<sourceb<<"\n"
	    <<"     source damping width= "<<source_dampw<<"\n"
	    <<"     sink rate  =    "<<omega_sink<<"\n"
            <<"     sink boundary = "<<sourceb<<"\n"
	    <<"     damping width =   "<<dampw<<"\n"
	    <<"     fluxmode =   "<<fluxmode<<"\n";
        displayBC( os, bc_x, bc_y);
	    displayBC( os, bc_x_phi, bc_y);
        os << std::flush;//the endl is for the implicit flush 
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

}//namespace eule

#endif//_DG_PARAMETERS_

    

