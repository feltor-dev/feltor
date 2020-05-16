#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <complex>

#include "dg/algorithm.h"
#include "dg/poisson.h"

#include "dg/functors.h"

#include "dg/file/file.h"
#include "reco2D/parameters.h"
int main( int argc, char* argv[])
{
    if( argc != 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc]\n";
        return -1;
    }

    ///////////////////read in and show inputfile//////////////////
    file::NC_Error_Handle err;
    int ncid;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    size_t length;
    err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
    std::string input(length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
    std::cout << "input "<<input<<std::endl;
    Json::Value js;
    file::string2Json( input, js, "strict");
    const asela::Parameters p(js);

    //////////////////////////////Grids//////////////////////////////////////
    //input grid
    dg::Grid2d g2d( -p.lxhalf, p.lxhalf, -p.lyhalf, p.lyhalf , p.n, p.Nx, p.Ny, dg::DIR, dg::PER);
    
    //2d field netcdf vars of input.nc
    size_t count2d[3]  = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start2d[3]  = {0, 0, 0};
    std::string names[8] = {"electrons", "ions", "Ue", "Ui", "potential","Aparallel","Vor","Jparallel"}; 
    int dataIDs[8];
    
    //dg stuff
    dg::HVec tempH(dg::evaluate(dg::one,g2d));
    dg::DVec one(dg::evaluate(dg::one,g2d));
    dg::DVec w2d = dg::create::weights( g2d);
    dg::DVec apar(dg::evaluate(dg::zero,g2d));
    dg::InvCoshXsq init0( p.amp0, 2.*M_PI/p.lxhalf);
    dg::DVec apareq(dg::evaluate( init0, g2d));

    //open netcdf files
    //////////////////////////////open nc file//////////////////////////////////
    //set min and max timesteps
    double time = 0.;
    unsigned imin,imax;    
    imin=0; //set min time    
    //get max time
    err = nc_inq_varid(ncid, names[0].data(), &dataIDs[0]); 
    size_t steps;
    err = nc_inq_dimlen(ncid, dataIDs[0], &steps);
    steps-=1;
    imax = steps/p.itstp;
    double deltaT = p.dt*p.itstp;     //define timestep
    double apar0=0.,gamma=0.,psiX=0.,logpsiX=0.,logpsiXold=0.;

    for( unsigned i=imin; i<=imax; i++)//timestepping
    {
        
        start2d[0] = i;
        //get input.nc data
        err = nc_inq_varid(ncid, names[5].data(), &dataIDs[5]);
        err = nc_get_vara_double( ncid, dataIDs[5], start2d, count2d, tempH.data());
        apar=tempH;
        dg::HVec x0(1,0.);
        dg::HVec y0(1,0.);
        dg::DVec aparx0y0(1,0.);
        dg::DVec apareqx0y0(1,0.);

        dg::IDMatrix interp = dg::create::interpolation( x0, y0, g2d, dg::NEU);
        dg::blas2::gemv(interp, apar, aparx0y0);
            
        
//         apar0=aparx0y0[0];
        dg::blas2::gemv(interp, apareq, apareqx0y0);
            
//            std::cout << time << " " <<  aparx0y0[0]<< " " << apareqx0y0[0] <<"\n";

        psiX=std::fabs(aparx0y0[0]-apareqx0y0[0]);
        
//         if (i>=1) {
        logpsiX=log(psiX);
        if (i==imin) logpsiXold=logpsiX;
        gamma=(logpsiX-logpsiXold)/deltaT;
//         }
        logpsiXold=logpsiX;
        //write norm data
        std::cout << time << " " <<  psiX<< " " << gamma <<"\n";

        time += deltaT;
    }


    err = nc_close(ncid);
    return 0;
}


