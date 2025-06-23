#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <thrust/extrema.h>
#include "spectral/drt_dft.h"
// #include "spectral/drt_drt.h"

#include "dg/algorithm.h"

#include "dg/file/file.h"
#include "feltorShw/parameters.h"
int main( int argc, char* argv[])
{
    if( argc != 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc]\n";
        return -1;
    }

    ///////////////////read in and show inputfile//////////////////
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_open( argv[1], NC_NOWRITE, &ncid);
    size_t length;
    err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
    std::string input(length, 'x');
    err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
    std::cout << "input "<<input<<std::endl;
    auto js = dg::file::string2Json( input, dg::file::comments::are_forbidden);
    const eule::Parameters p(js);
    p.display(std::cout);
    
    //////////////////////////////Grids//////////////////////////////////////
    //input grid
    dg::Grid2d g2d( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);
    
    //2d field netcdf vars of input.nc
    size_t count2d[3]  = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start2d[3]  = {0, 0, 0};
    std::string names[4] = {"electrons", "ions",  "potential","vor"}; 
    int dataIDs[4];
    
    //dg stuff
    dg::HVec phi(dg::evaluate(dg::zero,g2d));
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

    
    double phisupnorm =0.;	
    double xsum=0.;
    double ysum=0.;
    double x2sum=0.;
    double xysum=0.;
    double n=0.;
    //double a=0.;
    double gamma=0.;
    double b=0.;
    double c=0.;
    unsigned bias =10; //due to initialisation
    for( unsigned i=imin; i<=imax; i++)//timestepping
    {
	  start2d[0] = i;
	  //get input.nc data
	  err = nc_inq_varid(ncid, names[2].data(), &dataIDs[2]);
	  err = nc_get_vara_double( ncid, dataIDs[2], start2d, count2d, phi.data());
	  //get max phi value
	  phisupnorm=*thrust::max_element(phi.begin(),phi.end()); 
//     std::cout << p.sigma << " " <<  gamma<< " " <<  b<< " " <<  c<<" " <<  p.lx<<" " <<  p.ly<<" " <<  p.alpha<<" " <<  p.invkappa<<" " <<  p.nu_perp<<" " <<  time<<" " << phisupnorm<<"\n";
	  //compute fit helpers
	  if (i>bias)
	  {
	      xsum+=time;                        //calculate sigma(xi)
	      ysum+=log(phisupnorm);                        //calculate sigma(yi)
	      x2sum+=pow(time,2);                //calculate sigma(x^2i)
	      xysum+=time*log(phisupnorm);                    //calculate sigma(xi*yi)
	  }
   	  //advance time
	  time += deltaT;
    }
    //compute fit vars
    n = imax-bias;
    gamma=(n*xysum-xsum*ysum)/(n*x2sum-xsum*xsum);            		//calculate slope(or the the power of exp)
    b=(x2sum*ysum-xsum*xysum)/(x2sum*n-xsum*xsum);            //calculate intercept
    c=pow(2.71828,b);
    std::cout << p.sigma << " " <<  gamma<< " " <<  b<< " " <<  c<<" " <<  p.lx<<" " <<  p.ly<<" " <<  p.alpha<<" " <<  p.invkappa<<" " <<  p.nu_perp<<"\n";

    err = nc_close(ncid);
    return 0;
}

