#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <iterator>
#include "dg/algorithm.h"

#include "dg/file/file.h"
#include "parameters.h"

int main( int argc, char* argv[])
{
    if( argc == 1)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input1.nc] [input2.nc] ... \n";
        return -1;
    }
    //nc defs
    dg::file::NC_Error_Handle err;
    int ncid;
    int dataIDs[14];
    std::string names[14] = {"Rx","Guynx","Tnx","A","Rfn","difffauy1","difffauy2","Sfauy",
      "dxfaux","fauxik","difflnn","Slnn","dtfauy","ln(ne)avg"}; 
    //input nc files
    for( int i=1; i< argc; i++)
    {
        err = nc_open( argv[i], NC_NOWRITE, &ncid);
        ///////////////////read in and show inputfile und geomfile//////////////////
        size_t length;
        err = nc_inq_attlen( ncid, NC_GLOBAL,   "inputfile", &length);
        std::string input( length, 'x');
        err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
        
        auto js = dg::file::string2Json( input, dg::file::comments::are_forbidden);
        const eule::Parameters p(js);
        
        dg::Grid1d g1d( 0., p.lx,p.n_out, p.Nx_out, p.bc_x);
	size_t count1d[2]  = {1, g1d.n()*g1d.N()};
	size_t start1d[2]  = {0, 0};    
	dg::HVec temp1d(dg::evaluate(dg::zero,g1d));
	dg::HVec temp1d_int(dg::evaluate(dg::zero,g1d));
	dg::HVec temp1d_old(dg::evaluate(dg::zero,g1d));
	dg::HVec w1d = dg::create::weights( g1d);

	double deltaT = p.itstp*p.dt;
	double fac= 0.5*deltaT;
	
	size_t start0d  = 0;    
        //get maxtime of input file
        int timeID;
	size_t numOut;
        err = nc_inq_dimid(ncid, "time", &timeID);        
        err = nc_inq_dimlen(ncid, timeID, &numOut);

	std::vector<double> vt(numOut);
	
	
	err = nc_get_vara_double( ncid, timeID,     &start0d, &numOut, vt.data());
        //Timestepping
	double timepointexact_min=0.*p.invkappa; //in units omega_ci 
	double timepointexact_max=300.*p.invkappa; //in units omega_ci 

	std::vector<double>::iterator timepoint_min,timepoint_max;
	timepoint_min=std::lower_bound (vt.begin(), vt.end(), timepointexact_min);
	timepoint_max=std::lower_bound (vt.begin(), vt.end(), timepointexact_max);
	unsigned timepos_min = std::distance( vt.begin(),timepoint_min);
	unsigned timepos_max = std::distance( vt.begin(),timepoint_max);
    
	if (timepos_max > vt.size()) {
	    timepos_max = vt.size();
	}
    
	std::cout << p.alpha << " " << p.invkappa<< " " ;
	//read and write data
	
	for( unsigned m=0; m<12; m++) 
	{
	for (unsigned n=timepos_min; n<timepos_max; n++)
	{
	    start1d[0] = n;

	    //read 1d profiles
	    err = nc_inq_varid(ncid, names[m].data(), &dataIDs[m]);
	    err = nc_get_vara_double(ncid, dataIDs[m], start1d, count1d, temp1d.data());
	    
	    //integrate with trapez rule
	    if (n==timepos_min) dg::blas1::axpby(fac,temp1d,0.0,temp1d_int); // Int = temp
	    if (n>timepos_min && n<timepos_max-1) dg::blas1::axpby(2.0*fac,temp1d,1.0,temp1d_int);  
	    if (n==timepos_max)  dg::blas1::axpby(fac,temp1d,1.0,temp1d_int);  
	    
	}
	//Compute norm
	std::cout << sqrt(dg::blas2::dot(temp1d_int,w1d,temp1d_int))/p.lx << " ";
	}
	
	//Write \Delta [[u_y]]
	start1d[0] = timepos_min;
	err = nc_inq_varid(ncid, names[12].data(), &dataIDs[12]);
	err = nc_get_vara_double(ncid, dataIDs[12], start1d, count1d, temp1d_old.data());
	start1d[0] = timepos_max-1;
	err = nc_inq_varid(ncid, names[12].data(), &dataIDs[12]);
	err = nc_get_vara_double(ncid, dataIDs[12], start1d, count1d, temp1d.data());
	dg::blas1::axpby(1.0,temp1d,-1.0,temp1d_old,temp1d_int);
	std::cout << sqrt(dg::blas2::dot(temp1d_int,w1d,temp1d_int))/p.lx << " ";
	
	//Write \Delta ln<n>
	start1d[0] = timepos_min;
	err = nc_inq_varid(ncid, names[13].data(), &dataIDs[13]);
	err = nc_get_vara_double(ncid, dataIDs[13], start1d, count1d, temp1d_old.data());
	start1d[0] = timepos_max-1;
	err = nc_inq_varid(ncid, names[13].data(), &dataIDs[13]);
	err = nc_get_vara_double(ncid, dataIDs[13], start1d, count1d, temp1d.data());
	dg::blas1::axpby(1.0,temp1d,-1.0,temp1d_old,temp1d_int);
	std::cout << sqrt(dg::blas2::dot(temp1d_int,w1d,temp1d_int))/p.lx << " ";
	
	//write normalised time
        std::cout << " " << vt[timepos_max-1]/p.invkappa;
	std::cout << " " << vt[timepos_min]/p.invkappa;
        std::cout << "\n";
        err = nc_close(ncid);
        
    }    
    return 0;
}

