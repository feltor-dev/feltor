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

double Mean(std::vector<double> v, unsigned imin, unsigned imax)
{     
       double sum=0;
       for(unsigned i=imin;i<imax;i++)
               sum+=v[i];
       return sum/(imax-imin);
}
double StdDev(std::vector<double> v, double mean, unsigned imin, unsigned imax)
{
       double E=0.;
       for(unsigned i=imin;i<imax;i++)
               E+=(v[i] - mean)*(v[i] - mean);
       return sqrt(E/(imax-imin));
}

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
    int dataIDs[31];
    std::string names[31] = {"Rfxnorm","Anorm","Rfnnorm","Annorm","dtfauynorm","Rxnorm","invkappaavg","Rnxnorm","Guyxnorm","Txnorm","Guynxnorm","Tnxnorm","neatnorm","Gamma","Rxnormscal","Guynxnormscal","Tnxnormscal","Anormscal","Annormscal","Rfnnormscal","neatsupnorm","nuturbnorm","Rnnormscal","dfnormscal","Rnffnormscal","difflnnnorm","difffauy1norm","difffauy2norm","Slnnnorm","Sfauynorm","vyfavgnorm"}; 
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
        
	size_t start0d  = 0;    
        //get maxtime of input file
        int timeID;
	size_t numOut;
        err = nc_inq_dimid(ncid, "time", &timeID);        
        err = nc_inq_dimlen(ncid, timeID, &numOut);
	std::vector<double> temp(numOut);
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
    
    if (timepos_max > temp.size()) {
        timepos_max = temp.size();
    }
    
	std::cout << p.alpha << " " << p.invkappa;
	//read and write data

	for( unsigned m=0; m<30; m++) {
	    err = nc_inq_varid(ncid, names[m].data(), &dataIDs[m]);
	    err = nc_get_vara_double( ncid, dataIDs[m], &start0d, &numOut, temp.data());

	    double mean   = Mean(  temp,      timepos_min, timepos_max);
	    double stddev = StdDev(temp, mean, timepos_min, timepos_max); 
	    std::cout << " " << mean << " " << stddev; // << " " << stddev/mean;
	}
	//[[vy]]_norm

    err = nc_inq_varid(ncid, names[30].data(), &dataIDs[30]);
    err = nc_get_vara_double( ncid, dataIDs[30], &start0d, &numOut, temp.data());
    std::cout << " " << temp[timepos_max-1]-temp[timepos_min];
    
    std::cout << " " << vt[timepos_min]/p.invkappa;
    std::cout << " " << vt[timepos_max-1]/p.invkappa;
    std::cout << "\n";
    err = nc_close(ncid);
        
    }    
    return 0;
}

