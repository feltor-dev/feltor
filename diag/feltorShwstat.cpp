#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <iterator>
#include "dg/algorithm.h"
#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"

#include "file/nc_utilities.h"
#include "feltorShw/parameters.h"

double Mean(std::vector<double> v, unsigned imin, unsigned imax)
{     
       double sum=0;
       for(unsigned i=imin;i<imax;i++)
               sum+=v[i];
       return sum/imax;
}
double StdDev(std::vector<double> v, double mean, unsigned imin, unsigned imax)
{
       double E=0.;
       for(unsigned i=imin;i<imax;i++)
               E+=(v[i] - mean)*(v[i] - mean);
       return sqrt(E/imax);
}

int main( int argc, char* argv[])
{
    if( argc == 1)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input1.nc] [input2.nc] ... \n";
        return -1;
    }
    //nc defs
    file::NC_Error_Handle err;
    int ncid;
    int dataIDs[13];
    std::string names[13] =  {"Rfxnorm","Anorm","Rfnnorm","Annorm","dtfauynorm","Rxnorm","invkappaavg","Rnxnorm","Guyxnorm","Txnorm","Guynxnorm","Tnxnorm","neatnorm"}; 
    //input nc files
    for( int i=1; i< argc; i++)
    {
        err = nc_open( argv[i], NC_NOWRITE, &ncid);
        ///////////////////read in and show inputfile und geomfile//////////////////
        size_t length;
        err = nc_inq_attlen( ncid, NC_GLOBAL,   "inputfile", &length);
        std::string input( length, 'x');
        err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
        
        Json::Reader reader;
        Json::Value js;
        reader.parse( input, js, false);
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
	double timepointexact= 100.*p.invkappa; //in units omega_ci 
	std::vector<double>::iterator timepoint;
	timepoint=std::lower_bound (vt.begin(), vt.end(), timepointexact);
	unsigned timepos = std::distance( vt.begin(),timepoint);
	std::cout << p.alpha << " " << p.invkappa;
	//read and write data
	for( unsigned m=0; m<13; m++) {
	    err = nc_inq_varid(ncid, names[m].data(), &dataIDs[m]);
	    err = nc_get_vara_double( ncid, dataIDs[m], &start0d, &numOut, temp.data());

            
	    double mean = Mean(temp,timepos,temp.size());
	    double stddev = StdDev(temp,mean,timepos,temp.size()); 
	    std::cout << " " << mean << " " << stddev; // << " " << stddev/mean;
	}
        std::cout << "\n";
        err = nc_close(ncid);
        
    }    
    return 0;
}
