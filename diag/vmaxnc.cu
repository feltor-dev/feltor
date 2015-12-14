#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>

#include "file/nc_utilities.h"

//scan all imputfiles for maximum radial velocity and write to std::out
int main( int argc, char* argv[])
{
    if( argc == 1)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input1.nc] [input2.nc] ...\n";
        return -1;
    }
    file::NC_Error_Handle err;
    int timeID, varID;
    size_t start = {0}, numOut;
    for( int i=1; i< argc; i++)
    {
        int ncid;
        err = nc_open( argv[i], NC_NOWRITE, &ncid);
        err = nc_inq_dimid( ncid, "time", &timeID);
        err = nc_inq_dimlen( ncid, timeID, &numOut);
        err = nc_inq_varid( ncid, "Depsi_total", &varID);
        std::vector<double> v(numOut);
        err = nc_get_vara_double( ncid, varID, &start, &numOut, v.data());

        std::cout << argv[i]<< " " << *std::max_element(v.begin(), v.end()) <<"\n";

    }

    
    return 0;
}

