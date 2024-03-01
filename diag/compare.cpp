#include <iostream>
#include <iomanip>

#include "dg/blas1.h"
#include "dg/file/nc_utilities.h"


// DEPRECATED: Write a python script!
//This program reads in two netcdf files and tries to compare the fields 
//named electrons in both files at the same times.
//The fields must both be either two or three-dimensional and must
// be of the same size. 
// The physical boundaries and the coordinate system is assumed to be the same. 
//Only fields at the same time can be compared.


int main( int argc, char** argv)
{
    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [file1.nc file2.nc]\n";
        return -1;
    }
    std::cout << "Compare "<<argv[1]<<" with "<<argv[2]<<"\n";
    //////////////////////////////open nc files//////////////////////////////////
    dg::file::NC_Error_Handle err;
    int ncid1, ncid2;
    err = nc_open( argv[1], NC_NOWRITE, &ncid1);
    err = nc_open( argv[2], NC_NOWRITE, &ncid2);
    
    int dimIDs1[3], dimIDs2[3], timeID1, timeID2;
    int numDims1=3, numDims2=3;

    err = nc_inq_dimid( ncid1, "time", &timeID1);
    err = nc_inq_dimid( ncid2, "time", &timeID2);
    err = nc_inq_dimid( ncid1, "x", &dimIDs1[0]);
    err = nc_inq_dimid( ncid2, "x", &dimIDs2[0]);
    err = nc_inq_dimid( ncid1, "y", &dimIDs1[1]);
    err = nc_inq_dimid( ncid2, "y", &dimIDs2[1]);
    try{ err = nc_inq_dimid( ncid1, "z", &dimIDs1[2]);}
    catch( dg::file::NC_Error&) { numDims1=2; }
    try{ err = nc_inq_dimid( ncid2, "z", &dimIDs2[2]);}
    catch( dg::file::NC_Error&) { numDims2=2; }
    if( numDims1 != numDims2)
    {
        std::cerr << "Files not of same dimensionality!!\n";
        return -1;
    }
    size_t length1[numDims1+1], length2[numDims2+1];
    err = nc_inq_dimlen(ncid1, timeID1, &length1[numDims1]);
    err = nc_inq_dimlen(ncid2, timeID2, &length2[numDims2]);
    for(int i=0; i<numDims1; i++)
    {
        err = nc_inq_dimlen(ncid1, dimIDs1[i], &length1[i]);
        err = nc_inq_dimlen(ncid2, dimIDs2[i], &length2[i]);
    }
    for( int i=0; i<numDims1+1; i++)
    {
        std::cout << "Dimension "<<i<<" has "<<length1[i]<<" points!\n";
        if( length1[i] != length2[i])
        {
            std::cerr << "Dimension lengths not equal!! "<<length1[i]<<" "<<length2[i]<<"\n";
            return -1;
        }
    }
    int dataID1, dataID2;
    try{
        err = nc_inq_varid(ncid1, "electrons", &dataID1);
        err = nc_inq_varid(ncid2, "electrons", &dataID2);
    }
    catch( dg::file::NC_Error&)
    {
        try{
            err = nc_inq_varid(ncid1, "T", &dataID1);
            err = nc_inq_varid(ncid2, "T", &dataID2);
        }
        catch( dg::file::NC_Error&)
        {
            std::cerr <<"Neither electrons nor T found!\n";
            return -1;
        }
    }
    size_t start[numDims1+1], count[numDims1+1], size=1;
    for( int i=0; i<numDims1; i++) {
        start[numDims1-i] = 0;
        count[numDims1-i] = length1[i];
        size*=length1[i];
    }
    start[0] = 0, count[0] = 1;
    thrust::host_vector<double> input1( size), input2(input1);
    for( size_t i=0; i<length1[numDims1]; i++)
    {
        start[0] = i;
        err = nc_get_vara_double( ncid1, dataID1, start, count, input1.data());
        err = nc_get_vara_double( ncid2, dataID2, start, count, input2.data());
        dg::blas1::axpby( 1., input1, -1., input2, input2);
        double norm = dg::blas1::dot( input1, input1);
        double diff = dg::blas1::dot( input2, input2);
        //double norm2 = *thrust::max_element( input2.begin(), input2.end());
        std::cout << 
//"Abs. and rel. difference at timestep \t"<<
i<<"\t"<<std::setprecision(17)<<sqrt(diff)<<"\t"<<sqrt(diff/norm)<<"\n";
    }
    err = nc_close(ncid1);
    err = nc_close(ncid2);

    return 0;
}
