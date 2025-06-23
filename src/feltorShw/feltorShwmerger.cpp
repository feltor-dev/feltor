#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

#include "dg/algorithm.h"

#include "dg/file/file.h"
#include "parameters.h"

//MW: the command line argument ncrcat should do the same doesn't it?
//merge inputfiles together to a new output file
//be aware of grids!!! Should be equal for all input files
int main( int argc, char* argv[])
{
    if( argc == 1)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input1.nc] [input2.nc] ... [output.nc]\n";
        return -1;
    }
    
    double time =0;
    
    
    //nc defs
    dg::file::NC_Error_Handle err, err_out;
    int ncid, ncid_out, tvarIDout = 0,EtimeID, EtimevarID = 0;
    int dim_ids2d[3];
    size_t start2d_out[3]  = {0, 0, 0};
    int dataIDs[4],dataIDs_out[4];     
    std::string names[4] = {"electrons", "ions",  "potential","vor"}; 
    
    //input nc files
    for( int i=1; i< argc-1; i++)
    {

        err = nc_open( argv[i], NC_NOWRITE, &ncid);
        ///////////////////read in and show inputfile und geomfile//////////////////
        size_t length;
        err = nc_inq_attlen( ncid, NC_GLOBAL,   "inputfile", &length);
        std::string input( length, 'x');
        err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
        
        auto js = dg::file::string2Json( input, dg::file::comments::are_forbidden);
        const eule::Parameters p(js);
        
        dg::Grid2d g2d( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);
        size_t count2d[3]  = {1, g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
        size_t start2d[3]  = {0, 0, 0};
        //output nc file
        if (i==1) {
            err_out = nc_create(argv[argc-1],NC_NETCDF4|NC_CLOBBER, &ncid_out);
            err_out = nc_put_att_text( ncid_out, NC_GLOBAL, "inputfile", input.size(), input.data());
            err_out = dg::file::define_dimensions( ncid_out, dim_ids2d, &tvarIDout, g2d);
            err_out = dg::file::define_time( ncid_out, "energy_time", &EtimeID, &EtimevarID);
            for( unsigned j=0; j<4; j++) {
                err_out  = nc_def_var(ncid_out, names[j].data(),  NC_DOUBLE, 3, dim_ids2d, &dataIDs_out[j]);
            }
            err_out = nc_enddef(ncid_out);               
        }

        dg::HVec temp(dg::evaluate(dg::zero,g2d));

        //get maxtime of input file
        int timeID;
        err = nc_inq_dimid(ncid, "time", &timeID);
        size_t steps;
        err = nc_inq_dimlen(ncid, timeID, &steps);

        steps-=1;
        unsigned kmax = steps;
        unsigned kmin=0;
        if (i>1) {  kmin=1;  }

        err_out = nc_open( argv[argc-1], NC_WRITE, &ncid_out);
        //Timestepping
        for( unsigned k=kmin; k<=kmax; k++)
        {
            start2d[0] = k;
            std::cout <<" time = " <<  time << "  start2d_out = " <<start2d_out[0] << "  start2d = " <<start2d[0] <<  std::endl;
            //read and write data
            for( unsigned m=0; m<4; m++) {
                err     = nc_inq_varid(ncid, names[m].data(), &dataIDs[m]);
                err     = nc_get_vara_double( ncid,     dataIDs[m],     start2d,     count2d, temp.data());
                err_out = nc_put_vara_double( ncid_out, dataIDs_out[m], start2d_out, count2d, temp.data()); 
            }
            //write time data
            err_out = nc_put_vara_double( ncid_out, tvarIDout,  start2d_out, count2d, &time);      
            err_out = nc_put_vara_double( ncid_out, EtimevarID,  start2d_out, count2d, &time);

            start2d_out[0] += 1;
            time += p.itstp*p.dt;
        }
        err = nc_close(ncid);
        err_out = nc_close(ncid_out);
    }    
    return 0;
}

