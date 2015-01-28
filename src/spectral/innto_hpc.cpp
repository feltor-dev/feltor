#include <iostream>
#include <iomanip>
#include <sstream>
#include <omp.h>
#include <netcdf.h>

#include "toefl/toefl.h"
#include "file/read_input.h"
#include "file/file.h"
#include "dg/backend/grid.h"
#include "file/nc_utilities.h"

#include "particle_density.h"
#include "dft_dft_solver.h"
#include "drt_dft_solver.h"
#include "blueprint.h"
/*
 * Same as innblobs but outputs results in netcdf - file
 */

using namespace std;
using namespace toefl;

const unsigned n = 2;
typedef DFT_DFT_Solver<n> Sol;
typedef Sol::Matrix_Type Mat;
    
unsigned itstp; //initialized by init function
unsigned max_out;
double amp, imp_amp; //
double blob_width, posX, posY;
unsigned reduction;


Blueprint read( char const * file)
{
    std::cout << "Reading from "<<file<<"\n";
    std::vector<double> para;
    try{ para = file::read_input( file); }
    catch (Message& m) 
    {  
        m.display(); 
        throw m;
    }
    Blueprint bp( para);
    amp = para[10];
    imp_amp = para[14];
    itstp = para[19];
    omp_set_num_threads( para[20]);
    blob_width = para[21];
    max_out = para[22];
    posX = para[23];
    posY = para[24];
    reduction = para[25];
    std::cout<< "With "<<omp_get_max_threads()<<" threads\n";
    return bp;
}

void copyAndReduceMatrix( const Mat& src, std::vector<double> & dst)
{
    unsigned num = 0;
    for( unsigned i=0; i<src.rows(); i+= reduction)
        for( unsigned j=0; j<src.cols(); j+= reduction)
        {
            dst[num] = src(i,j);
            num ++;
        }
}
double integral( const Mat& src, double h)
{
    double sum=0;
    for( unsigned i=0; i<src.rows(); i++)
        for( unsigned j=0; j<src.cols(); j++)
            sum+=h*h*src(i,j);
    return sum;
}

void xpa( std::vector<double>& x, double a)
{
    for( unsigned i =0; i<x.size(); i++)
        x[i] += a;
}


    
int main( int argc, char* argv[])
{
    //Parameter initialisation
    Blueprint bp_mod;
    std::vector<double> v;
    std::string input;
    if( argc != 3)
    {
        cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [outputfile]\n";
        return -1;
    }
    else 
    {
        bp_mod = read(argv[1]);
        input = file::read_file( argv[1]);
    }
    const Blueprint bp = bp_mod;
    bp.display( );
    if( bp.boundary().bc_x != TL_PERIODIC)
    {
        cerr << "Only periodic boundaries allowed!\n";
        return -1;
    }
    //construct solvers 
    try{
    Sol solver( bp);
    }catch( Message& m){m.display();}
    Sol solver (bp);

    const Algorithmic& alg = bp.algorithmic();
    Mat ne{ alg.ny, alg.nx, 0.}, phi{ ne};
    const Boundary& bound = bp.boundary();
    // place some gaussian blobs in the field
    try{
        init_gaussian( ne, posX, posY, blob_width/bound.lx, blob_width/bound.ly, amp);
        std::array< Mat, n> arr{{ ne, phi}};
        //now set the field to be computed
        solver.init( arr, TL_IONS);
    }catch( Message& m){m.display();}
    double meanMassE = integral( ne, alg.h)/bound.lx/bound.ly;
    //Energetics<n> energetics( bp);

    /////////////////////////////////////////////////////////////////////////
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( argv[2], NC_NETCDF4|NC_CLOBBER, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "input", input.size(), input.data());
    int dim_ids[3], tvarID;
    dg::Grid1d<double> gx( 0, bound.lx, 1, alg.nx/reduction);
    dg::Grid1d<double> gy( 0, bound.ly, 1, alg.ny/reduction);
    dg::Grid2d<double> g2d( gx, gy);
    err = file::define_dimensions( ncid, dim_ids, &tvarID, g2d);
    int id_ne, id_ni, id_phi;
    err = nc_def_var( ncid, "n_e", NC_DOUBLE, 3, dim_ids, &id_ne);
    err = nc_def_var( ncid, "n_i", NC_DOUBLE, 3, dim_ids, &id_ni);
    err = nc_def_var( ncid, "phi", NC_DOUBLE, 3, dim_ids, &id_phi);
    err = nc_enddef( ncid);
    size_t count[3] = {1, alg.ny/reduction, alg.nx/reduction};
    size_t start[3] = {0, 0, 0};

    //file::T5trunc t5file( argv[2], input);
    double time = 0.0;
    std::vector<double> out( alg.nx/reduction*alg.ny/reduction);
    std::vector<double> output[3] = {out, out, out};
    for( unsigned i=0; i<max_out; i++)
    {
        //output all three fields
        copyAndReduceMatrix( solver.getField( TL_ELECTRONS), output[0]);
        xpa( output[0], meanMassE); //mean mass gets lost through the timestep
        copyAndReduceMatrix( solver.getField( TL_IONS), output[1]);
        copyAndReduceMatrix( solver.getField( TL_POTENTIAL), output[2]);
        //t5file.write( output[0], output[1], output[2], time, alg.nx/reduction, alg.ny/reduction);
        start[0] = i;
        err = nc_put_vara_double( ncid, id_ne, start, count, output[0].data());
        err = nc_put_vara_double( ncid, id_ni, start, count, output[1].data());
        err = nc_put_vara_double( ncid, id_phi, start, count, output[2].data());
        const size_t Tcount = 1, Tstart = i;
        err = nc_put_vara_double( ncid, tvarID, &Tstart, &Tcount, &time);
        //std::vector<double> exb = energetics.exb_energies( solver.getField(TL_POTENTIAL));
        //std::vector<double> thermal = energetics.thermal_energies( solver.getDensity());
        //std::cout<< thermal[0] << " "<< thermal[1]<<" "<<exb[0]<<"\n";
        //t5file.append( meanMassE, 0, exb[0]+thermal[0]+thermal[1], 0);

        for( unsigned j=0; j<itstp; j++)
        {
            if( i==0 && j==0)
                solver.first_step();
            else if( i==0 && j==1)
                solver.second_step();
            else 
                solver.step();
            time += alg.dt;
        }
    }
    copyAndReduceMatrix( solver.getField( TL_ELECTRONS), output[0]);
    xpa( output[0], meanMassE);
    copyAndReduceMatrix( solver.getField( TL_IONS), output[1]);
    copyAndReduceMatrix( solver.getField( TL_POTENTIAL), output[2]);
    //t5file.write( output[0], output[1], output[2], time, alg.nx/reduction, alg.ny/reduction);
    start[0] = max_out;
    err = nc_put_vara_double( ncid, id_ne, start, count, output[0].data());
    err = nc_put_vara_double( ncid, id_ni, start, count, output[1].data());
    err = nc_put_vara_double( ncid, id_phi, start, count, output[2].data());
    const size_t Tcount = 1, Tstart = max_out;
    err = nc_put_vara_double( ncid, tvarID, &Tstart, &Tcount, &time);
    err = nc_close( ncid);
    //////////////////////////////////////////////////////////////////
    fftw_cleanup();
    return 0;

}
