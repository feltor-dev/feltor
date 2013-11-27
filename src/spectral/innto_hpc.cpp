#include <iostream>
#include <iomanip>
#include <sstream>
#include <omp.h>

#include "toefl/toefl.h"
#include "file/read_input.h"
#include "file/file.h"
//#include "utility.h"
#include "particle_density.h"
#include "dft_dft_solver.h"
//#include "drt_dft_solver.h"
#include "blueprint.h"
/*
 * Same as innblobs but outputs results in hdf5 - file
 */

using namespace std;
using namespace toefl;

const unsigned n = 2;
typedef DFT_DFT_Solver<n> Sol;
typedef typename Sol::Matrix_Type Mat;
    
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

    /////////////////////////////////////////////////////////////////////////
    file::T5trunc t5file( argv[2], input);
    double time = 3.*alg.dt;
    std::vector<double> out( alg.nx/reduction*alg.ny/reduction);
    std::vector<double> output[3] = {out, out, out};
    for( unsigned i=0; i<max_out; i++)
    {
        //output all three fields
        copyAndReduceMatrix( solver.getField( TL_ELECTRONS), output[0]);
        xpa( output[0], meanMassE); //mean mass gets lost through the timestep
        copyAndReduceMatrix( solver.getField( TL_IONS), output[1]);
        copyAndReduceMatrix( solver.getField( TL_POTENTIAL), output[2]);
        t5file.write( output[0], output[1], output[2], time, alg.nx/reduction, alg.ny/reduction);
        for( unsigned i=0; i<itstp; i++)
            solver.step();
        
        time += itstp*alg.dt;
    }
    copyAndReduceMatrix( solver.getField( TL_ELECTRONS), output[0]);
    xpa( output[0], meanMassE);
    copyAndReduceMatrix( solver.getField( TL_IONS), output[1]);
    copyAndReduceMatrix( solver.getField( TL_POTENTIAL), output[2]);
    t5file.write( output[0], output[1], output[2], time, alg.nx/reduction, alg.ny/reduction);
    //////////////////////////////////////////////////////////////////
    fftw_cleanup();
    return 0;

}
