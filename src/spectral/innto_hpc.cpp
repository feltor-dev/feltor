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
    std::cout<< "With "<<omp_get_max_threads()<<" threads\n";
    return bp;
}

void copyMatrix( const Mat& src, std::vector<double> & dst)
{
    for( unsigned i=0; i<src.rows(); i++)
        for( unsigned j=0; j<src.cols(); j++)
            dst[i*src.cols()+j] = src(i,j);
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
    // place some gaussian blobs in the field
    try{
        init_gaussian( ne, posX, posY, blob_width/bp.boundary().lx, blob_width/bp.boundary().ly, amp);
        std::array< Mat, n> arr{{ ne, phi}};
        //now set the field to be computed
        solver.init( arr, TL_IONS);
    }catch( Message& m){m.display();}

    /////////////////////////////////////////////////////////////////////////
    hid_t   file, grp;
    herr_t  status;
    hsize_t dims[] = { alg.ny, alg.nx };
    file = H5Fcreate( argv[2], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    //std::stringstream title; 
    hsize_t size = input.size();
    status = H5LTmake_dataset_char( file, "inputfile", 1, &size, input.data()); //name should precede t so that reading is easier
    double time = 3.*alg.dt;
    std::vector<double> output( alg.nx*alg.ny);
    for( unsigned i=0; i<max_out; i++)
    {
        grp = H5Gcreate( file, file::setTime( time).data(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT  );
        //output all three fields
        copyMatrix( solver.getField( TL_ELECTRONS), output);
        status = H5LTmake_dataset_double( grp, "electrons", 2,  dims, output.data());
        copyMatrix( solver.getField( TL_IONS), output);
        status = H5LTmake_dataset_double( grp, "ions", 2,  dims, output.data());
        copyMatrix( solver.getField( TL_POTENTIAL), output);
        status = H5LTmake_dataset_double( grp, "potential", 2,  dims, output.data());
        H5Gclose( grp);
        for( unsigned i=0; i<itstp; i++)
            solver.step();
        
        time += itstp*alg.dt;
    }
    grp = H5Gcreate( file, file::setTime( time).data(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT  );
    //output all three fields
    copyMatrix( solver.getField( TL_ELECTRONS), output);
    status = H5LTmake_dataset_double( grp, "electrons", 2,  dims, output.data());
    copyMatrix( solver.getField( TL_IONS), output);
    status = H5LTmake_dataset_double( grp, "ions", 2,  dims, output.data());
    copyMatrix( solver.getField( TL_POTENTIAL), output);
    status = H5LTmake_dataset_double( grp, "potential", 2,  dims, output.data());
    H5Gclose( grp);
    //////////////////////////////////////////////////////////////////
    H5Fclose( file);
    fftw_cleanup();
    return 0;

}
