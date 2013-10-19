#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

#include "dg/xspacelib.cuh"
#include "file/read_input.h"
#include "file/file.h"

#include "galerkin/parameters.h"

double X( double x, double y) {return x;}
double Y( double x, double y) {return y;}

int main( int argc, char* argv[])
{
    if( argc != 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [inputfile]\n";
        return;
    }
    //open file for com - data
    std::string outputfile( argv[1]);
    outputfile.erase( outputfile.end()-2, outputfile.end());
    outputfile+="com";
    std::ofstream os( outputfile.c_str());

    hid_t file = H5Fopen( argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    hsize_t nlinks = file::getNumObjs( file);
    std::string name = file::getName( file, 0);
    std::string in;
    
    herr_t  status;
    hsize_t dims[2]; 
    in.resize( 10000);
    status = H5LTread_dataset_string( file, name.data(), &in[0]); //name should precede t so that reading is easier


    const Parameters p( file::read_input( in));
    p.display();
    dg::Grid<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    dg::HVec input_h( grid.size());
    dg::DVec input( input_h);
    dg::DVec xvec = dg::evaluate( X, grid);
    dg::DVec yvec = dg::evaluate( Y, grid);
    dg::DVec one = dg::evaluate( dg::one, grid);
    dg::DVec w2d = dg::create::w2d( grid);

    double mass, posX, posY, velX;
    double posX_old = 0;
    for( unsigned i=1; i<=p.maxout+1; i++)
    {
        name = file::getName( file, i);
        hid_t group = H5Gopen( file, name.data(), H5P_DEFAULT);
        status = H5LTread_dataset_double( group, "electrons", &input_h[0] );
        input = input_h;
        if( p.global)
            thrust::transform( input.begin(), input.end(), input.begin(), dg::PLUS<double>(-1));
        mass = dg::blas2::dot( one, w2d, input);
        os << file::getTime( name)<<" ";

        posX = dg::blas2::dot( xvec, w2d, input)/mass - p.posX*p.lx;
        posY = dg::blas2::dot( yvec, w2d, input)/mass - p.posY*p.ly;
        velX = (posX - posX_old)/p.dt;
        posX_old = posX;
        os << posX << " " << posY << " "<<velX<<"\n";
    }
    std::cout << "Format is:\n"
        << " time posX posY velX\n";

    os.close();
    H5Fclose( file);
    return 0;
}
