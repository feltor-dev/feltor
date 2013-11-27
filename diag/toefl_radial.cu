#include <iostream>
#include <iomanip>
#include <vector>

#include "dg/xspacelib.cuh"
#include "dg/timer.cuh"
#include "file/read_input.h"
#include "file/file.h"

#include "galerkin/parameters.h"
//#include "lamb_dipole/parameters.h"


int main( int argc, char* argv[])
{
    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [inputfile.h5 output.dat]\n";
        return -1;
    }

    std::string in;
    std::ofstream os( argv[2]);
    file::T5rdonly t5file( argv[1], in);

    int layout = 0;
    if( in.find( "TOEFL") != std::string::npos)
        layout = 0;
    else if( in.find( "INNTO") != std::string::npos)
        layout = 1;
    else 
        std::cerr << "Unknown input file format: default to 0"<<std::endl;
    const Parameters p( file::read_input( in), layout);
    p.display();
    dg::Grid<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    dg::HVec visual(  grid.size(), 0.), input( visual);
    dg::HMatrix equi = dg::create::backscatter( grid);
    unsigned index = 1;
    while ( index < t5file.get_size() + 1 )
    {
        t5file.get_field( input, "electrons", index);
        index++;
        //if( p.global)
        //    thrust::transform( input.begin(), input.end(), input.begin(), dg::PLUS<double>(-1));
        dg::blas2::gemv( equi, input, visual);
        const double hx = grid.hx()/(double)grid.n();
        double x = hx/2.;
        unsigned j=p.Ny*p.n/2;
        os << x <<" "<< visual[j*p.Nx*p.n] <<"\n";
        for( unsigned i=0; i<p.Nx*p.n; i++)
        {
            x+=hx;
            os << x <<" "<<visual[j*p.Nx*p.n+i]<<"\n";
        }
        os << std::endl<<std::endl;

    }
    os.close( );
    return 0;
}
