#include <iostream>
#include <iomanip>
#include <vector>

#include "dg/backend/xspacelib.cuh"
#include "dg/backend/timer.cuh"
#include "file/read_input.h"
#include "file/file.h"

#include "toefl/parameters.h"
//#include "lamb_dipole/parameters.h"

//t2s10a020 50, 100, 150, 200, 250 posX und posY_max
//local
//double py[5] = { -5.16667, -21.16667, -28.8333, -38.8333, -44.8333};//global
double py[5] = { -14., -24.17, -35.1075, -35.4982, -27.295};//local
//Annahme dass Punkte alle äquidistant sind
unsigned position_to_index( double x, double y, const dg::Grid2d<double>& grid)
{
    const unsigned Nx = grid.Nx()*grid.n(); 
    const double hx = grid.hx()/(double)grid.n();
    const double hy = grid.hy()/(double)grid.n();

    unsigned idxX = (x-0.5*hx)/hx;
    unsigned idxY = (y-0.5*hy)/hy;
    return idxX + Nx*idxY;
}

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
    dg::Grid2d<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    dg::HVec visual(  grid.size(), 0.), input( visual);
    dg::HMatrix equi = dg::create::backscatter( grid);
    unsigned index = 1;
    while ( index < t5file.get_size() + 1 )
    {
        t5file.get_field( input, "electrons", index);
        index++;
        dg::blas2::gemv( equi, input, visual);
        const double hx = grid.hx()/(double)grid.n();
        
        //std::cout << idx << std::endl;
        //unsigned j=p.Ny*p.n/2;
        //std::cout << j*p.Nx*p.n << std::endl;
        for( unsigned k = 0; k<5; k++)
        {
            double x = hx/2.;
            unsigned idx = position_to_index( x, grid.ly()/2.+py[k], grid);
            os << x <<" "<< visual[idx] <<"\n";
            for( unsigned i=1; i<p.Nx*p.n; i++)
            {
                x+=hx;
                os << x <<" "<<visual[idx+i]<<"\n";
            }
            os << std::endl<<std::endl;
        }

    }
    os.close( );
    return 0;
}
