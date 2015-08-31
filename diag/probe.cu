#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

#include "dg/algorithm.h"
#include "file/read_input.h"
#include "file/file.h"
#include "dg/timer.cuh"

#include "galerkin/parameters.h"


//read and evaluate TOEFL & INNTO h5 files
//t2s10a020 50, 100, 150, 200, 250 posX und posY_max
//global
    //double p1x = 22.1667, p1y = -5.16667 ; 
    //double p2x = 61.1667, p2y = -21.1667;
    //double p3x = 96.5, p3y = -28.833;
    //double p4x = 138.167, p4y = -38.8333;
    //double p5x = 176.167, p5y = -44.8333;

//local
    //double p1x = 28.1667, p1y = -14. ;
    //double p2x = 72.3, p2y = -24.17;
    //double p3x = 114.893, p3y = -35.1075;
    //double p4x = 146.924, p4y = -35.4982;
    //double p5x = 196.924, p5y = -27.295;

    //tau = 0 
    double p1x = 25, p1y = 0. ;
    double p2x = 50, p2y = 0.;
    double p3x = 75, p3y = 0.;
    double p4x = 100, p4y = 0.;
    double p5x = 125, p5y = 0.;

//x, y absolute positions
//Annahme dass Punkte alle äquidistant sind
unsigned position_to_index( double x, double y, const dg::Grid<double>& grid)
{
    const unsigned Nx = grid.Nx()*grid.n(); 
    const double hx = grid.hx()/(double)grid.n();
    const double hy = grid.hy()/(double)grid.n();

    unsigned idxX = (x-0.5*hx)/hx;
    unsigned idxY = (y-0.5*hy)/hy;
    return idxX + Nx*idxY;
}

//x, y absolute positions
//Annahme dass Punkte alle äquidistant sind
void index_to_position( unsigned idx, double& x, double& y, const dg::Grid<double>& grid)
{
    //Annahme dass Punkte alle äquidistant sind
    const unsigned Nx = grid.Nx()*grid.n(); 
    const double hx = grid.hx()/(double)grid.n();
    const double hy = grid.hy()/(double)grid.n();

    x = hx*(1./2. + (double)(idx%Nx));
    y = hy*(1./2. + (double)(idx/Nx));
}



double X( double x, double y) {return x;}
double Y( double x, double y) {return y;}
template< class container>
void log( const container& y, container& target)
{
    thrust::transform( y.begin(), y.end(), target.begin(), dg::LN<double>());
}

int main( int argc, char* argv[])
{
    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.h5] [output.dat]\n";
        return -1;
    }
    std::ofstream os( argv[2]);
    std::cout << argv[1]<< " -> "<<argv[2]<<std::endl;

    std::string in;
    file::T5rdonly t5file( argv[1], in);
    const unsigned num_out = t5file.get_size();

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

    dg::HVec input( grid.size());
    dg::HVec visual( input);

    dg::HMatrix equi = dg::create::backscatter( grid);

    const double posX_init = p.lx*p.posX;
    const double posY_init = p.ly*p.posY;
    unsigned p1 = position_to_index( p1x+posX_init, p1y+posY_init, grid);
    unsigned p2 = position_to_index( p2x+posX_init, p2y+posY_init, grid);
    unsigned p3 = position_to_index( p3x+posX_init, p3y+posY_init, grid);
    unsigned p4 = position_to_index( p4x+posX_init, p4y+posY_init, grid);
    unsigned p5 = position_to_index( p5x+posX_init, p5y+posY_init, grid);

    os << "#Time(1) n50(2) n100(3) n150(4) n200(5) n250(6)\n";
    //dg::Timer t;
    for( unsigned idx=1; idx<=num_out; idx++)
    {
        t5file.get_field( input, "electrons", idx);
        os << t5file.get_time( idx);//(1)
        dg::blas2::gemv( equi, input, visual);
        os << " "<<visual[p1]<<" "<<visual[p2]<<" "<<visual[p3]<<" "<<visual[p4]<<" "<<visual[p5];
        os <<"\n";
    }
    os.close();
    return 0;
}

