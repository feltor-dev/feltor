#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"
#include "dg/xspacelib.cuh"
#include "dg/timer.cuh"
#include "dg/functors.cuh"
#include "dg/projection.cuh"
#include "file/read_input.h"
#include "file/file.h"

#include "galerkin/parameters.h"

//compare two TOEFL h5 files with equal physical parameters

int main( int argc, char* argv[])
{
    dg::Timer t;
    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [file1.h5 file2.h5]\n";
        return -1;
    }

    std::string in1, in2;
    file::T5rdonly t5file1( argv[1], in1);
    file::T5rdonly t5file2( argv[2], in2);
    unsigned nlinks1 = t5file1.get_size();
    unsigned nlinks2 = t5file2.get_size();

    const Parameters p1( file::read_input( in1));
    const Parameters p2( file::read_input( in2));
    dg::Grid<double> grid1( 0, p1.lx, 0, p1.ly, p1.n, p1.Nx, p1.Ny, p1.bc_x, p1.bc_y);
    dg::Grid<double> grid2( 0, p2.lx, 0, p2.ly, p2.n, p2.Nx, p2.Ny, p2.bc_x, p2.bc_y);
    if( p1.lx != p2.lx || p1.ly != p2.ly)
    {
        std::cerr << " Domain not equal!\n";
        return -1;
    }
    if( p1.bc_x != p2.bc_x || p1.bc_y != p2.bc_y)
    {
        std::cerr << "Warning: Boundary conditions not equal!\n";
    }
    dg::DifferenceNorm<dg::HVec> diff( grid1, grid2);
    //compute common timesteps
    unsigned DT1 = (unsigned)1e5*p1.dt*p1.itstp;
    unsigned DT2 = (unsigned)1e5*p2.dt*p2.itstp;
    unsigned DT = dg::lcm( DT1, DT2); //least common multiple
    unsigned D1 = DT/DT1, D2 = DT/DT2;
    //std::cout << DT1 << " "<<DT2<<" "<<DT<<" D1 "<<D1<<"D2 "<<D2<<std::endl;
    unsigned idx1=1, idx2=1;
    dg::HVec field1( grid1.size()), field2( grid2.size());
    dg::HVec w1 = dg::create::w2d( grid1), w2 = dg::create::w2d( grid2);
    std::cout << "Begin computation...\n";
    std::cout << "Differences \tTime \tElectrons \tIons    \tPotential\n";
    while( (idx1 <= t5file1.get_size()) && (idx2<=t5file2.get_size()))
    {
        t5file1.get_field( field1, "electrons", idx1);
        t5file2.get_field( field2, "electrons", idx2);
        thrust::transform( field1.begin(), field1.end(), field1.begin(), dg::PLUS<double>(-1));
        thrust::transform( field2.begin(), field2.end(), field2.begin(), dg::PLUS<double>(-1));

        std::cout << "Differences \t"<< t5file1.get_time( idx1)<< "\t"<<2.*diff( field1, field2)/diff.sum( field1, field2);

        t5file1.get_field( field1, "ions", idx1);
        t5file2.get_field( field2, "ions", idx2);
        thrust::transform( field1.begin(), field1.end(), field1.begin(), dg::PLUS<double>(-1));
        thrust::transform( field2.begin(), field2.end(), field2.begin(), dg::PLUS<double>(-1));

        std::cout << "\t"<<2.*diff( field1, field2)/diff.sum( field1, field2);
        t5file1.get_field( field1, "potential", idx1);
        t5file2.get_field( field2, "potential", idx2);

        std::cout << "\t"<<2.*diff( field1, field2)/diff.sum( field1, field2)<<std::endl;
        idx1 += 100*D1, idx2 += 100*D2;
    }

    return 0;
}
