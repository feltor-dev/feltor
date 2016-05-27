#include <iostream>
#include <iomanip>
#include <vector>

#include "dg/blas.h"
#include "dg/backend/xspacelib.cuh"
#include "dg/backend/timer.cuh"
#include "dg/functors.h"
#include "file/read_input.h"
#include "file/file.h"


#include "parameters.h"

int main( int argc, char* argv[])
{

    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.h5 output.dat]\n";
        return 0;
    }

    std::string in;
    file::T5rdonly t5file( argv[1], in);
    unsigned nlinks = t5file.get_size();
    const Parameters p( file::read_input(in));
    dg::Grid2d<double> g( 0, p.lx, 0., p.ly, p.n, p.Nx, p.Ny);
    const dg::HVec w2d( dg::create::weights(g));
    const dg::HVec one( g.size(), 1.);

    std::vector<double> vort, enstrophy, energy, variation; 
    t5file.get_xfile( vort, "mass");
    t5file.get_xfile( enstrophy, "diffusion");
    t5file.get_xfile( energy, "energy");
    t5file.get_xfile( variation, "dissipation");

    std::ofstream ofs( argv[2]);
    ofs << "# Time(1) vort(2) enstrophy(3) energy(4) variation(5) eps_enstrophy(6) eps_energy(7)\n";
    ofs << 0*p.dt <<" "<< vort[0] <<" "<<enstrophy[0]<<" "<<energy[0]<<" "<<variation[0]<<" 0 0\n";
    for(unsigned i=1; i<vort.size(); i++ )
    {
        ofs << i*p.dt <<" "<< vort[i] <<" "<<enstrophy[i]<<" "<<energy[i]<<" "<<variation[i]<<" ";
        enstrophy[i] = fabs((enstrophy[i]-enstrophy[0])/enstrophy[0]);
        energy[i] = fabs((energy[i]-energy[0])/energy[0]);
        ofs << enstrophy[i]<<" "<<energy[i]<<"\n";
    }
    for( unsigned i=1; i<70; i++)
    {
        dg::HVec input(g.size());
        t5file.get_field( input, "electrons", i);
        std::cout << dg::blas2::dot( input, w2d, one)<<std::endl;
    }
    ofs.close();

    return 0;
}
