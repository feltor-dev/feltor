#include <iostream>
#include <iomanip>
#include <vector>

#include "dg/xspacelib.cuh"
#include "dg/timer.cuh"
#include "dg/functors.cuh"
#include "file/read_input.h"
#include "file/file.h"


#include "parameters.h"

int main( int argc, char* argv[])
{

    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.h5 output.dat]\n";
        return;
    }

    std::string in;
    file::T5rdonly t5file( argv[1], in);
    unsigned nlinks = t5file.get_size();
    const Parameters p( file::read_input(in));

    std::vector<double> vort, enstrophy, energy; 
    t5file.get_xfile( vort, "mass");
    t5file.get_xfile( enstrophy, "energy");
    t5file.get_xfile( energy, "diffusion");
    std::ofstream ofs( argv[2]);
    ofs << "# Time vort enstrophy energy eps_enstrophy eps_energy\n";
    ofs << 0*p.dt <<" "<< vort[0] <<" "<<enstrophy[0]<<" "<<energy[0]<<" 0 0\n";
    for(unsigned i=1; i<vort.size(); i++ )
    {
        ofs << i*p.dt <<" "<< vort[i] <<" "<<enstrophy[i]<<" "<<energy[i]<<" ";
        enstrophy[i] = fabs((enstrophy[i]-enstrophy[0])/enstrophy[0]);
        energy[i] = fabs((energy[i]-energy[0])/energy[0]);
        ofs << enstrophy[i]<<" "<<energy[i]<<"\n";
    }
    ofs.close();

    return 0;
}
