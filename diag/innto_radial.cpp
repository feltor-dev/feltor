#include <iostream>
#include <iomanip>
#include <vector>

#include "file/read_input.h"
#include "file/file.h"

#include "spectral/dft_dft_solver.h"
#include "spectral/blueprint.h"
#include "spectral/particle_density.h"

typedef toefl::DFT_DFT_Solver<2> Sol;
typedef typename Sol::Matrix_Type Mat;

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

    const toefl::Blueprint bp( file::read_input( in));
    const toefl::Algorithmic alg( bp.algorithmic());
    std::vector<double> input( alg.nx*alg.ny);
    bp.display();
    unsigned index = 1;
    while ( index < t5file.get_size() + 1 )
    {
        t5file.get_field( input, "electrons", index);
        index++;
        const double hx = alg.h;
        double x = hx/2.;
        unsigned j=alg.ny/2;
        os << x <<" "<< input[j*alg.nx] <<"\n";
        for( unsigned i=0; i<alg.nx; i++)
        {
            x+=hx;
            os << x <<" "<<input[j*alg.nx+i]<<"\n";
        }
        os << std::endl<<std::endl;

    }
    os.close( );
    return 0;
}
