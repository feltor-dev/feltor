#include <iostream>
#include "grid.h"
#include "derivatives.cuh"

/*
 * Create a cusp tensor product and try to recreate with 
 * new tensor produc for dx_matrix
 *
 */

 
int main(void)
{
    unsigned int P = 0;
    unsigned int Nx = 0;
    unsigned int Ny = 0;
    std::cout << "Type in P, Nx, and Ny" << std::endl;
    std::cin >> P >> Nx >> Ny; 

    dg::Grid2d<double> g2(0.0, 1.0, // x0:x1
                          0.0, 1.0, //y0:y1
                          P, Nx, Ny, //P, Nx, Ny
                          dg::PER, dg::PER);

    g2.display();

    cusp::coo_matrix<int, double, cusp::host_memory> dx2_matrix = dg::create::dx(g2, dg::PER);
    cusp::print(dx2_matrix);

}

