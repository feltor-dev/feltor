#include <iostream>

#include <cusp/print.h>
#include "../blas.h"
#include "xspacelib.cuh"
#include "interpolation.cuh"

 unsigned n = 3;
 unsigned Nx = 30; 
 unsigned Ny = 50; 
 unsigned Nz = 2; 

typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;

double sinex( double x, double y) {return sin(x)*sin(x)*sin(y)*sin(y)*x*x*y*y;}
double sinez( double x, double y, double z) {return sin(x)*sin(x)*sin(y)*sin(y)*x*x*y*y;}

int main()
{
    std::cout << "type n, Nx, Ny, Nz\n";
    std::cin >> n >> Nx >> Ny >> Nz;


    {
    dg::Grid2d<double> g( -10, 10, -5, 5, n, Nx, Ny);
    Matrix A = dg::create::backscatter( g);
    //A.sort_by_row_and_column();

    thrust::host_vector<double> x( g.size()), y(x);
    for( unsigned i=0; i<g.Ny()*g.n(); i++)
        for( unsigned j=0; j<g.Nx()*g.n(); j++)
        {
            x[i*g.Nx()*g.n() + j] = 
                    g.x0() + (j+0.5)*g.hx()/(double)(g.n());
            y[i*g.Nx()*g.n() + j] = 
                    g.y0() + (i+0.5)*g.hy()/(double)(g.n());
        }
    Matrix B = dg::create::interpolation( x, y, g);
    bool passed = true;
    //cusp::print(A);
    //cusp::print(B);
    //ATTENTION: backscatter might delete zeroes in matrices
    for( unsigned i=0; i<A.values.size(); i++)
    {
        if( (A.values[i] - B.values[i]) > 1e-10)
        {
            std::cerr << "NOT EQUAL "<<A.row_indices[i] <<" "<<A.column_indices[i]<<" "<<A.values[i] << "\t "<<B.row_indices[i]<<" "<<B.column_indices[i]<<" "<<B.values[i]<<"\n";
            passed = false;
        }
    }
    if( A.num_entries != B.num_entries)
    {
        std::cerr << "Number of entries not equal!\n";
        passed = false;
    }
    if( passed)
        std::cout << "2D TEST PASSED!\n";


    passed = true;
    thrust::host_vector<double> xs = dg::evaluate( dg::coo1, g); 
    thrust::host_vector<double> ys = dg::evaluate( dg::coo2, g); 
    thrust::host_vector<double> xF = dg::create::forward_transform( xs, g);
    thrust::host_vector<double> yF = dg::create::forward_transform( ys, g);
    for( unsigned i=0; i<x.size(); i++)
    {
        double xi = dg::interpolate(x[i],y[i], xF, g);
        double yi = dg::interpolate(x[i],y[i], yF, g);
        if( x[i] - xi > 1e-14)
        {
            std::cerr << "X NOT EQUAL "<<i<<"\t"<<x[i]<<"  \t"<<xi<<"\n";
            passed = false;
        }
        if( y[i] - yi > 1e-14)
        {
            std::cerr << "Y NOT EQUAL "<<i<<"\t"<<y[i]<<"  \t"<<yi<<"\n";
            passed = false;
        }
    }
    if( passed)
        std::cout << "2D INTERPOLATE TEST PASSED!\n";

    dg::Grid2d<double> gfine( -10, 10, -5, 5, n, n*Nx, n*Ny);
    thrust::host_vector<double> xfine = dg::evaluate( sinex, gfine);
    thrust::host_vector<double> xcoarse = dg::evaluate( sinex, g);
    thrust::host_vector<double> wfine = dg::create::weights( gfine);
    thrust::host_vector<double> wcoarse = dg::create::weights( g);
    double coarse = dg::blas2::dot( xcoarse, wcoarse, xcoarse);
    std::cout << "coar integral: "<<coarse<<"\n";

    Matrix f2c = dg::create::interpolation( g, gfine); 
    dg::blas2::symv( f2c, xfine, xcoarse);
    double fine = dg::blas2::dot( xfine, wfine, xfine);
    coarse = dg::blas2::dot( xcoarse, wcoarse, xcoarse);
    //double fine = dg::blas1::dot( wfine, xfine);
    //coarse = dg::blas1::dot( wcoarse, xcoarse);
    std::cout << "Fine integral: "<<fine<<" \n";
    std::cout << "coar integral: "<<coarse<<"\n";
    std::cout << "Rel Difference "<<fabs(fine-coarse)/fabs(fine)<<"\n";

    }
    ////////////////////////////////////////////////////////////////////////////
    {
    dg::Grid3d<double> g( -10, 10, -5, 5, -7, -3, n, Nx, Ny, Nz);
    Matrix A = dg::create::backscatter( g);
    //A.sort_by_row_and_column();

    std::vector<double> x( g.size()), y(x), z(x);
    for( unsigned k=0; k<g.Nz(); k++)
        for( unsigned i=0; i<g.Ny()*g.n(); i++)
            for( unsigned j=0; j<g.Nx()*g.n(); j++)
            {
                x[(k*g.Ny()*g.n() + i)*g.Nx()*g.n() + j] = 
                        g.x0() + (j+0.5)*g.hx()/(double)(g.n());
                y[(k*g.Ny()*g.n() + i)*g.Nx()*g.n() + j] = 
                        g.y0() + (i+0.5)*g.hy()/(double)(g.n());
                z[(k*g.Ny()*g.n() + i)*g.Nx()*g.n() + j] = 
                        g.z0() + (k+0.5)*g.hz();
            }
    Matrix B = dg::create::interpolation( x, y, z, g);
    bool passed = true;
    for( unsigned i=0; i<A.values.size(); i++)
    {
        if( (A.values[i] - B.values[i]) > 1e-10)
        {
            std::cerr << "NOT EQUAL "<<A.row_indices[i] <<" "<<A.column_indices[i]<<" "<<A.values[i] << "\t "<<B.row_indices[i]<<" "<<B.column_indices[i]<<" "<<B.values[i]<<"\n";
            passed = false;
        }
    }
    if( passed)
        std::cout << "3D TEST PASSED!\n";
    dg::Grid3d<double> gfine( -10, 10, -5, 5, -7, -3,  n, n*Nx, n*Ny, Nz);
    thrust::host_vector<double> xfine = dg::evaluate( sinez, gfine);
    thrust::host_vector<double> xcoarse = dg::evaluate( sinez, g);
    thrust::host_vector<double> wfine = dg::create::weights( gfine);
    thrust::host_vector<double> wcoarse = dg::create::weights( g);

    Matrix f2c = dg::create::scalar_interpolation( g, gfine); 
    dg::blas2::symv( f2c, xfine, xcoarse);
    double fine = dg::blas1::dot( xfine, wfine);
    double coarse = dg::blas1::dot( xcoarse, wcoarse);
    std::cout << "Fine integral: "<<fine<<" \n";
    std::cout << "coar integral: "<<coarse<<"\n";
    std::cout << "Difference   : "<<fine-coarse<<"\n";
    }

    return 0;
}
