#include <iostream>

#include <cusp/print.h>
#include "../blas.h"
#include "xspacelib.cuh"
#include "interpolation.cuh"
#include "../blas.h"
#include "evaluation.cuh"

double function( double x, double y){return sin(x)*sin(y);}
double function( double x, double y, double z){return sin(x)*sin(y)*sin(z);}

 unsigned n = 3;
 unsigned Nx = 30; 
 unsigned Ny = 50; 
 unsigned Nz = 2; 

typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;

double sinex( double x, double y) {return sin(x)*sin(x)*sin(y)*sin(y)*x*x*y*y;}
double sinex( double x, double y, double z) {return sin(x)*sin(x)*sin(y)*sin(y)*x*x*y*y;}

int main()
{
    std::cout << "type n, Nx, Ny, Nz\n";
    std::cin >> n >> Nx >> Ny >> Nz;
    std::cout << "type nfine, Nmultiply (fine grid is nf, NfNx, NfNy, Nz)\n";
    unsigned nf, Nf;
    std::cin >> nf >> Nf;


    {
    dg::Grid2d g( -10, 10, -5, 5, n, Nx, Ny);
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
    thrust::host_vector<double> vec = dg::evaluate( function, g), inter1(vec), inter2(vec);
    dg::blas2::symv( A, vec, inter1);
    dg::blas2::symv( B, vec, inter2);
    dg::blas1::axpby( 1., inter1, -1., inter2, vec);
    double error = dg::blas1::dot( vec, vec);
    std::cout << "Error is "<<error<<" (should be small)!\n";
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


    bool passed = true;
    thrust::host_vector<double> xs = dg::evaluate( dg::cooX2d, g); 
    thrust::host_vector<double> ys = dg::evaluate( dg::cooY2d, g); 
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

    dg::Grid2d gfine( -10, 10, -5, 5, nf, Nf*Nx, Nf*Ny);
    const thrust::host_vector<double> xfine = dg::evaluate( sinex, gfine);
    thrust::host_vector<double> xcoarseI = dg::evaluate( sinex, g);
    const thrust::host_vector<double> xcoarse = dg::evaluate( sinex, g);
    const thrust::host_vector<double> wfine = dg::create::weights( gfine);
    const thrust::host_vector<double> wcoarse = dg::create::weights( g);
    double coarseL2 = dg::blas2::dot( xcoarse, wcoarse, xcoarse);
    double fineL2 =   dg::blas2::dot( xfine, wfine, xfine);
    std::cout << "coarse L2 norm:       "<<coarseL2<<"\n";
    std::cout << "Fine L2 norm:         "<<fineL2<<" \n";

    Matrix f2c = dg::create::projection( g, gfine); 
    dg::blas2::symv( f2c, xfine, xcoarseI);
    coarseL2 = dg::blas2::dot( xcoarseI, wcoarse, xcoarseI);
    std::cout << "interpolated L2 norm: "<<coarseL2<<"\n";
    std::cout << "Difference in L2      "<<fabs(fineL2-coarseL2)/fabs(fineL2)<<"\n";
    //integrals
    double coarseI = dg::blas1::dot( wcoarse, xcoarse);
    double fineI = dg::blas1::dot( wfine, xfine);
    std::cout << "coarse integral:      "<<coarseI<<"\n";
    std::cout << "Fine integral:        "<<fineI<<" \n";
    coarseI = dg::blas1::dot( wcoarse, xcoarseI);
    std::cout << "interpolated integral "<<coarseI<<"\n";
    std::cout << "Difference Integral   "<<fabs(fineI-coarseI)/fabs(fineI)<<"\n";
    dg::blas1::axpby( 1., xcoarseI, -1., xcoarse, xcoarseI);
    double norm = dg::blas2::dot( xcoarseI, wcoarse, xcoarseI);
    std::cout << "Difference evaluated to interpolated: "<<norm/coarseL2<<"\n";

    }
    ////////////////////////////////////////////////////////////////////////////
    {
    dg::Grid3d g( -10, 10, -5, 5, -7, -3, n, Nx, Ny, Nz);
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
    dg::Grid3d gfine( -10, 10, -5, 5, -7, -3,  nf, Nf*Nx, Nf*Ny, Nz);
    const thrust::host_vector<double> xfine = dg::evaluate( sinex, gfine);
    thrust::host_vector<double> xcoarseI = dg::evaluate( sinex, g);
    const thrust::host_vector<double> xcoarse = dg::evaluate( sinex, g);
    const thrust::host_vector<double> wfine = dg::create::weights( gfine);
    const thrust::host_vector<double> wcoarse = dg::create::weights( g);
    double coarseL2 = dg::blas2::dot( xcoarse, wcoarse, xcoarse);
    double fineL2 =   dg::blas2::dot( xfine, wfine, xfine);
    std::cout << "coarse L2 norm:       "<<coarseL2<<"\n";
    std::cout << "Fine L2 norm:         "<<fineL2<<" \n";

    Matrix f2c = dg::create::projection( g, gfine); 
    dg::blas2::symv( f2c, xfine, xcoarseI);
    coarseL2 = dg::blas2::dot( xcoarseI, wcoarse, xcoarseI);
    std::cout << "interpolated L2 norm: "<<coarseL2<<"\n";
    std::cout << "Difference in L2      "<<fabs(fineL2-coarseL2)/fabs(fineL2)<<"\n";
    //integrals
    double coarseI = dg::blas1::dot( wcoarse, xcoarse);
    double fineI = dg::blas1::dot( wfine, xfine);
    std::cout << "coarse integral:      "<<coarseI<<"\n";
    std::cout << "Fine integral:        "<<fineI<<" \n";
    coarseI = dg::blas1::dot( wcoarse, xcoarseI);
    std::cout << "interpolated integral "<<coarseI<<"\n";
    std::cout << "Difference Integral   "<<fabs(fineI-coarseI)/fabs(fineI)<<"\n";
    dg::blas1::axpby( 1., xcoarseI, -1., xcoarse, xcoarseI);
    double norm = dg::blas2::dot( xcoarseI, wcoarse, xcoarseI);
    std::cout << "Difference evaluated to interpolated: "<<norm/coarseL2<<"\n";
    }

    return 0;
}
