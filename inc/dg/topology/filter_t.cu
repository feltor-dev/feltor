#include <iostream>

#include "dg/blas.h"
#include "dg/functors.h"

#include "interpolation.h"
#include "evaluation.h"
#include "filter.h"

double function( double x, double y){return sin(x)*sin(y);}
double function( double x, double y, double z){return sin(x)*sin(y)*sin(z);}

const unsigned Nx = 8, Ny = 10, Nz = 6;

int main()
{

    std::vector<double> x = {7,2,3, -20,4,7, 2.3,2.1,4, 4,15,-2};
    std::vector<double> medians = {7,4.5,3,2.5,3,3.5,3,2.65,3,3.5,4,3.5};
    std::vector<double> mad = {0,2.5,1,2.5,1,2.5,1,1,1,1.3,1.9,1.45};
    int row_offsets[2] = {0, 3};
    std::vector<int> column_indices = {0,1,2,3,4,5,6,7,8,9,10,11,12};
    double result;
    dg::CSRMedianFilter median_filter;
    for( unsigned i=1; i<x.size()+1; i+=2)
    {
        row_offsets[1] = i;
        median_filter( 0, row_offsets, &column_indices[0], &x[0], &x[0], &result);
        std::cout << "Median of [";
        int k=0;
        for( k=0; k<row_offsets[1]-1; k++)
            std::cout << x[k]<<", ";
        std::cout<<x[k]<<"] is "<<result<<" ("<<medians[i-1]<<")"<<std::endl;
    }
    //We test if the Modal Filter on 3 polynomials has the same effect as the
    //projection matrix
    {
    std::cout << "Test 2d exponential filter: \n";
    dg::Grid2d g3( -M_PI, 0, -5*M_PI, -4*M_PI, 3, Nx, Ny);
    dg::Grid2d g2( -M_PI, 0, -5*M_PI, -4*M_PI, 2, Nx, Ny);

    const dg::DVec vec = dg::evaluate( function, g3);
    const dg::DVec weights = dg::create::weights( g3);
    dg::DVec filtered_vec(vec), projected_vec(dg::evaluate( dg::zero, g2)), inter_vec( vec);
    dg::ModalFilter<dg::DMatrix, dg::DVec> filter( dg::ExponentialFilter(36, 0.5, 8, g3.nx()), g3);
    dg::IDMatrix project = dg::create::projection( g2,g3);
    dg::IDMatrix interpo = dg::create::interpolation( g3,g2);

    dg::blas2::symv( project, vec, projected_vec);
    dg::blas2::symv( interpo, projected_vec, inter_vec);
    filter( vec, filtered_vec);
    dg::blas1::axpby( 1., filtered_vec, -1., inter_vec);
    double error = sqrt(dg::blas2::dot( inter_vec, weights, inter_vec)/ dg::blas2::dot( vec, weights, vec));
    std::cout << "Error by filtering: "<<error<<std::endl;

    if( error > 1e-14)
        std::cout << "2D TEST FAILED!\n";
    else
        std::cout << "2D TEST PASSED!\n";
    }
    {
    std::cout << "Test 3d exponential filter: \n";
    dg::Grid3d g3( -M_PI, 0, -5*M_PI, -4*M_PI, 0., 2.*M_PI, 3, Nx, Ny, Nz);
    dg::Grid3d g2( -M_PI, 0, -5*M_PI, -4*M_PI, 0., 2.*M_PI, 2, Nx, Ny, Nz);

    const dg::DVec vec = dg::evaluate( function, g3);
    const dg::DVec weights = dg::create::weights( g3);
    dg::DVec filtered_vec(vec), projected_vec(dg::evaluate( dg::zero, g2)), inter_vec( vec);
    dg::ModalFilter<dg::DMatrix, dg::DVec> filter( dg::ExponentialFilter(36, 0.5, 8, g3.nx()), g3);
    dg::IDMatrix project = dg::create::projection( g2,g3);
    dg::IDMatrix interpo = dg::create::interpolation( g3,g2);

    dg::blas2::symv( project, vec, projected_vec);
    dg::blas2::symv( interpo, projected_vec, inter_vec);
    filter( vec, filtered_vec);
    dg::blas1::axpby( 1., filtered_vec, -1., inter_vec);
    double error = sqrt(dg::blas2::dot( inter_vec, weights, inter_vec)/ dg::blas2::dot( vec, weights, vec));
    std::cout << "Error by filtering: "<<error<<std::endl;

    if( error > 1e-14)
        std::cout << "3D TEST FAILED!\n";
    else
        std::cout << "3D TEST PASSED!\n";
    //Test recursive filter
    dg::ModalFilter<dg::DMatrix, std::vector<dg::DVec>> vec_filter( dg::ExponentialFilter(36, 0.5, 8, g3.nx()), g3, 3);
    const std::vector<dg::DVec> vec_vec ( 3, vec);
    std::vector<dg::DVec> filtered_vec_vec ( vec_vec);
    vec_filter( vec_vec, filtered_vec_vec);
    dg::blas1::axpby( 1., filtered_vec, -1., filtered_vec_vec[2]);
    error = sqrt(dg::blas2::dot( filtered_vec_vec[2], weights, filtered_vec_vec[2])/ dg::blas2::dot( filtered_vec, weights, filtered_vec));
    std::cout << "Error by filtering: "<<error<<std::endl;

    if( error > 1e-14)
        std::cout << "Vector TEST FAILED!\n";
    else
        std::cout << "Vector TEST PASSED!\n";
    }

    return 0;
}
