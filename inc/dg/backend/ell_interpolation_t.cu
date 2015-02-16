#include <cusp/print.h>
#include "xspacelib.cuh"
#include "ell_interpolation.cuh"
#include "typedefs.cuh"
const unsigned n = 3;
const unsigned Nx = 5; 
const unsigned Ny = 7; 
//const unsigned Nz = 2; 

double sinus( double x, double y) {return sin(x)*sin(y);}

int main()
{

    dg::Grid2d<double> g( -10, 10, -5, 5, n, Nx, Ny);
    thrust::host_vector<double> vector = dg::evaluate( sinus, g);
    dg::Matrix A = dg::create::backscatter( g);
    A.sort_by_row_and_column();

    thrust::host_vector<double> x( g.size()), y(x);
    for( unsigned i=0; i<g.Ny()*g.n(); i++)
        for( unsigned j=0; j<g.Nx()*g.n(); j++)
        {
            x[i*g.Nx()*g.n() + j] = 
                    g.x0() + (j+0.5)*g.hx()/(double)(g.n());
            y[i*g.Nx()*g.n() + j] = 
                    g.y0() + (i+0.5)*g.hy()/(double)(g.n());
        }
    thrust::device_vector<double> xd(x), yd(y);
    //cusp::ell_matrix<int, double, cusp::host_memory> dA = A;
    //cusp::coo_matrix<int, double, cusp::host_memory> C = dA;
    cusp::ell_matrix<int, double, cusp::device_memory> dB = dg::create::ell_interpolation( xd, yd, g);
    //cusp::coo_matrix<int, double, cusp::host_memory> B = dB;
    //cusp::print(C);
    //cusp::print(A);
    //bool passed = true;
    //for( unsigned i=0; i<C.values.size(); i++)
    //{
    //    if( (C.values[i] - B.values[i]) > 1e-14)
    //    {
    //        std::cerr << "NOT EQUCL "<<C.row_indices[i] <<" "<<C.column_indices[i]<<" "<<C.values[i] << "\t "<<B.row_indices[i]<<" "<<B.column_indices[i]<<" "<<B.values[i]<<"\n";
    //        passed = false;
    //    }
    //}
    //if( C.num_entries != B.num_entries)
    //{
    //    std::cerr << "Number of entries not equal!\n";
    //    passed = false;
    //}
    //if( passed)
    //    std::cout << "2D TEST PASSED!\n";
    dg::DVec dv( vector), w2( vector);
    dg::HVec w(vector);
    dg::blas2::symv( dB, dv, w2);
    dg::blas2::symv( A, vector, w);
    dg::blas1::axpby( 1., (dg::DVec)w, -1., w2, w2);
    std::cout << "Error is: "<<dg::blas1::dot( w2, w2)<<std::endl;
    
    return 0;
}
