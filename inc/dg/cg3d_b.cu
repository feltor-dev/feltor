#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "timer.cuh"
#include "evaluation.cuh"
#include "cg.h"
#include "derivatives.cuh"

#include "typedefs.cuh"


//leo3 can do 350 x 350 but not 375 x 375
const double ly = 2.*M_PI;

const double eps = 1e-6; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

const double lx = 2.*M_PI;
const double lz = 1.;
double fct(double x, double y){ return sin(y)*sin(x);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
dg::bc bcx = dg::DIR;
//const double lx = 2./3.*M_PI;
//double fct(double x, double y){ return sin(y)*sin(3.*x/4.);}
//double laplace_fct( double x, double y) { return 25./16.*sin(y)*sin(3.*x/4.);}
//dg::bc bcx = dg::DIR_NEU;
double initial( double x, double y) {return sin(0);}

double fct(double x, double y, double z){ return sin(y)*sin(x);}
double laplace_fct( double x, double y, double z) { return 2*sin(y)*sin(x);}
//const double lx = 2./3.*M_PI;
//double fct(double x, double y){ return sin(y)*sin(3.*x/4.);}
//double laplace_fct( double x, double y) { return 25./16.*sin(y)*sin(3.*x/4.);}
//dg::bc bcx = dg::DIR_NEU;
double initial( double x, double y, double z) {return sin(0);}

using namespace std;

int main()
{
    dg::Timer t;
    unsigned n, Nx, Ny, Nz; 
    std::cout << "Type n, Nx, Ny and Nz\n";
    std::cin >> n >> Nx >> Ny>> Nz;

    cout << "TEST 3D VERSION\n";
    dg::Grid3d<double> g3d( 0, lx, 0, ly, 0, lz, n, Nx, Ny, Nz, bcx, dg::PER);
    dg::HVec w3d = dg::create::w3d( g3d);
    dg::HVec v3d = dg::create::v3d( g3d);
    dg::HVec x3 = dg::evaluate( initial, g3d);
    dg::HVec b3 = dg::evaluate ( laplace_fct, g3d);
    dg::blas2::symv( w3d, b3, b3);

    dg::DVec w3d_d(w3d), v3d_d(v3d), x3_d(x3), b3_d(b3);

    dg::HMatrix A3 = dg::create::laplacianM_perp( g3d, dg::not_normed); 
    dg::DMatrix A3_d(A3);
    dg::CG<dg::HVec > pcg3_host( x3, g3d.size());
    dg::CG<dg::DVec > pcg3_d( x3_d, g3d.size());
    t.tic();
    cout << "Number of pcg iterations "<< pcg3_d( A3_d, x3_d, b3_d, v3d_d, eps)<<endl;
    t.toc();
    cout << "... for a precision of "<< eps<<endl;
    cout << "... on the device took "<< t.diff()<<"s\n";
    t.tic();
    cout << "Number of pcg iterations "<< pcg3_host( A3, x3, b3, v3d, eps)<<endl;
    t.toc();
    cout << "... for a precision of "<< eps<<endl;
    cout << "... on the host took   "<< t.diff()<<"s\n";
    //compute error
    const dg::HVec solution3 = dg::evaluate ( fct, g3d);
    dg::HVec error3( solution3);
    dg::blas1::axpby( 1.,x3,-1.,error3);

    double eps3 = dg::blas2::dot(w3d , error3);
    double norm3 = dg::blas2::dot(w3d , solution3);
    //cout << "L2 Norm2 of Solution is        " << norm3 << endl;
    cout << "L2 Norm of relative error is:  " <<sqrt( eps3/norm3)<<endl;
    /////////////////STD_VECTOR MATRIX//////////////////////////
    //dg::Grid2d<double> g2d( 0, lx, 0, ly, n, Nx, Ny, bcx, dg::PER);
    //dg::DVec w2d = dg::create::w2d( g2d), v2d( dg::create::v2d(g2d));
    //dg::DMatrix A2  =dg::create::laplacianM( g2d, dg::not_normed);
    //dg::DVec x2 = dg::evaluate( initial, g2d);
    //dg::DVec b2 = dg::evaluate( laplace_fct, g2d);
    //dg::blas2::symv( w2d, b2, b2);
    //std::vector<dg::DVec> v2_( g3d.Nz(), v2d);
    //std::vector<dg::DVec> b2_( g3d.Nz(), b2);
    //std::vector<dg::DVec> x2_( g3d.Nz(), x2);
    //dg::CG<std::vector<dg::DVec> > pcg2( x2_, g2d.size());
    //t.tic();
    //cout << "Number of pcg iterations "<< pcg2( A2, x2_, b2_, v2_, eps)<<endl;
    //t.toc();
    //cout << "... for a precision of "<< eps<<endl;
    //cout << "... on the device took "<< t.diff()<<"s\n";


    return 0;
}
