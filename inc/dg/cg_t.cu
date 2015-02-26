#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "backend/evaluation.cuh"
#include "cg.h"
#include "backend/dxx.cuh"
#include "backend/typedefs.cuh"


const double lx = 2.*M_PI;

const double eps = 1e-7; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double sine(double x){ return sin( x);}
double sol(double x){ return sin(x);}
dg::bc bcx = dg::DIR;
//double sine(double x){ return 9./16.*sin( 3./4.*x);}
//double sol(double x){ return sin(3./4.*x);}
//dg::bc bcx = dg::DIR_NEU;
//double sine(double x){ return 9./16.*cos( 3./4.*x);}
//double sol(double x){ return cos(3./4.*x);}
//dg::bc bcx = dg::NEU_DIR;
double initial( double x) {return sin(0);}

using namespace std;
int main()
{
    unsigned n; //global relative error in L2 norm is O(h^P)
    unsigned N;  //more N means less iterations for same error
    cout << "Type n and N\n";
    cin >> n >> N;
    dg::Grid1d<double > g( 0, lx, n, N, bcx);
    dg::DVec x = dg::evaluate( initial, g);
    dg::DVec w1d = dg::create::weights( g);
    dg::DVec v1d = dg::create::inv_weights( g);
    dg::DMatrix A = dg::create::laplace1d( g,bcx, dg::not_normed, dg::centered); 

    dg::CG< dg::DVec > cg( x, x.size());
    dg::DVec b = dg::evaluate ( sine, g);
    dg::DVec error(b);
    const dg::DVec solution = dg::evaluate( sol, g);

    //copy data to device memory
    std::cout << "# of polynomial coefficients: "<< n <<std::endl;
    std::cout << "# of intervals                "<< N <<std::endl;
    //compute W b
    dg::blas2::symv( w1d, b, b);
    std::cout << "Number of pcg iterations "<< cg( A, x, b, v1d, eps)<<std::endl;
    std::cout << "For a precision of "<< eps<<std::endl;
    //compute error
    dg::blas1::axpby( 1.,x,-1., solution, error);

    double eps = dg::blas2::dot(w1d, error);
    std::cout << "L2 Norm of Error is " << sqrt(eps) << std::endl;
    double norm = dg::blas2::dot( w1d, solution);
    std::cout << "L2 Norm of relative error is "<<sqrt( eps/norm)<<std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)

    return 0;
}
