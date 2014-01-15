#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "evaluation.cuh"
#include "cg.cuh"
#include "laplace.cuh"
#include "preconditioner.cuh"
#include "typedefs.cuh"

unsigned n = 3; //global relative error in L2 norm is O(h^P)
unsigned N = 200;  //more N means less iterations for same error

const double lx = 2.*M_PI;

const double eps = 1e-7; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus


typedef dg::T1D<double> Preconditioner;


double sine(double x){ return sin( x);}
double initial( double x) {return sin(0);}
dg::bc bcx = dg::PER;

using namespace std;
int main()
{
    cout << "Type n and N\n";
    cin >> n >> N;
    dg::Grid1d<double > g( 0, lx, n, N, bcx);
    dg::HVec x = dg::expand( initial, g);
    dg::DMatrix A = dg::create::laplace1d( g); 

    dg::CG< dg::DVec > cg( x, x.size());
    dg::HVec b = dg::expand ( sine, g);
    dg::HVec error(b);
    const dg::HVec solution(b);

    //copy data to device memory
    dg::DVec dx( x), db( b), derror( error);
    const dg::DVec dsolution( solution);

    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of intervals                "<< N <<endl;
    //compute S b
    dg::blas2::symv( dg::S1D<double>(g), db, db);
    std::cout << "Number of pcg iterations "<< cg( A, dx, db, Preconditioner(g), eps)<<endl;
    std::cout << "Number of cg iterations "<< cg( A, dx, db, eps)<<endl;
    cout << "For a precision of "<< eps<<endl;
    //compute error
    dg::blas1::axpby( 1.,dx,-1.,derror);
    /*
    //and Ax
    DArrVec dbx(dx);
    dg::blas2::symv(  A, dx.data(), dbx.data());

    cout<< dx <<endl;
    */

    double eps = dg::blas2::dot( dg::S1D<double>(g), derror);
    cout << "L2 Norm2 of Error is " << eps << endl;
    double norm = dg::blas2::dot( dg::S1D<double>(g), dsolution);
    std::cout << "L2 Norm of relative error is "<<sqrt( eps/norm)<<std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)



    return 0;
}
