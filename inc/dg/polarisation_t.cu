#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cusp/print.h>

#include "polarisation.cuh"
#include "evaluation.cuh"
#include "cg.cuh"
#include "laplace.cuh"
#include "preconditioner.cuh"
#include "functions.h"

//const unsigned n = 3; //global relative error in L2 norm is O(h^P)
//const unsigned N = 100;  //more N means less iterations for same error

const double lx = M_PI;
//const double eps = 1e-1; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;

typedef dg::T1D<double> Preconditioner;

typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;
typedef cusp::device_memory Memory;

double initial( double x) {return sin(0);}
//double pol( double x) {return 1. + sin(x); } //must be strictly positive
//double pol( double x) {return 1.; }
double grad = 1.000;
double pol( double x) {return 1. + sin(x) + grad*x; } //must be strictly positive

//double rhs( double x) { return sin(x) + 1.-2.*cos(x)*cos(x);}
//double rhs( double x) { return sin( x);}
double rhs( double x) { return -(cos(x)+grad)*cos(x)+(1.+sin(x)+grad*x)*sin(x);}
//solution to -\d_x ( \pol(x)*\d_x \phi) = \rhs
double sol(double x)  { return sin( x);}

using namespace std;

int main()
{
    unsigned n, N;
    double eps;
    std::cout << "Write n N and eps!\n";
    std::cin>> n >> N >> eps;

    //create functions A(chi) x = b
    dg::Grid1d<double> g( 0, lx, n, N, dg::DIR);
    HVec x = dg::expand ( initial, g);
    HVec b = dg::expand ( rhs, g);
    HVec chi = dg::expand( pol,g);
    const HVec solution = dg::expand (sol, g);
    HVec error(solution);

    //copy data to device memory
    DVec dx( x), db( b), derror( error), dchi( chi);
    const DVec dsolution( solution);
    cusp::array1d_view<DVec::iterator> dchi_view( dchi.begin(), dchi.end());

    cout << "Create Polarisation object!\n";
    dg::Polarisation<double, Memory> pol( g);
    cout << "Create Polarisation matrix!\n";
    cusp::coo_matrix<int, double, cusp::device_memory> A_ = pol.create( dchi_view ); 
    DMatrix A = A_;
    //DMatrix B = dg::create::laplace1d_dir<double, n>( N, h); 
    cout << "A is sorted?"<<A_.is_sorted_by_row_and_column()<<endl;
    cout << "Create conjugate gradient!\n";
    dg::CG< DVec > pcg( dx, n*N);

    cout << "# of polynomial coefficients: "<< n <<endl;
    cout << "# of intervals                "<< N <<endl;
    //compute S b
    dg::blas2::symv( dg::S1D<double>(g), db, db);
    std::cout << "Number of pcg iterations "<< pcg( A, dx, db, Preconditioner(g), eps)<<endl;
    cout << "For a precision of "<< eps<<endl;
    //compute error
    dg::blas1::axpby( 1.,dx,-1.,derror);

    double epsl = dg::blas2::dot( dg::S1D<double>(g), derror);
    cout << "L2 Norm2 of Error is " << epsl << endl;
    double norm = dg::blas2::dot( dg::S1D<double>(g), dsolution);
    std::cout << "L2 Norm of relative error is "<<sqrt( epsl/norm)<<std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)

    return 0;
}

