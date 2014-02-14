#include <iostream>
#include <iomanip>

#include "evaluation.cuh"

#include "dg.h"



const unsigned n = 1; //global relative error in L2 norm is O(h^P)
const unsigned Nx = 30;  //more N means less iterations for same error
const unsigned Ny = 40;  //more N means less iterations for same error

const double lx = M_PI;
const double ly = M_PI;
const double eps = 1e-3; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double initial( double x, double y) {return 0.;}
double pol( double x, double y) {return 1. + sin(x)*sin(y); } //must be strictly positive
double sol(double x, double y)  { return sin( x)*sin(y);}

double rhs( double x, double y) { return 4.*sol(x,y)*sol(x,y) + 2.*sol(x,y);}

using namespace std;

int main()
{
    dg::Grid<double> grid( 0, lx, 0, ly, n, Nx, Ny, dg::DIR, dg::DIR);
    //create functions A(chi) x = b
    dg_container x =    dg::evaluate( initial, grid);
    dg_container b =    dg::evaluate( rhs, grid);
    const dg_container chi =  dg::evaluate( pol, grid);
    const dg_container solution = dg::evaluate( sol, grid);
    dg_container error( solution);

    cout << "Create Polarization object!\n";
    dg_workspace* w = dg_create_workspace( Nx, Ny, lx/Nx, ly/Ny, dg::DIR, dg::DIR);
    cout << "Update Polarization matrix!\n";
    dg_update_polarizability( w, chi.data());
    cout << "# of 2d cells                 "<< Nx*Ny <<endl;
    std::cout << "Number of pcg iterations "<< dg_solve( w, x.data(), b.data(), eps)<<endl;
    cout << "For a precision of "<< eps<<endl;
    //compute error
    dg::blas1::axpby( 1.,x,-1., error);

    double eps = 1./grid.hx()/grid.hy()*dg::blas1::dot( error, error);
    cout << "L2 Norm2 of Error is " << eps << endl;
    double norm = 1./grid.hx()/grid.hy()*dg::blas1::dot( solution, solution);
    std::cout << "L2 Norm of relative error is "<<sqrt( eps/norm)<<std::endl;

    return 0;
}

