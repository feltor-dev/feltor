#include <iostream>
#include <iomanip>

#include "dg.h"
#include "evaluation.cuh"

#include "../toefl/timer.h"


const double lx = M_PI;
const double ly = M_PI;

double initial( double x, double y) {return 0.;}
double amp = 1;
double pol( double x, double y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
double sol(double x, double y)  { return sin( x)*sin(y);}

int main()
{
    toefl::Timer t;
    unsigned Nx, Ny;
    double eps;
    std::cout << "Type Nx and Ny and epsilon! \n";
    std::cin >> Nx >> Ny >> eps; 

    //create functions A(chi) x = b
    dg::Grid<double> grid( 0, lx, 0, ly, 1, Nx, Ny, dg::DIR, dg::DIR);
    dg_container x =    dg::evaluate( initial, grid);
    const dg_container b =    dg::evaluate( rhs, grid);
    const dg_container chi =  dg::evaluate( pol, grid);
    const dg_container solution = dg::evaluate( sol, grid);
    dg_container error( solution);

    std::cout << "# of 2d cells                 "<< Nx*Ny <<"\n";
    std::cout << "Create Polarization object!\n";
    t.tic();
    //create a workspace
    dg_workspace* w = dg_create_workspace( Nx, Ny, lx/Nx, ly/Ny, dg::DIR, dg::DIR);
    t.toc();
    std::cout << "... took "<<t.diff()<<"s\n";
    std::cout << "Update Polarization matrix!\n";
    t.tic();
    //assemble the polarization matrix
    dg_update_polarizability( w, chi.data());
    t.toc();
    std::cout << "... took "<<t.diff()<<"s\n";
    t.tic();
    //solve
    unsigned number = dg_solve( w, x.data(), b.data(), eps);
    if( number >= Nx*Ny)
    {
        std::cerr << "Failed to converge\n";
        return -1;
    }
    t.toc();
    std::cout << "Number of pcg iterations "<<number<<"\n";
    std::cout << "Solution took "<<t.diff()<<"s\n";
    std::cout << "For a precision of "<< eps<<"\n";
    //compute errors
    dg::blas1::axpby( 1.,x,-1., error);
    double eps_ = grid.hx()*grid.hy()*dg::blas1::dot( error, error);
    std::cout << "L2 Norm2 of Error is " << eps << "\n";
    double norm = grid.hx()*grid.hy()*dg::blas1::dot( solution, solution);
    std::cout << "L2 Norm of relative error is "<<sqrt( eps_/norm)<<std::endl;
    dg_free_workspace( w);

    return 0;
}

