#include <iostream>
#include <iomanip>
#include <mpi.h>


#include "elliptic.h"
#include "pcg.h"
#include "multigrid.h"

#include "backend/timer.h"
#include "backend/mpi_init.h"

//
//global relative error in L2 norm is O(h^P)
//as a rule of thumb with n=4 the true error is err = 1e-3 * eps as long as eps > 1e3*err
//using value_type = float;
//using Vector = dg::fMDVec;
//using Matrix = dg::fMDMatrix;
using value_type = double;
using Vector = dg::MDVec;
using Matrix = dg::MDMatrix;

const value_type lx = M_PI;
const value_type ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;

value_type initial( value_type x, value_type y) {return 0.;}
value_type amp = 0.9999;
value_type pol( value_type x, value_type y) {return 1. + amp*sin(x)*sin(y); } //must be strictly positive
//value_type pol( value_type x, value_type y) {return 1.; }
//value_type pol( value_type x, value_type y) {return 1. + sin(x)*sin(y) + x; } //must be strictly positive

value_type rhs( value_type x, value_type y) { return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);}
//value_type rhs( value_type x, value_type y) { return 2.*sin( x)*sin(y);}
//value_type rhs( value_type x, value_type y) { return 2.*sin(x)*sin(y)*(sin(x)*sin(y)+1)-sin(x)*sin(x)*cos(y)*cos(y)-cos(x)*cos(x)*sin(y)*sin(y)+(x*sin(x)-cos(x))*sin(y) + x*sin(x)*sin(y);}
value_type sol(value_type x, value_type y)  { return sin( x)*sin(y);}
value_type derX(value_type x, value_type y)  { return cos( x)*sin(y);}
value_type derY(value_type x, value_type y)  { return sin( x)*cos(y);}
value_type vari(value_type x, value_type y)  { return pol(x,y)*pol(x,y)*(derX(x,y)*derX(x,y) + derY(x,y)*derY(x,y));}


int main(int argc, char* argv[] )
{
    /////////////////////MPI INIT//////////////////////////
    MPI_Init( &argc, &argv);
    unsigned n, Nx, Ny;
    MPI_Comm comm;
    dg::mpi_init2d( bcx, bcy, n, Nx, Ny, comm);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    dg::Timer t;
    value_type eps = 1e-6;

    //if(rank==0)std::cout << "Type epsilon! \n";
    //if(rank==0)std::cin >> eps;
    MPI_Bcast(  &eps,1 , dg::getMPIDataType<value_type>(), 0, comm);
    //////////////////////begin program///////////////////////
    //create functions A(chi) x = b
    dg::RealCartesianMPIGrid2d<value_type> grid( 0., lx, 0, ly, n, Nx, Ny, bcx, bcy, comm);
    const Vector w2d = dg::create::weights( grid);
    const Vector v2d = dg::create::inv_weights( grid);
    Vector x =    dg::evaluate( initial, grid);
    Vector b =    dg::evaluate( rhs, grid);
    Vector chi =  dg::evaluate( pol, grid);



    if(rank==0)std::cout <<rank<< "Create Polarisation object and set chi!\n";
    t.tic();
    unsigned stages = 3;

    dg::MultigridCG2d<dg::aRealMPIGeometry2d<value_type>, Matrix, Vector > multigrid( grid, stages, 0);

    std::vector<Vector> chi_ = multigrid.project( chi);
    std::vector<dg::Elliptic<dg::aRealMPIGeometry2d<value_type>, Matrix, Vector> > multi_pol( stages);

    for(unsigned u=0; u<stages; u++)
    {
        multi_pol[u].construct( multigrid.grid(u), dg::centered);
        multi_pol[u].set_chi( chi_[u]);
    }
    t.toc();
    if(rank==0)std::cout << "Creation of polarisation object took: "<<t.diff()<<"s\n";

    t.tic();
    std::vector<unsigned> number = multigrid.solve( multi_pol, x, b, eps);
    t.toc();
    for( unsigned u=0; u<number.size(); u++)
    	if(rank==0)std::cout << " # iterations stage "<< number.size()-1-u << " " << number[number.size()-1-u] << " \n";
    if(rank==0)std::cout << "For a precision of "<< eps<<std::endl;
    if(rank==0)std::cout << " took "<<t.diff()<<"s\n";

    //compute error
    const Vector solution = dg::evaluate( sol, grid);
    const Vector derivati = dg::evaluate( derX, grid);
    const Vector variatio = dg::evaluate( vari, grid);
    Vector error( solution);
    dg::blas1::axpby( 1.,x,-1., error);

    value_type err = dg::blas2::dot( w2d, error);
    value_type norm = dg::blas2::dot( w2d, solution);
    if(rank==0)std::cout << "L2 Norm of relative error is               "<<sqrt( err/norm)<<std::endl;
    Matrix DX = dg::create::dx( grid);
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1.,derivati,-1., error);
    err = dg::blas2::dot( w2d, error);
    norm = dg::blas2::dot( w2d, derivati);
    if(rank==0)std::cout << "L2 Norm of relative error in derivative is "<<sqrt( err/norm)<<std::endl;
    //derivative converges with p-1, for p = 1 with 1/2
    if(rank==0)std::cout << "Compute variation in forward Elliptic      ";
    multi_pol[0].variation( 1., chi, x, 0., error);
    dg::blas1::axpby( 1., variatio, -1., error);
    err = dg::blas2::dot( w2d, error);
    norm = dg::blas2::dot( w2d, variatio);
    if(rank==0)std::cout <<sqrt( err/norm) << "\n";

    MPI_Finalize();
    return 0;
}

