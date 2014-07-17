#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include "mpi_vector.h"
#include "mpi_matrix.h"

#include "cg.h"

//leo3 can do 350 x 350 but not 375 x 375
const double ly = 2.*M_PI;

const double eps = 1e-6; //# of pcg iterations increases very much if 
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

const double lx = M_PI;
double fct(double x, double y){ return sin(y)*sin(x);}
double derivative( double x, double y){return cos(x)*sin(y);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
dg::bc bcx = dg::DIR;
double initial( double x, double y) {return sin(0);}

typedef dg::MPI_Vector MVec;
typedef dg::MPI_Matrix MMatrix;
typedef dg::MPI_Precon MPrecon;


int main()
{
    MPI_Init();
    dg::Timer t;
    unsigned n, Nx, Ny; 
    std::cout << "Type n, Nx and Ny\n";
    std::cin >> n >> Nx >> Ny;
    int np[2];
    int periods[2] = {0,0};
    if( bcx == PER) periods[0] = 1;
    if( bcy == PER) periods[1] = 1;
    std::cout << "Type npx and npy\n";
    std::cin >> np[0] >> np[1];

    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, comm);
    dg::MPIGrid2d<double> grid( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER, comm);
    const dg::MPrecon w2d = dg::create::weights( grid);
    const dg::MPrecon v2d = dg::create::precond( grid);
    std::cout<<"Expand initial condition\n";
    dg::MVec x = dg::evaluate( initial, grid);

    std::cout << "Create symmetric Laplacian\n";
    t.tic();
    dg::MMatrix A = dg::create::laplacianM( grid, dg::not_normed, dg::forward); 
    dg::MMatrix DX = dg::create::dx( grid);
    t.toc();
    std::cout<< "Creation took "<<t.diff()<<"s\n";

    dg::CG< dg::MVec > pcg( x, n*n*Nx*Ny);
    dg::CG< dg::MVec > pcg_host( x, n*n*Nx*Ny);

    std::cout<<"Expand right hand side\n";
    const dg::MVec solution = dg::evaluate ( fct, grid);
    const dg::MVec deriv = dg::evaluate( derivative, grid);
    dg::MVec b = dg::evaluate ( laplace_fct, grid);
    //compute S b
    dg::blas2::symv( w2d, b, b);

    //copy data to device memory
    t.tic();
    dg::MVec dsolution( solution);
    dg::MVec db( b), dx( x);
    dg::MVec db_(b), dx_(x);
    dg::MVec b_(b), x_(x);
    t.toc();
    std::cout << "Allocation and copy to device "<<t.diff()<<"s\n";
    //////////////////////////////////////////////////////////////////////
    std::cout << "# of polynomial coefficients: "<< n <<std::endl;
    std::cout << "# of 2d cells                 "<< Nx*Ny <<std::endl;
    
    t.tic();
    std::cout << "Number of pcg iterations "<< pcg( A, dx, db, t2d_d, eps)<<std::endl;
    t.toc();
    std::cout << "... for a precision of "<< eps<<std::endl;
    std::cout << "... on the device took "<< t.diff()<<"s\n";
    t.tic();
    dg::cg( dA, dx_, db_, t2d_d, eps, dx_.size());
    t.toc();
    std::cout << "... with function took "<< t.diff()<<"s\n";

    dg::MVec derror( dsolution);
    dg::MVec  error(  solution);
    dg::blas1::axpby( 1.,dx,-1.,derror);
    dg::blas1::axpby( 1., x,-1., error);

    double normerr = dg::blas2::dot( w2d, derror);
    double norm = dg::blas2::dot( w2d, dsolution);
    std::cout << "L2 Norm of relative error is:               " <<sqrt( normerr/norm)<<std::endl;
    dg::blas2::gemv( DX, dsolution, derror);
    dg::blas1::axpby( 1., deriv, -1., derror);
    normerr = dg::blas2::dot( s2d_d, derror); 
    norm = dg::blas2::dot( s2d_d, deriv);
    std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;
    //both functiona and derivative converge with order P 

    return 0;
}
