#include <iostream>
#include <iomanip>

#include <mpi.h>

#include "pcg.h"
#include "elliptic.h"

#include "backend/timer.h"
#include "backend/mpi_init.h"
#include "topology/split_and_join.h"

//using value_type = float;
//using Matrix = dg::fMDMatrix;
//using Vector = dg::fMDVec;
//using LVector = dg::fDVec;
using value_type = double;
using Matrix = dg::MDMatrix;
using Vector = dg::MDVec;
using LVector = dg::DVec;

const value_type R_0 = 10;
const value_type lx = 2.*M_PI;
const value_type ly = 2.*M_PI;
const value_type lz = 2.*M_PI;
value_type fct(value_type x, value_type y, value_type z){ return sin(x-R_0)*sin(y)*sin(z);}
value_type fctX( value_type x, value_type y, value_type z){return cos(x-R_0)*sin(y)*sin(z);}
value_type fctY(value_type x, value_type y, value_type z){ return sin(x-R_0)*cos(y)*sin(z);}
value_type fctZ(value_type x, value_type y, value_type z){ return sin(x-R_0)*sin(y)*cos(z);}
value_type laplace2d_fct( value_type x, value_type y, value_type z) { return -1./x*cos(x-R_0)*sin(y)*sin(z) + 2.*fct(x,y,z);}
value_type laplace3d_fct( value_type x, value_type y, value_type z) { return -1./x*cos(x-R_0)*sin(y)*sin(z) + 2.*fct(x,y,z) + 1./x/x*fct(x,y,z);}
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::DIR;
dg::bc bcz = dg::PER;
value_type initial( value_type x, value_type y, value_type z) {return sin(0);}
value_type variation3d( value_type x, value_type y, value_type z) {
    return (fctX(x,y,z)*fctX(x,y,z)
        + fctY(x,y,z)*fctY(x,y,z)
        + fctZ(x,y,z)*fctZ(x,y,z)/x/x)*fct(x,y,z)*fct(x,y,z);
}


int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    unsigned n, Nx, Ny, Nz;
    MPI_Comm comm;
    dg::mpi_init3d( bcx, bcy, dg::PER, n, Nx, Ny, Nz, comm);

    dg::RealCylindricalMPIGrid3d<value_type> grid( R_0, R_0+lx, 0, ly, 0,lz, n, Nx, Ny,Nz, bcx, bcy, dg::PER, comm);
    const Vector w3d = dg::create::volume( grid);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    value_type eps=1e-6;
    if(rank==0)std::cout << "Type epsilon! \n";
    if(rank==0)std::cin >> eps;
    MPI_Bcast(  &eps,1 , dg::getMPIDataType<value_type>(), 0, comm);
    /////////////////////////////////////////////////////////////////
    if(rank==0)std::cout<<"TEST CYLINDRIAL LAPLACIAN!\n";
    dg::Timer t;
    Vector x = dg::evaluate( initial, grid);

    if(rank==0)std::cout << "Create Laplacian\n";
    t.tic();
    dg::Elliptic3d<dg::aRealMPIGeometry3d<value_type>, Matrix, Vector> laplace(grid, dg::centered);
    Matrix DX = dg::create::dx( grid);
    t.toc();
    if(rank==0)std::cout<< "Creation took "<<t.diff()<<"s\n";

    dg::PCG< Vector > pcg( x, n*n*Nx*Ny);

    if(rank==0)std::cout<<"Expand right hand side\n";
    const Vector solution = dg::evaluate ( fct, grid);
    const Vector deriv = dg::evaluate( fctX, grid);
    Vector b = dg::evaluate ( laplace3d_fct, grid);

    if(rank==0)std::cout << "For a precision of "<< eps<<" ..."<<std::endl;
    t.tic();
    unsigned num = pcg.solve( laplace, x, b, 1., w3d, eps);
    t.toc();
    if(rank==0)std::cout << "Number of pcg iterations "<< num<<std::endl;
    if(rank==0)std::cout << "... took                 "<< t.diff()<<"s\n";
    Vector  error(  solution);
    dg::blas1::axpby( 1., x,-1., error);

    value_type normerr = dg::blas2::dot( w3d, error);
    value_type norm = dg::blas2::dot( w3d, solution);
    dg::exblas::udouble res;
    norm = sqrt(normerr/norm); res.d = norm;
    if(rank==0)std::cout << "L2 Norm of relative error is:               " <<res.d<<"\t"<<res.i<<std::endl;
    dg::blas2::gemv( DX, x, error);
    dg::blas1::axpby( 1., deriv, -1., error);
    normerr = dg::blas2::dot( w3d, error);
    norm = dg::blas2::dot( w3d, deriv);
    if(rank==0)std::cout << "L2 Norm of relative error in derivative is: " <<sqrt( normerr/norm)<<std::endl;
    if(rank==0)std::cout << "Compute variation in Elliptic               ";
    const Vector variatio = dg::evaluate ( variation3d, grid);
    laplace.variation( solution, x, error);
    dg::blas1::axpby( 1., variatio, -1., error);
    norm = dg::blas2::dot( w3d, variatio);
    normerr = dg::blas2::dot( w3d, error);
    if(rank==0)std::cout <<sqrt( normerr/norm) << "\n";

    if(rank==0)std::cout << "TEST SPLIT SOLUTION\n";
    x = dg::evaluate( initial, grid);
    b = dg::evaluate ( laplace2d_fct, grid);
    //create grid and perp and parallel volume
    dg::ClonePtr<dg::aRealMPIGeometry2d<value_type>> grid_perp = grid.perp_grid();
    const Vector w2d = dg::create::volume( *grid_perp);
    Vector g_parallel = grid.metric().value(2,2);
    dg::blas1::transform( g_parallel, g_parallel, dg::SQRT<>());
    Vector chi = dg::evaluate( dg::one, grid);
    dg::blas1::pointwiseDivide( chi, g_parallel, chi);
    //create split Laplacian
    std::vector< dg::Elliptic<dg::aRealMPIGeometry2d<value_type>, Matrix, Vector> > laplace_split(
            grid.local().Nz(), dg::Elliptic<dg::aRealMPIGeometry2d<value_type>, Matrix, Vector>(*grid_perp, dg::centered));
    // create split  vectors and solve
    std::vector<dg::MPI_Vector<dg::View<LVector>>> b_split, x_split, chi_split;
    pcg.construct( w2d, w2d.size());
    std::vector<unsigned>  number(grid.local().Nz());
    t.tic();
    dg::blas1::pointwiseDivide( b, g_parallel, b);
    b_split = dg::split( b, grid);
    chi_split = dg::split( chi, grid);
    x_split = dg::split( x, grid);
    for( unsigned i=0; i<grid.local().Nz(); i++)
    {
        laplace_split[i].set_chi( chi_split[i]);
        number[i] = pcg.solve( laplace_split[i], x_split[i], b_split[i], 1., w2d, eps);
    }
    t.toc();
    if(rank==0)std::cout << "Number of iterations in split     "<< number[0]<<"\n";
    if(rank==0)std::cout << "Split solution on the device took "<< t.diff()<<"s\n";
    dg::blas1::axpby( 1., x,-1., solution, error);
    normerr = dg::blas2::dot( w3d, error);
    norm = dg::blas2::dot( w3d, solution);
    if(rank==0)std::cout << "L2 Norm of relative error is:     " <<sqrt( normerr/norm)<<std::endl;
    //both function and derivative converge with order P

    MPI_Finalize();
    return 0;
}
