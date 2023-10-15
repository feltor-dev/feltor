#include <iostream>
#include <memory>
#include <mpi.h>

#include "dg/file/file.h"

#include "dg/algorithm.h"

#include "solovev.h"
//#include "guenter.h"
#include "mpi_curvilinear.h"
#include "simple_orthogonal.h"
#include "flux.h"
#include "testfunctors.h"

typedef  dg::geo::CurvilinearMPIGrid2d Geometry;

int main(int argc, char**argv)
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz;
    MPI_Comm comm;
    dg::mpi_init3d( dg::DIR, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( comm, 3, dims, periods, coords);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( dims[2] != 1)
    {
        // because of netcdf output
        if(rank==0) std::cout << "Please do not parallelize in z!\n";
        MPI_Finalize();
        return 0;
    }
    auto js = dg::file::file2Json( argc == 1 ? "geometry_params_Xpoint.json" : argv[1]);
    dg::geo::solovev::Parameters gp(js);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    if(rank==0)std::cout << "Psi min "<<mag.psip()(gp.R_0, 0)<<"\n";
    if(rank==0)std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    if(rank==0)std::cin >> psi_0>> psi_1;
    MPI_Bcast( &psi_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast( &psi_1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank==0)gp.display( std::cout);
    if(rank==0)std::cout << "Constructing grid ... \n";
    dg::Timer t;
    t.tic();
    //dg::geo::SimpleOrthogonal generator( mag.get_psip(), psi_0, psi_1, gp.R_0, 0., 1);
    dg::geo::FluxGenerator generator( mag.get_psip(), mag.get_ipol(), psi_0, psi_1, gp.R_0, 0., 1);
    dg::geo::CurvilinearProductMPIGrid3d g3d( generator, n, Nx, Ny,Nz, dg::DIR, dg::PER, dg::PER, comm);
    std::unique_ptr<dg::aMPIGeometry2d> g2d(g3d.perp_grid());
    dg::Elliptic<dg::aMPIGeometry2d, dg::MDMatrix, dg::MDVec> pol( *g2d, dg::forward);
    t.toc();
    if(rank==0)std::cout << "Construction took "<<t.diff()<<"s\n";
    ///////////////////////////////////////////////////////////////////////////
    int ncid;
    dg::file::NC_Error_Handle ncerr;
    if(rank==0)ncerr = nc_create( "testE_mpi.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    if(rank==0)ncerr = dg::file::define_dimensions(  ncid, dim2d, *g2d);
    int coordsID[2], psiID, functionID, function2ID;
    if(rank==0)ncerr = nc_def_var( ncid, "xc", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    if(rank==0)ncerr = nc_def_var( ncid, "yc", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    if(rank==0)ncerr = nc_def_var( ncid, "error", NC_DOUBLE, 2, dim2d, &psiID);
    if(rank==0)ncerr = nc_def_var( ncid, "num_solution", NC_DOUBLE, 2, dim2d, &functionID);
    if(rank==0)ncerr = nc_def_var( ncid, "ana_solution", NC_DOUBLE, 2, dim2d, &function2ID);

    dg::MHVec X( g2d->map()[0]), Y( g2d->map()[1]);
    dg::file::put_var_double( ncid, coordsID[0], *g2d, X);
    dg::file::put_var_double( ncid, coordsID[1], *g2d, Y);
    ///////////////////////////////////////////////////////////////////////////
    dg::MDVec x =    dg::evaluate( dg::zero, *g2d);
    const dg::MDVec b =    dg::pullback( dg::geo::EllipticDirPerM(mag, psi_0, psi_1, 4), *g2d);
    const dg::MDVec chi =  dg::pullback( dg::geo::Bmodule(mag), *g2d);
    const dg::MDVec solution = dg::pullback( dg::geo::FuncDirPer(mag, psi_0, psi_1, 4), *g2d);
    const dg::MDVec vol3d = dg::create::volume( *g2d);
    pol.set_chi( chi);
    //compute error
    dg::MDVec error( solution);
    const double eps = 1e-10;
    dg::PCG<dg::MDVec > pcg( x, n*n*Nx*Ny*Nz);
    if(rank==0)std::cout << "eps \t # iterations \t error \t hx_max\t hy_max \t time/iteration \n";
    if(rank==0)std::cout << eps<<"\t";
    t.tic();
    unsigned number = pcg.solve(pol, x,b, pol.precond(), pol.weights(), eps);
    if(rank==0)std::cout <<number<<"\t";
    t.toc();
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( vol3d, error);
    const double norm = dg::blas2::dot( vol3d, solution);
    if(rank==0)std::cout << sqrt( err/norm) << "\t";

    dg::SparseTensor<dg::MDVec> metric = g2d->metric();
    dg::MDVec gyy = metric.value(1,1), gxx=metric.value(0,0), volume = dg::tensor::volume(metric);
    dg::blas1::transform( gxx, gxx, dg::SQRT<double>());
    dg::blas1::transform( gyy, gyy, dg::SQRT<double>());
    dg::blas1::pointwiseDot( gxx, volume, gxx);
    dg::blas1::pointwiseDot( gyy, volume, gyy);
    dg::blas1::scal( gxx, g2d->hx());
    dg::blas1::scal( gyy, g2d->hy());
    if(rank==0)std::cout << "(Max elements on first process)\t";
    if(rank==0)std::cout << *thrust::max_element( gxx.data().begin(), gxx.data().end()) << "\t";
    if(rank==0)std::cout << *thrust::max_element( gyy.data().begin(), gyy.data().end()) << "\t";
    if(rank==0)std::cout<<t.diff()/(double)number<<"s"<<std::endl;
    ///////////////////////////////////////////////////////////////////////
    if(rank==0)std::cout << "TESTING VARIATION\n";
    pol.variation( x, x);
    const dg::MDVec variation = dg::pullback( dg::geo::VariationDirPer( mag, psi_0, psi_1), *g2d);
    dg::blas1::axpby( 1., variation, -1., x);
    double result = dg::blas2::dot( x, vol3d, x);
    if(rank==0)std::cout << "               distance to solution "<<sqrt( result)<<std::endl; //don't forget sqrt when comuting errors

    dg::MHVec transfer;
    dg::assign( error, transfer);
    dg::file::put_var_double( ncid, psiID, *g2d, transfer);
    dg::assign( x, transfer);
    dg::file::put_var_double( ncid, functionID, *g2d, transfer);
    dg::assign( solution, transfer);
    dg::file::put_var_double( ncid, function2ID, *g2d, transfer);
    if(rank==0)ncerr = nc_close( ncid);
    MPI_Finalize();


    return 0;
}
