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
    dg::file::NcFile file( "testE_mpi.nc", dg::file::nc_clobber);
    file.defput_dim( "x", {{"axis", "X"}}, g2d->abscissas(0));
    file.defput_dim( "y", {{"axis", "Y"}}, g2d->abscissas(1));
    file.defput_var( "xc", {"y", "x"}, {}, {*g2d}, g2d->map()[0]);
    file.defput_var( "yc", {"y", "x"}, {}, {*g2d}, g2d->map()[1]);
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

    file.defput_var( "error", {"y", "x"}, {}, {*g2d}, error);
    file.defput_var( "num_solution", {"y", "x"}, {}, {*g2d}, x);
    file.defput_var( "ana_solution", {"y", "x"}, {}, {*g2d}, solution);
    file.close();

    MPI_Finalize();


    return 0;
}
