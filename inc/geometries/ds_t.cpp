#include <iostream>
#include <iomanip>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "dg/algorithm.h"
#include "magnetic_field.h"
#include "testfunctors.h"
#include "ds.h"
#include "toroidal.h"

const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

int main(
#ifdef WITH_MPI
    int argc, char * argv[]
#endif
)
{
#ifdef WITH_MPI
    MPI_Init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    DG_RANK0 std::cout << "# Test the parallel derivative DS in cylindrical coordinates for circular flux surfaces with DIR and NEU boundary conditions.\n";
    unsigned n, Nx, Ny, Nz, mx[2], max_iter = 1e4;
    std::string method = "cubic";

#ifdef WITH_MPI
    MPI_Comm comm;
    dg::mpi_init3d( dg::NEU, dg::NEU, dg::PER, n, Nx, Ny, Nz, comm);
    unsigned letters = 0;
    if( rank == 0)
    {
        std::cout <<"# Type mx (10) and my (10)\n";
        std::cin >> mx[0] >> mx[1];
        std::cout << "# Type method (dg, nearest, linear, cubic) \n";
        std::cin >> method;
        method.erase( std::remove( method.begin(), method.end(), '"'), method.end());
        letters = method.size();
    }
    MPI_Bcast( mx, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( &letters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    method.resize( letters);
    MPI_Bcast( &method[0], letters, MPI_CHAR, 0, MPI_COMM_WORLD);
#else
    std::cout << "# Type n (3), Nx(20), Ny(20), Nz(20)\n";
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "# Type mx (10) and my (10)\n";
    std::cin >> mx[0]>> mx[1];
    std::cout << "# Type method (dg, nearest, linear, cubic) \n";
    std::cin >> method;
    method.erase( std::remove( method.begin(), method.end(), '"'), method.end());
#endif
    DG_RANK0 std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"Nz: "<<Nz<<"\n"
              <<"mx: "<<mx[0]<<"\n"
              <<"my: "<<mx[1]<<"\n"
              <<"method: "<< method<<std::endl;
    DG_RANK0 std::cout << "# Create parallel Derivative!\n";

    //![doxygen]
    const dg::x::CylindricalGrid3d g3d( R_0-a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    //create magnetic field
    const dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    auto bhat = dg::geo::createBHat(mag);
    //create Fieldaligned object and construct DS from it
    dg::geo::Fieldaligned<dg::x::aProductGeometry3d,dg::x::IDMatrix,dg::x::DVec>  dsFA(
            bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, mx[0], mx[1],
            -1, method);
    dg::geo::DS<dg::x::aProductGeometry3d, dg::x::IDMatrix, dg::x::DVec>
        ds( dsFA );
    //![doxygen]
    ///##########################################################///
    auto ff = dg::geo::TestFunctionDirNeu(mag);
    const dg::x::DVec fun = dg::pullback( ff, g3d);
    dg::x::DVec derivative(fun);
    const dg::x::DVec divb = dg::pullback( dg::geo::Divb(mag), g3d);
    const dg::x::DVec sol0 = dg::pullback( dg::geo::DsFunction<dg::geo::TestFunctionDirNeu>(mag,ff), g3d);
    const dg::x::DVec sol1 = dg::pullback( dg::geo::DssFunction<dg::geo::TestFunctionDirNeu>(mag,ff), g3d);
    const dg::x::DVec sol2 = dg::pullback( dg::geo::DsDivFunction<dg::geo::TestFunctionDirNeu>(mag,ff), g3d);
    const dg::x::DVec sol3 = dg::pullback( dg::geo::DsDivDsFunction<dg::geo::TestFunctionDirNeu>(mag,ff), g3d);
    const dg::x::DVec sol4 = dg::pullback( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionDirNeu>(mag,ff), g3d);
    std::vector<std::pair<std::string, std::array<const dg::x::DVec*,2>>> names{
         {"forward",{&fun,&sol0}},          {"backward",{&fun,&sol0}},
         {"forward2",{&fun,&sol0}},         {"backward2",{&fun,&sol0}},
         {"centered",{&fun,&sol0}},         {"centered_bc_along",{&fun,&sol0}},
         {"dss",{&fun,&sol1}},              {"dss_bc_along",{&fun,&sol1}},
         {"divForward",{&fun,&sol2}},       {"divBackward",{&fun,&sol2}},
         {"divCentered",{&fun,&sol2}},      {"directLap",{&fun,&sol3}},
         {"directLap_bc_along",{&fun,&sol3}}, {"invCenteredLap",{&sol4,&fun}}
    };
    DG_RANK0 std::cout << "# TEST NEU Boundary conditions!\n";
    DG_RANK0 std::cout << "# TEST ADJOINT derivatives do unfortunately not fulfill Neumann BC!\n";
    ///##########################################################///
    DG_RANK0 std::cout <<"Neumann:\n";
    dg::x::DVec vol3d = dg::create::volume( g3d);
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::x::DVec& function = *std::get<1>(tuple)[0];
        const dg::x::DVec& solution = *std::get<1>(tuple)[1];
        if( name.find("inv") != std::string::npos ||
                name.find( "div") != std::string::npos)
            callDS( ds, name, function, derivative, max_iter,1e-8);
        else
        {
            // test aliasing
            dg::blas1::copy( function, derivative);
            callDS( ds, name, derivative, derivative, max_iter,1e-8);
        }
        double sol = dg::blas2::dot( vol3d, solution);
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        DG_RANK0 std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    ///##########################################################///
    DG_RANK0 std::cout << "# Reconstruct parallel derivative!\n";
    dsFA.construct( bhat, g3d, dg::DIR, dg::DIR, dg::geo::NoLimiter(), 1e-8, mx[0], mx[1], -1, method);
    ds.construct( dsFA);
    DG_RANK0 std::cout << "# TEST DIR Boundary conditions!\n";
    ///##########################################################///
    DG_RANK0 std::cout << "Dirichlet: \n";
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::x::DVec& function = *std::get<1>(tuple)[0];
        const dg::x::DVec& solution = *std::get<1>(tuple)[1];
        if( name.find("inv") != std::string::npos ||
                name.find( "div") != std::string::npos)
            callDS( ds, name, function, derivative, max_iter,1e-8);
        else
        {
            // test aliasing
            dg::blas1::copy( function, derivative);
            callDS( ds, name, derivative, derivative, max_iter,1e-8);
        }

        double sol = dg::blas2::dot( vol3d, solution);
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        DG_RANK0 std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }

    ///##########################################################///
    DG_RANK0 std::cout << "# TEST FIELDALIGNED EVALUATION of a Gaussian\n";
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    dg::Timer t;
    t.tic();
    dg::x::DVec aligned = dg::geo::fieldaligned_evaluate( g3d, bhat, init0, modulate, Nz/2, 2);
    t.toc();
    DG_RANK0 std::cout << "# took "<<t.diff()<<"s\n";
    ds.ds( dg::centered, aligned, derivative);
    double norm = dg::blas2::dot(vol3d, derivative);
    DG_RANK0 std::cout << "# Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_mpit)\n";
    t.tic();
    aligned = dsFA.evaluate( init0, modulate, Nz/2, 2);
    t.toc();
    DG_RANK0 std::cout << "# took "<<t.diff()<<"s\n";
    ds.ds( dg::centered, aligned, derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    DG_RANK0 std::cout << "# Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_mpit)\n";
    ///##########################################################///
    DG_RANK0 std::cout << "# TEST STAGGERED GRID DERIVATIVE\n";
    dg::x::DVec zMinus(fun), eMinus(fun), zPlus(fun), ePlus(fun);
    dg::x::DVec funST(fun);
    dg::geo::Fieldaligned<dg::x::aProductGeometry3d,dg::x::IDMatrix,dg::x::DVec>  dsFAST(
            bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, mx[0], mx[1],
            g3d.hz()/2., method);
    DG_RANK0 std::cout <<"Time: ";
    t.tic();
    for( unsigned i=0; i<10; i++)
        ds.centered( fun, derivative);
    t.toc();
    double gbytes=fun.size()*sizeof(double)/1e9;
    DG_RANK0 std::cout << t.diff()/10 << " #\t "<<gbytes*83*10/t.diff()<<"GB/s\n" ;

    for( auto bc : {dg::NEU, dg::DIR})
    {
        if( bc == dg::DIR)
            DG_RANK0 std::cout << "DirichletST:\n";
        if( bc == dg::NEU)
            DG_RANK0 std::cout << "NeumannST:\n";
        dsFAST( dg::geo::zeroMinus, fun, zMinus);
        dsFAST( dg::geo::einsPlus,  fun, ePlus);
        dg::geo::assign_bc_along_field_1st( dsFAST, zMinus, ePlus, zMinus, ePlus,
            bc, {0,0});
        dg::geo::ds_average( dsFAST, 1., zMinus, ePlus, 0., funST);
        dsFAST( dg::geo::zeroPlus, funST, zPlus);
        dsFAST( dg::geo::einsMinus, funST, eMinus);
        dg::geo::assign_bc_along_field_1st( dsFAST, eMinus, zPlus, eMinus, zPlus,
            bc, {0,0});
        dg::geo::ds_slope( dsFAST, 1., eMinus, zPlus, 0., derivative);
        double sol = dg::blas2::dot( vol3d, sol0);
        dg::blas1::axpby( 1., sol0, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        std::string name = "forward";
        DG_RANK0 std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n";

        // now try the adjoint direction (should be exactly the same result)
        dsFAST( dg::geo::zeroPlus, fun, zPlus);
        dsFAST( dg::geo::einsMinus, fun, eMinus);
        dg::geo::assign_bc_along_field_1st( dsFAST, eMinus, zPlus, eMinus, zPlus,
            bc, {0,0});
        dg::geo::ds_average( dsFAST, 1., eMinus, zPlus, 0., funST);
        dsFAST( dg::geo::einsPlus, funST, ePlus);
        dsFAST( dg::geo::zeroMinus, funST, zMinus);
        dg::geo::assign_bc_along_field_1st( dsFAST, zMinus, ePlus, zMinus, ePlus,
            bc, {0,0});
        dg::geo::ds_slope( dsFAST, 1., zMinus, ePlus, 0., derivative);
        dg::blas1::axpby( 1., sol0, -1., derivative);
        norm = dg::blas2::dot( derivative, vol3d, derivative);
        name = "backward";
        DG_RANK0 std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
