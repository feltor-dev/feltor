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
#include "catch2/catch_all.hpp"

const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

TEST_CASE( "DS")
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {false,false,true});
#endif
    INFO( "# Test the parallel derivative DS in cylindrical coordinates for \
circular flux surfaces with DIR and NEU boundary conditions.");
    std::string method = GENERATE( "dg", "linear", "cubic");
    unsigned mm = GENERATE( 1, 10);
    dg::bc bcx = GENERATE( dg::NEU, dg::DIR);
    unsigned n = 3, Nx = 26, Ny = 26, Nz = 20, mx[2] = {mm,mm}, max_iter = 1e4;

    INFO( "Combination:\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"Nz: "<<Nz<<"\n"
              <<"mx: "<<mx[0]<<"\n"
              <<"my: "<<mx[1]<<"\n"
              <<"method: "<< method << "\n"
              <<"bc: "<<dg::bc2str(bcx)<<"\n"
              );
    const dg::x::CylindricalGrid3d g3d( R_0-a, R_0+a, -a, a, 0, 2.*M_PI,
        n, Nx, Ny, Nz, bcx, bcx, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    dg::x::DVec vol3d = dg::create::volume( g3d);
    //create magnetic field
    const dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    auto bhat = dg::geo::createBHat(mag);
    auto ff = dg::geo::TestFunctionDirNeu(mag);
    const dg::x::DVec fun = dg::pullback( ff, g3d);
    dg::x::DVec derivative(fun);
    const dg::x::DVec divb = dg::pullback( dg::geo::Divb(mag), g3d);
    const dg::x::DVec sol0 = dg::pullback( dg::geo::DsFunction<dg::geo::TestFunctionDirNeu>(mag,ff), g3d);
    const dg::x::DVec sol1 = dg::pullback( dg::geo::DssFunction<dg::geo::TestFunctionDirNeu>(mag,ff), g3d);
    const dg::x::DVec sol2 = dg::pullback( dg::geo::DsDivFunction<dg::geo::TestFunctionDirNeu>(mag,ff), g3d);
    const dg::x::DVec sol3 = dg::pullback( dg::geo::DsDivDsFunction<dg::geo::TestFunctionDirNeu>(mag,ff), g3d);
    const dg::x::DVec sol4 = dg::pullback( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionDirNeu>(mag,ff), g3d);
    std::vector<std::tuple<std::string, std::array<const dg::x::DVec*,2>, double>> names{
         {"forward",{&fun,&sol0}, 0.21},          {"backward",{&fun,&sol0}, 0.21},
         {"forward2",{&fun,&sol0}, 0.17},         {"backward2",{&fun,&sol0}, 0.17},
         {"centered",{&fun,&sol0}, 0.08},         {"centered_bc_along",{&fun,&sol0}, 0.08},
         {"dss",{&fun,&sol1}, 0.36},              {"dss_bc_along",{&fun,&sol1}, 0.24},
         {"divForward",{&fun,&sol2}, 0.21},       {"divBackward",{&fun,&sol2}, 0.21},
         {"divCentered",{&fun,&sol2}, 0.08},      {"directLap",{&fun,&sol3}, 0.36},
         {"directLap_bc_along",{&fun,&sol3}, 0.24}//, {"invCenteredLap",{&sol4,&fun}} //inversion may take too long for test
    };
    ///##########################################################///
    //create Fieldaligned object and construct DS from it
    dg::geo::Fieldaligned<dg::x::aProductGeometry3d,dg::x::IDMatrix,dg::x::DVec>  dsFA(
            bhat, g3d, bcx, bcx, dg::geo::NoLimiter(), 1e-8, mx[0], mx[1],
            -1, method, false);
    dg::geo::DS<dg::x::aProductGeometry3d, dg::x::IDMatrix, dg::x::DVec>
        ds( dsFA );
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
        INFO("    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol));
        CHECK( sqrt( norm/sol) <= std::get<2>(tuple));
    }
}

TEST_CASE( "Fieldaligned construction")
{
    // This test is taken from a failed MAST simulation which failed on construction
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {false,false,true});
#endif
    const dg::x::CylindricalGrid3d g3d( 96.2339, 752.624, -682.386, 682.386, 0, 2.*M_PI,
        3, 32, 67, 32, dg::NEU, dg::NEU, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    double R_0 = 469.921339;
    const dg::geo::TokamakMagneticField mag = dg::geo::createToroidalField( R_0);
    auto bhat = dg::geo::createBHat(mag);
    std::string method = "linear-nearest";
    dg::geo::Fieldaligned<dg::x::aProductGeometry3d,dg::x::IDMatrix,dg::x::DVec>  dsFA(
            bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, 12, 12,
            -1, method, false);
    CHECK(true);
}

TEST_CASE("Fieldaligned")
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {false,false,true});
#endif
    const dg::x::CylindricalGrid3d g3d( R_0-a, R_0+a, -a, a, 0, 2.*M_PI,
        3, 26, 26, 20, dg::NEU, dg::NEU, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    const dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    auto bhat = dg::geo::createBHat(mag);
    std::string method = "dg";
    dg::geo::Fieldaligned<dg::x::aProductGeometry3d,dg::x::IDMatrix,dg::x::DVec>  dsFA(
            bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, 10, 10,
            -1, method, false);
    dg::geo::DS<dg::x::aProductGeometry3d, dg::x::IDMatrix, dg::x::DVec>
        ds( dsFA );
    dg::x::DVec derivative = dg::evaluate( dg::zero, g3d);
    dg::x::DVec vol3d = dg::create::volume( g3d);
    ///##########################################################///
    INFO( "FIELDALIGNED EVALUATION of a Gaussian");
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    dg::x::DVec aligned = dg::geo::fieldaligned_evaluate( g3d, bhat, init0, modulate, 10, 2);
    ds.ds( dg::centered, aligned, derivative);
    double norm = dg::blas2::dot(vol3d, derivative);
    INFO( "# Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_mpit)");
    CHECK( norm < 0.099);
    aligned = dsFA.evaluate( init0, modulate, 10, 2);
    ds.ds( dg::centered, aligned, derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    INFO( "# Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_mpit)");
    CHECK( norm < 0.099);
    ///##########################################################///
    INFO( "STAGGERED GRID DERIVATIVE");
    auto ff = dg::geo::TestFunctionDirNeu(mag);
    const dg::x::DVec fun = dg::pullback( ff, g3d);
    const dg::x::DVec sol0 = dg::pullback( dg::geo::DsFunction<dg::geo::TestFunctionDirNeu>(mag,ff), g3d);
    dg::x::DVec zMinus(fun), eMinus(fun), zPlus(fun), ePlus(fun), eZero(fun);
    dg::x::DVec funST(fun);
    dg::geo::Fieldaligned<dg::x::aProductGeometry3d,dg::x::IDMatrix,dg::x::DVec>  dsFAST(
            bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, 10, 10,
            g3d.hz()/2., method, false);

    dsFAST( dg::geo::zeroMinus, fun, zMinus);
    dsFAST( dg::geo::einsPlus,  fun, ePlus);
    dg::geo::assign_bc_along_field_1st( dsFAST, zMinus, ePlus, zMinus, ePlus,
        dg::NEU, {0,0});
    dg::geo::ds_average( dsFAST, 1., zMinus, ePlus, 0., funST);
    dsFAST( dg::geo::zeroPlus, funST, zPlus);
    dsFAST( dg::geo::einsMinus, funST, eMinus);
    dg::geo::assign_bc_along_field_1st( dsFAST, eMinus, zPlus, eMinus, zPlus,
        dg::NEU, {0,0});
    dg::geo::ds_slope( dsFAST, 1., eMinus, zPlus, 0., derivative);
    double sol = dg::blas2::dot( vol3d, sol0);
    dg::blas1::axpby( 1., sol0, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::string name = "forward";
    INFO("    "<<name<<":" <<std::setw(18-name.size())
              <<" "<<sqrt(norm/sol));
    CHECK( sqrt(norm/sol) < 0.0711);

    // now try the adjoint direction (should be exactly the same result)
    dsFAST( dg::geo::zeroPlus, fun, zPlus);
    dsFAST( dg::geo::einsMinus, fun, eMinus);
    dg::geo::assign_bc_along_field_1st( dsFAST, eMinus, zPlus, eMinus, zPlus,
        dg::NEU, {0,0});
    dg::geo::ds_average( dsFAST, 1., eMinus, zPlus, 0., funST);
    dsFAST( dg::geo::einsPlus, funST, ePlus);
    dsFAST( dg::geo::zeroMinus, funST, zMinus);
    dg::geo::assign_bc_along_field_1st( dsFAST, zMinus, ePlus, zMinus, ePlus,
        dg::NEU, {0,0});
    dg::geo::ds_slope( dsFAST, 1., zMinus, ePlus, 0., derivative);
    dg::blas1::axpby( 1., sol0, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    name = "backward";
    INFO("    "<<name<<":" <<std::setw(18-name.size())
              <<" "<<sqrt(norm/sol));
    CHECK( sqrt(norm/sol) < 0.0711);
}
