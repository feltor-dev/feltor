#include <iostream>
#include <iomanip>
#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "dg/algorithm.h"
#include "ds.h"
#include "guenter.h"
#include "magnetic_field.h"
#include "testfunctors.h"
#include "ds_generator.h"
#ifdef WITH_MPI
#include "mpi_curvilinear.h"
#endif
#include "catch2/catch_all.hpp"

const double R_0 = 3;
const double I_0 = 10; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

TEST_CASE( "DS Guenter")
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {false,false,true});
#endif
    INFO( "# Test the parallel derivative DS in cylindrical coordinates for the \
guenter flux surfaces. Fieldlines do not cross boundaries.");
    std::string method = GENERATE( "dg", "linear-nearest", "linear-linear");
    unsigned mm = GENERATE( 1, 12);
    unsigned n = 3, Nx = 26, Ny = 26, Nz = 20, mx[2] = {mm,mm}, max_iter = 1e4;
    if( "dg" != method)
    {
        Nx = Ny = 30;
        if( mm == 1)
            mx[0] = mx[1] = n;
    }

    INFO( "Combination\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"Nz: "<<Nz<<"\n"
              <<"mx: "<<mx[0]<<"\n"
              <<"my: "<<mx[1]<<"\n"
              <<"method: "<< method << "\n");
    ////////////////////////////////initialze fields /////////////////////
    const dg::x::CylindricalGrid3d g3d( R_0 - a, R_0+a, -a, a, 0, 2.*M_PI, n,
    Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    const dg::geo::TokamakMagneticField mag = dg::geo::createGuenterField(R_0, I_0);
    dg::geo::DS<dg::x::aProductGeometry3d, dg::x::IDMatrix, dg::x::DVec> ds(
        mag, g3d, dg::NEU, dg::NEU, dg::geo::FullLimiter(),
        1e-8, mx[0], mx[1], -1, method, false);

    ///##########################################################///
    auto ff = dg::geo::TestFunctionPsi2(mag,a);
    const dg::x::DVec fun = dg::evaluate( ff, g3d);
    dg::x::DVec derivative(fun);
    dg::x::DVec sol0 = dg::evaluate( dg::geo::DsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol1 = dg::evaluate( dg::geo::DssFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol2 = dg::evaluate( dg::geo::DsDivFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol3 = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol4 = dg::evaluate( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    std::vector<std::tuple<std::string, std::array<const dg::x::DVec*,2>, double,double>> names{
         {"forward",{&fun,&sol0},0.33,0.06},          {"backward",{&fun,&sol0},0.33,0.07},
         {"forward2",{&fun,&sol0},0.15,0.06},         {"backward2",{&fun,&sol0},0.15,0.06},
         {"centered",{&fun,&sol0},0.075,0.06},         {"dss",{&fun,&sol1},0.41,0.01},
         {"centered_bc_along",{&fun,&sol0},0.08,0.06},{"dss_bc_along",{&fun,&sol1},0.05,0.01},
         {"divForward",{&fun,&sol2},0.33,1e-5},       {"divBackward",{&fun,&sol2},0.33,1e-5},
         {"divCentered",{&fun,&sol2},0.08,1e-5},      {"directLap",{&fun,&sol3},0.05,5e-5}//,
         //{"invCenteredLap",{&sol4,&fun}}
    };

    ///##########################################################///
    const dg::x::DVec vol3d = dg::create::volume( g3d);
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::x::DVec& function = *std::get<1>(tuple)[0];
        const dg::x::DVec& solution = *std::get<1>(tuple)[1];
        callDS( ds, name, function, derivative, max_iter,1e-8);
        double sol = dg::blas2::dot( vol3d, solution);
        double vol = dg::blas1::dot( vol3d, derivative)/sqrt( dg::blas2::dot( vol3d, function)); // using function in denominator makes entries comparable
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        INFO("    "<<name<<":" <<" "<<sqrt(norm/sol)
                  <<"    "<<name+"_vol:"<<std::setw(30-name.size())
                  <<" "<<vol);
        CHECK( sqrt(norm/sol) < std::get<2>(tuple));
        CHECK( fabs(vol) < std::get<3>(tuple));
    }
}

TEST_CASE("Staggered")
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
    const dg::geo::TokamakMagneticField mag = dg::geo::createGuenterField(R_0, I_0);
    auto bhat = dg::geo::createBHat(mag);
    std::string method = "dg";
    ///##########################################################///
    auto ff = dg::geo::TestFunctionPsi2(mag,a);
    const dg::x::DVec fun = dg::pullback( ff, g3d);
    dg::x::DVec derivative(fun);
    const dg::x::DVec vol3d = dg::create::volume( g3d);
    dg::x::DVec sol0 = dg::evaluate( dg::geo::DsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol3 = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec zMinus(fun), eMinus(fun), zPlus(fun), ePlus(fun), eZero(fun);
    dg::x::DVec funST(fun);
    dg::geo::Fieldaligned<dg::x::aProductGeometry3d,dg::x::IDMatrix,dg::x::DVec>  dsFAST(
            mag, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, 12, 12,
            g3d.hz()/2., method, false);
    dsFAST( dg::geo::zeroMinus, fun, zMinus);
    dsFAST( dg::geo::einsPlus,  fun, ePlus);
    dg::geo::ds_slope( dsFAST, 1., zMinus, ePlus, 0., funST);
    dsFAST( dg::geo::zeroPlus, funST, zPlus);
    dsFAST( dg::geo::einsMinus, funST, eMinus);
    dg::geo::ds_average( dsFAST, 1., eMinus, zPlus, 0., derivative);

    double sol = dg::blas2::dot( vol3d, sol0);
    double vol = dg::blas1::dot( vol3d, derivative)/sqrt( dg::blas2::dot( vol3d, fun));
    dg::blas1::axpby( 1., sol0, -1., derivative);
    double norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::string name  = "centeredST";
    INFO("    "<<name<<":" <<std::setw(18-name.size())
              <<" "<<sqrt(norm/sol)<<"\n"
              <<"    "<<name+"_vol:"<<std::setw(30-name.size())
              <<" "<<vol);
    CHECK( sqrt(norm/sol) < 0.075);
    CHECK( fabs(vol) < 0.06);

    dsFAST( dg::geo::zeroMinus, fun, zMinus);
    dsFAST( dg::geo::einsPlus,  fun, ePlus);
    dg::geo::ds_centered( dsFAST, 1., zMinus, ePlus, 0., funST);
    dsFAST( dg::geo::einsMinus, funST, zMinus);
    dsFAST( dg::geo::zeroPlus,  funST, ePlus);
    dg::geo::ds_divCentered( dsFAST, 1., zMinus, ePlus, 0., derivative);
    sol = dg::blas2::dot( vol3d, sol3);
    vol = dg::blas1::dot( vol3d, derivative)/sqrt( dg::blas2::dot( vol3d, fun));
    dg::blas1::axpby( 1., sol3, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    name  = "staggeredLapST";
    INFO("    "<<name<<":" <<std::setw(18-name.size())
              <<" "<<sqrt(norm/sol)<<"\n"
              <<"    "<<name+"_vol:"<<std::setw(30-name.size())
              <<" "<<vol);
    CHECK( sqrt(norm/sol) < 0.075);
    CHECK( fabs(vol) < 1e-5);

    ///##########################################################///
    INFO( "# TEST VOLUME FORMS");
    double volume = dg::blas1::dot( 1., dsFAST.sqrtG());
    double volumeM = dg::blas1::dot( 1., dsFAST.sqrtGm());
    double volumeP = dg::blas1::dot( 1., dsFAST.sqrtGp());
    INFO( "volume_error    minus: "<<fabs(volumeM-volume)/volume);
    INFO( "volume_error    minus: "<<fabs(volumeP-volume)/volume);
    CHECK( fabs( volumeM - volume)/volume < 1e-12);
    CHECK( fabs( volumeP - volume)/volume < 1e-12);
}

TEST_CASE( "Toroidal DS Guenter")
{
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm comm = dg::mpi_cart_create( MPI_COMM_WORLD, {0,0,0}, {false,false,true});
#endif
    INFO( "# Test the parallel derivative DS in cylindrical coordinates for the \
guenter flux surfaces. Fieldlines do not cross boundaries.");
    std::string method = "dg";
    unsigned mm = 3;
    unsigned n = 7, Nx = 20, Ny = 20, Nz = 2, mx[2] = {mm,mm}, max_iter = 1e4;

    INFO( "Combination\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"Nz: "<<Nz<<"\n"
              <<"mx: "<<mx[0]<<"\n"
              <<"my: "<<mx[1]<<"\n"
              <<"method: "<< method << "\n");
    ////////////////////////////////initialze fields /////////////////////
    const dg::x::CylindricalGrid3d g3d( R_0 - a, R_0+a, -a, a, 0, 2.*M_PI, n,
    Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    const dg::geo::TokamakMagneticField mag = dg::geo::createGuenterField(R_0, I_0);
    auto bhat = dg::geo::createToroidalBHat(mag);
    dg::geo::Fieldaligned<dg::x::aProductGeometry3d,dg::x::IDMatrix,dg::x::DVec>  dsFA(
            bhat, g3d, dg::NEU, dg::NEU, dg::geo::FullLimiter(), 1e-8, mx[0], mx[1],
            2.*M_PI/2000, method, false);
    dg::geo::DS<dg::x::aProductGeometry3d, dg::x::IDMatrix, dg::x::DVec>
        ds( dsFA );

    ///##########################################################///
    auto ff = dg::geo::TestFunctionPsi2(mag,a, 0.); //  kphi = 0
    const dg::x::DVec fun = dg::evaluate( ff, g3d);
    dg::x::DVec derivative(fun);
    dg::x::DVec sol0 = dg::evaluate( dg::geo::ToroidalDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol1 = dg::evaluate( dg::geo::ToroidalDssFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol2 = dg::evaluate( dg::geo::ToroidalDsDivFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol3 = dg::evaluate( dg::geo::ToroidalDsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol4 = dg::evaluate( dg::geo::ToroidalOMDsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    std::vector<std::tuple<std::string, std::array<const dg::x::DVec*,2>>> names{
         {"forward2",{&fun,&sol0}},         {"backward2",{&fun,&sol0}},
         {"centered",{&fun,&sol0}},         {"dss",{&fun,&sol1}},
         {"centered_bc_along",{&fun,&sol0}},{"dss_bc_along",{&fun,&sol1}},
         {"divCentered",{&fun,&sol2}},      {"directLap",{&fun,&sol3}}//,
    };

    ///##########################################################///
    const dg::x::DVec vol3d = dg::create::volume( g3d);
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::x::DVec& function = *std::get<1>(tuple)[0];
        const dg::x::DVec& solution = *std::get<1>(tuple)[1];
        callDS( ds, name, function, derivative, max_iter,1e-8);
        double sol = dg::blas2::dot( vol3d, solution);
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        INFO("    "<<name<<":" <<" "<<sqrt(norm/sol));
        CHECK( sqrt(norm/sol) < 1e-5);
    }
}
