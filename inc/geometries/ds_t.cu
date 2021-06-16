#include <iostream>
#include <iomanip>

#define DG_BENCHMARK
#include "dg/algorithm.h"
#include "magnetic_field.h"
#include "testfunctors.h"
#include "ds.h"
#include "toroidal.h"

const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

int main(int argc, char * argv[])
{
    std::cout << "# Test the parallel derivative DS in cylindrical coordinates for circular flux surfaces with DIR and NEU boundary conditions.\n";
    std::cout << "# Type n (3), Nx(20), Ny(20), Nz(20)\n";
    unsigned n, Nx, Ny, Nz, mx[2], max_iter = 1e4;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"Nz: "<<Nz<<std::endl;
    std::cout << "# Type mx (10) and my (10)\n";
    std::cin >> mx[0]>> mx[1];
    std::cout << "# You typed\n"
              <<"mx: "<<mx[0]<<"\n"
              <<"my: "<<mx[1]<<std::endl;
    std::cout << "# Create parallel Derivative!\n";

    //![doxygen]
    const dg::CylindricalGrid3d g3d( R_0-a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU, dg::PER);
    //create magnetic field
    const dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    const dg::geo::CylindricalVectorLvl0 bhat = dg::geo::createBHat(mag);
    //create Fieldaligned object and construct DS from it
    dg::geo::Fieldaligned<dg::aProductGeometry3d,dg::IDMatrix,dg::DVec>  dsFA(
            bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, mx[0], mx[1]);
    dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec>
        ds( dsFA, dg::centered );
    //![doxygen]
    ///##########################################################///
    const dg::DVec fun = dg::pullback( dg::geo::TestFunctionDirNeu(mag), g3d);
    dg::DVec derivative(fun);
    const dg::DVec divb = dg::pullback( dg::geo::Divb(mag), g3d);
    const dg::DVec sol0 = dg::pullback( dg::geo::DsFunction<dg::geo::TestFunctionDirNeu>(mag), g3d);
    const dg::DVec sol1 = dg::pullback( dg::geo::DssFunction<dg::geo::TestFunctionDirNeu>(mag), g3d);
    const dg::DVec sol2 = dg::pullback( dg::geo::DsDivFunction<dg::geo::TestFunctionDirNeu>(mag), g3d);
    const dg::DVec sol3 = dg::pullback( dg::geo::DsDivDsFunction<dg::geo::TestFunctionDirNeu>(mag), g3d);
    const dg::DVec sol4 = dg::pullback( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionDirNeu>(mag), g3d);
    std::vector<std::pair<std::string, std::array<const dg::DVec*,2>>> names{
         {"forward",{&fun,&sol0}},          {"backward",{&fun,&sol0}},
         {"forward2",{&fun,&sol0}},         {"backward2",{&fun,&sol0}},
         {"centered",{&fun,&sol0}},         {"dss",{&fun,&sol1}},
         {"centered_bc_along",{&fun,&sol0}},{"dss_bc_along",{&fun,&sol1}},
         {"divForward",{&fun,&sol2}},       {"divBackward",{&fun,&sol2}},
         {"divCentered",{&fun,&sol2}},      {"divDirectForward",{&fun,&sol2}},
         {"divDirectBackward",{&fun,&sol2}},{"divDirectCentered",{&fun,&sol2}},
         {"forwardLap",{&fun,&sol3}},       {"backwardLap",{&fun,&sol3}},
         {"centeredLap",{&fun,&sol3}},      {"directLap",{&fun,&sol3}},
         {"directLap_bc_along",{&fun,&sol3}},
         {"invForwardLap",{&sol4,&fun}},    {"invBackwardLap",{&sol4,&fun}},
         {"invCenteredLap",{&sol4,&fun}}
    };
    std::cout << "# TEST NEU Boundary conditions!\n";
    std::cout << "# TEST ADJOINT derivatives do unfortunately not fulfill Neumann BC!\n";
    ///##########################################################///
    std::cout <<"Neumann:\n";
    dg::DVec vol3d = dg::create::volume( g3d);
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::DVec& function = *std::get<1>(tuple)[0];
        const dg::DVec& solution = *std::get<1>(tuple)[1];
        if( name.find("inv") != std::string::npos ||
                name.find( "div") != std::string::npos)
            callDS( ds, name, function, derivative, divb, max_iter,1e-8);
        else
        {
            // test aliasing
            dg::blas1::copy( function, derivative);
            callDS( ds, name, derivative, derivative, divb, max_iter,1e-8);
        }
        double sol = dg::blas2::dot( vol3d, solution);
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    ///##########################################################///
    std::cout << "# Reconstruct parallel derivative!\n";
    dsFA.construct( bhat, g3d, dg::DIR, dg::DIR, dg::geo::NoLimiter(), 1e-8, mx[0], mx[1]);
    ds.construct( dsFA, dg::centered);
    std::cout << "# TEST DIR Boundary conditions!\n";
    ///##########################################################///
    std::cout << "Dirichlet: \n";
    for( const auto& tuple :  names)
    {
        std::string name = std::get<0>(tuple);
        const dg::DVec& function = *std::get<1>(tuple)[0];
        const dg::DVec& solution = *std::get<1>(tuple)[1];
        if( name.find("inv") != std::string::npos ||
                name.find( "div") != std::string::npos)
            callDS( ds, name, function, derivative, divb, max_iter,1e-8);
        else
        {
            // test aliasing
            dg::blas1::copy( function, derivative);
            callDS( ds, name, derivative, derivative, divb, max_iter,1e-8);
        }

        double sol = dg::blas2::dot( vol3d, solution);
        dg::blas1::axpby( 1., solution, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }

    ///##########################################################///
    std::cout << "# TEST FIELDALIGNED EVALUATION of a Gaussian\n";
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    dg::DVec aligned = dsFA.evaluate( init0, modulate, Nz/2, 2);
    ds( aligned, derivative);
    double norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "# Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_mpit)\n";
    ///##########################################################///
    std::cout << "# TEST STAGGERED GRID DERIVATIVE\n";
    dg::DVec zMinus(fun), eMinus(fun), zPlus(fun), ePlus(fun);
    dg::DVec funST(fun);
    dg::geo::Fieldaligned<dg::aProductGeometry3d,dg::IDMatrix,dg::DVec>  dsFAST(
            bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, mx[0], mx[1],
            g3d.hz()/2.);
    for( auto bc : {dg::NEU, dg::DIR})
    {
        if( bc == dg::DIR)
            if(rank==0)std::cout << "DirichletST:\n";
        if( bc == dg::NEU)
            if(rank==0)std::cout << "NeumannST:\n";
        dsFAST( dg::geo::zeroMinus, fun, zMinus);
        dsFAST( dg::geo::einsPlus,  fun, ePlus);
        dg::geo::assign_bc_along_field_1st( dsFAST, zMinus, ePlus, zMinus, ePlus,
            bc, {0,0});
        dg::blas1::axpby( 0.5, zMinus, 0.5, ePlus, funST);
        dsFAST( dg::geo::zeroPlus, funST, zPlus);
        dsFAST( dg::geo::einsMinus, funST, eMinus);
        dg::geo::assign_bc_along_field_1st( dsFAST, eMinus, zPlus, eMinus, zPlus,
            bc, {0,0});
        dg::blas1::subroutine( []DG_DEVICE( double& df, double fm, double fp,
                    double hp, double hm){
                df = (fp-fm)/(hp+hm);
                }, derivative, eMinus, zPlus, dsFAST.hp(), dsFAST.hm());
        double sol = dg::blas2::dot( vol3d, sol0);
        dg::blas1::axpby( 1., sol0, -1., derivative);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        std::string name = "forward";
        std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n";

        // now try the adjoint direction (should be exactly the same result)
        dsFAST( dg::geo::zeroPlus, fun, zPlus);
        dsFAST( dg::geo::einsMinus, fun, eMinus);
        dg::geo::assign_bc_along_field_1st( dsFAST, eMinus, zPlus, eMinus, zPlus,
            bc, {0,0});
        dg::blas1::axpby( 0.5, eMinus, 0.5, zPlus, funST);
        dsFAST( dg::geo::einsPlus, funST, ePlus);
        dsFAST( dg::geo::zeroMinus, funST, zMinus);
        dg::geo::assign_bc_along_field_1st( dsFAST, zMinus, ePlus, zMinus, ePlus,
            bc, {0,0});
        dg::blas1::subroutine( []DG_DEVICE( double& df, double fm, double fp,
                    double hp, double hm){
                df = (fp-fm)/(hp+hm);
                }, derivative, zMinus, ePlus, dsFAST.hp(), dsFAST.hm());
        dg::blas1::axpby( 1., sol0, -1., derivative);
        norm = dg::blas2::dot( derivative, vol3d, derivative);
        name = "backward";
        std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    return 0;
}
