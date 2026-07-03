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

const double R_0 = 3;
const double I_0 = 10; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius

int main(
#ifdef WITH_MPI
    int argc, char * argv[]
#endif
)
{
#ifdef WITH_MPI
    dg::mpi_init( &argc, &argv);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif
    DG_RANK0 std::cout << "# Test the parallel derivative DS in cylindrical coordinates for the guenter flux surfaces. Fieldlines do not cross boundaries.\n";
    unsigned n, Nx, Ny, Nz, mx, my, max_iter = 1e4;
    std::string method = "cubic";
#ifdef WITH_MPI
    MPI_Comm comm;
    dg::mpi_init3d( dg::NEU, dg::NEU, dg::PER, n, Nx, Ny, Nz, comm);
    unsigned letters = 0;
    if( rank == 0)
    {
        std::cout <<"# Type mx (10) and my (10)\n";
        std::cin >> mx >> my;
        std::cout << "# Type method (dg, nearest, linear, cubic) \n";
        std::cin >> method;
        method.erase( std::remove( method.begin(), method.end(), '"'), method.end());
        letters = method.size();
    }
    MPI_Bcast( &mx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( &my, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast( &letters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    method.resize( letters);
    MPI_Bcast( &method[0], letters, MPI_CHAR, 0, MPI_COMM_WORLD);
#else
    std::cout << "# Type n (3), Nx(20), Ny(20), Nz(20)\n";
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "# Type mx (10) and my (10)\n";
    std::cin >> mx>> my;
    std::cout << "# Type method (dg, nearest, linear, cubic) \n";
    std::cin >> method;
    method.erase( std::remove( method.begin(), method.end(), '"'), method.end());
#endif
    DG_RANK0 std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<"\n"
              <<"Nz: "<<Nz<<"\n"
              <<"mx: "<<mx<<"\n"
              <<"my: "<<my<<"\n"
              <<"method: "<< method<<std::endl;
    DG_RANK0 std::cout << "# Create parallel Derivative!\n";
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
        1e-8, mx, my, -1, method);

    ///##########################################################///
    auto ff = dg::geo::TestFunctionPsi2(mag,a);
    const dg::x::DVec fun = dg::evaluate( ff, g3d);
    dg::x::DVec derivative(fun);
    dg::x::DVec sol0 = dg::evaluate( dg::geo::DsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol1 = dg::evaluate( dg::geo::DssFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol2 = dg::evaluate( dg::geo::DsDivFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol3 = dg::evaluate( dg::geo::DsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    dg::x::DVec sol4 = dg::evaluate( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionPsi2>(mag,ff), g3d);
    std::vector<std::pair<std::string, std::array<const dg::x::DVec*,2>>> names{
         {"forward",{&fun,&sol0}},          {"backward",{&fun,&sol0}},
         {"forward2",{&fun,&sol0}},         {"backward2",{&fun,&sol0}},
         {"centered",{&fun,&sol0}},         {"dss",{&fun,&sol1}},
         {"centered_bc_along",{&fun,&sol0}},{"dss_bc_along",{&fun,&sol1}},
         {"divForward",{&fun,&sol2}},       {"divBackward",{&fun,&sol2}},
         {"divCentered",{&fun,&sol2}},      {"directLap",{&fun,&sol3}}//,
         //{"invCenteredLap",{&sol4,&fun}}
    };

    ///##########################################################///
    DG_RANK0 std::cout << "# TEST Guenter (No Boundary conditions)!\n";
    DG_RANK0 std::cout <<"Guenter : #rel_Error rel_Volume_integral(should be zero for div and Lap)\n";
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
        DG_RANK0 std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
                  <<" "<<sqrt(norm/sol)<<std::endl
                  <<"    "<<name+"_vol:"<<std::setw(30-name.size())
                  <<" "<<vol<<"\n";
    }
    ///##########################################################///
    DG_RANK0 std::cout << "# TEST TOTAL VARIATION DIMINISHING\n";
    ds.fieldaligned()(dg::geo::zeroPlus, fun, derivative);
    double mass_before = dg::blas1::dot( vol3d, fun);
    DG_RANK0 std::cout << "# mass before: "<<mass_before<<"\n";
    double mass_after = dg::blas1::dot( ds.fieldaligned().sqrtGp(), derivative);
    DG_RANK0 std::cout << "# mass after   "<<mass_after<<"\n";
    DG_RANK0 std::cout << "    Tp_mass_err:   "<<(mass_after-mass_before)/mass_before<<"\n";
    mass_before = sqrt(dg::blas2::dot( vol3d, fun));
    DG_RANK0 std::cout << "# l2 norm before: "<<mass_before<<"\n";
    mass_after = sqrt(dg::blas2::dot( ds.fieldaligned().sqrtGp(), derivative));
    DG_RANK0 std::cout << "# l2 norm after   "<<mass_after<<"\n";
    DG_RANK0 std::cout << "    Tp_l2_err:   "<<(mass_after-mass_before)/mass_before<<"\n";

    dg::geo::DSPGenerator generator( mag, g3d.x0(), g3d.x1(), g3d.y0(),
            g3d.y1(), g3d.hz());
    dg::geo::x::CurvilinearProductGrid3d g3dP( generator, {g3d.nx(), g3d.Nx(),
            g3d.bcx()}, {g3d.ny(), g3d.Ny(), g3d.bcy()}, g3d.gz()
#ifdef WITH_MPI
    , comm
#endif
    );
    dg::x::DVec vol3dP = dg::create::volume( g3dP);

    dg::Elliptic<dg::x::aProductGeometry3d, dg::x::DMatrix, dg::x::DVec> elliptic(g3d);
    dg::x::DVec variation(fun);
    elliptic.variation( fun, variation);
    dg::blas1::transform( variation, variation, []DG_DEVICE( double var){
            return sqrt(var);});
    double var_before = dg::blas1::dot( vol3d, variation);
    DG_RANK0 std::cout << "# variation before: "<<var_before<<"\n";
    dg::x::DVec var0 = dg::evaluate( dg::geo::Variation<dg::geo::TestFunctionPsi2>(ff), g3d);
    dg::blas1::axpby( 1., variation, -1., var0);
    double errVar0 = dg::blas2::dot( vol3d, var0)/ dg::blas2::dot( vol3d, variation);
    DG_RANK0 std::cout << "# error variation before: "<<sqrt(errVar0)<<"\n";
    // convergence: converges order P
    elliptic.construct(g3dP);
    elliptic.variation( derivative, variation);
    dg::blas1::transform( variation, variation, []DG_DEVICE( double var){
            return sqrt(var);});
    double var_after = dg::blas1::dot( vol3dP, variation);
    DG_RANK0 std::cout << "# variation after   "<<var_after<<"\n";
    var0 = dg::pullback( dg::geo::Variation<dg::geo::TestFunctionPsi2>(ff), g3dP);
    dg::blas1::axpby( 1., variation, -1., var0);
    errVar0 = dg::blas2::dot( vol3dP, var0)/dg::blas2::dot( vol3dP, variation);
    DG_RANK0 std::cout << "# error variation after : "<<sqrt(errVar0)<<"\n";
    // supraconvergence: converges order P-1 for dg interpolation
    // supraconvergence: converges order 1/2 for linear interpolation
    DG_RANK0 std::cout << "    Tp_TV_err: "<<(var_after-var_before)/var_before<<"\n";
    ///##########################################################///
    DG_RANK0 std::cout << "# TEST STAGGERED GRID DERIVATIVE\n";
    dg::x::DVec zMinus(fun), eMinus(fun), zPlus(fun), ePlus(fun), eZero(fun);
    dg::x::DVec funST(fun);
    dg::geo::Fieldaligned<dg::x::aProductGeometry3d,dg::x::IDMatrix,dg::x::DVec>  dsFAST(
            mag, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, mx, my,
            g3d.hz()/2., method);
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
    DG_RANK0 std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
              <<" "<<sqrt(norm/sol)<<"\n"
              <<"    "<<name+"_vol:"<<std::setw(30-name.size())
              <<" "<<vol<<"\n";

    ds.fieldaligned()(dg::geo::einsPlus, fun, ePlus);
    ds.fieldaligned()(dg::geo::zeroForw, fun, eZero);
    ds.fieldaligned()(dg::geo::einsMinus, fun, eMinus);
    dg::blas1::pointwiseDot ( 1./2./dsFAST.deltaPhi(), dsFAST.bphiM(),
            eZero, -1./2./dsFAST.deltaPhi(), dsFAST.bphiM(),
            eMinus, 0., eMinus);
    dg::blas1::pointwiseDot( 1./2./dsFAST.deltaPhi(), ePlus,
            dsFAST.bphiP(), -1./2./dsFAST.deltaPhi(), eZero,
            dsFAST.bphiP(), 0., ePlus);
    dg::geo::ds_divCentered( dsFAST, 1., eMinus, ePlus, 0., derivative);
    sol = dg::blas2::dot( vol3d, sol3);
    vol = dg::blas1::dot( vol3d, derivative)/sqrt( dg::blas2::dot( vol3d, fun));
    dg::blas1::axpby( 1., sol3, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    name  = "directLapST"; // works as well as directLap (MW: actually works almost exactly the same, also in diffusion equation test, it seems that the volume element can be well interpolated along the fieldlines)
    DG_RANK0 std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
              <<" "<<sqrt(norm/sol)<<"\n"
              <<"    "<<name+"_vol:"<<std::setw(30-name.size())
              <<" "<<vol<<"\n";

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
    DG_RANK0 std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
              <<" "<<sqrt(norm/sol)<<"\n"
              <<"    "<<name+"_vol:"<<std::setw(30-name.size())
              <<" "<<vol<<"\n";

    DG_RANK0 std::cout << "# TEST Inverse of I^+ is I^-\n";
    ds.fieldaligned()(dg::geo::einsPlus, fun, ePlus);
    ds.fieldaligned()(dg::geo::einsMinus, ePlus, eMinus);
    dg::blas1::axpby( 1., eMinus, -1., fun, eMinus);
    vol = sqrt( dg::blas2::dot( vol3d, eMinus)/dg::blas2::dot( vol3d, fun));
    name  = "IpIm"; // works as well as directLap
    DG_RANK0 std::cout <<"    "<<name<<":" <<std::setw(18-name.size())
              <<" "<<vol<<"\n";

    ///##########################################################///
    DG_RANK0 std::cout << "# TEST VOLUME FORMS\n";
    double volume = dg::blas1::dot( 1., dsFAST.sqrtG());
    double volumeM = dg::blas1::dot( 1., dsFAST.sqrtGm());
    double volumeP = dg::blas1::dot( 1., dsFAST.sqrtGp());
    double volumeG = dg::blas1::dot( 1., vol3dP);
    DG_RANK0 std::cout << "volume_error:\n";
    DG_RANK0 std::cout <<"    minus:"<<std::setw(13)<<" "<<fabs(volumeM-volume)/volume<<"\n";
    DG_RANK0 std::cout <<"    plus:" <<std::setw(14)<<" "<<fabs(volumeP-volume)/volume<<"\n";
    DG_RANK0 std::cout <<"    grid:" <<std::setw(14)<<" "<<fabs(volumeG-volume)/volume<<"\n";


    dg::x::DVec f(dg::evaluate( dg::one, g3d)), temp1(f), temp2(f), temp3(f);
    dsFAST(dg::geo::einsPlus, f, temp1);

    dg::blas1::pointwiseDot( dsFAST.sqrtG(), temp1, temp3);
    dsFAST(dg::geo::einsPlusT, temp3, temp2);
    dg::blas1::pointwiseDivide( temp2, dsFAST.sqrtGm(), temp2);
    dg::blas1::axpby( 1., temp2, -1., 1., temp2);
    dsFAST(dg::geo::einsPlus, temp2, temp3);

    double error = dg::blas2::dot( temp3, vol, temp3);
    //norm = dg::blas2::dot( 1., vol, 1.);
    norm = dg::blas2::dot( temp1, vol, temp1);
    DG_RANK0 std::cout <<"    Inv:"<<std::setw(15)<<" "<<sqrt(error/norm)<<"\n";
#ifdef WITH_MPI
    MPI_Finalize();
#endif

    return 0;
}
