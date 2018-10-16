#include <iostream>

#include <cusp/print.h>
#include "dg/algorithm.h"
#include "magnetic_field.h"
#define DG_BENCHMARK
#include "ds.h"
#include "toroidal.h"

template<class DS, class container>
struct DSS{
    DSS( DS& ds):m_ds(ds){}
    void symv( const container& x, container& y){
        m_ds.symv( -1., x, 0., y);
    }
    const container& weights()const{return m_ds.weights();}
    const container& inv_weights()const{return m_ds.inv_weights();}
    const container& precond()const{return m_ds.precond();}
    private:
    DS& m_ds;
};
namespace dg{
template< class DS, class container>
struct TensorTraits< DSS<DS, container> >
{
    using value_type = double;
    using tensor_category = SelfMadeMatrixTag;
};
}

const double R_0 = 10;
const double I_0 = 20; //q factor at r=1 is I_0/R_0
const double a  = 1; //small radius
double funcNEU(double R, double Z, double phi)
{
    return sin(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*sin(phi);
}
double deriNEU(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z; //(grad psi)^2
    return ( Z     *M_PI/2.*cos(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*sin(phi)
           -(R-R_0)*M_PI/2.*sin(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*sin(phi)
           + I_0/R*sin(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*cos(phi)
           )/sqrt(I_0*I_0+r2);
}
double deriAdjNEU(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z; //(grad psi)^2
    return Z/R/(I_0*I_0+r2)*funcNEU(R,Z,phi) + deriNEU(R,Z,phi);
}
double funcDIR(double R, double Z, double phi)
{
    return cos(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*sin(phi);
}
double deriDIR(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z; //(grad psi)^2
    return (-Z      *M_PI/2.*sin(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*sin(phi)
            +(R-R_0)*M_PI/2.*cos(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*sin(phi)
            +I_0/R*cos(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*cos(phi)
           )/sqrt(I_0*I_0+r2);
}
double deriAdjDIR(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z; //(grad psi)^2
    return Z/R/(I_0*I_0+r2)*funcDIR(R,Z,phi) + deriDIR(R,Z,phi);
}

double dssFuncNEU( double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    double bR = Z/sqrt(I_0*I_0+r2),
           bZ = -(R-R_0)/sqrt(I_0*I_0+r2),
           bP = I_0/R/sqrt(I_0*I_0+r2);
    double gradbR = - (R-R_0)/(I_0*I_0+r2);
    double gradbZ = - Z/(I_0*I_0+r2);
    double gradbP = - I_0/(R*R)/(I_0*I_0+r2);
    double f = sin(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*sin(phi);
    double dRf = M_PI/2.*cos(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*sin(phi);
    double dZf = M_PI/2.*sin(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*sin(phi);
    double dPf =         sin(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*cos(phi);
    double dRdZf = M_PI*M_PI/4.*cos(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*sin(phi);
    double dRdPf = M_PI/2.*cos(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*cos(phi);
    double dZdPf = M_PI/2.*sin(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*cos(phi);
    double dRdRf = -M_PI*M_PI/4.*f;
    double dZdZf = -M_PI*M_PI/4.*f;
    double dPdPf = -f;
    return (bR*bR*dRdRf + bZ*bZ*dZdZf + bP*bP*dPdPf)
            +2.*(bR*bZ*dRdZf + bR*bP*dRdPf + bZ*bP*dZdPf)
            +dRf*gradbR + dZf*gradbZ + dPf*gradbP;
}
double dssFuncDIR( double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    double bR = Z/sqrt(I_0*I_0+r2),
           bZ = -(R-R_0)/sqrt(I_0*I_0+r2),
           bP = I_0/R/sqrt(I_0*I_0+r2);
    double gradbR = - (R-R_0)/(I_0*I_0+r2);
    double gradbZ = - Z/(I_0*I_0+r2);
    double gradbP = - I_0/(R*R)/(I_0*I_0+r2);
    double f = cos(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*sin(phi);
    double dRf = -M_PI/2.*sin(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*sin(phi);
    double dZf = -M_PI/2.*cos(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*sin(phi);
    double dPf =          cos(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*cos(phi);
    double dRdZf = M_PI*M_PI/4.*sin(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*sin(phi);
    double dRdPf = -M_PI/2.*sin(M_PI*(R-R_0)/2.)*cos(M_PI*Z/2.)*cos(phi);
    double dZdPf = -M_PI/2.*cos(M_PI*(R-R_0)/2.)*sin(M_PI*Z/2.)*cos(phi);
    double dRdRf = -M_PI*M_PI/4.*f;
    double dZdZf = -M_PI*M_PI/4.*f;
    double dPdPf = -f;
    return (bR*bR*dRdRf + bZ*bZ*dZdZf + bP*bP*dPdPf)
            +2.*(bR*bZ*dRdZf + bR*bP*dRdPf + bZ*bP*dZdPf)
            +dRf*gradbR + dZf*gradbZ + dPf*gradbP;
}

double laplaceFuncNEU(double R, double Z, double phi)
{
    double dssf = dssFuncNEU(R,Z,phi);
    double dsf  = deriNEU(R,Z,phi);
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    double divb = Z/R/(I_0*I_0+r2);
    return divb*dsf + dssf;
}
double laplaceFuncDIR(double R, double Z, double phi)
{
    double dssf = dssFuncDIR(R,Z,phi);
    double dsf  = deriDIR(R,Z,phi);
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    double divb = Z/R/(I_0*I_0+r2);
    return divb*dsf + dssf;
}


int main(int argc, char * argv[])
{
    std::cout << "This program tests the parallel derivative DS in cylindrical coordinates for circular flux surfaces with DIR and NEU boundary conditions.\n";
    std::cout << "Type n (3), Nx(20), Ny(20), Nz(20)\n";
    unsigned n, Nx, Ny, Nz, mx, my;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "You typed "<<n<<" "<<Nx<<" "<<Ny<<" "<<Nz<<std::endl;
    std::cout << "Type mx (10) and my (10)\n";
    std::cin >> mx>> my;
    std::cout << "You typed "<<mx<<" "<<my<<std::endl;
    std::cout << "Create parallel Derivative!\n";

    //![doxygen]
    const dg::CylindricalGrid3d g3d( R_0 - a, R_0+a, -a, a, 0, 2.*M_PI, n, Nx, Ny, Nz, dg::NEU, dg::NEU);
    //create magnetic field
    const dg::geo::TokamakMagneticField mag = dg::geo::createCircularField( R_0, I_0);
    const dg::geo::BinaryVectorLvl0 bhat( (dg::geo::BHatR)(mag), (dg::geo::BHatZ)(mag), (dg::geo::BHatP)(mag));
    //create Fieldaligned object and construct DS from it
    dg::geo::Fieldaligned<dg::aProductGeometry3d,dg::IDMatrix,dg::DVec>  dsFA( bhat, g3d, dg::NEU, dg::NEU, dg::geo::NoLimiter(), 1e-8, mx, my);
    dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ds( dsFA, dg::normed, dg::centered);
    ///##########################################################///
    //apply to function
    const dg::DVec function = dg::evaluate( funcNEU, g3d);
    dg::DVec derivative(function);
    ds.centered( function, derivative);
    //![doxygen]
    std::cout << "TEST NEU Boundary conditions!\n";
    dg::DVec solution = dg::evaluate( deriNEU, g3d);
    const dg::DVec vol3d = dg::create::volume( g3d);
    double sol = dg::blas2::dot( vol3d, solution);
    dg::blas1::axpby( 1., solution, -1., derivative);
    double norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "Error centered derivative \t"<< sqrt( norm/sol )<<"\n";
    ds.forward( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Error Forward  Derivative \t"<<sqrt( norm/sol)<<"\n";
    ds.backward( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Error Backward Derivative \t"<<sqrt( norm/sol)<<"\n";
    ///##########################################################///
    //std::cout << "TEST DSS derivative!\n";
    solution = dg::evaluate( dssFuncNEU, g3d);
    sol = dg::blas2::dot( vol3d, solution);

    ds.dss( function, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "Error dss                 \t"<< sqrt( norm/sol )<<"\n";
    ///##########################################################///
    ///We unfortunately cannot test convergence of adjoint because
    ///b and therefore bf does not fulfill Neumann boundary conditions
    std::cout << "TEST ADJOINT derivatives (do unfortunately not fulfill Neumann BC!)\n";
    solution = dg::evaluate( deriAdjNEU, g3d);
    sol = dg::blas2::dot( vol3d, solution);

    ds.centeredDiv( function, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "Error centered divergence \t"<< sqrt( norm/sol )<<"\n";
    ds.forwardDiv( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Error Forward  divergence \t"<<sqrt( norm/sol)<<"\n";
    ds.backwardDiv( 1., function, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Error Backward divergence \t"<<sqrt( norm/sol)<<"\n";
    ///##########################################################///
    solution = dg::evaluate( laplaceFuncNEU, g3d);
    sol = dg::blas2::dot( vol3d, solution);

    ds.set_direction( dg::centered);
    ds.symv( function, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "Error centered Laplace    \t"<< sqrt( norm/sol )<<"\n";
    ds.set_direction( dg::forward);
    ds.symv( function, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "Error backward Laplace    \t"<< sqrt( norm/sol )<<"\n";
    ///##########################################################///
    solution = dg::evaluate( laplaceFuncNEU, g3d);
    ds.set_norm( dg::not_normed);
    DSS< dg::geo::DS<dg::aProductGeometry3d, dg::IDMatrix, dg::DMatrix, dg::DVec>, dg::DVec> dss( ds);
    unsigned max_iter;
    std::cin >> max_iter;
    dg::Invert<dg::DVec> invert( solution, max_iter, 1e-5);
    invert( dss, derivative, solution);

    sol = dg::blas2::dot( vol3d, function);
    dg::blas1::axpby( 1., function, 1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "Error Laplace_parallel    \t"<< sqrt( norm/sol )<<"\n";

    ///##########################################################///
    std::cout << "Reconstruct parallel derivative!\n";
    dsFA.construct( bhat, g3d, dg::DIR, dg::DIR, dg::geo::NoLimiter(), 1e-8, mx, my);
    ds.construct( dsFA, dg::normed, dg::centered);
    std::cout << "TEST DIR Boundary conditions!\n";
    //apply to function
    const dg::DVec functionDIR = dg::evaluate( funcDIR, g3d);
    solution = dg::evaluate( deriDIR, g3d);
    sol = dg::blas2::dot( vol3d, solution);

    ds.centered( functionDIR, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "Error centered derivative \t"<< sqrt( norm/sol )<<"\n";
    ds.forward( 1., functionDIR, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Error Forward  Derivative \t"<<sqrt( norm/sol)<<"\n";
    ds.backward( 1., functionDIR, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Error Backward Derivative \t"<<sqrt( norm/sol)<<"\n";
    ///##########################################################///
    //std::cout << "TEST DSS derivative!\n";
    solution = dg::evaluate( dssFuncDIR, g3d);
    sol = dg::blas2::dot( vol3d, solution);

    ds.dss( 1., functionDIR,  0., derivative);
    //ds.forward( functionDIR, derivative);
    //ds.backward( derivative, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "Error dss                 \t"<< sqrt( norm/sol )<<"\n";

    ///##########################################################///
    std::cout << "TEST ADJOINT derivatives!\n";
    solution = dg::evaluate( deriAdjDIR, g3d);
    sol = dg::blas2::dot( vol3d, solution);

    ds.centeredDiv( functionDIR, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "Error centered divergence \t"<< sqrt( norm/sol )<<"\n";
    ds.forwardDiv( 1., functionDIR, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Error Forward  divergence \t"<<sqrt( norm/sol)<<"\n";
    ds.backwardDiv( 1., functionDIR, 0., derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Error Backward divergence \t"<<sqrt( norm/sol)<<"\n";
    ///##########################################################///
    solution = dg::evaluate( laplaceFuncDIR, g3d);
    sol = dg::blas2::dot( vol3d, solution);

    ds.set_direction( dg::centered);
    ds.symv( functionDIR, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "Error centered Laplace    \t"<< sqrt( norm/sol )<<"\n";
    ds.set_direction( dg::forward);
    ds.symv( functionDIR, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "Error backward Laplace    \t"<< sqrt( norm/sol )<<"\n";
    ///##########################################################///
    solution = dg::evaluate( laplaceFuncDIR, g3d);
    ds.set_norm( dg::not_normed);
    dg::Invert<dg::DVec> invertDIR( solution, max_iter, 1e-5);
    invertDIR( dss, derivative, solution);
    sol = dg::blas2::dot( vol3d, functionDIR);

    ////ds.symv( functionDIR, derivative);
    dg::blas1::axpby( 1., functionDIR, 1., derivative);
    norm = dg::blas2::dot( derivative, vol3d, derivative);
    std::cout << "Error Laplace_parallel    \t"<< sqrt( norm/sol )<<"\n";

    ///##########################################################///
    std::cout << "TEST FIELDALIGNED EVALUATION of a Gaussian\n";
    dg::Gaussian init0(R_0+0.5, 0, 0.2, 0.2, 1);
    dg::GaussianZ modulate(0., M_PI/3., 1);
    dg::DVec aligned = dsFA.evaluate( init0, modulate, Nz/2, 2);
    ds( aligned, derivative);
    norm = dg::blas2::dot(vol3d, derivative);
    std::cout << "Norm Centered Derivative "<<sqrt( norm)<<" (compare with that of ds_mpit)\n";

    return 0;
}
