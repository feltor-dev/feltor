#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>

#include "feltor.h"
#include "implicit.h"
using HVec = dg::HVec;
using DVec = dg::DVec;
using DMatrix = dg::DMatrix;
using IDMatrix = dg::IDMatrix;
using IHMatrix = dg::IHMatrix;
using Geometry = dg::CylindricalGrid3d;
#define MPI_OUT

#include "init.h"
#include "feltordiag.h"

namespace detail{
template<class Explicit, class Implicit, class Container >
struct FullSystem
{
    FullSystem() = default;
    FullSystem( Explicit exp, Implicit imp, Container temp):
            m_exp( exp), m_imp( imp), m_temp(temp){}

    template<class Container2>
    void operator()( const Container2& y, Container2& yp)
    {
        m_exp( 0, y, m_temp);
        m_imp( 0, y, yp);
        dg::blas1::axpby( 1., m_temp, 1., yp);
    }
    private:
    Explicit m_exp;
    Implicit m_imp;
    Container m_temp;
};
}//namespace detail

template<class Explicit, class Implicit, class Container, class ContainerType2>
void solve_steady_state( Explicit& ex, Implicit& im, Container& x, const Container& b, const ContainerType2& weights)
{
    Container tmp(x);
    detail::FullSystem<Explicit&, Implicit&, Container&> full(ex,im,tmp);
    // allocate memory
    unsigned mMax =3, restart = mMax;
    dg::AndersonAcceleration<Container> acc( x, mMax);
    // Evaluate right hand side and solution on the grid
    const double eps = 1e-1;
    unsigned max_iter =10;
    double damping = 1e-2;
    std::cout << "Number of iterations "<< acc.solve( full, x, b, weights, eps, eps, max_iter, damping, restart, true)<<std::endl;

}

int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Value js, gs;
    if( argc == 1)
    {
        std::ifstream is("input.json");
        std::ifstream ks("geometry_params.json");
        is >> js;
        ks >> gs;
    }
    else if( argc == 3)
    {
        std::ifstream is(argv[1]);
        std::ifstream ks(argv[2]);
        is >> js;
        ks >> gs;
    }
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "
                  << argv[0]<<" [inputfile] [geomfile] \n";
        return -1;
    }
    const feltor::Parameters p( js);
    const dg::geo::solovev::Parameters gp(gs);
    p.display( std::cout);
    gp.display( std::cout);
    /////////////////////////////////////////////////////////////////////////
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a;
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Make grid
    dg::CylindricalGrid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.symmetric ? 1 : p.Nz, p.bcxN, p.bcyN, dg::PER);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    if( p.alpha_mag > 0.)
        mag = dg::geo::createModifiedSolovevField(gp, (1.-p.rho_damping)*mag.psip()(mag.R0(),0.), p.alpha_mag);

    //create RHS
    std::cout << "Constructing Explicit...\n";
    feltor::Explicit<Geometry, IDMatrix, DMatrix, DVec> feltor( grid, p, mag);
    std::cout << "Constructing Implicit...\n";
    feltor::Implicit<Geometry, IDMatrix, DMatrix, DVec> im( grid, p, mag);
    std::cout << "Done!\n";

    DVec result = dg::evaluate( dg::zero, grid);
    std::array<std::array<DVec,2>,2> y0;
    y0[0][0] = y0[0][1] =y0[1][0] =y0[1][1] = result;
    std::array<std::array<DVec,2>,2> b(y0);
    DVec weights = dg::create::weights( grid);
    solve_steady_state( feltor, im, y0, b, weights);

    ////////////////////////////////////////////////////////////////////
    return 0;

}
