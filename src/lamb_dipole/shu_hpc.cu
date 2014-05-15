#include <iostream>
#include <iomanip>

#include "dg/timer.cuh"
#include "dg/functors.cuh"
#include "dg/evaluation.cuh"
#include "dg/rk.cuh"
#include "dg/karniadakis.cuh"
#include "dg/xspacelib.cuh"
#include "dg/typedefs.cuh"

#include "file/read_input.h"
#include "file/file.h"

#include "shu.cuh"
#include "parameters.h"



const unsigned k = 3;

double delta =0.05;
double rho =M_PI/15.;
double shearLayer(double x, double y){
    if( y<= M_PI)
        return delta*cos(x) - 1./rho/cosh( (y-M_PI/2.)/rho)/cosh( (y-M_PI/2.)/rho);
    return delta*cos(x) + 1./rho/cosh( (3.*M_PI/2.-y)/rho)/cosh( (3.*M_PI/2.-y)/rho);
}

int main( int argc, char * argv[])
{
    dg::Timer t;
    //input files
    std::vector<double> v;
    std::string input;
    if( argc != 3)
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [outputfile]\n";
        return -1;
    }
    else 
    {
        v = file::read_input( argv[1]);
        input = file::read_file( argv[1]);
    }
    const Parameters p( v);
    p.display();
    if( p.k != k)
    {
        std::cerr << "Time stepper needs recompilation!\n";
        return -1;
    }

    //initiate solver 
    dg::Grid2d<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    dg::DVec w2d( dg::create::w2d(grid));
    dg::Lamb lamb( p.posX*p.lx, p.posY*p.ly, p.R, p.U);
    //dg::HVec omega = dg::evaluate ( lamb, grid);
    dg::HVec omega = dg::evaluate ( shearLayer, grid);
    dg::DVec y0( omega );
    //subtract mean mass 
    const dg::DVec one = dg::evaluate( dg::one, grid);
    //if( p.bc_x == dg::PER && p.bc_y == dg::PER)
    //{
    //    double meanMass = dg::blas2::dot( y0, w2d, one)/(double)(p.lx*p.ly);
    //    dg::blas1::axpby( -meanMass, one, 1., y0);
    //}
    //make solver and stepper
    dg::Shu<dg::DVec> shu( grid, p.eps);
    //dg::AB< k, dg::DVec > ab( y0);
    dg::Diffusion< dg::DVec > diff( grid, p.D);
    dg::Karniadakis< dg::DVec> ab( y0, y0.size(), 1e-10);
    ab.init( shu, diff, y0, p.dt);
    //ab( shu, y0, y1, p.dt);
    ab( shu, diff, y0); //make potential ready
    //y0.swap( y1); //y1 now contains value at zero time

    dg::DVec varphi( grid.size());
    double vorticity = dg::blas2::dot( one , w2d, ab.last());
    double enstrophy = 0.5*dg::blas2::dot( ab.last(), w2d, ab.last());
    double energy =    0.5*dg::blas2::dot( ab.last(), w2d, shu.potential()) ;
    shu.arakawa().variation( shu.potential(), varphi);
    double variation = dg::blas2::dot( varphi, w2d, one );
    /////////////////////////////////////////////////////////////////
    file::T5trunc t5file( argv[2], input);
    double time = 0;
    unsigned step = 0;
    dg::HVec output[3] = { ab.last(), ab.last(), shu.potential()}; //intermediate transport locations
    t5file.write( output[0], output[1], output[2], time, grid.n()*grid.Nx(), grid.n()*grid.Ny());
    t5file.append( vorticity, enstrophy, energy, variation);
    try{
    for( unsigned i=0; i<p.maxout; i++)
    {

        dg::Timer ti;
        ti.tic();
        for( unsigned j=0; j<p.itstp; j++)
        {
            //ab( shu, y0, y1, p.dt);
            ab( shu, diff, y0);//one step further
            vorticity = dg::blas2::dot( one , w2d, ab.last());
            enstrophy = 0.5*dg::blas2::dot( ab.last(), w2d, ab.last());
            energy    = 0.5*dg::blas2::dot( ab.last(), w2d, shu.potential()) ;
            shu.arakawa().variation( shu.potential(), varphi);
            variation = dg::blas2::dot( varphi, w2d, one );
            t5file.append( vorticity, enstrophy, energy, variation);
            if( energy>1e6) throw dg::Fail(p.eps);
        }
        step+=p.itstp;
        time += p.itstp*p.dt;
        //output all fields
        output[0] = ab.last(), output[1] = ab.last(), output[2] = shu.potential(); 
        t5file.write( output[0], output[1], output[2], time, grid.n()*grid.Nx(), grid.n()*grid.Ny());
        ti.toc();
        std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
    }
    }
    catch( dg::Fail& fail) { 
        std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
        std::cerr << "Does Simulation respect CFL condition?\n";
    }

    return 0;

}
