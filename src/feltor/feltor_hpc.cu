#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG

// #include "draw/host_window.h"
//#include "draw/device_window.cuh"
#include "dg/backend/timer.cuh"
#include "dg/backend/xspacelib.cuh"
#include "file/read_input.h"
#include "file/file.h"


#include "feltor.cuh"
#include "bessel.h"
#include "parameters.h"
#include "geometry.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - writes outputs to a given outputfile using hdf5. 
        density fields are the real densities in XSPACE ( not logarithmic values)
*/

const unsigned k = 3;//!< a change in k needs a recompilation

int main( int argc, char* argv[])
{
    //Parameter initialisation
    std::vector<double> v,v3;
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
    p.display( std::cout);

    ////////////////////////////////set up computations///////////////////////////
    try{ v3 = file::read_input( "geometry_params.txt"); }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
//             std::cout << v3[i] << " ";
//             std::cout << std::endl;
        return -1;}

     const solovev::GeomParameters gp(v3);
    gp.display( std::cout);
    double Rmin=gp.R_0-(gp.boxscale)*gp.a;
    double Zmin=-(gp.boxscale)*gp.a*gp.elongation;
    double Rmax=gp.R_0+(gp.boxscale)*gp.a; 
    double Zmax=(gp.boxscale)*gp.a*gp.elongation;
    //Make grid
     dg::Grid3d<double > grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n, p.Nx, p.Ny, p.Nz, dg::DIR, dg::DIR, dg::PER);  
     
    //create RHS 
    eule::Feltor< dg::DVec > feltor( grid, p,gp); 
    eule::Rolkar< dg::DVec > rolkar( grid, p,gp);

    
    //with bath
      dg::BathRZ init0(16,16,p.Nz,Rmin,Zmin, 30.,15.,p.amp);
     //with zonal flow field
//       solovev::ZonalFlow init0(gp,p.amp);
    //with gaussians
//     dg::Gaussian3d init0( p.R_0, p.posY*p.a,    M_PI, p.sigma, p.sigma, M_PI/8.*p.m_par, p.amp );     
//     dg::Gaussian3d init1( p.R_0, -p.a*p.posY,   M_PI, p.sigma, p.sigma, M_PI/8.*p.m_par, p.amp ); 
//     dg::Gaussian3d init2( p.R_0+p.posX*p.a, 0., M_PI, p.sigma, p.sigma, M_PI/8.*p.m_par, p.amp ); 
//     dg::Gaussian3d init3( p.R_0-p.a*p.posX, 0., M_PI, p.sigma, p.sigma, M_PI/8.*p.m_par, p.amp ); 
    
//     solovev::Gradient grad(gp); //background gradient
    solovev::Nprofile grad(gp); //initial profile


    std::vector<dg::DVec> y0(4, dg::evaluate( grad, grid)), y1(y0); 

    dg::blas1::axpby( 1., (dg::DVec)dg::evaluate(init0, grid), 1., y0[0]);
    
//     dg::blas1::axpby( 1., (dg::DVec)dg::evaluate(init1, grid), 1., y0[0]);
//     dg::blas1::axpby( 1., (dg::DVec)dg::evaluate(init2, grid), 1., y0[0]);
//     dg::blas1::axpby( 1., (dg::DVec)dg::evaluate(init3, grid), 1., y0[0]);
   
    dg::blas1::axpby( 1., (dg::DVec)dg::evaluate(init0, grid), 1., y0[1]);
   
//     dg::blas1::axpby( 1., (dg::DVec)dg::evaluate(init1, grid), 1., y0[1]);
//     dg::blas1::axpby( 1., (dg::DVec)dg::evaluate(init2, grid), 1., y0[1]);
//     dg::blas1::axpby( 1., (dg::DVec)dg::evaluate(init3, grid), 1., y0[1]);

    dg::blas1::axpby( 0., y0[2], 0., y0[2]); //set U = 0
    dg::blas1::axpby( 0., y0[3], 0., y0[3]); //set U = 0

    feltor.log( y0, y0, 2); //transform to logarithmic values (ne and ni)
    
    dg::blas1::pointwiseDot(rolkar.iris(),y0[0],y0[0]); //is pupil on bath
    dg::blas1::pointwiseDot(rolkar.iris(),y0[1],y0[1]); //is pupil on bath
    
    dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    karniadakis.init( feltor, rolkar, y0, p.dt);
    double time = 0;
    unsigned step = 0;

    /////////////////////////////set up hdf5/////////////////////////////////
    file::T5trunc t5file( argv[2], input);
    std::vector<std::string> names(5); 
    names[0] = "electrons", names[1] = "ions", names[2] = "Ue", names[3] = "Ui";
    names[4] = "potential";
    std::vector<unsigned> dims( 3);
    dims[0] = grid.Nz(), dims[1] = grid.n()*grid.Ny(), dims[2] = grid.n()*grid.Nx();
    std::vector<dg::HVec> output(5); 
    ///////////////////////////////////first output/////////////////////////
    feltor.exp( y0,y0,2); //transform to correct values
    for( unsigned i=0; i<4; i++)
        output[i] = y0[i];
    output[4] = feltor.potential()[0];
    t5file.write( output, names, dims, time );
    t5file.append( feltor.mass(), feltor.mass_diffusion(), feltor.energy(), feltor.energy_diffusion());
    ///////////////////////////////////////Timeloop/////////////////////////////////
    dg::Timer t;
    t.tic();
    try
    {
#ifdef DG_BENCHMARK
    unsigned step = 0;
#endif //DG_BENCHMARK
    for( unsigned i=0; i<p.maxout; i++)
    {

#ifdef DG_BENCHMARK
        dg::Timer ti;
        ti.tic();
#endif//DG_BENCHMARK
        for( unsigned j=0; j<p.itstp; j++)
        {
            try{ karniadakis( feltor, rolkar, y0);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                break;
            }
            t5file.append( feltor.mass(), feltor.mass_diffusion(), feltor.energy(), feltor.energy_diffusion());
        }
        time += p.itstp*p.dt;
        feltor.exp( y0,y0,2); //transform to correct values
        for( unsigned i=0; i<4; i++)
            output[i] = y0[i];
        output[4] = feltor.potential()[0];
        t5file.write( output, names, dims, time );
#ifdef DG_BENCHMARK
        ti.toc();
        step+=p.itstp;
        std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
#endif//DG_BENCHMARK
    }
    }
    catch( dg::Fail& fail) { 
        std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
        std::cerr << "Does Simulation respect CFL condition?\n";
    }
    t.toc(); 
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout<<"s/step\n";

    return 0;

}

