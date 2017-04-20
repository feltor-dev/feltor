#include <iostream>
#include <iomanip>

#include "dg/backend/timer.cuh"
#include "dg/functors.h"
#include "dg/backend/evaluation.cuh"
#include "dg/runge_kutta.h"
#include "dg/multistep.h"
#include "dg/backend/xspacelib.cuh"
#include "dg/backend/typedefs.cuh"

#include "dg/exceptions.h"

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
    ////////////////////////Parameter initialisation//////////////////////////
    Json::Reader reader;
    Json::Value js;
    if( argc != 3)
    {
        if(rank==0)std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [outputfile]\n";
        return -1;
    }
    else 
    {
        std::ifstream is(argv[1]);
        reader.parse( is, js, false); //read input without comments
    }
    std::string input = js.toStyledString(); //save input without comments, which is important if netcdf file is later read by another parser
    const Parameters p( js);
    p.display( std::cout);

    //////////////initiate solver/////////////////////////////////////////
    dg::Grid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    dg::DVec w2d( dg::create::weights(grid));
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
    dg::Shu<dg::DMatrix, dg::DVec> shu( grid, p.eps);
    //dg::AB< k, dg::DVec > ab( y0);
    dg::Diffusion< dg::DMatrix, dg::DVec > diff( grid, p.D);
    dg::Karniadakis< dg::DVec> ab( y0, y0.size(), 1e-10);
    ab.init( shu, diff, y0, p.dt);
    //ab( shu, y0, y1, p.dt);
    ab( shu, diff, y0); //make potential ready
    //y0.swap( y1); //y1 now contains value at zero time

    dg::DVec varphi( grid.size()), potential;
    double vorticity = dg::blas2::dot( one , w2d, ab.last());
    double enstrophy = 0.5*dg::blas2::dot( ab.last(), w2d, ab.last());
    double energy =    0.5*dg::blas2::dot( ab.last(), w2d, shu.potential()) ;
    potential = shu.potential();
    shu.arakawa().variation( potential, varphi);
    double variation = dg::blas2::dot( varphi, w2d, one );
    /////////////////////////////////////////////////////////////////
    file::T5trunc t5file( argv[2], input);
    double time = 0;
    unsigned step = 0;
    dg::DVec transferD[3] = { ab.last(), ab.last(), shu.potential()}; //intermediate transport locations
    dg::HVec output[3];
    for( int i=0;i<3; i++)
        dg::blas1::transfer( transferD[i], output[i]);
    t5file.write( output[0], output[1], output[2], time, grid.n()*grid.Nx(), grid.n()*grid.Ny());
    t5file.append( vorticity, enstrophy, energy, variation);
    /////////////////////////////set up netcdf/////////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    err = nc_create( argv[2],NC_NETCDF4|NC_CLOBBER, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    int dim_ids[3], tvarID;
    err = file::define_dimensions( ncid, dim_ids, &tvarID, grid);
    //field IDs
    std::string names[2] = {"electrons", "potential"}; 
    int dataIDs[2]; 
    for( unsigned i=0; i<2; i++){
        err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}

    //energy IDs
    int EtimeID, EtimevarID;
    err = file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    int energyID, massID, dissID, dEdtID;
    err = nc_def_var( ncid, "energy",      NC_DOUBLE, 1, &EtimeID, &energyID);
    err = nc_def_var( ncid, "vorticity",        NC_DOUBLE, 1, &EtimeID, &massID);
    err = nc_def_var( ncid, "enstrophy", NC_DOUBLE, 1, &EtimeID, &dissID);
    err = nc_def_var( ncid, "dEdt",        NC_DOUBLE, 1, &EtimeID, &dEdtID);
    err = nc_enddef(ncid);
    size_t start[3] = {0, 0, 0};
    size_t count[3] = {1, grid.n()*grid.Ny(), grid.n()*grid.Nx()};
    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    ///////////////////////////////////first output/////////////////////////
    //output all three fields
    std::vector<dg::DVec> transferD(4);
    std::vector<dg::HVec> output(4);
    transferD[0] = y1[0], transferD[1] = y1[1], transferD[2] = test.potential()[0], transferD[3] = test.potential()[0]; //electrons
    start[0] = 0;
    for( int k=0;k<4; k++)
    {
        dg::blas1::transfer( transferD[k], output[k]);
        err = nc_put_vara_double( ncid, dataIDs[k], start, count, output[k].data() );
    }
    err = nc_put_vara_double( ncid, tvarID, start, count, &time);
    err = nc_close(ncid);
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
            potential = shu.potential();
            energy    = 0.5*dg::blas2::dot( ab.last(), w2d, potential) ;
            shu.arakawa().variation(potential, varphi);
            variation = dg::blas2::dot( varphi, w2d, one );
            t5file.append( vorticity, enstrophy, energy, variation);
            if( energy>1e6) throw dg::Fail(p.eps);
        }
        step+=p.itstp;
        time += p.itstp*p.dt;
        //output all fields
        transferD[0] = ab.last(), transferD[1] = ab.last(), transferD[2] = shu.potential(); 
        for( int i=0;i<3; i++)
            dg::blas1::transfer( transferD[i], output[i]);
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
