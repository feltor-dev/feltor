#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG

#include <cusp/coo_matrix.h>
#include <cusp/print.h>

#include "dg/algorithm.h"
#include "dg/file/file.h"
#include "dg/geometries/geometries.h"

#include "parameters.h"
#include "heat.h"

int main( int argc, char* argv[])
{
    ////Parameter initialisation ///////////////////////////////////////
    //read input without comments
    if(!(( argc == 4) || ( argc == 5)) )
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [geomfile] [output.nc] [input.nc]\n";
        std::cerr << "OR "<< argv[0]<<" [inputfile] [geomfile] [output.nc] \n";
        return -1;
    }
    dg::file::WrappedJsonValue js = dg::file::file2Json(argv[1], dg::file::comments::are_forbidden);
    dg::file::WrappedJsonValue gs = dg::file::file2Json(argv[2], dg::file::comments::are_forbidden);
    const heat::Parameters p( js); p.display( std::cout);
    const dg::geo::solovev::Parameters gp(gs); gp.display( std::cout);
    ////////////////////////////set up computations//////////////////////

    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a;
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;

    //Make grids
    dg::CylindricalGrid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.Nz, p.bcx, p.bcy, dg::PER);
    dg::CylindricalGrid3d grid_out( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,
        p.n, p.Nx/p.cx, p.Ny/p.cy, p.Nz, p.bcx, p.bcy, dg::PER);
    dg::IDMatrix project = dg::create::projection( grid_out, grid);
    dg::DVec w3d    =  dg::create::volume(grid);
    dg::DVec w3dout =  dg::create::volume(grid_out);

    // /////////////////////get last temperature field of sim
    dg::DVec Tend( dg::evaluate(dg::zero,grid));
    //////////////////////////////open nc file//////////////////////////////////
    if (argc == 5)
    {
        dg::file::NC_Error_Handle errin;
        int ncidin;
        errin = nc_open( argv[4], NC_NOWRITE, &ncidin);
        //////////////read in and show inputfile und geomfile////////////
        size_t length;
        errin = nc_inq_attlen( ncidin, NC_GLOBAL, "inputfile", &length);
        std::string inputin(length, 'x');
        errin = nc_get_att_text( ncidin, NC_GLOBAL, "inputfile", &inputin[0]);
        errin = nc_inq_attlen( ncidin, NC_GLOBAL, "geomfile", &length);
        std::string geomin(length, 'x');
        errin = nc_get_att_text( ncidin, NC_GLOBAL, "geomfile", &geomin[0]);
        dg::file::WrappedJsonValue js = dg::file::string2Json(inputin, dg::file::comments::are_forbidden);
        dg::file::WrappedJsonValue gs = dg::file::string2Json(geomin, dg::file::comments::are_forbidden);
        std::cout << "input in"<<inputin<<std::endl;
        std::cout << "geome in"<<geomin <<std::endl;
        const heat::Parameters pin(js);
        const dg::geo::solovev::Parameters gpin(gs);
        size_t start3din[4]  = {pin.maxout, 0, 0, 0};
        size_t count3din[4]  = {1, pin.Nz, pin.n*pin.Ny, pin.n*pin.Nx};
        dg::CylindricalGrid3d grid_in( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,
            pin.n_out, pin.Nx_out, pin.Ny_out, pin.Nz_out, p.bcx, p.bcy, dg::PER);
        dg::IHMatrix interpolatef2c = dg::create::interpolation( grid, grid_in);//f2c
        dg::HVec TendIN = dg::evaluate( dg::zero, grid_in);
        //Now read Tend and interpolate from input grid to our grid
        int dataIDin;
        errin = nc_inq_varid(ncidin, "T", &dataIDin);
        errin = nc_get_vara_double( ncidin, dataIDin, start3din, count3din,
            TendIN.data());
        dg::HVec TendH = dg::evaluate( dg::zero, grid);
        dg::blas2::symv( interpolatef2c, TendIN, TendH);
        dg::assign( TendH, Tend);
        //now Tend lives on grid
        errin = nc_close(ncidin);
    }
    // /////////////////////create RHS
    std::cout << "Constructing Feltor...\n";
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField( gp);
    heat::Explicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec> ex( grid, p,mag); //initialize before diffusion!
    std::cout << "initialize implicit" << std::endl;
    heat::Implicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec > diffusion( grid, p,mag);
    std::cout << "Done!\n";

    //////////////////The initial field/////////////////////////////////
    //initial perturbation
    dg::Gaussian3d init0(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma_z, p.amp);
    dg::DVec y0 = dg::evaluate( init0, grid);
    ///////////////////TIME STEPPER
    heat::ImplicitSolver<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec> solver(
        diffusion, y0, p);
    dg::Adaptive<dg::ARKStep<dg::DVec>> adaptive(
        "ARK-4-2-3", y0);
    double dt = p.dt, dt_new = dt;

    ex.energies( y0);//now energies and potential are at time 0
    dg::DVec T0 = y0, T1(T0);
    double normT0 = dg::blas2::dot( T0, w3d, T0);
    //Now map quantities to values
    std::map<std::string, const dg::DVec*> v3d{ {"T", &y0} };
    const heat::Quantities& q = ex.quantities();
    double entropy0 = q.entropy, heat0 = q.energy; //at time 0
    double E0 = entropy0, accuracy = 0;
    dg::blas1::axpby( 1., y0, -1.,T0, T1);

    //Compute error to zero timestep
    double error = sqrt(dg::blas2::dot( w3d, T1)/normT0);
    double relerror=0.;
    if (argc==5)
    {
        dg::DVec Tdiff = Tend;
        dg::blas1::axpby( 1., y0, -1., Tend, Tdiff);
        relerror = sqrt(dg::blas2::dot( w3d, Tdiff)/dg::blas2::dot(w3dout,Tend));
    }
    std::map<std::string, const double*> v0d{
        {"heat", &q.energy}, {"entropy", &q.entropy},
        {"dissipation", &q.energy_diffusion},
        {"entropy_dissipation", &q.entropy_diffusion},
        {"accuracy", &accuracy}, {"error", &error}, {"relerror", &relerror}
    };
    //////////////////set up netcdf for output/////////////////////////////////////

    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_create( argv[3],NC_NETCDF4|NC_CLOBBER, &ncid);
    std::string input = js.toStyledString();
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    std::string geom = gs.toStyledString();
    err = nc_put_att_text( ncid, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim_ids[4], tvarID;
    err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid_out);

    //energy IDs
    int EtimeID, EtimevarID;
    err = dg::file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    std::map<std::string, int> id0d;
    for( auto name_value : v0d)
        err = nc_def_var( ncid, name_value.first.data(), NC_DOUBLE, 1, &EtimeID, &id0d[name_value.first]);
    std::map<std::string, int> id3d;
    for( auto name_value : v3d)
        err = nc_def_var( ncid, name_value.first.data(), NC_DOUBLE, 4, dim_ids, &id3d[name_value.first]);

    err = nc_enddef(ncid);
    ///////////////////////////////////first output/////////////////////////
    std::cout << "First output ... \n";
    size_t start[4] = {0, 0, 0, 0};
    size_t count[4] = {1, grid_out.Nz(), grid_out.n()*grid_out.Ny(),
                            grid_out.n()*grid_out.Nx()};

    //interpolate fine 2 coarse grid
    dg::DVec transferD( dg::evaluate(dg::zero, grid_out));
    dg::HVec transferH( dg::evaluate(dg::zero, grid_out));

    err = nc_open(argv[3], NC_WRITE, &ncid);
    dg::blas2::symv( project, *v3d["T"], transferD);
    dg::assign( transferD, transferH);
    err = nc_put_vara_double( ncid, id3d["T"], start, count, transferH.data());

    double time = 0;
    err = nc_put_vara_double( ncid, tvarID, start, count, &time);
    err = nc_put_vara_double( ncid, EtimevarID, start, count, &time);

    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    for( auto name_value : v0d)
        err = nc_put_vara_double( ncid, id0d[name_value.first], Estart, Ecount, name_value.second);
    err = nc_close(ncid);
    std::cout << "First write successful!\n";
    ///////////////////////////////////////Timeloop/////////////////////////////////
    dg::Timer t;
    t.tic();
    unsigned step = 0;
    for( unsigned i=1; i<=p.maxout; i++)
    {

#ifdef DG_BENCHMARK
        dg::Timer ti;
        ti.tic();
#endif//DG_BENCHMARK
        for( unsigned j=0; j<p.itstp; j++)
        {
            try{
                do
                {
                    dt = dt_new;
                    adaptive.step(std::tie(ex,diffusion,solver),time,y0,time,y0,dt_new,
                        dg::pid_control, dg::l2norm, p.rtol, 1e-10);
                    if( adaptive.failed())
                        std::cout << "Step Failed! REPEAT!\n";
                 }
                 while( adaptive.failed());
            }
            catch( dg::Fail& fail) {
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                err = nc_close(ncid);
                return -1;
            }
            step++;
            ex.energies(y0);//advance energies
            Estart[0] = step;
            double dEdt = (*v0d["entropy"] - E0)/dt;
            accuracy = 2.*fabs(
                            (dEdt - *v0d["entropy_dissipation"])/
                            (dEdt + *v0d["entropy_dissipation"]));
            E0 = *v0d["entropy"];
            //compute errors
            dg::blas1::axpby( 1., y0, -1.,T0, T1);
            error = sqrt(dg::blas2::dot( w3d, T1)/normT0);
            if (argc==5)
            {
                dg::DVec Tdiff = Tend;
                dg::blas1::axpby( 1., y0, -1., Tend, Tdiff);
                relerror = sqrt(dg::blas2::dot( w3d, Tdiff)/dg::blas2::dot(w3dout,Tend));
            }
            err = nc_open(argv[3], NC_WRITE, &ncid);
            err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
            for( auto name_value : v0d)
                err = nc_put_vara_double( ncid, id0d[name_value.first], Estart, Ecount, name_value.second);

            std::cout <<"(Q_tot-Q_0)/Q_0: "
                      << (q.energy-heat0)/heat0<<"\t";
            std::cout <<"(E_tot-E_0)/E_0: "
                      << (q.entropy-entropy0)/entropy0<<"\t";
            std::cout <<" d E/dt = " << dEdt
                      <<" Lambda = " << q.entropy_diffusion
                      <<" -> Accuracy: "<< accuracy
                      <<" -> error2t0: "<< error
                      <<" -> error2ref: "<< relerror <<"\n";
            err = nc_close(ncid);
        }
#ifdef DG_BENCHMARK
        ti.toc();
        std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
#endif//DG_BENCHMARK
        //////////////////////////write fields////////////////////////
        start[0] = i;
        dg::blas2::symv( project, *v3d["T"], transferD);
        dg::assign( transferD, transferH);

        err = nc_open(argv[3], NC_WRITE, &ncid);
        err = nc_put_vara_double( ncid, id3d["T"], start, count, transferH.data());
        err = nc_put_vara_double( ncid, tvarID, start, count, &time);
        err = nc_close(ncid);
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

