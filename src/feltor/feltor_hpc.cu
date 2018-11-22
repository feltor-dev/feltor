#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <sstream>
#include <cmath>

#include "dg/file/nc_utilities.h"
#include "feltor.cuh"

int main( int argc, char* argv[])
{
    ////////////////////////Parameter initialisation//////////////////////////
    Json::Value js, gs;
    Json::CharReaderBuilder parser;
    parser["collectComments"] = false;
    std::string errs;
    if( argc != 4)
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [geomfile] [outputfile]\n";
        return -1;
    }
    else
    {
        std::ifstream is(argv[1]);
        std::ifstream ks(argv[2]);
        parseFromStream( parser, is, &js, &errs); //read input without comments
        parseFromStream( parser, ks, &gs, &errs); //read input without comments
    }
    const feltor::Parameters p( js);
    const dg::geo::solovev::Parameters gp(gs);
    p.display( std::cout);
    gp.display( std::cout);
    std::string input = js.toStyledString(), geom = gs.toStyledString();
    ////////////////////////////////set up computations///////////////////////////
    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a;
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Make grids
    dg::CylindricalGrid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.Nz, p.bcxN, p.bcyN, dg::PER);
    dg::CylindricalGrid3d grid_out( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI,
        p.n_out, p.Nx_out, p.Ny_out, p.Nz_out, p.bcxN, p.bcyN, dg::PER);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);

    //create RHS
    std::cout << "Constructing Explicit...\n";
    feltor::Explicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec>
        feltor( grid, p, mag); //initialize before im!
    std::cout << "Constructing Implicit...\n";
    feltor::Implicit< dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec >
        im( grid, p, mag);
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    //First the profile and the source (on the host since we want to output those)
    dg::HVec profile = dg::pullback( dg::geo::Compose<dg::LinearX>( mag.psip(), p.nprofamp/mag.psip()(mag.R0(), 0.), 0.), grid);
    dg::HVec xpoint_damping = dg::evaluate( dg::one, grid);
    if( gp.hasXpoint() )
        xpoint_damping = dg::pullback(
            dg::geo::ZCutter(-1.1*gp.elongation*gp.a), grid);
    dg::HVec source_damping = dg::pullback( dg::geo::TanhDamping(
        mag.psip(), -3.*p.alpha, p.alpha, -1.), grid);
    dg::blas1::pointwiseDot( xpoint_damping, source_damping, source_damping);
    if( p.omega_source != 0)
    {
        feltor.set_source( p.omega_source, profile, source_damping);
    }
    dg::HVec profile_damping = dg::pullback(dg::geo::TanhDamping(
        //first change coordinate from psi to (psi_0 - psip)/psi_0
        dg::geo::Compose<dg::LinearX>( mag.psip(), -1./mag.psip()(mag.R0(), 0.),1.),
        //then shift tanh
        p.rho_source-3.*p.alpha, p.alpha, -1.), grid);

    dg::blas1::pointwiseDot( xpoint_damping, profile_damping, profile_damping);
    dg::blas1::pointwiseDot( profile_damping, profile, profile);

    //Now perturbation
    dg::DVec ntilde = dg::evaluate(dg::zero,grid);
    if( p.initne == "blob" || p.initne == "straight blob")
    {
        dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
        dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
        if( p.initne == "blob")
            ntilde = feltor.ds().fieldaligned().evaluate( init0, gaussianZ,
                (unsigned)p.Nz/2, 3); //rounds =3 ->2*3-1
        if( p.initne == "straight blob")
            ntilde = feltor.ds().fieldaligned().evaluate( init0, gaussianZ,
                (unsigned)p.Nz/2, 1); //rounds =1 ->2*1-1
    }
    else if( p.initne == "turbulence")
    {
        dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1);
        dg::BathRZ init0(16,16,Rmin,Zmin, 30.,5.,p.amp);
        ntilde = feltor.ds().fieldaligned().evaluate( init0, gaussianZ,
            (unsigned)p.Nz/2, 1);
    }
    else if( p.initne == "zonal")
    {
        dg::geo::ZonalFlow init0(mag.psip(), p.amp, 0., p.k_psi);
        ntilde = dg::pullback( init0, grid);
    }
    else
        std::cerr <<"WARNING: Unknown initial condition!\n";
    std::array<std::array<dg::DVec,2>,2> y0;
    y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::construct<dg::DVec>(profile);
    dg::blas1::axpby( 1., ntilde, 1., y0[0][0]); //sum up background and perturbation
    std::cout << "initialize ni" << std::endl;
    if( p.initphi == "zero")
        feltor.initializeni( y0[0][0], y0[0][1]);
    else if( p.initphi == "balance")
        dg::blas1::copy( y0[0][0], y0[0][1]); //set N_i = n_e
    else
        std::cerr <<"WARNING: Unknown initial condition for phi!\n";

    dg::blas1::copy( 0., y0[1][0]); //set Ue = 0
    dg::blas1::copy( 0., y0[1][1]); //set Ui = 0
    ////////////map quantities to output/////////////////
    //since we map pointers we don't need to update those later
    std::map<std::string, const dg::DVec* > v4d;
    v4d["electrons"] = &feltor.fields()[0][0], v4d["ions"] = &feltor.fields()[0][1];
    v4d["Ue"] = &feltor.fields()[1][0],        v4d["Ui"] = &feltor.fields()[1][1];
    v4d["potential"] = &feltor.potential()[0];
    v4d["induction"] = &feltor.induction();
    const feltor::Quantities& q = feltor.quantities();
    double dEdt = 0, accuracy = 0, dMdt = 0, accuracyM  = 0;
    std::map<std::string, const double*> v0d{
        {"energy", &q.energy}, {"ediff", &q.ediff},
        {"mass", &q.mass}, {"diff", &q.diff}, {"Apar", &q.Apar},
        {"Se", &q.S[0]}, {"Si", &q.S[1]}, {"Uperp", &q.Tperp},
        {"Upare", &q.Tpar[0]}, {"Upari", &q.Tpar[1]},
        {"dEdt", &dEdt}, {"accuracy", &accuracy},
        {"aligned", &q.aligned}
    };
    /////////////////////////////set up netcdf/////////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    err = nc_create( argv[3],NC_NETCDF4|NC_CLOBBER, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    err = nc_put_att_text( ncid, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim_ids[4], tvarID;
    {   //output 3d variables into file
        dg::geo::BFieldR fieldR(mag);
        dg::geo::BFieldZ fieldZ(mag);
        dg::geo::BFieldP fieldP(mag);

        dg::HVec vecR = dg::pullback( fieldR, grid_out);
        dg::HVec vecZ = dg::pullback( fieldZ, grid_out);
        dg::HVec vecP = dg::pullback( fieldP, grid_out);
        dg::HVec psip = dg::pullback( mag.psip(), grid_out);
        std::map<std::string, const dg::HVec*> v3d{
            {"BR", &vecR}, {"BZ", &vecZ}, {"BP", &vecP},
            {"Psip", &psip}, {"Nprof", &profile }, {"Source", &source_damping }
        };
        err = file::define_dimensions( ncid, dim_ids, &tvarID, grid_out);
        for( auto pair : v3d)
        {
            int vecID;
            err = nc_def_var( ncid, pair.first.data(), NC_DOUBLE, 3, &dim_ids[1], &vecID);
            err = nc_enddef( ncid);
            err = nc_put_var_double( ncid, vecID, pair.second->data());
            err = nc_redef(ncid);

        }
    }

    //field IDs
    int EtimeID, EtimevarID;
    err = file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    std::map<std::string, int> id0d, id4d;
    for( auto pair : v0d)
        err = nc_def_var( ncid, pair.first.data(), NC_DOUBLE, 1, &EtimeID, &id0d[pair.first]);
    for( auto pair : v4d)
        err = nc_def_var( ncid, pair.first.data(), NC_DOUBLE, 4, dim_ids, &id4d[pair.first]);
    err = nc_enddef(ncid);
    ///////////////////////////////////first output/////////////////////////
    double time = 0, dt_new = p.dt, dt = 0;
    std::cout << "First output ... \n";
    //first, update quantities in feltor
    {
        std::array<std::array<dg::DVec,2>,2> y1(y0);
        feltor( time, y0, y1);
        feltor.update_quantities();
    }
    q.display(std::cout);
    double energy0 = q.energy, mass0 = q.mass, E0 = energy0, M0 = mass0;
    size_t start[4] = {0, 0, 0, 0};
    size_t count[4] = {1, grid_out.Nz(), grid_out.n()*grid_out.Ny(),
        grid_out.n()*grid_out.Nx()};
    dg::DVec transferD( dg::evaluate(dg::zero, grid_out));
    dg::HVec transferH( dg::evaluate(dg::zero, grid_out));
    dg::IDMatrix project = dg::create::projection( grid_out, grid);
    for( auto pair : v4d)
    {
        dg::blas2::symv( project, *pair.second, transferD);
        dg::assign( transferD, transferH);
        err = nc_put_vara_double( ncid, id4d[pair.first], start, count, transferH.data() );
    }
    err = nc_put_vara_double( ncid, tvarID, start, count, &time);
    err = nc_put_vara_double( ncid, EtimevarID, start, count, &time);
    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    for( auto pair : v0d)
        err = nc_put_vara_double( ncid, id0d[pair.first], Estart, Ecount, pair.second);
    err = nc_close(ncid);
    std::cout << "First write successful!\n";
    ///////////////////////////////////////Timeloop/////////////////////////////////
    dg::Adaptive< dg::ARKStep<std::array<std::array<dg::DVec,2>,2>> > adaptive(
        "ARK-4-2-3", y0, y0[0][0].size(), p.eps_time);
    dg::Timer t;
    t.tic();
    unsigned step = 0;
    q.display(std::cout);
    for( unsigned i=1; i<=p.maxout; i++)
    {

        dg::Timer ti;
        ti.tic();
        for( unsigned j=0; j<p.itstp; j++)
        {
            try{
                do
                {
                    dt = dt_new;
                    adaptive.step( feltor, im, time, y0, time, y0, dt_new,
                        dg::pid_control, dg::l2norm, p.rtol, 1e-10);
                    if( adaptive.failed())
                        std::cout << "FAILED STEP! REPEAT!\n";
                }while ( adaptive.failed());
            }
            catch( dg::Fail& fail) {
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                err = nc_close(ncid);
                return -1;
            }
            step++;

            feltor.update_quantities();
            dEdt = (*v0d["energy"] - E0)/dt, dMdt = (*v0d["mass"] - M0)/dt;
            E0 = *v0d["energy"], M0 = *v0d["mass"];
            accuracy  = 2.*fabs( (dEdt - *v0d["ediff"])/( dEdt + *v0d["ediff"]));
            accuracyM = 2.*fabs( (dMdt - *v0d["diff"])/( dMdt + *v0d["diff"]));
            err = nc_open(argv[3], NC_WRITE, &ncid);
            Estart[0] = step;
            err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
            for( auto pair : v0d)
                err = nc_put_vara_double( ncid, id0d[pair.first], Estart, Ecount, pair.second);

            q.display(std::cout);
            std::cout << "(m_tot-m_0)/m_0: "<< (*v0d["mass"]-mass0)/mass0<<"\t";
            std::cout << "(E_tot-E_0)/E_0: "<< (*v0d["energy"]-energy0)/energy0<<"\t";
            std::cout <<" d E/dt = " << dEdt
                      <<" Lambda = " << *v0d["ediff"]
                      <<" -> Accuracy: " << accuracy << "\n";
            std::cout <<" d M/dt = " << dMdt
                      <<" Lambda = " << *v0d["diff"]
                      <<" -> Accuracy: " << accuracyM << "\n";
            err = nc_close(ncid);

        }
        ti.toc();
        std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout
                  << " at time "<<time;
        std::cout << "\n\t Average time for one step: "
                  << ti.diff()/(double)p.itstp<<"s";
        ti.tic();
        //////////////////////////write fields////////////////////////
        start[0] = i;
        err = nc_open(argv[3], NC_WRITE, &ncid);
        err = nc_put_vara_double( ncid, tvarID, start, count, &time);
        for( auto pair : v4d)
        {
            dg::blas2::symv( project, *pair.second, transferD);
            dg::assign( transferD, transferH);
            err = nc_put_vara_double( ncid, id4d[pair.first], start, count, transferH.data() );
        }
        err = nc_close(ncid);
        ti.toc();
        std::cout << "\n\t Time for output: "<<ti.diff()<<"s\n\n"<<std::flush;
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
