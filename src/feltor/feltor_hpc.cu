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

    //create RHS
    std::cout << "Constructing Explicit...\n";
    feltor::Explicit<dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec>
        feltor( grid, p, gp); //initialize before im!
    std::cout << "Constructing Implicit...\n";
    feltor::Implicit< dg::CylindricalGrid3d, dg::IDMatrix, dg::DMatrix, dg::DVec >
        im( grid, p, gp);
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    std::array<std::array<dg::DVec,2>,2> y0;
    dg::DVec helper(dg::evaluate(dg::zero,grid));
    //perturbation
    dg::GaussianZ gaussianZ( 0., p.sigma_z*M_PI, 1); //modulation along fieldline
    if( p.initni == "blob" || p.initni == "straight blob")
    {
        dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);
        if( p.initni == "blob")
            helper = feltor.ds().fieldaligned().evaluate( init0, gaussianZ,
                (unsigned)p.Nz/2, 3); //rounds =3 ->2*3-1
        if( p.initni == "straight blob")
            helper = feltor.ds().fieldaligned().evaluate( init0, gaussianZ,
                (unsigned)p.Nz/2, 1); //rounds =1 ->2*1-1
    }
    else if( p.initni == "turbulence")
    {
        dg::BathRZ init0(16,16,Rmin,Zmin, 30.,5.,p.amp);
        helper = feltor.ds().fieldaligned().evaluate( init0, gaussianZ,
            (unsigned)p.Nz/2, 1);
    }
    else if( p.initni == "zonal")
    {
        dg::geo::ZonalFlow init0(p.amp, p.k_psi, gp, dg::geo::solovev::Psip(gp));
        helper = feltor.ds().fieldaligned().evaluate( init0, gaussianZ,
            (unsigned)p.Nz/2, 1);
    }
    else
        std::cerr <<"WARNING: Unknown initial condition for Ni!\n";
    dg::geo::Nprofile prof(p.bgprofamp, p.nprofileamp, gp, dg::geo::solovev::Psip(gp)); //initial background profile
    y0[0][0] = y0[0][1] = y0[1][0] = y0[1][1] = dg::evaluate( prof, grid);
    dg::blas1::axpby( 1., helper, 1., y0[0][1]); //sum up background and perturbation
    dg::blas1::plus(y0[0][1], -1); //initialize ni-1
    if( p.initni == "turbulence" || p.initni == "zonal") //Cut initialization outside separatrix
    {
        dg::DVec damping = dg::evaluate( dg::geo::GaussianProfXDamping(dg::geo::solovev::Psip(gp), gp), grid);
        dg::blas1::pointwiseDot(damping,y0[0][1], y0[0][1]);
    }
    std::cout << "intiialize ne" << std::endl;
    if( p.initphi == "zero")  feltor.initializene( y0[0][1], y0[0][0]);
    else if( p.initphi == "balance") dg::blas1::copy( y0[0][1], y0[0][0]); //set n_e = N_i
    else
        std::cerr <<"WARNING: Unknown initial condition for phi!\n";

    dg::blas1::copy( 0., y0[1][0]); //set Ue = 0
    dg::blas1::copy( 0., y0[1][1]); //set Ui = 0
    /////////////////////////////set up netcdf/////////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    err = nc_create( argv[3],NC_NETCDF4|NC_CLOBBER, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    err = nc_put_att_text( ncid, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim_ids[4], tvarID;
    {
        err = file::define_dimensions( ncid, dim_ids, &tvarID, grid_out);
        dg::geo::TokamakMagneticField c=dg::geo::createSolovevField(gp);
        dg::geo::BFieldR fieldR(c);
        dg::geo::BFieldZ fieldZ(c);
        dg::geo::BFieldP fieldP(c);

        dg::HVec vecR = dg::evaluate( fieldR, grid_out);
        dg::HVec vecZ = dg::evaluate( fieldZ, grid_out);
        dg::HVec vecP = dg::evaluate( fieldP, grid_out);
        int vecID[3];
        err = nc_def_var( ncid, "BR", NC_DOUBLE, 3, &dim_ids[1], &vecID[0]);
        err = nc_def_var( ncid, "BZ", NC_DOUBLE, 3, &dim_ids[1], &vecID[1]);
        err = nc_def_var( ncid, "BP", NC_DOUBLE, 3, &dim_ids[1], &vecID[2]);
        err = nc_enddef( ncid);
        err = nc_put_var_double( ncid, vecID[0], vecR.data());
        err = nc_put_var_double( ncid, vecID[1], vecZ.data());
        err = nc_put_var_double( ncid, vecID[2], vecP.data());
        err = nc_redef(ncid);
    }

    //field IDs
    std::string names4d [] = {"electrons", "ions", "Ue", "Ui", "potential"};
    std::map<std::string, int> id4d;
    for( auto name : names4d)
        err = nc_def_var( ncid, name.data(), NC_DOUBLE, 4, dim_ids, &id4d[name]);
    std::map<std::string, int> id0d;
    std::string names0d [] = {"energy", "ediff", "mass", "diff", "Se", "Si",
        "Uperp", "Upare", "Upari", "dEdt", "accuracy", "aligned"};
    int EtimeID, EtimevarID;
    err = file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    for( auto name : names0d)
        err = nc_def_var( ncid, name.data(), NC_DOUBLE, 1, &EtimeID, &id0d[name]);
    err = nc_enddef(ncid);
    ////////////map quantities to output/////////////////
    //since values take references we don't need to update those later
    std::map<std::string, const dg::DVec* > v4d;
    for( unsigned i=0; i<2; i++)
    {
        v4d[names4d[  i]] = &(y0[0][i]);
        v4d[names4d[2+i]] = &(y0[1][i]);
    }
    v4d["potential"] = &feltor.potential()[0];
    const feltor::Quantities& q = feltor.quantities();
    double dEdt = 0, accuracy = 0;
    std::map<std::string, const double*> v0d{
        {"energy", &q.energy}, {"ediff", &q.ediff},
        {"mass", &q.mass}, {"diff", &q.diff},
        {"Se", &q.S[0]}, {"Si", &q.S[1]}, {"Uperp", &q.Tperp},
        {"Upare", &q.Tpar[0]}, {"Upari", &q.Tpar[1]},
        {"dEdt", &dEdt}, {"accuracy", &accuracy},
        {"aligned", &q.aligned}
    };
    ///////////////////////////////////first output/////////////////////////
    double time = 0, dt_new = p.dt, dt = 0;
    std::cout << "First output ... \n";
    //first, update quantities in feltor
    {
        std::array<std::array<dg::DVec,2>,2> y1(y0);
        feltor( time, y0, y1);
    }
    double energy0 = q.energy, mass0 = q.mass, E0 = energy0;
    size_t start[4] = {0, 0, 0, 0};
    size_t count[4] = {1, grid_out.Nz(), grid_out.n()*grid_out.Ny(),
        grid_out.n()*grid_out.Nx()};
    dg::DVec transferD( dg::evaluate(dg::zero, grid_out));
    dg::HVec transferH( dg::evaluate(dg::zero, grid_out));
    dg::IDMatrix project = dg::create::projection( grid_out, grid);
    for( auto name : names4d)
    {
        dg::blas2::symv( project, *v4d[name], transferD);
        dg::assign( transferD, transferH);
        err = nc_put_vara_double( ncid, id4d[name], start, count, transferH.data() );
    }
    err = nc_put_vara_double( ncid, tvarID, start, count, &time);
    err = nc_put_vara_double( ncid, EtimevarID, start, count, &time);
    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    for( auto name : names0d)
        err = nc_put_vara_double( ncid, id0d[name], Estart, Ecount, v0d[name]);
    err = nc_close(ncid);
    std::cout << "First write successful!\n";
    ///////////////////////////////////////Timeloop/////////////////////////////////
    dg::Adaptive< dg::ARKStep<std::array<std::array<dg::DVec,2>,2>> > adaptive(
        "ARK-4-2-3", y0, y0[0][0].size(), p.eps_time);
    dg::Timer t;
    t.tic();
    unsigned step = 0;
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

            dEdt = (*v0d["energy"] - E0)/dt;
            E0 = *v0d["energy"];
            accuracy = 2.*fabs( (dEdt - *v0d["ediff"])/( dEdt + *v0d["ediff"]));
            err = nc_open(argv[3], NC_WRITE, &ncid);
            Estart[0] = step;
            err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
            for( auto name : names0d)
                err = nc_put_vara_double( ncid, id0d[name], Estart, Ecount, v0d[name]);

            q.display(std::cout);
            std::cout << "(m_tot-m_0)/m_0: "<< (*v0d["mass"]-mass0)/mass0<<"\t";
            std::cout << "(E_tot-E_0)/E_0: "<< (*v0d["energy"]-energy0)/energy0<<"\t";
            std::cout <<" d E/dt = " << dEdt <<" Lambda = "
                      << *v0d["ediff"] <<" -> Accuracy: "
                      << accuracy << "\n";
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
        for( auto name : names4d)
        {
            dg::blas2::symv( project, *v4d[name], transferD);
            dg::assign( transferD, transferH);
            err = nc_put_vara_double( ncid, id4d[name], start, count, transferH.data() );
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
