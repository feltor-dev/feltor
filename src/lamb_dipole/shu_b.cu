#include <iostream>
#include <iomanip>
#include <sstream>
#include <thrust/host_vector.h>

#include "dg/algorithm.h"
#include "dg/file/json_utilities.h"

#ifndef WITHOUT_GLFW
#include "draw/host_window.h"
#endif
#include "dg/file/file.h"

#include "init.h"
#include "shu.cuh"


int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Value js;
    enum file::error mode = file::error::is_warning;
    if( argc == 1)
        file::file2Json( "input/default.json", js, file::comments::are_discarded);
    else
        file::file2Json( argv[1], js);
    std::cout << js <<std::endl;

    /////////////////////////////////////////////////////////////////
    dg::CartesianGrid2d grid = shu::createGrid( js, mode);
    dg::DVec w2d( dg::create::weights(grid));
    /////////////////////////////////////////////////////////////////

    std::string initial = file::get( mode, js, "init", "type", "lamb").asString();
    dg::HVec omega = shu::initial_conditions.at( initial)(js, mode, grid);


    dg::DVec y0( omega ), y1( y0);
    //subtract mean mass
    if( grid.bcx() == dg::PER && grid.bcy() == dg::PER)
    {
        double meanMass = dg::blas1::dot( y0, w2d)/(double)(grid.lx()*grid.ly());
        dg::blas1::axpby( -meanMass, 1., 1., y0);
    }
    //make solver and stepper
    shu::Shu<dg::CartesianGrid2d, dg::IDMatrix, dg::DMatrix, dg::DVec>
        shu( grid, js, mode);
    shu::Diffusion<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> diffusion( grid, js, mode);

    dg::Timer t;
    t.tic();
    shu( 0., y0, y1);
    t.toc();
    std::cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";
    double vorticity = dg::blas1::dot( w2d, y0);
    double enstrophy = 0.5*dg::blas2::dot( y0, w2d, y0);
    double energy =    0.5*dg::blas2::dot( y0, w2d, shu.potential()) ;
    dg::DVec varphi( grid.size());
    shu.variation( shu.potential(), varphi);
    double variation = 0.5*dg::blas1::dot( varphi, w2d);

    std::cout << "Total energy:     "<<energy<<"\n";
    std::cout << "Total enstrophy:  "<<enstrophy<<"\n";
    std::cout << "Total vorticity:  "<<vorticity<<"\n";
    std::cout << "Total variation:  "<<variation<<"\n";

    double time = 0;
    std::string stepper = file::get( mode, js, "timestepper", "stepper", "FilteredMultistep").asString();
    std::string regularization = file::get( mode, js, "regularization", "type", "moddal").asString();
    dg::ModalFilter<dg::DMatrix, dg::DVec> filter;
    dg::Karniadakis<dg::DVec> karniadakis;
    dg::ShuOsher<dg::DVec> shu_osher;
    dg::FilteredExplicitMultistep<dg::DVec> multistep;
    if( regularization == "modal")
    {
        double alpha = file::get( mode, js, "regularization", "alpha", 36).asDouble();
        double order = file::get( mode, js, "regularization", "order", 8).asDouble();
        double eta_c = file::get( mode, js, "regularization", "eta_c", 0.5).asDouble();
        filter.construct( alpha, eta_c, order, grid);
    }
    double dt = file::get( mode, js, "timestepper", "dt", 2e-3).asDouble();
    if( "Karniadakis" == stepper)
    {
        if( regularization != "viscosity")
        {
            throw dg::Error(dg::Message(_ping_)<<"Warning! Karniadakis only works with viscosity regularization! Exit now!");

            return -1;
        }
        double eps_time = file::get( mode, js, "timestepper", "eps_time", 1e-10).asDouble();
        karniadakis.construct( y0, y0.size(), eps_time);
        karniadakis.init( shu, diffusion, time, y0, dt);
    }
    else if( "Shu-Osher" == stepper)
    {
        shu_osher.construct( "SSPRK-3-3", y0);
    }
    else if( "FilteredMultistep" == stepper)
    {
        multistep.construct( "eBDF", 3, y0);
        multistep.init( shu, filter, time, y0, dt);
    }
    else
    {
        throw dg::Error(dg::Message(_ping_)<<"Error! Timestepper not recognized!\n");

        return -1;
    }
    unsigned maxout = file::get( mode, js, "output", "maxout", 100).asUInt();
    unsigned itstp = file::get( mode, js, "output", "itstp", 5).asUInt();
    std::string output = file::get( mode, js, "output", "type", "glfw").asString();
#ifndef WITHOUT_GLFW
    if( "glfw" == output)
    {
        ////////////////////////////////glfw//////////////////////////////
        //create visualisation vectors
        dg::DVec visual( grid.size());
        dg::HVec hvisual( grid.size());
        //transform vector to an equidistant grid
        std::stringstream title;
        GLFWwindow* w = draw::glfwInitAndCreateWindow(600, 600, "");
        draw::RenderHostData render( 1,1);
        draw::ColorMapRedBlueExt colors( 1.);
        dg::IDMatrix equidistant = dg::create::backscatter( grid );
        while (!glfwWindowShouldClose(w) && time < maxout*itstp*dt)
        {
            dg::blas2::symv( equidistant, y0, visual);
            colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
            //draw and swap buffers
            dg::blas1::transfer( visual, hvisual);
            render.renderQuad( hvisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
            title << "Time "<<time<< " \ttook "<<t.diff()/(double)itstp<<"\t per step";
            glfwSetWindowTitle(w, title.str().c_str());
            title.str("");
            glfwPollEvents();
            glfwSwapBuffers(w);
            //step
            t.tic();
            try{
            if( "Karniadakis" == stepper)
                for( unsigned i=0; i<itstp; i++)
                    karniadakis.step( shu, diffusion, time, y0);
            else if ( "FilteredMultistep" == stepper)
                for( unsigned i=0; i<itstp; i++)
                    multistep.step( shu, filter, time, y0);
            else if ( "Shu-Osher" == stepper)
                for( unsigned i=0; i<itstp; i++)
                    shu_osher.step( shu, filter, time, y0, time, y0, dt);
            }
            catch( dg::Fail& fail) {
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                return -1;
            }
            t.toc();
            //std::cout << "Timer for one step: "<<t.diff()/N<<"s\n";
        }
        glfwTerminate();
    }
    else
#endif //WITHOUT_GLFW
    {
        std::string outputfile;
        if( argc == 1 || argc == 2)
            outputfile = "shu.nc";
        else
            outputfile = argv[2];
        ////////////////////////////set up netcdf/////////////////////////////////////
        file::NC_Error_Handle err;
        int ncid;
        err = nc_create( outputfile.c_str(),NC_NETCDF4|NC_CLOBBER, &ncid);
        std::string input = js.toStyledString(); //save input without comments, which is important if netcdf file is later read by another parser
        err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
        int dim_ids[3], tvarID;
        int EtimeID, EtimevarID;
        err = file::define_dimensions( ncid, dim_ids, &tvarID, grid);
        err = file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
        //field IDs
        std::string names3d[2] = {"vorticity_field", "potential"};
        std::string names1d[4] = {"vorticity", "enstrophy", "energy", "variation"};
        int dataIDs[2], variableIDs[4];
        for( unsigned i=0; i<2; i++){
            err = nc_def_var( ncid, names3d[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}
        for( unsigned i=0; i<4; i++){
            err = nc_def_var( ncid, names1d[i].data(), NC_DOUBLE, 1, &EtimeID, &variableIDs[i]);}
        err = nc_enddef(ncid);
        size_t start[3] = {0, 0, 0};
        size_t count[3] = {1, grid.n()*grid.Ny(), grid.n()*grid.Nx()};
        size_t Estart[] = {0};
        size_t Ecount[] = {1};
        ///////////////////////////////////first output/////////////////////////
        std::vector<dg::HVec> transferH(2);
        dg::blas1::transfer( y0, transferH[0]);
        dg::blas1::transfer( shu.potential(), transferH[1]);
        for( int k=0;k<2; k++)
            err = nc_put_vara_double( ncid, dataIDs[k], start, count, transferH[k].data() );
        err = nc_put_vara_double( ncid, tvarID, start, count, &time);
        double output1d[4] = {vorticity, enstrophy, energy, variation};
        for( int k=0;k<4; k++)
            err = nc_put_vara_double( ncid, variableIDs[k], Estart, Ecount, &output1d[k] );
        err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
        ///////////////////////////////////timeloop/////////////////////////
        unsigned step=0;
        try{
        for( unsigned i=1; i<=maxout; i++)
        {

            dg::Timer ti;
            ti.tic();
            for( unsigned j=0; j<itstp; j++)
            {
                if( "Karniadakis" == stepper)
                    karniadakis.step( shu, diffusion, time, y0);
                else if ( "FilteredMultistep" == stepper)
                    multistep.step( shu, filter, time, y0);
                else if ( "Shu-Osher" == stepper)
                    shu_osher.step( shu, filter, time, y0, time, y0, dt);

                output1d[0] = dg::blas1::dot( w2d, y0);
                output1d[1] = 0.5*dg::blas2::dot( y0, w2d, y0);
                output1d[2] = 0.5*dg::blas2::dot( y0, w2d, shu.potential()) ;
                shu.variation(shu.potential(), varphi);
                output1d[3] = 0.5*dg::blas1::dot( varphi, w2d);
                Estart[0] += 1;
                for( int k=0;k<4; k++)
                    err = nc_put_vara_double( ncid, variableIDs[k], Estart, Ecount, &output1d[k] );
                err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
                if( energy>1e6)
                    throw dg::Error(dg::Message(_ping_)<<"Energy exploded! Exit\n");
            }
            step+=itstp;
            //output all fields
            dg::blas1::transfer( y0, transferH[0]);
            dg::blas1::transfer( shu.potential(), transferH[1]);
            start[0] = i;
            for( int k=0;k<2; k++)
                err = nc_put_vara_double( ncid, dataIDs[k], start, count, transferH[k].data() );
            err = nc_put_vara_double( ncid, tvarID, start, count, &time);
            ti.toc();
            std::cout << "\n\t Step "<<step <<" of "<<itstp*maxout <<" at time "<<time;
            std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)itstp<<"s\n\n"<<std::flush;
        }
        }
        catch( dg::Fail& fail) {
            std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
            std::cerr << "Does Simulation respect CFL condition?\n";
            err = nc_close(ncid);
            return -1;
        }
        err = nc_close(ncid);
    }
    ////////////////////////////////////////////////////////////////////
    std::cout << "Time "<<time<<std::endl;
    if( "lamb" == initial)
    {
        double posX = file::get( mode, js, "init", "posX", 0.5).asDouble();
        double posY = file::get( mode, js, "init", "posY", 0.8).asDouble();
        double R = file::get( mode, js, "init", "sigma", 0.1).asDouble();
        double U = file::get( mode, js, "init", "velocity", 1).asDouble();
        dg::Lamb lamb( posX*grid.lx(), posY*grid.ly() - U*time, R, U);
        dg::DVec sol = dg::evaluate( lamb, grid);
        dg::blas1::axpby( 1., y0, -1., sol);
        double error = dg::blas2::dot( sol, w2d, sol)/dg::blas2::dot( y0 , w2d, y0);
        std::cout << "Analytic error to solution "<<error<<std::endl;
        std::cout << "Relative enstrophy error is: "<<(0.5*dg::blas2::dot( w2d, y0) - lamb.enstrophy())/lamb.enstrophy()<<"\n";
        std::cout << "Relative energy error    is: "<<(0.5*dg::blas2::dot( shu.potential(), w2d, y0) - lamb.energy())/lamb.energy()<<"\n";
    }
    std::cout << "Absolute vorticity error is: "<<dg::blas1::dot( w2d, y0) - vorticity << "\n";
    std::cout << "Relative enstrophy error is: "<<(0.5*dg::blas2::dot( w2d, y0) - enstrophy)/enstrophy<<"\n";
    std::cout << "Relative energy error    is: "<<(0.5*dg::blas2::dot( shu.potential(), w2d, y0) - energy)/energy<<"\n";
    shu.variation(shu.potential(), varphi);
    std::cout << "Relative variation error is: "<<(0.5*dg::blas1::dot( varphi, w2d)-variation)/variation << std::endl;

    return 0;

}
