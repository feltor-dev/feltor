#include <iostream>
#include <iomanip>
#include <sstream>
#include <thrust/host_vector.h>

#include "dg/algorithm.h"

#ifndef WITHOUT_GLFW
#include "draw/host_window.h"
#endif
#include "dg/file/file.h"

#include "init.h"
#include "diag.h"
#include "shu.h"


int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Value js;
    if( argc == 1)
        dg::file::file2Json( "input/default.json", js, dg::file::comments::are_discarded);
    else
        dg::file::file2Json( argv[1], js);
    std::cout << js <<std::endl;
    dg::file::WrappedJsonValue ws( js, dg::file::error::is_throw);

    /////////////////////////////////////////////////////////////////
    dg::CartesianGrid2d grid = shu::createGrid( ws["grid"]);
    dg::DVec w2d( dg::create::weights(grid));
    /////////////////////////////////////////////////////////////////

    dg::HVec omega;
    try{
        omega = shu::initial_conditions( ws["init"], grid);
    }
    catch ( std::exception& e) {
        std::cerr << e.what()<<"\n";
        std::cerr << "Exit now!\n";
        return -1;
    }

    dg::DVec y0( omega ), y1( y0);
    //subtract mean mass
    if( grid.bcx() == dg::PER && grid.bcy() == dg::PER)
    {
        double meanMass = dg::blas1::dot( y0, w2d)/(double)(grid.lx()*grid.ly());
        dg::blas1::axpby( -meanMass, 1., 1., y0);
    }
    //make solver and stepper
    shu::Shu<dg::CartesianGrid2d, dg::DMatrix, dg::DVec>
        shu( grid, ws);
    shu::Diffusion<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> diffusion( shu);
    if( "mms" == ws["init"]["type"].asString())
    {
        double sigma = ws["init"].get( "sigma", 0.2).asDouble();
        double velocity = ws["init"].get( "velocity", 0.1).asDouble();
        shu.set_mms_source( sigma, velocity, grid.ly());
    }

    double time = 0;
    shu::Variables var = {shu, grid, y0, time, w2d, 0., ws};
    dg::Timer t;
    t.tic();
    shu( 0., y0, y1);
    t.toc();
    var.duration = t.diff();
    for( auto record : shu::diagnostics1d_list)
    {
        double result = record.function( var);
        std::cout  << "Diagnostics "<<record.name<<" "<<result<<"\n";
    }

    /// ////////////// Initialize timestepper ///////////////////////
    std::string stepper = ws[ "timestepper"].get( "type", "FilteredExplicitMultistep").asString();
    std::string tableau = ws[ "timestepper"].get( "tableau", "ImEx-BDF-3-3").asString();
    std::string regularization = ws[ "regularization"].get( "type", "modal").asString();
    dg::ModalFilter<dg::DMatrix, dg::DVec> filter;
    dg::IdentityFilter identity;
    bool apply_filter = false;
    dg::ImExMultistep<dg::DVec, dg::AndersonSolver<dg::DVec>> imex;
    dg::ShuOsher<dg::DVec> shu_osher;
    dg::FilteredExplicitMultistep<dg::DVec> multistep;
    dg::FilteredImplicitMultistep<dg::DVec, dg::AndersonSolver<dg::DVec>> multistep_implicit;
    dg::Adaptive<dg::ERKStep<dg::DVec> > adaptive;
    dg::Adaptive<dg::ARKStep<dg::DVec> > adaptive_imex;
    dg::Adaptive<dg::DIRKStep<dg::DVec, dg::AndersonSolver<dg::DVec> >> adaptive_implicit;
    if( regularization == "modal")
    {
        double alpha = ws[ "regularization"].get( "alpha", 36).asDouble();
        double order = ws[ "regularization"].get( "order", 8).asDouble();
        double eta_c = ws[ "regularization"].get( "eta_c", 0.5).asDouble();
        filter.construct( dg::ExponentialFilter(alpha, eta_c, order, grid.n()), grid);
        apply_filter = true;
        if( stepper != "FilteredExplicitMultistep" && stepper != "Shu-Osher" && stepper != "FilteredImplicitMultistep" )
        {
            throw std::runtime_error( "Error: modal regularization only works with either FilteredExplicit- or -ImplicitMultistep or Shu-Osher!");
            return -1;
        }
    }
    else if( regularization != "viscosity")
        throw std::runtime_error( "ERROR: Unkown regularization type "+regularization);

    double dt = 0.;
    double rtol = 1., atol = 1.;
    if( "ImExMultistep" == stepper)
    {
        dt = ws[ "timestepper"].get( "dt", 2e-3).asDouble();
        unsigned mMax = ws["timestepper"].get( "mMax", 8).asUInt();
        unsigned max_iter = ws["timestepper"].get( "max_iter", 1000).asUInt();
        double damping = ws["timestepper"].get( "damping", 1e-5).asDouble();
        double eps_time = ws[ "timestepper"].get( "eps_time", 1e-10).asDouble();
        //imex.construct( tableau, y0, y0.size(), eps_time);
        imex.construct( tableau, y0, mMax, eps_time, max_iter, damping, mMax );
        imex.init( shu, diffusion, time, y0, dt);
    }
    else if( "Shu-Osher" == stepper)
    {
        dt = ws[ "timestepper"].get( "dt", 2e-3).asDouble();
        shu_osher.construct( tableau, y0);
    }
    else if( "FilteredExplicitMultistep" == stepper)
    {
        dt = ws[ "timestepper"].get( "dt", 2e-3).asDouble();
        multistep.construct( tableau, y0);
        if( apply_filter)
            multistep.init( shu, filter, time, y0, dt);
        else
            multistep.init( shu, identity, time, y0, dt);
    }
    else if( "FilteredImplicitMultistep" == stepper)
    {
        dt = ws[ "timestepper"].get( "dt", 2e-3).asDouble();
        unsigned mMax = ws["timestepper"].get( "mMax", 8).asUInt();
        unsigned max_iter = ws["timestepper"].get( "max_iter", 1000).asUInt();
        double damping = ws["timestepper"].get( "damping", 1e-5).asDouble();
        double eps_time = ws[ "timestepper"].get( "eps_time", 1e-10).asDouble();
        multistep_implicit.construct( tableau, y0, mMax, eps_time, max_iter, damping, mMax );
        if( apply_filter)
            multistep_implicit.init( shu, filter, time, y0, dt);
        else
            multistep_implicit.init( shu, identity, time, y0, dt);
    }
    else if( "ERK" == stepper)
    {
        rtol = ws["timestepper"].get(rtol, 1e-5).asDouble();
        atol = ws["timestepper"].get(atol, 1e-5).asDouble();
        adaptive.construct( tableau, y0);
    }
    else if( "ARK" == stepper)
    {
        double eps_time = ws[ "timestepper"].get( "eps_time", 1e-10).asDouble();
        rtol = ws["timestepper"].get(rtol, 1e-5).asDouble();
        atol = ws["timestepper"].get(atol, 1e-5).asDouble();
        adaptive_imex.construct( tableau, y0, y0.size(), eps_time);
    }
    else if ( "DIRK" == stepper)
    {
        unsigned mMax = ws["timestepper"].get( "mMax", 8).asUInt();
        unsigned max_iter = ws["timestepper"].get( "max_iter", 1000).asUInt();
        double damping = ws["timestepper"].get( "damping", 1e-5).asDouble();
        double eps_time = ws[ "timestepper"].get( "eps_time", 1e-10).asDouble();
        rtol = ws["timestepper"].get(rtol, 1e-5).asDouble();
        atol = ws["timestepper"].get(atol, 1e-5).asDouble();
        adaptive_implicit.construct( tableau, y0, mMax, eps_time, max_iter, damping, mMax);
    }
    else
    {
        throw std::runtime_error( "Error! Timestepper "+stepper+" not recognized!\n");

        return -1;
    }
    unsigned maxout =    ws[ "output"].get( "maxout", 100).asUInt();
    unsigned itstp =     ws[ "output"].get( "itstp", 5).asUInt();
    std::string output = ws[ "output"].get( "type", "glfw").asString();
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
            dg::assign( visual, hvisual);
            render.renderQuad( hvisual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
            title << "Time "<<time<< " \ttook "<<t.diff()/(double)itstp<<"\t per step";
            glfwSetWindowTitle(w, title.str().c_str());
            title.str("");
            glfwPollEvents();
            glfwSwapBuffers(w);
            //step
            t.tic();
            try{
                for( unsigned j=0; j<itstp; j++)
                {
                    if( "ImExMultistep" == stepper)
                        imex.step( shu, diffusion, time, y0);
                    else if ( "FilteredExplicitMultistep" == stepper)
                    {
                        if( apply_filter)
                            multistep.step( shu, filter, time, y0);
                        else
                            multistep.step( shu, identity, time, y0);
                    }
                    else if ( "Shu-Osher" == stepper)
                    {
                        if( apply_filter)
                            shu_osher.step( shu, filter, time, y0, time, y0, dt);
                        else
                            shu_osher.step( shu, identity, time, y0, time, y0, dt);
                    }
                    else if ( "FilteredImplicitMultistep" == stepper)
                    {
                        if( apply_filter)
                            multistep_implicit.step( shu, filter, time, y0);
                        else
                            multistep_implicit.step( shu, identity, time, y0);
                    }
                }
            } catch( dg::Fail& fail) {
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                return -1;
            }
            t.toc();
            var.duration = t.diff()/(double)itstp;
        }
        glfwTerminate();
    }
#endif //WITHOUT_GLFW
    if( "netcdf" == output)
    {
        std::string inputfile = js.toStyledString(); //save input without comments, which is important if netcdf file is later read by another parser
        std::string outputfile;
        if( argc == 1 || argc == 2)
            outputfile = "shu.nc";
        else
            outputfile = argv[2];
        /// //////////////////////set up netcdf/////////////////////////////////////
        dg::file::NC_Error_Handle err;
        int ncid=-1;
        try{
            err = nc_create( outputfile.c_str(),NC_NETCDF4|NC_CLOBBER, &ncid);
        }catch( std::exception& e)
        {
            std::cerr << "ERROR creating file "<<outputfile<<std::endl;
            std::cerr << e.what()<<std::endl;
           return -1;
        }
        /// Set global attributes
        std::map<std::string, std::string> att;
        att["title"] = "Output file of feltor/src/lamb_dipole/shu_b.cu";
        att["Conventions"] = "CF-1.7";
        ///Get local time and begin file history
        auto ttt = std::time(nullptr);
        auto tm = *std::localtime(&ttt);

        std::ostringstream oss;
        ///time string  + program-name + args
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        for( int i=0; i<argc; i++) oss << " "<<argv[i];
        att["history"] = oss.str();
        att["comment"] = "Find more info in feltor/src/lamb_dipole/shu.tex";
        att["source"] = "FELTOR";
        att["references"] = "https://github.com/feltor-dev/feltor";
        att["inputfile"] = inputfile;
        for( auto pair : att)
            err = nc_put_att_text( ncid, NC_GLOBAL,
                pair.first.data(), pair.second.size(), pair.second.data());

        int dim_ids[3], tvarID;
        std::map<std::string, int> id1d, id3d;
        err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid,
                {"time", "y", "x"});

        //Create field IDs
        for( auto& record : shu::diagnostics2d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            id3d[name] = 0;
            err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids,
                    &id3d.at(name));
            err = nc_put_att_text( ncid, id3d.at(name), "long_name", long_name.size(),
                long_name.data());
        }
        for( auto& record : shu::diagnostics1d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            id1d[name] = 0;
            err = nc_def_var( ncid, name.data(), NC_DOUBLE, 1, &dim_ids[0],
                &id1d.at(name));
            err = nc_put_att_text( ncid, id1d.at(name), "long_name", long_name.size(),
                long_name.data());
        }
        // Output static vars
        dg::DVec resultD = dg::evaluate( dg::zero, grid);
        dg::HVec resultH( resultD);
        for( auto& record : shu::diagnostics2d_static_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            int staticID = 0;
            err = nc_def_var( ncid, name.data(), NC_DOUBLE, 2, &dim_ids[1],
                &staticID);
            err = nc_put_att_text( ncid, staticID, "long_name", long_name.size(),
                long_name.data());
            err = nc_enddef(ncid);
            record.function( resultD, var);
            dg::assign( resultD, resultH);
            dg::file::put_var_double( ncid, staticID, grid, resultH);
            err = nc_redef(ncid);
        }
        err = nc_enddef(ncid);
        size_t start[3] = {0, 0, 0};
        size_t count[3] = {1, grid.n()*grid.Ny(), grid.n()*grid.Nx()};
        ///////////////////////////////////first output/////////////////////////
        for( auto& record : shu::diagnostics2d_list)
        {
            record.function( resultD, var);
            dg::assign( resultD, resultH);
            dg::file::put_vara_double( ncid, id3d.at(record.name), start[0], grid, resultH);
        }
        for( auto& record : shu::diagnostics1d_list)
        {
            double result = record.function( var);
            nc_put_vara_double( ncid, id1d.at(record.name), start, count, &result);
        }
        err = nc_put_vara_double( ncid, tvarID, start, count, &time);
        ///////////////////////////////////timeloop/////////////////////////
        unsigned step=0;
        try{
        for( unsigned i=1; i<=maxout; i++)
        {

            dg::Timer ti;
            ti.tic();
            for( unsigned j=0; j<itstp; j++)
            {
                if( "ImExMultistep" == stepper)
                    imex.step( shu, diffusion, time, y0);
                else if ( "FilteredExplicitMultistep" == stepper)
                {
                    if( apply_filter)
                        multistep.step( shu, filter, time, y0);
                    else
                        multistep.step( shu, identity, time, y0);
                }
                else if ( "Shu-Osher" == stepper)
                {
                    if( apply_filter)
                        shu_osher.step( shu, filter, time, y0, time, y0, dt);
                    else
                        shu_osher.step( shu, identity, time, y0, time, y0, dt);
                }
                else if ( "FilteredImplicitMultistep" == stepper)
                {
                    if( apply_filter)
                        multistep_implicit.step( shu, filter, time, y0);
                    else
                        multistep_implicit.step( shu, identity, time, y0);
                }
            }
            step+=itstp;
            ti.toc();
            var.duration = ti.diff() / (double) itstp;
            //output all fields
            start[0] = i;
            for( auto& record : shu::diagnostics2d_list)
            {
                record.function( resultD, var);
                dg::assign( resultD, resultH);
                dg::file::put_vara_double( ncid, id3d.at(record.name), start[0], grid, resultH);
            }
            for( auto& record : shu::diagnostics1d_list)
            {
                double result = record.function( var);
                nc_put_vara_double( ncid, id1d.at(record.name), start, count, &result);
            }
            err = nc_put_vara_double( ncid, tvarID, start, count, &time);
            std::cout << "\n\t Step "<<step <<" of "<<itstp*maxout <<" at time "<<time;
            std::cout << "\n\t Average time for one step: "<<var.duration<<"s\n\n"<<std::flush;
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
    if( !("netcdf" == output) && !("glfw" == output))
    {
        std::cerr<<"Error: Wrong value for output type "<<output<<" Must be glfw or netcdf! Exit now!";
        return -1;
    }
    ////////////////////////////////////////////////////////////////////
    std::cout << "Time "<<time<<std::endl;
    for( auto record : shu::diagnostics1d_list)
    {
        double result = record.function( var);
        std::cout  << "Diagnostics "<<record.name<<" "<<result<<"\n";
    }

    return 0;

}
