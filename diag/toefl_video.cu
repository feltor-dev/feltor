#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"
#include "dg/xspacelib.cuh"
#include "dg/timer.cuh"
#include "file/read_input.h"
#include "file/file.h"

#include "galerkin/parameters.h"



int main( int argc, char* argv[])
{
    dg::Timer t;
    std::vector<double> v = file::read_input( "window_params.txt");
    draw::HostWindow w(v[3], v[4]);
    w.set_multiplot( v[1], v[2]);

    if( argc != 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [inputfile]\n";
        return;
    }

    hid_t file = H5Fopen( argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    hsize_t nlinks = file::getNumObjs( file);
    std::string name = file::getName( file, 0);
    std::string in;
    
    herr_t  status;
    hsize_t dims[2]; 
    in.resize( 10000);
    status = H5LTread_dataset_string( file, name.data(), &in[0]); //name should precede t so that reading is easier


    const Parameters p( file::read_input( in));
    p.display();
    dg::Grid<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    dg::HVec visual(  grid.size(), 0.), input( visual);
    dg::HMatrix equi = dg::create::backscatter( grid);
    dg::HMatrix laplacianM = dg::create::laplacianM( grid, dg::normed, dg::XSPACE);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    bool running = true;
    unsigned index = 1;
    std::cout << "PRESS N FOR NEXT FRAME!\n";
    std::cout << "PRESS P FOR PREVIOUS FRAME!\n";
    unsigned num_entries = (p.maxout+1)*p.itstp;
    std::vector<double> mass( 2*num_entries+4, 0.), energy( mass), massAcc( num_entries), energyAcc( num_entries);
    hid_t group;
    //read xfiles
    group = H5Gopen( file, "xfiles", H5P_DEFAULT);
    H5LTread_dataset_double( group, "mass", &mass[2] );
    H5LTread_dataset_double( group, "energy", &energy[2] );
    H5Gclose( file);
    for(unsigned i=0; i<num_entries; i++ )
    {
        massAcc[i] = (mass[2*(i+2)]-mass[2*i])/2./p.dt; //first column
        energyAcc[i] = (energy[2*(i+2)]-energy[2*i])/2./p.dt;
        //massAcc[i] = 2.*(massAcc[i]-mass[2*(i+1)+1])/(massAcc[i]+mass[2*(i+1)+1]); //2nd column
        energyAcc[i] = 2.*(energyAcc[i]-energy[2*(i+1)+1])/(energyAcc[i]+energy[2*(i+1)+1]);
    }

    while (running && index < nlinks )
    {
        std::cout << "Mass loss: "<<massAcc[(index-1)*p.itstp]<<"\t energy accuracy: "<<energyAcc[(index-1)*p.itstp]<<std::endl;
        t.tic();
        name = file::getName( file, index);
        group = H5Gopen( file, name.data(), H5P_DEFAULT);
        status = H5LTread_dataset_double( group, "electrons", &input[0] );
        t.toc();
        //std::cout << "Reading of electrons took "<<t.diff()<<"s\n";
        t.tic();
        if( p.global)
            thrust::transform( input.begin(), input.end(), input.begin(), dg::PLUS<double>(-1));
        dg::blas2::gemv( equi, input, visual);

        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        t.toc();
        //std::cout << "Computing colorscale took "<<t.diff()<<"s\n";
        //draw ions
        w.title() << std::setprecision(2) << std::scientific;
        w.title() <<"ne / "<<colors.scale()<<"\t";
        t.tic();
        w.draw( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        t.toc();
        //std::cout << "Drawing took              "<<t.diff()<<"s\n";

        //transform phi
        t.tic();
        status = H5LTread_dataset_double(group, "potential", &input[0] );
        //Vorticity is \curl \bm{u}_E \approx \frac{\Delta\phi}{B}
        dg::blas2::gemv( laplacianM, input, visual);
        input.swap( visual);
        dg::blas2::gemv( equi, input, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        if(colors.scale() == 0) { colors.scale() = 1;}
        //draw phi and swap buffers
        w.title() <<"omega / "<<colors.scale()<<"\t";
        w.title() << std::fixed; 
        w.title() << " &&  time = "<<file::getTime( name); //read time as double from string
        w.draw( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        t.toc();
        //std::cout <<"2nd half took          "<<t.diff()<<"s\n";
        bool waiting = true;
        do
        {
            glfwPollEvents();
            if( glfwGetKey( 'B')||glfwGetKey( 'P') ){
                index -= v[5];
                waiting = false;
            }
            else if( glfwGetKey( 'N') ){
                index +=v[5];
                waiting = false;
            }
            //glfwWaitEvents();
        }while( waiting && !glfwGetKey( GLFW_KEY_ESC) && glfwGetWindowParam( GLFW_OPENED));

        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    H5Fclose( file);
    return 0;
}
