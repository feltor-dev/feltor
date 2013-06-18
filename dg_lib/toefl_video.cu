#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"
#include "nvcc/xspacelib.cuh"
#include "nvcc/timer.cuh"
#include "lib/read_input.h"

#include "parameters.h"

#include "file.h"

const unsigned n = 4;

int main( int argc, char* argv[])
{
    dg::Timer t;
    std::vector<double> v = toefl::read_input( "window_params.txt");
    draw::HostWindow w(v[3], v[4]);
    w.set_multiplot( v[1], v[2]);

    if( argc != 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [inputfile]\n";
        return;
    }

    hid_t file = H5Fopen( argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    hsize_t nlinks = file::getNumObjs( file);
    //H5G_info_t group_info;
    //H5Gget_info( file, &group_info);
    //hsize_t nlinks = group_info.nlinks;
    //std::cout << "Number of groups "<< nlinks<<"\n";
    
    std::string name = file::getName( file, 0);
    //hsize_t length = H5Lget_name_by_idx( file, ".", H5_INDEX_NAME, H5_ITER_INC, 0, NULL, 10, H5P_DEFAULT);
    //std::cout << "Length of name "<<length<<"\n";
    //name.resize( length+1);
    //H5Lget_name_by_idx( file, ".", H5_INDEX_NAME, H5_ITER_INC, 0, &name[0], length+1, H5P_DEFAULT); //creation order
    std::cout << "Name of first link "<<name<<"\n";
    std::string in;
    
    herr_t  status;
    hsize_t dims[2]; 
    in.resize( 10000);
    status = H5LTread_dataset_string( file, name.data(), &in[0]); //name should precede t so that reading is easier
    const Parameters p( toefl::read_input( in));
    p.display();
    if( p.n != n )
    {
        std::cerr << "ERROR: n doesn't match: "<<n<<" vs. "<<p.n<<"\n";
        return -1;
    }

    dg::Grid<double, n> grid( 0, p.lx, 0, p.ly, p.Nx, p.Ny, p.bc_x, p.bc_y);


    dg::HVec visual(  grid.size(), 0.), input( visual);
    dg::HMatrix equi = dg::create::backscatter( grid);
    dg::HMatrix laplacianM = dg::create::laplacianM( grid, dg::normed, dg::XSPACE);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    bool running = true;
    unsigned index = 1;
    while (running && index < nlinks )
    {
        t.tic();
        glfwPollEvents();
        if( glfwGetKey( 'S')/*||((unsigned)t%100 == 0)*/) 
        {
            do
            {
                glfwWaitEvents();
            } while( !glfwGetKey('R') && 
                     !glfwGetKey( GLFW_KEY_ESC) && 
                      glfwGetWindowParam( GLFW_OPENED) );
        }
        hid_t group;
        name = file::getName( file, index);
        //length = H5Lget_name_by_idx( file, ".", H5_INDEX_NAME, H5_ITER_INC, index, NULL, 10, H5P_DEFAULT);
        //name.resize( length+1);
        //H5Lget_name_by_idx( file, ".", H5_INDEX_NAME, H5_ITER_INC, index, &name[0], length+1, H5P_DEFAULT); 
        index += v[5];
        //std::cout << "Index "<<index<<" "<<name<<"\n";
        group = H5Gopen( file, name.data(), H5P_DEFAULT);
        //std::cout << "Read electrons\n";
        status = H5LTread_dataset_double(group, "electrons", &input[0] );
        //transform field to an equidistant grid
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
        w.draw( visual, n*grid.Nx(), n*grid.Ny(), colors);
        t.toc();
        //std::cout << "Drawing took              "<<t.diff()<<"s\n";

        //transform phi
        t.tic();
        //std::cout << "Read potential\n";
        status = H5LTread_dataset_double(group, "potential", &input[0] );
        dg::blas2::gemv( laplacianM, input, visual);
        input.swap( visual);
        dg::blas2::gemv( equi, input, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        w.title() <<"phi / "<<colors.scale()<<"\t";
        w.title() << std::fixed; 
        w.title() << " &&  time = "<<file::getTime( name); //read time as double from string
        w.draw( visual, n*grid.Nx(), n*grid.Ny(), colors);
#ifdef DG_DEBUG
        glfwWaitEvents();
#endif //DG_DEBUG
        t.toc();
        //std::cout <<"2nd half took          "<<t.diff()<<"s\n";

        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    H5Fclose( file);
    return 0;
}
