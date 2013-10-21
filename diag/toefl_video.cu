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

    std::string in;
    file::T5rdonly t5file( argv[1], in);
    unsigned nlinks = t5file.get_size();
    /*
    //hid_t file = H5Fopen( argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    //hsize_t nlinks = file::getNumObjs( file);
    //std::string name = file::getName( file, 0); 
    
    //herr_t  status;
    //hsize_t dims[2]; 
    //in.resize( 10000);
    //status = H5LTread_dataset_string( file, name.data(), &in[0]); //name should precede t so that reading is easier
    */

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
    /*
    //unsigned num_entries = (p.maxout+1)*p.itstp;//actually too large
    //std::cout << num_entries<<"\n";
    //std::vector<double> mass( num_entries+2, 0.), energy( mass); 
    //std::vector<double> diffusion( num_entries), dissipation( num_entries), massAcc( num_entries), energyAcc( num_entries);
    //hid_t group;
    //read xfiles
    */
    std::vector<double> mass, energy, diffusion, dissipation, massAcc, energyAcc;
    t5file.get_xfile( mass, "mass");
    t5file.get_xfile( energy, "energy");
    t5file.get_xfile( diffusion, "diffusion");
    t5file.get_xfile( dissipation, "dissipation");
    massAcc.resize(mass.size()), energyAcc.resize(mass.size());
    mass.insert(mass.begin(), 0), mass.push_back(0);
    energy.insert( energy.begin(), 0), energy.push_back(0);
    /*
    //group = H5Gopen( file, "xfiles", H5P_DEFAULT);
    //H5LTread_dataset_double( group, "mass", &mass[1] );
    //H5LTread_dataset_double( group, "energy", &energy[1] );
    //H5LTread_dataset_double( group, "diffusion", &diffusion[0] );
    //H5LTread_dataset_double( group, "dissipation", &dissipation[0] );
    //H5Gclose( group);
    */
    for(unsigned i=0; i<massAcc.size(); i++ )
    {
        massAcc[i] = (mass[i+2]-mass[i])/2./p.dt; //first column
        //if( i < 10 || i > num_entries - 20)
        //    std::cout << "i "<<i<<"\t"<<massAcc[i]<<"\t"<<mass[i+1]<<std::endl;
        massAcc[i] = fabs(2.*(massAcc[i]-diffusion[i])/(massAcc[i]+diffusion[i]));
        energyAcc[i] = (energy[i+2]-energy[i])/2./p.dt;
        energyAcc[i] = fabs(2.*(energyAcc[i]-dissipation[i])/(energyAcc[i]+dissipation[i]));
        //energyAcc[i] = fabs(energyAcc[i]-dissipation[i])/p.nu;
    }

    std::cout << std::scientific << std::setprecision( 2);
    /*
        bool waiting = true;
        do
        {
            glfwPollEvents();
            if( glfwGetKey( 'S')){
                waiting = false;
            }
        }while( waiting && !glfwGetKey( GLFW_KEY_ESC) && glfwGetWindowParam( GLFW_OPENED));
        */
    std::cout<< "Hello world\n";
    while (running && index < nlinks + 1 )
    {
        t.tic();
        //name = file::getName( file, index);
        //group = H5Gopen( file, name.data(), H5P_DEFAULT);
        //status = H5LTread_dataset_double( group, "electrons", &input[0] );
        t5file.get_field( input, "electrons", index);
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
        //status = H5LTread_dataset_double(group, "potential", &input[0] );
        t5file.get_field( input, "potential", index);
        //Vorticity is \curl \bm{u}_E \approx \frac{\Delta\phi}{B}
        dg::blas2::gemv( laplacianM, input, visual);
        input.swap( visual);
        dg::blas2::gemv( equi, input, visual);
        dg::blas1::axpby( -1., visual, 0., visual);//minus laplacian
        std::cout <<"(m_tot-m_0)/m_0: "<<(mass[(index-1)*p.itstp]-mass[1])/(mass[1]-grid.lx()*grid.ly()) //blob mass is mass[] - Area
                  <<"\t(E_tot-E_0)/E_0: "<<(energy[(index-1)*p.itstp]-energy[1])/energy[1]
                  <<"\tAccuracy: "<<energyAcc[(index-1)*p.itstp]<<std::endl;

        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        if(colors.scale() == 0) { colors.scale() = 1;}
        //draw phi and swap buffers
        w.title() <<"omega / "<<colors.scale()<<"\t";
        w.title() << std::fixed; 
        //w.title() << " &&  time = "<<file::getTime( name); //read time as double from string
        w.title() << " && time = "<<t5file.get_time( index);
        w.draw( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        t.toc();
        //std::cout <<"2nd half took          "<<t.diff()<<"s\n";
        //H5Gclose( group);
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
    //H5Fclose( file);
    return 0;
}
