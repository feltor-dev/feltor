#include <iostream>
#include <iomanip>
#include <vector>

#include "cuda_texture.cuh"

#include "toefl.cuh"
#include "rk.cuh"

#include "timer.cuh"

using namespace std;
using namespace dg;

const unsigned n = 3;

const double lx = 128.;
const double ly = 128.;

//const Parameter p = {0.005, 0.999, 0.001, 1, 48};

const unsigned k = 1;
const double eps = 1e-4; //The condition for conjugate gradient

const unsigned N = 10;// only every Nth computation is visualized

using namespace std;


int main()
{
    //do a cin for gridpoints
    unsigned Nx, Ny;
    cout << "Type number of grid points in each direction! \n";
    cin >> Nx; 
    Ny = Nx;
    double dt;
    cout << "Type timestep \n";
    cin >> dt;
    bool global = true;
    cout << "Type local(0) or global(1) \n";
    cin >> global;
    dg::HostWindow w(400, 400);
    glfwSetWindowTitle( "Behold the blob!\n");

    /////////////////////////////////////////////////////////////////////////
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;

    dg::Grid<double,n > grid( 0, lx, 0, ly, Nx, Ny);
    //create initial vector
    dg::Gaussian g( 0.3*lx, 0.5*ly, 10., 10., 5.);
    dg::DVec ne = dg::evaluate ( g, grid);
    if( global)
        thrust::transform( ne.begin(), ne.end(), ne.begin(), dg::PLUS<double>(1));
    std::vector<dg::DVec> y0(2, ne), y1(y0); // n_e = n_i 

    //create RHS and RK
    dg::Toefl<double, n, dg::DVec > test( grid, global, eps, 0.005, 0.001); 
    if( global)
        test.log( y0,y0); //transform to logarithmic values
    dg::RK< k, dg::Toefl<double, n, dg::DVec> > rk( y0);

    dg::HVec visual( n*n*Nx*Ny);
    dg::DMatrix equi = dg::create::backscatter( grid);
    dg::ColorMapRedBlueExt colors( 1.);
    //create timer
    Timer t;
    bool running = true;
    double time = 0;
    while (running)
    {
        t.tic();
        //transform field to an equidistant grid
        if( global)
        {
            test.exp( y0, y1);
            thrust::transform( y1[0].begin(), y1[0].end(), y1[0].begin(), dg::PLUS<double>(-1));
            dg::blas2::gemv( equi, y1[0], y1[1]);
        }
        else
            dg::blas2::gemv( equi, y0[0], y1[1]);
        t.toc();
        //std::cout << "Equilibration took        "<<t.diff()<<"s\n";
        t.tic();
        visual = y1[1]; //transfer to host
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        //draw and swap buffers
        w.title() << scientific;
        w.title() <<"Scale "<<colors.scale()<<"\t";
        w.title() << setprecision(2) << fixed;
        w.title() << " &&   time = "<<time;
        w.draw( visual, n*Nx, n*Ny, colors);
        t.toc();
        //std::cout << "Color scale " << colors.scale() <<"\n";
        //std::cout << "Visualisation time        " <<t.diff()<<"s\n";
        //step 
        t.tic();
        for( unsigned i=0; i<N; i++)
        {
            rk( test, y0, y1, dt);
            for( unsigned i=0; i<y0.size(); i++)
                thrust::swap( y0[i], y1[i]);
        }
        time += N*dt;
        t.toc();
        //std::cout << "Time for "<<N<<" step(s)      "<<t.diff()<<"s\n";
        //glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    ////////////////////////////////////////////////////////////////////

    return 0;

}
