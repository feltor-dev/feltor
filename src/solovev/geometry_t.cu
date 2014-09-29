#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"
#include "file/read_input.h"
#include "draw/host_window.h"

#include "geometry.h"
#include "init.h"

int main( int argc, char* argv[])
{
    std::cout << "Type n, Nx, Ny\n";
    unsigned n, Nx, Ny;
    std::cin >> n>> Nx>>Ny;   
    std::vector<double> v;
    try{ 
        if( argc==1)
            v = file::read_input( "geometry_params.txt"); 
        else
            v = file::read_input( argv[1]); 
    }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
            std::cout << v[i] << " ";
            std::cout << std::endl;
        return -1;}
    for( unsigned i = 1; i<v.size(); i++)
    std::cout <<  v[i] << " ";
    std::cout << std::endl;
    std::cout << "Total number of parameters read is: "<<v.size()-1 <<"\n";
    std::stringstream title;
    //write parameters from file into variables

    const solovev::GeomParameters gp(v);
    gp.display( std::cout);
    double Rmin=gp.R_0-(gp.boxscale)*gp.a;
    double Zmin=-(gp.boxscale)*gp.a*gp.elongation;
    double Rmax=gp.R_0+(gp.boxscale)*gp.a; 
    double Zmax=(gp.boxscale)*gp.a*gp.elongation;
    //construct all geometry quantities
    solovev::Psip psip(gp.R_0,gp.A,gp.c);
    solovev::PsipR psipR(gp.R_0,gp.A,gp.c);
    solovev::PsipRR psipRR(gp.R_0,gp.A,gp.c);  
    solovev::PsipZ psipZ(gp.R_0,gp.A,gp.c);  
    solovev::PsipZZ psipZZ(gp.R_0,gp.A,gp.c);   
    solovev::PsipRZ psipRZ(gp.R_0,gp.A,gp.c);  
    solovev::Ipol ipol(gp.R_0,gp.A,psip);
    solovev::InvB invB(gp.R_0,ipol,psipR,psipZ);
    solovev::BR bR(gp.R_0,gp.A,psipR,psipRR,psipZ,psipRZ,invB);
    solovev::BZ bZ(gp.R_0,gp.A,psipR,psipZ,psipZZ,psipRZ,invB);
    //Feltor quantities
    solovev::CurvatureR curvatureR(gp);
    solovev::CurvatureZ curvatureZ(gp);
    solovev::GradLnB gradLnB(gp);
    solovev::Field field(gp);
    solovev::Iris iris(gp);
    solovev::Pupil pupil(gp);
    solovev::PsiLimiter limiter(gp);
    solovev::GaussianDamping dampgauss(gp);
    solovev::ZonalFlow zonalflow(gp,0.5);
    solovev::Gradient gradient(gp);
    solovev::Nprofile prof(gp);
    solovev::TanhDampingIn damp2(gp);
    solovev::TanhDampingProf dampcut(gp);
    solovev::TanhDampingInv source(gp);
    dg::BathRZ bath1(16,16,0,Rmin,Zmin, 30.,3.,1.);
    dg::BathRZ bath2(16,16,0,Rmin,Zmin, 30.,30.,10.);
    dg::Gaussian3d init0(gp.R_0+0.75*gp.a, 0.,M_PI/20., 2., 2., 2.,2.);

    dg::Grid2d<double> grid(Rmin,Rmax,Zmin,Zmax, n,Nx,Ny,dg::PER,dg::PER);
    std::vector<dg::HVec> hvisual(19);
    hvisual[1 ] = dg::evaluate( psip, grid);
    hvisual[2 ] = dg::evaluate( ipol, grid);
    hvisual[3 ] = dg::evaluate( invB, grid);
    hvisual[4 ] = dg::evaluate( curvatureR, grid);
    hvisual[5 ] = dg::evaluate( curvatureZ, grid);
    hvisual[6 ] = dg::evaluate( gradLnB, grid);
    hvisual[7 ] = dg::evaluate( iris, grid);
    hvisual[8 ] = dg::evaluate( pupil, grid);
    hvisual[9 ] = dg::evaluate( dampgauss, grid);
    hvisual[10] = dg::evaluate( zonalflow, grid);
    hvisual[11] = dg::evaluate( gradient, grid);
    hvisual[12] = dg::evaluate( field, grid);
    hvisual[13] = dg::evaluate( prof, grid);
    hvisual[14] = dg::evaluate( limiter, grid);
    hvisual[15] = dg::evaluate( dampcut, grid);
    hvisual[16] = dg::evaluate( bath1, grid);
    hvisual[17] = dg::evaluate( bath1,grid);
    dg::blas1::pointwiseDot(hvisual[8], hvisual[17],hvisual[17]);
    hvisual[18] = dg::evaluate( init0,grid);
    dg::blas1::pointwiseDot(hvisual[9], hvisual[18], hvisual[18]);

//     dg::blas1::pointwiseDot(hvisual8, hvisual18, hvisual18);
    dg::blas1::axpby( 1.,hvisual[13] , 1.,hvisual[18],hvisual[18]);
    //allocate mem for visual
    std::vector<dg::HVec> visual(hvisual);

    //make equidistant grid from dggrid
    dg::HMatrix equigrid = dg::create::backscatter(grid);
    //evaluate on valzues from devicevector on equidistant visual hvisual vector
    for( unsigned i=1; i<=18; i++)
        dg::blas2::gemv( equigrid, hvisual[i], visual[i]);

    //Create Window and set window title
    GLFWwindow* w = draw::glfwInitAndCreateWindow( 1800, 900, "");
    draw::RenderHostData render( 3, 6);
  
    //create a colormap
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);

    std::string names[] = { "", "psip", "ipol", "1/B", "K^R", "K_Z", 
           "gradLnB", "iris", "pupil", "damping", "zonal", 
           "grad", "invbf", "nprof", "limiter", "tanhcut", 
           "source", "bath", "bath"};


    title << std::setprecision(2) << std::scientific;
    while (!glfwWindowShouldClose( w ))
    {
        for(unsigned i=1; i<=18; i++)
        {

        
            colors.scalemax() = (float)thrust::reduce( visual[i].begin(), visual[i].end(), -100., thrust::maximum<double>()   );
            colors.scalemin() =  (float)thrust::reduce( visual[i].begin(), visual[i].end(), colors.scalemax() ,thrust::minimum<double>() );
            if(i==1) colors.scalemax() = - colors.scalemin();
            if(i==18) colors.scalemin() = 1.0;
            title <<names[i]<<" / "/*<<colors.scalemin()<<"  "*/ << colors.scalemax()<<"\t";
            render.renderQuad( visual[i], grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        title << std::fixed; 
        glfwSetWindowTitle(w,title.str().c_str());
        title.str("");
        glfwSwapBuffers(w);
        glfwWaitEvents();
    }

    glfwTerminate();
    return 0;
}
