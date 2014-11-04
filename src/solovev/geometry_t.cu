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
    //std::cout << "Type n, Nx, Ny\n";
    unsigned n, Nx, Ny;
    //std::cin >> n>> Nx>>Ny;   
    std::vector<double> v, v2;

    try{ 
        if( argc==1)
        {
            v = file::read_input( "../feltor/input.txt");
            v2 = file::read_input( "geometry_params.txt"); 
        }
        else
        {
            v = file::read_input( argv[1]); 
            v2 = file::read_input( argv[2]);
        }
    }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
            std::cout << v[i] << " ";
            std::cout << std::endl;
        return -1;}
    //write parameters from file into variables
    const eule::Parameters p(v);
    const solovev::GeomParameters gp(v2);
    p.display( std::cout);
    gp.display( std::cout);
    n = p.n, Nx = p.Nx, Ny = p.Ny;
    double Rmin=gp.R_0-p.boxscale*gp.a;
    double Zmin=-p.boxscale*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscale*gp.a; 
    double Zmax=p.boxscale*gp.a*gp.elongation;

 
    //construct all geometry quantities
    solovev::Psip psip(gp);
    solovev::PsipR psipR(gp);
    solovev::PsipRR psipRR(gp);  
    solovev::PsipZ psipZ(gp);  
    solovev::PsipZZ psipZZ(gp);   
    solovev::PsipRZ psipRZ(gp);  
    solovev::Ipol ipol(gp);
    solovev::InvB invB(gp);
    solovev::BR bR(gp);
    solovev::BZ bZ(gp);

    //Feltor quantities
    solovev::CurvatureR curvatureR(gp);
    solovev::CurvatureZ curvatureZ(gp);
    solovev::GradLnB gradLnB(gp);
    solovev::Field field(gp);
    solovev::FieldR fieldR(gp);
    solovev::FieldZ fieldZ(gp);
    solovev::Iris iris(gp);
    solovev::Pupil pupil(gp);
    solovev::GaussianDamping dampgauss(gp);
    solovev::GaussianProfDamping dampprof(gp);
    solovev::ZonalFlow zonalflow(p, gp);
    solovev::PsiLimiter psilimiter(gp);
    solovev::Nprofile prof(p, gp);

    dg::BathRZ bath(16,16,p.Nz,Rmin,Zmin, 30.,5.,p.amp);
//     dg::Gaussian3d bath(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma, p.amp);
    dg::Gaussian3d blob(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma, p.amp);
    dg::Grid2d<double> grid(Rmin,Rmax,Zmin,Zmax, n,Nx,Ny,dg::PER,dg::PER);

    std::vector<dg::HVec> hvisual(21);
        //allocate mem for visual
    std::vector<dg::HVec> visual(21);

    //B field functions
    hvisual[1] = dg::evaluate( psip, grid);
    hvisual[2] = dg::evaluate( ipol, grid);
    hvisual[3] = dg::evaluate( invB, grid);
    hvisual[4] = dg::evaluate( field, grid);
    hvisual[5] = dg::evaluate( curvatureR, grid);
    hvisual[6] = dg::evaluate( curvatureZ, grid);
    hvisual[7] = dg::evaluate( gradLnB, grid);
    //cut functions
    hvisual[8] = dg::evaluate( iris, grid);
    hvisual[9] = dg::evaluate( pupil, grid);
    hvisual[10] = dg::evaluate( dampprof, grid);
    hvisual[11] = dg::evaluate( dampgauss, grid);
    hvisual[12] = dg::evaluate( psilimiter, grid);
    //initial functions
    hvisual[13] = dg::evaluate( zonalflow, grid);
    hvisual[14] = dg::evaluate( prof, grid);
    hvisual[15] = dg::evaluate( blob, grid);
    hvisual[16] = dg::evaluate( bath, grid);

    //initial functions damped and with profile
    hvisual[17] = dg::evaluate( dg::one, grid);
    hvisual[18] = dg::evaluate( dg::one, grid);
    hvisual[19] = dg::evaluate( dg::one, grid);
    hvisual[20] = dg::evaluate( dg::one, grid);            
    dg::blas1::axpby( 1.,hvisual[16] , 1.,hvisual[14],hvisual[17]); //prof + bath
    dg::blas1::axpby( 1.,hvisual[13] , 1.,hvisual[14],hvisual[18]); //prof + zonal
    dg::blas1::axpby( 1.,hvisual[15] , 1.,hvisual[14],hvisual[19]); //prof + blob
    dg::blas1::transform(hvisual[17], hvisual[17], dg::PLUS<>(-1)); //to n -1
    dg::blas1::transform(hvisual[18], hvisual[18], dg::PLUS<>(-1)); //to n -1
    dg::blas1::transform(hvisual[19], hvisual[19], dg::PLUS<>(-1)); //to n -1
    dg::blas1::pointwiseDot(hvisual[10], hvisual[17], hvisual[17]); //damped 
    dg::blas1::pointwiseDot(hvisual[10], hvisual[18], hvisual[18]); //damped 
    dg::blas1::pointwiseDot(hvisual[10], hvisual[19], hvisual[19]); //damped 



//         Compute flux average
    solovev::Alpha alpha(gp); // = B^phi / |nabla psip |
    
    std::cout << "Compute flux average of psi   "<< "\n";
    dg::DVec psipog2d   = dg::evaluate( psip, grid);
    dg::DVec alphaog2d   = dg::evaluate( alpha, grid); 
    
    double psipmin = (float)thrust::reduce( psipog2d .begin(), psipog2d .end(), 0.0,thrust::minimum<double>()  );
    unsigned Npsi = 100;//set number of psivalues
    dg::Grid1d<double> g1d(psipmin ,0.0, 1,Npsi,dg::DIR);
    
    solovev::FluxSurfaceAverage<dg::HVec> fsa1(grid,gp,psipog2d );
    solovev::SafetyFactor<dg::HVec> qprof(grid,gp,alphaog2d );
    dg::HVec fsaofpsip = dg::evaluate(fsa1,g1d);
    dg::HVec sf = dg::evaluate(qprof,g1d);
    dg::HVec abs = dg::evaluate( dg::coo1, g1d);

    
for (unsigned i=0;i<g1d.size() ;i++) {
    std::cout << "psip_ref = " << abs[i] << "  psip_fsa = " << fsaofpsip[i]<< " rel error = " << ( fsaofpsip[i]-abs[i])/abs[i] << "  q = " << sf[i]<<"\n";
}
    
    //make equidistant grid from dggrid
    dg::HMatrix equigrid = dg::create::backscatter(grid);               

    //evaluate on valzues from devicevector on equidistant visual hvisual vector
    for( unsigned i=1; i<=20; i++){
        visual[i] = dg::evaluate( dg::one, grid);
        dg::blas2::gemv( equigrid, hvisual[i], visual[i]);
    }
    //Create Window and set window title
    GLFWwindow* w = draw::glfwInitAndCreateWindow( 1500, 1200, "");
    draw::RenderHostData render(4 , 5);
  
    //create a colormap
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);

    std::string names[] = { "", "psip", "ipol", "1/B","invbf", "K^R", "K_Z", "gradLnB", 
        "iris", "pupil", "dampprof", "damp", "lim",  
        "zonal", "prof", "blob", "bath", "ini1","ini2","ini3","ini3"};


    std::stringstream title;
    title << std::setprecision(2) << std::scientific;
    while (!glfwWindowShouldClose( w ))
    {
        for(unsigned i=1; i<=20; i++)
        {

            colors.scalemax() = (float)thrust::reduce( visual[i].begin(), visual[i].end(), -100., thrust::maximum<double>()   );
            colors.scalemin() =  (float)thrust::reduce( visual[i].begin(), visual[i].end(), colors.scalemax() ,thrust::minimum<double>() );
//             if(i==1) colors.scalemax() = - colors.scalemin();
//             if(i<=6 && i>=4) colors.scalemax() = - colors.scalemin();
//             if(i==18) colors.scalemin() = 1.0;
            title <<names[i]<<" / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
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
