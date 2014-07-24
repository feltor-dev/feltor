#include <iostream>
#include <iomanip>
#include <vector>
#include "xspacelib.cuh"
#include "geometry.cuh"
#include <fstream>
#include <sstream>
#include "file/read_input.h"
#include <cmath>

#include "draw/host_window.h"
int main()
{
    unsigned Nx,Ny,n;
//     double Nxh = Nx/2.,Nyh=Ny/2.;
//     double a, elongation,triangularity;
    double Rmin,Zmin,Rmax,Zmax;
//     double A,R_0,psipmin,psipmax;
//     std::vector<double> c(13);
        std::cout << "Type n, Nx, Ny\n";
    std::cin >> n>> Nx>>Ny;
    
    std::vector<double> v;
    try{ v = file::read_input( "geometry_params.txt"); }
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
//     A=v[1];
//     for (unsigned i=2;i<14;i++) c[i-2]=v[i];
//     R_0 = v[14];
//     psipmin= v[15];
//     psipmax= v[16];
//     a=R_0*v[17];
//     elongation=v[18];
//     triangularity=v[19];
    const dg::GeomParameters gp(v);
    gp.display( std::cout);
    Rmin=gp.R_0-1.1*gp.a;
    Zmin=-1.1*gp.a*gp.elongation;
    Rmax=gp.R_0+1.1*gp.a; 
    Zmax=1.1*gp.a*gp.elongation;

    //construct all geometry quantities
    dg::Psip psip(gp.R_0,gp.A,gp.c);
    dg::PsipR psipR(gp.R_0,gp.A,gp.c);
    dg::PsipRR psipRR(gp.R_0,gp.A,gp.c);  
    dg::PsipZ psipZ(gp.R_0,gp.A,gp.c);  
    dg::PsipZZ psipZZ(gp.R_0,gp.A,gp.c);   
    dg::PsipRZ psipRZ(gp.R_0,gp.A,gp.c);  
    dg::Ipol ipol(gp.R_0,gp.A,psip);
    dg::InvB invB(gp.R_0,ipol,psipR,psipZ);
    dg::BR bR(gp.R_0,gp.A,psipR,psipRR,psipZ,psipRZ,invB);
    dg::BZ bZ(gp.R_0,gp.A,psipR,psipZ,psipZZ,psipRZ,invB);
    //Feltor quantities
    dg::CurvatureR curvatureR(gp);
    dg::CurvatureZ curvatureZ(gp);
    dg::GradLnB gradLnB(gp);
    dg::Field field(gp);
    dg::Iris iris(gp);
    dg::Pupil pupil(gp);
    dg::Damping damping(gp);
    dg::Gradient grad(gp,0.05);
    dg::ZonalFlow zonalflow(gp,7);
//     //make dggrid
    dg::Grid2d<double> grid(Rmin,Rmax,Zmin,Zmax, n,Nx,Ny,dg::PER,dg::PER);
    
    dg::HVec hvisual1 = dg::evaluate( psip, grid);
    dg::HVec hvisual2 = dg::evaluate( ipol, grid);
    dg::HVec hvisual3 = dg::evaluate( invB, grid);
    dg::HVec hvisual4 = dg::evaluate( curvatureR, grid);
    dg::HVec hvisual5 = dg::evaluate( curvatureZ, grid);
    dg::HVec hvisual6 = dg::evaluate( gradLnB, grid);
    dg::HVec hvisual7 = dg::evaluate( iris, grid);
    dg::HVec hvisual8 = dg::evaluate( pupil, grid);
    dg::HVec hvisual9 = dg::evaluate( damping, grid);
    dg::HVec hvisual10 = dg::evaluate( grad, grid);
    dg::HVec hvisual11 = dg::evaluate( zonalflow, grid);
    //allocate mem for visual
    dg::HVec visual1( grid.size());
    dg::HVec visual2( grid.size());
    dg::HVec visual3( grid.size());
    dg::HVec visual4( grid.size());
    dg::HVec visual5( grid.size());
    dg::HVec visual6( grid.size());
    dg::HVec visual7( grid.size());
    dg::HVec visual8( grid.size());
    dg::HVec visual9( grid.size());
    dg::HVec visual10( grid.size());
    dg::HVec visual11( grid.size());
    //make equidistant grid from dggrid
    dg::HMatrix equigrid = dg::create::backscatter(grid);
    //evaluate on valzues from devicevector on equidistant visual hvisual vector
    dg::blas2::gemv( equigrid, hvisual1, visual1);
    dg::blas2::gemv( equigrid, hvisual2, visual2);
    dg::blas2::gemv( equigrid, hvisual3, visual3);
    dg::blas2::gemv( equigrid, hvisual4, visual4);
    dg::blas2::gemv( equigrid, hvisual5, visual5);
    dg::blas2::gemv( equigrid, hvisual6, visual6);
    dg::blas2::gemv( equigrid, hvisual7, visual7);
    dg::blas2::gemv( equigrid, hvisual8, visual8);
    dg::blas2::gemv( equigrid, hvisual9, visual9);
    dg::blas2::gemv( equigrid, hvisual10, visual10);
    dg::blas2::gemv( equigrid, hvisual11, visual11);

    //Create Window and set window title
    GLFWwindow* w = draw::glfwInitAndCreateWindow( 1800, 600, "");
    draw::RenderHostData render( 2, 6);
  
    //create a colormap
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);

    while (!glfwWindowShouldClose( w ))
    {
        colors.scalemax() =0.1;//  (float)thrust::reduce( visual1.begin(), visual1.end(), 0., dg::AbsMax<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual1.begin(), visual1.end(), colors.scalemax()  ,thrust::minimum<double>() );
        title << std::setprecision(2) << std::scientific;
  
        title <<"psip / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        render.renderQuad( visual1, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        colors.scalemax() = (float)thrust::reduce( visual2.begin(), visual2.end(), 0., dg::AbsMax<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual2.begin(), visual2.end(), colors.scalemax() ,thrust::minimum<double>() );
        title <<"ipol / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        render.renderQuad( visual2, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        colors.scalemax() = (float)thrust::reduce( visual3.begin(), visual3.end(), 0., dg::AbsMax<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual3.begin(), visual3.end(), colors.scalemax() ,thrust::minimum<double>() );
        title <<"1/B / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        render.renderQuad( visual3, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        colors.scalemax() = (float)thrust::reduce( visual4.begin(), visual4.end(),  -1.0, thrust::maximum<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual4.begin(), visual4.end(), colors.scalemax() ,thrust::minimum<double>() );
        title <<"K^R / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        render.renderQuad( visual4, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        colors.scalemax() = (float)thrust::reduce( visual5.begin(), visual5.end(), -1.0, thrust::maximum<double>()   );
        colors.scalemin() =  (float)thrust::reduce( visual5.begin(), visual5.end(), colors.scalemax() ,thrust::minimum<double>() );
//                 colors.scalemax() = -0.0058;
//         colors.scalemin() = -0.009;
        title <<"K^Z / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        render.renderQuad( visual5, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        colors.scalemax() = (float)thrust::reduce( visual6.begin(), visual6.end(), 0., dg::AbsMax<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual6.begin(), visual6.end(), colors.scalemax() ,thrust::minimum<double>() );
        title <<"gradLnB / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        render.renderQuad( visual6, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        colors.scalemax() = (float)thrust::reduce( visual7.begin(), visual7.end(), 0., dg::AbsMax<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual7.begin(), visual7.end(), colors.scalemax() ,thrust::minimum<double>() );
        title <<"iris / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        render.renderQuad( visual7, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        colors.scalemax() = (float)thrust::reduce( visual8.begin(), visual8.end(), 0., dg::AbsMax<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual8.begin(), visual8.end(), colors.scalemax() ,thrust::minimum<double>() );
        title <<"pupil / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        render.renderQuad( visual8, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        colors.scalemax() = (float)thrust::reduce( visual9.begin(), visual9.end(), 0., dg::AbsMax<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual9.begin(), visual9.end(), colors.scalemax() ,thrust::minimum<double>() );
        title <<"damping / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        render.renderQuad( visual9, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        colors.scalemax() = (float)thrust::reduce( visual10.begin(), visual10.end(), 0., dg::AbsMax<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual10.begin(), visual10.end(), colors.scalemax() ,thrust::minimum<double>() );
        title <<"gradient / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        render.renderQuad( visual10, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        colors.scalemax() = (float)thrust::reduce( visual11.begin(), visual11.end(), 0., dg::AbsMax<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual11.begin(), visual11.end(), colors.scalemax() ,thrust::minimum<double>() );
        title <<"gradient / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        render.renderQuad( visual11, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        colors.scalemax() = (float)thrust::reduce( visual11.begin(), visual11.end(), 0., dg::AbsMax<double>()  );
        colors.scalemin() =  (float)thrust::reduce( visual11.begin(), visual11.end(), colors.scalemax() ,thrust::minimum<double>() );
        title <<"gradient / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";
        render.renderQuad( visual11, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        
        title << std::fixed; 
        glfwSetWindowTitle(w,title.str().c_str());
        title.str("");
        glfwSwapBuffers(w);
        glfwWaitEvents();
    }

    glfwTerminate();
 return 0;
}