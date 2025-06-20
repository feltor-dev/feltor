#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>

#include "dg/algorithm.h"
#include "dg/file/file.h"
#include "parameters.h"

//scan all imputfiles for maximum radial velocity and write to std::out
int main( int argc, char* argv[])
{
    if( argc == 1)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input1.nc] [input2.nc] ...\n";
        return -1;
    }
    dg::file::NC_Error_Handle err;
    int timeID, varID,varID2,varID3;
    size_t start = {0}, numOut;
    for( int i=1; i< argc; i++)
    {
        int ncid;

        
        err = nc_open( argv[i], NC_NOWRITE, &ncid);
        //read input file
        size_t length;
        err = nc_inq_attlen( ncid, NC_GLOBAL, "inputfile", &length);
        std::string input( length, 'x');
        err = nc_get_att_text( ncid, NC_GLOBAL, "inputfile", &input[0]);
//         std::cout << "input "<<input<<std::endl;
        auto js = dg::file::string2Json( input, dg::file::comments::are_forbidden);
        const eule::Parameters p(js);
        err = nc_inq_dimid( ncid, "time", &timeID);
        err = nc_inq_dimlen( ncid, timeID, &numOut);
        err = nc_inq_varid( ncid, "velX", &varID);
        std::vector<double> v(numOut);
        std::vector<double> vt(numOut);
        err = nc_get_vara_double( ncid, varID, &start, &numOut, v.data());
        err = nc_get_vara_double( ncid, timeID, &start, &numOut, vt.data());
        
        size_t count1d[2]  = {1, p.n_out* p.Nx_out}; 
        size_t start1d[2] =  {0,0};
        dg::Grid1d g1d( 0., p.lx, p.n_out, p.Nx_out, p.bc_x);
        dg::HVec nemax(dg::evaluate( dg::one, g1d));
        dg::HVec temax(dg::evaluate( dg::one, g1d));
        err = nc_inq_varid( ncid, "ne_max", &varID2);
        err = nc_inq_varid( ncid, "te_max", &varID3);
        err = nc_get_vara_double( ncid, varID2, start1d, count1d,  nemax.data());
        err = nc_get_vara_double( ncid, varID3, start1d, count1d,  temax.data());

        double neamp = *std::max_element(nemax.begin(),nemax.end())-1;
        double teamp = *std::max_element(temax.begin(),temax.end())-1;
        
        //Find integer time value of t=17 gamma
        double dpe=neamp+teamp+neamp*teamp;
        //double gammath =20./sqrt((2+p.amp)*p.amp*p.mcv/(p.sigma*(1.)));
        double gammathreal = sqrt((2+neamp)*neamp*p.mcv/p.sigma);
        double gammathrealdpe = sqrt(dpe*p.mcv/p.sigma);
//         if (p.sigma==10.)
//         {
//             if (p.tau[1]==4.) gammathrealdpe =15./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==2.) gammathrealdpe =17./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==1.) gammathrealdpe =19./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==0.5) gammathrealdpe =19./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==0.1) gammathrealdpe =19./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==0.) gammathrealdpe =20./sqrt(dpe*p.mcv/p.sigma);
//         }
//         if (p.sigma==20.)
//         {
//             if (p.tau[1]==4.) gammathrealdpe =15./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==2.) gammathrealdpe =15./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==1.) gammathrealdpe =15./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==0.5) gammathrealdpe =15./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==0.1) gammathrealdpe =15./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==0.) gammathrealdpe =15./sqrt(dpe*p.mcv/p.sigma);
//         }
//         if (p.sigma==5.)
//         {
//             if (p.tau[1]==4.) 
//             {
//                 if (p.amp==2.) gammathrealdpe =16./sqrt(dpe*p.mcv/p.sigma);
//                 if (p.amp==1.) gammathrealdpe =13./sqrt(dpe*p.mcv/p.sigma);
//                 if (p.amp==0.5) gammathrealdpe =17./sqrt(dpe*p.mcv/p.sigma);
//                 if (p.amp==0.1) gammathrealdpe =17./sqrt(dpe*p.mcv/p.sigma);
//             }
//             if (p.tau[1]==2.) 
//             {
//                 if (p.amp==2.) gammathrealdpe =20./sqrt(dpe*p.mcv/p.sigma);
//                 if (p.amp==1.) gammathrealdpe =20./sqrt(dpe*p.mcv/p.sigma);
//                 if (p.amp==0.5) gammathrealdpe =20./sqrt(dpe*p.mcv/p.sigma);
//                 if (p.amp==0.1) gammathrealdpe =15./sqrt(dpe*p.mcv/p.sigma);
//             }
//             if (p.tau[1]==1.) gammathrealdpe =20./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==0.5) gammathrealdpe =20./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==0.1) gammathrealdpe =20./sqrt(dpe*p.mcv/p.sigma);
//             if (p.tau[1]==0.) gammathrealdpe =20./sqrt(dpe*p.mcv/p.sigma);
//         }

        gammathrealdpe = 10./sqrt(dpe*p.mcv/p.sigma);

        std::vector<double>::iterator timepoint;
        timepoint=std::lower_bound (vt.begin() ,vt.end() ,gammathrealdpe );
//         gammathrealdpe = sqrt(dpe*p.mcv/p.sigma);
        gammathrealdpe = sqrt((1.+p.tau[1])*dpe*p.mcv/p.sigma);
//         std::cout <<  " timep " << *timepoint << std::endl;
        unsigned timepos = std::distance(vt.begin(), timepoint );
//         std::cout <<  " timeposition " << std::distance(vt.begin(), timepoint ) << std::endl;        
//         std::cout << "  vCOM max"       <<*std::max_element(v.begin(),v.begin()+timepos)  << std::endl;

//         std::cout << argv[i]<< " tau = " << p.tau[1]<< " sigma = " << p.sigma<< " A = " << p.amp<< " " << *std::max_element(v.begin(), v.end()) <<"\n";
        std::cout <<p.tau[1]<< " "
                  <<p.sigma<< " "
                  <<neamp<< " "
                  <<*std::max_element(v.begin(),v.begin()+timepos) << " "
                  <<gammathreal <<" "
                  <<gammathrealdpe <<" "
                  <<dpe <<std::endl;   
    }

    return 0;
}

