#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm> 
#include <thrust/random.h>

#include "dg/algorithm.h"
#include "dg/file/nc_utilities.h"

/**
 * @brief normalizes input vector 
 */ 
void NormalizeToFluc(std::vector<double>& in) {
    double ex= 0.;
    double exx= 0.;
    double ex2= 0.;
    double sigma = 0.;    
    for (unsigned j=0;j<in.size();j++)
    {
        ex+=in[j];
        exx+=in[j]*in[j];
    }
    ex/=in.size();
    exx/=in.size();
    ex2=ex*ex;
    sigma=sqrt(exx-ex2);
    for (unsigned j=0;j<in.size();j++)
    {
        in[j] = (in[j]-  ex)/sigma; 
    }
    std::cout << "Sigma = " <<sigma << " Meanvalue = " << ex << std::endl;
}

int main( int argc, char* argv[])
{

    if( argc != 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input.nc] [output.nc]\n";
        return -1;
    }
    std::cout << argv[1]<< " -> "<<argv[2]<<std::endl;   
    //----------------
    const unsigned Nhist = 50; 
    const unsigned nhist = 1;
    const unsigned Ninput =100;
    const double Nsigma =4.;
    std::vector<double> input1(Ninput,0.);    
    std::vector<double> input2(Ninput,0.);    

    thrust::random::minstd_rand generator;
    thrust::random::normal_distribution<double> d1;
    thrust::random::normal_distribution<double> d2;
    std::vector<double> rand1(Ninput,0.);    
    std::vector<double> rand2(Ninput,0.);    
    for (unsigned i=0;i<rand1.size();i++)  {  rand1[i] = d1(generator); }
    for (unsigned i=0;i<rand2.size();i++)  {  rand2[i] = d2(generator); }

    for (unsigned i=0;i<input1.size();i++)  {
        double t = (double)(i/(input1.size()-1));
        double omega1 =2.*M_PI* 20.;
        input1[i] = (rand1[i]*0.1*cos( omega1*t)+1.); 
    }
    for (unsigned i=0;i<input2.size();i++)  {
        double t = (double)(i/(input2.size()-1));
        //double omega1 = 2.*M_PI*20.;
        double omega2= 2.*M_PI*30.;
        //double phase = 0.5*M_PI;
//         input2[i] =input1[i];  //perfectly correlated
//         input2[i] = (-rand1[i]*0.1*cos(omega1*t)+1.);//perfectly anticorrelated
//         input2[i] = (rand2[i]*0.001*cos(omega2*t)+3.);//perfectly uncorrelated
        input2[i] = (rand2[i]*0.001*cos(omega2*t)+3.);//uncorrelated
    } 

    //normalize grid and compute sigma
    NormalizeToFluc(input1);
    NormalizeToFluc(input2);
    dg::Grid1d  g1d1(-Nsigma,Nsigma, nhist, Nhist,dg::DIR);
    dg::Grid1d  g1d2(-Nsigma,Nsigma, nhist, Nhist,dg::DIR); 
    dg::Grid2d  g2d( -Nsigma,Nsigma,-Nsigma,Nsigma, nhist, Nhist,Nhist,dg::DIR,dg::DIR); 
    dg::Histogram<dg::HVec> hist1(g1d1,input1);  
    dg::Histogram<dg::HVec> hist2(g1d2,input2);    
    dg::Histogram2D<dg::HVec> hist12(g2d,input1,input2);    

 
    dg::HVec PA1 = dg::evaluate(hist1,g1d1);
    dg::HVec A1 = dg::evaluate(dg::cooX1d,g1d1);
    dg::HVec PA2= dg::evaluate(hist2,g1d2);
    dg::HVec A2 = dg::evaluate(dg::cooX1d,g1d2);
    dg::HVec PA1A2= dg::evaluate(hist12,g2d);
    
    //-----------------NC output start
    int dataIDs1[2],dataIDs2[2],dataIDs12[1];
    int dim_ids1[1],dim_ids2[1],dim_ids12[2];
    int ncid;
    dg::file::NC_Error_Handle err; 
    err = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid); 
    //plot 1
    err = dg::file::define_dimension( ncid, &dim_ids1[0],  g1d1,"A1_");
    err = nc_def_var( ncid, "P(A1)",   NC_DOUBLE, 1, &dim_ids1[0], &dataIDs1[0]);
    err = nc_def_var( ncid, "A1",    NC_DOUBLE, 1, &dim_ids1[0], &dataIDs1[1]);
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, dataIDs1[0], PA1.data() );
    err = nc_put_var_double( ncid, dataIDs1[1], A1.data() );
    err = nc_redef(ncid);
    //plot 2
    err = dg::file::define_dimension( ncid, &dim_ids2[0],  g1d2,"A2_");
    err = nc_def_var( ncid, "P(A2)",   NC_DOUBLE, 1, &dim_ids2[0], &dataIDs2[0]);
    err = nc_def_var( ncid, "A2",    NC_DOUBLE, 1, &dim_ids2[0], &dataIDs2[1]);
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, dataIDs2[0], PA2.data() );
    err = nc_put_var_double( ncid, dataIDs2[1], A2.data() );
    err = nc_redef(ncid);
    //plot12
//     dim_ids12[0]=dim_ids1[0];
//     dim_ids12[1]=dim_ids2[0];
    dim_ids12[0]=dataIDs1[0];
    dim_ids12[1]=dataIDs2[0];
    err = dg::file::define_dimensions( ncid, &dim_ids12[0],  g2d);
    err = nc_def_var( ncid, "P(A1,A2)",   NC_DOUBLE, 2, &dim_ids12[0], &dataIDs12[0]);
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, dataIDs12[0], PA1A2.data() );
    err = nc_redef(ncid);
    nc_close( ncid);

    return 0;
}

