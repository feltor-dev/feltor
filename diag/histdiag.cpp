#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

#include "dg/algorithm.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"
#include "file/read_input.h"
#include "file/nc_utilities.h"
// #include <thrust/random/linear_congruential_engine.h>
// #include <thrust/random/normal_distribution.h>

/**
 * @brief returns histogram 
 * @tparam container 
 */ 
template <class container = thrust::host_vector<double> >
struct Histogram
{
     /**
     * @brief Construct from number of bins and input vector
     * @param g1d   grid of output vector
     * @param input input vector
     */
    Histogram(const dg::Grid1d<double>& g1d, const std::vector<double>& in) :
    g1d_(g1d),
    in_(in),
    binwidth_(g1d_.h()),
    count_(dg::evaluate(dg::zero,g1d_))
    {
        for (unsigned j=0;j<in_.size();j++)
        {            
            unsigned bin =(unsigned) ((in_[j]-1e-14-g1d_.x0())/binwidth_) ;
            count_[bin ]+=1.;
//             #ifdef DGDEBUG
            std::cout << "input[" << j << "] = " << in_[j] << 
                         " bin = " << bin << " bincount = " << count_[bin ]<<std::endl;     
//             #endif
        }
        //Normalize
        unsigned Ampmax = (unsigned)thrust::reduce( count_.begin(), count_.end(),0.,   thrust::maximum<double>()  );
        dg::blas1::scal(count_,1./Ampmax);
        
    }
    double binwidth() {return binwidth_;}
    double operator()(double x)
    {    
        double bin = (unsigned) ((x-1e-14-g1d_.x0())/binwidth_+0.5);
        std::cout <<"x ="<<  x +0.5*binwidth_<< " bin=" <<bin << " count=" << count_[bin] <<std::endl;
        return count_[bin];
    }

    private:
    dg::Grid1d<double> g1d_;
    const std::vector<double> in_;
    double binwidth_;
    container  count_;
};
template <class container = thrust::host_vector<double> >
struct Histogram2D
{
     /**
     * @brief Construct from number of bins and input vector
     * @param g1d   grid of output vector
     * @param input input vector
     */
    Histogram2D(const dg::Grid2d<double>& g2d, const std::vector<double>& inx,const std::vector<double>& iny) :
    g2d_(g2d),
    inx_(inx),
    iny_(iny),
    binwidthx_(g2d_.hx()),
    binwidthy_(g2d_.hy()),
    g1dx_(g2d_.x0(),g2d_.x1(), g2d_.n(), g2d_.Nx(),dg::DIR),
    g1dy_(g2d_.y0(),g2d_.y1(), g2d_.n(), g2d_.Ny(),dg::DIR),
    countx_(dg::evaluate(dg::zero,g1dx_)),
    county_(dg::evaluate(dg::zero,g1dy_)),
    count_(dg::evaluate(dg::zero,g2d_))
    {
        for (unsigned j=0;j<inx_.size();j++)
        {            
            unsigned binx =(unsigned) ((inx_[j]-1e-14-g2d_.x0())/binwidthx_) ;
            countx_[binx ]+=1.;
        }

         for (unsigned j=0;j<iny_.size();j++)
        {
            unsigned biny =(unsigned) ((iny_[j]-1e-14-g2d_.y0())/binwidthy_) ;
            county_[biny ]+=1.;  
        }

        for (unsigned j=0;j<iny_.size();j++)
        {
            unsigned biny =(unsigned) ((iny_[j]-1e-14-g2d_.y0())/binwidthy_) ;
            for (unsigned i=0;i<inx_.size();i++)
            {  
                unsigned binx =(unsigned) ((inx_[i]-1e-14-g2d_.x0())/binwidthx_) ;
//                 std::cout << "x = " << x << " y =" << y << " binx =" << binx <<" biny =" << biny<< std::endl;
//                 if (abs(countx_[binx ] - county_[biny ])<2)  count_[biny*g2d_.Nx()+binx ]=countx_[binx ] + county_[biny ];                

//                 if (abs(countx_[binx ] - county_[biny ])<10)  count_[biny*g2d_.Nx()+binx ]+=1.;                
                 count_[biny*g2d_.Nx()+binx ]+=1.;
            }
        }
        //Normalize
        unsigned Ampmaxx = (unsigned)thrust::reduce( countx_.begin(), countx_.end(),0.,thrust::maximum<double>()  );
        unsigned Ampmaxy = (unsigned)thrust::reduce( county_.begin(), county_.end(),0.,thrust::maximum<double>()  ); 
        unsigned Ampmax =  (unsigned)thrust::reduce( count_.begin(),   count_.end(),0.,thrust::maximum<double>()  );   
        dg::blas1::scal(countx_, 1./Ampmaxx);
        dg::blas1::scal(county_, 1./Ampmaxy);
        dg::blas1::scal(count_,  1./Ampmax);

    }

    double operator()(double x, double y)
    {
        unsigned binx = (unsigned) ((x-1e-14-g2d_.x0())/binwidthx_+0.5) ;
        unsigned biny = (unsigned) ((y-1e-14-g2d_.y0())/binwidthy_+0.5) ;
//         std::cout << "x = " << x << " y =" << y << " binx =" << binx <<" biny =" << biny<< std::endl;

//         return countx_[binxmom]+county_[binymom];
        return count_[biny*g2d_.Nx()+binx ]; ///(county_[biny ]*countx_[binx ]+1);

    }
    private:
    dg::Grid2d<double> g2d_;
    const std::vector<double> inx_,iny_;
    dg::Grid1d<double> g1dx_,g1dy_;
    double binwidthx_,binwidthy_;
    container countx_,county_;
    container count_;
};
double NormalizeToFluc(std::vector<double>& in) {
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
    return sigma;
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
    const unsigned Nhist = 30; 
    const unsigned nhist = 1;
    const unsigned Ninput =1000;
    std::vector<double> input1(Ninput,0.);    
    std::vector<double> input2(Ninput,0.);    

    thrust::random::minstd_rand generator;
    thrust::random::normal_distribution<double> d1;
    thrust::random::normal_distribution<double> d2;

    for (unsigned i=0;i<input1.size();i++)  {  input1[i] = d1(generator); }
//     for (unsigned i=0;i<input1.size();i++)  {  input1[i] = d1(generator)*cos(100.*M_PI*i/input1.size()); }

//     for (unsigned i=0;i<input2.size();i++)  {  input2[i] =input1[i]; }
    for (unsigned i=0;i<input2.size();i++)  {  input2[i] =(d2(generator)-3.)*0.001; }

//         (3.*(input1[i]-2.))*cos(100.*M_PI*i/input2.size());}// d2(generator); }
    //normalize grid and compute sigma
    double sigma_1 = NormalizeToFluc(input1);
    double sigma_2 = NormalizeToFluc(input2);
    dg::Grid1d<double>  g1d1(-4.,4., nhist, Nhist,dg::DIR);
    dg::Grid1d<double>  g1d2(-4.,4., nhist, Nhist,dg::DIR); 
    dg::Grid2d<double>  g2d( -4.,4.,-4.,4., nhist, Nhist,Nhist,dg::DIR,dg::DIR); 
    Histogram<dg::HVec> hist1(g1d1,input1);  
    Histogram<dg::HVec> hist2(g1d2,input2);    
    Histogram2D<dg::HVec> hist12(g2d,input1,input2);    

 
    dg::HVec PA1 = dg::evaluate(hist1,g1d1);
    dg::HVec A1 = dg::evaluate(dg::coo1,g1d1);
    dg::HVec PA2= dg::evaluate(hist2,g1d2);
    dg::HVec A2 = dg::evaluate(dg::coo1,g1d2);
    dg::HVec PA1A2= dg::evaluate(hist12,g2d);
    
    //-----------------NC output start
    int dataIDs1[2],dataIDs2[2],dataIDs12[1];
    int dim_ids1[1],dim_ids2[1],dim_ids12[2];
    int ncid;
    file::NC_Error_Handle err; 
    err = nc_create(argv[2],NC_NETCDF4|NC_CLOBBER, &ncid); 
    //plot 1
    err = file::define_dimension( ncid,"A1_", &dim_ids1[0],  g1d1);
    err = nc_def_var( ncid, "P(A1)",   NC_DOUBLE, 1, &dim_ids1[0], &dataIDs1[0]);
    err = nc_def_var( ncid, "A1",    NC_DOUBLE, 1, &dim_ids1[0], &dataIDs1[1]);
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, dataIDs1[0], PA1.data() );
    err = nc_put_var_double( ncid, dataIDs1[1], A1.data() );
    err = nc_redef(ncid);
    //plot 2
    err = file::define_dimension( ncid,"A2_", &dim_ids2[0],  g1d2);
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
    err = file::define_dimensions( ncid, &dim_ids12[0],  g2d);
    err = nc_def_var( ncid, "P(A1,A2)",   NC_DOUBLE, 2, &dim_ids12[0], &dataIDs12[0]);
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, dataIDs12[0], PA1A2.data() );
    err = nc_redef(ncid);
  /*  err = file::define_dimensions( ncid, &dim_ids2[0],  g1d2);
    err = nc_def_var( ncid, names[1].data(), NC_DOUBLE, 1,&dim_ids2[0], &dataIDs[1]);
    err = nc_put_vara_double( ncid, dataIDs[1], hist2g1d2.data());    */    
            nc_close( ncid);

//     size_t count[2] = {1, Nhist};
//     size_t start[2] = {0, 0};
   //-----------------NC end
//     dg::HVec hist12g2d = dg::evaluate(hist12,g2d);
//     for (unsigned j=0; j < g1d1.size();j++)
//     {
//         std::cout << "PA1[" << j << "] = " << PA1[j] << std::endl;      
//     }
//     for (unsigned j=0; j < g1d2.size();j++)
//     {
//         std::cout << "PA2[" << j << "] = " << PA2[j] << std::endl;      
//     }
//     for (unsigned j=0; j < g1d1.size();j++)
//     {
//         for (unsigned i=0; i < g1d2.size();i++)
//         {
//         std::cout << "hist12g1d2[" << j << i << "] = " << hist12g2d[j+i*Nhist] << std::endl;    
//         }
//     }
    return 0;
}

