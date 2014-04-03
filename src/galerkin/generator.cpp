#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>


//edit these variables
const double tau = 0;
const std::string buero = "backup";
const double sigma = 10;
//const double Ra = 2e5;
const double itstp = 85; //85 for s10 and s05
std::vector<double> a = {0.1, 0.5, 1, 2, 4 };
const int global = 1;
//v2!!! edit at nu and names 

const double T_max = 30;
const double maxout = 300;
const double kappa = 0.0005;

int main()
{
    std::string name;
    std::stringstream title;
    for( unsigned i = 0; i<a.size(); i++)
    {
        double amp = a[i];
        title.str("");
        title<<std::setfill('0');
        title<<"t"<<tau<<"s"<<std::setw(2)<<sigma<<"a";
        if( amp < 0.1)
            title<<std::setw(4)<< amp*100;
        else if( amp < 100)
            title<<std::setw(3)<<(int)(amp*10);
        else
            title<<std::setw(3)<<990;
        name = title.str();
        std::ofstream os( name+".in" );
        std::ofstream job(name+".sh" );
        double gamma = sqrt( kappa*amp/sigma*(1+tau));

        os << "TOEFL\n";
        os << "n = 4\n";
        os << "Nx = 300\n";//Nx
        os << "Ny = 300\n";//Ny
        os << "k = 3\n";//Ny
        os << "dt = "<<T_max/gamma/(maxout*itstp)<<"\n"; //dt
        os << "eps_pol = 1e-6\n"; //eps_pol
        os << "eps_gamma = 1e-10\n"; //eps_gamma
        os << "lx = "<<40*sigma<<"\n";//lx
        os << "ly = "<<40*sigma<<"\n";//ly
        os << "bc_x = 1\n";//bc_x
        os << "bc_y = 0\n";//bc_y
        os << "global = "<<global<<"\n";//global
        //os << "= "<<sqrt((1+tau)*pow(sigma,3)*kappa*amp/Ra)<<"\n";//nu
        os << "nu = 1e-2\n";//nu
        os << "kappa = "<<kappa<<"\n";//kappa
        os << "tau = "<<tau<<"\n"; //tau
        os << "amp = "<<amp<<"\n";//amp
        os << "sigma = "<<sigma<<"\n";
        os << "posX = 0.25\n"; //posX
        os << "posY = 0.5\n"; //posY
        os << "itstp = "<<itstp<<"\n";
        os << "maxout = "<<maxout<<"\n";
        os << std::endl;

        //edit backup or blobs directory!
        job << 
"#!/bin/bash\n\
# submit with qsub.py \n\
        \n\
export NAME="<<name<<"\n\
export DATA_DIR=data/$NAME\n\
export DESCR=n4N300v2\n\
export INPUT="<<name<<".in\n\
#$ -N "<<name<<"_TOE\n\
#$ -o "<<name<<".out\n\
#$ -P fermi\n\
#$ -q fermi\n\
#$ -j yes\n\
#$ -cwd\n\
            \n\
#$ -pe smp 6 \n\
            \n\
#make toefl_hpc\n\
export OMP_NUM_THREADS=$NSLOTS\n\
export LD_LIBRARY_PATH=/opt/intel/lib/intel64:$LD_LIBRARY_PATH\n\
./toefl_hpc $INPUT $DATA_DIR/$DESCR.h5   &> $DATA_DIR/$DESCR.info\n\
(cd ../../diag && exec ./com $DATA_DIR/$DESCR.h5   $DATA_DIR.v2dat)\n\
scp $DATA_DIR.v2dat $BUERO:paper/globalgf/blob/latex/data/global/$NAME.v2dat\n\
scp $DATA_DIR/$DESCR.h5 $BUERO:"<<buero<<"/$NAME/$DESCR.h5\n\
scp $DATA_DIR/$DESCR.info $BUERO:"<<buero<<"/$NAME/$DESCR.info\n\
";

       
    }

    
    return 0;
}
