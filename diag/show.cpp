
#include <iostream>
#include <iomanip>
#include <string>

#include "file/file.h"



int main( int argc, char** argv)
{
    if( argc != 2)
    {
        std::cerr << "Usage: "<<argv[0]<<" [inputfile.h5]\n";
        return -1;
    }
    hid_t file = H5Fopen( argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    std::string name = file::getName( file, 0);
    std::string in;
    herr_t  status;
    in.resize( 10000);
    status = H5LTread_dataset_string( file, name.data(), &in[0]); //name should precede t so that reading is easier

    std::cout << "Inputfile for data in '"<<argv[1]<<"':\n";
    std::cout << in<<std::endl;
    H5Fclose( file);
    return 0;
}
