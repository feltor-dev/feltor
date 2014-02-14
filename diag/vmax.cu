#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>

//scan all imputfiles for maximum radial velocity and write to std::out
int main( int argc, char* argv[])
{
    if( argc == 1)
    {
        std::cerr << "Usage: "<<argv[0]<<" [input1.dat] [input2.dat] ...\n";
        return -1;
    }
    std::ifstream is;
    double x, y;
    for( int i=1; i< argc; i++)
    {
        std::vector<double> v;
        is.open( argv[i]);
        std::string s; 
        std::getline( is, s);
        while( is.good())
        {
            is >> x >> x>> x>>y; //4th data is VelX
            std::getline( is, s); //throw away until newline
            v.push_back(y);
        }
        std::cout << *std::max_element(v.begin(), v.end()) <<"\n";
        is.close();

    }

    
    return 0;
}

