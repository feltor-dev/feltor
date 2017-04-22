#include "read_input.h"

using namespace std;
using namespace file;

int main()
{
    vector<double> v;
    try{ v = read_input( "input.test"); }
    catch (spectral::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
            cout << v[i] << " ";
        cout << endl;
        return -1;}
    for( unsigned i = 1; i<v.size(); i++)
        cout << v[i] << " ";
    cout << endl;
    cout << "Total number of parameters read is: "<<v.size()-1 <<"\n";
    return 0;
}

