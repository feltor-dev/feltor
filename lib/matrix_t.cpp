#include <iostream>
#include <complex>
#include "matrix.h"

using namespace std;
using namespace toefl;

typedef double type;
typedef Matrix<type, TL_DFT_DFT> DoubMat;
type value = 0; //{5,2};

int main()
{
    cout << "Test der TL Matrixklasse\n";
    cout << "Construction m1(2,8)\n";
    DoubMat m1(2,8);
    cout << "Set indices to value ++\n";
    for( size_t i=0; i < 2; i++)
        for( size_t j=0; j < 8; j++)
            m1(i,j) = (value+=1);
    cout << "Copy m2(m1)\n";
    DoubMat m2(m1);
    cout << "zero m1\n";
    m1.zero();
    cout << "swap m1 m2\n";
    swap_fields( m1, m2);
    cout << "const DoubMat\n";
    const DoubMat mc( m1);
    cout << "try Error handling on wrong index\n";
    try{ 
        cout << mc(9,2);
    }
    catch ( Message& message){ 
        message.display();}
    cout << "Assignment m3 = m1\n";
    DoubMat m3(2,8);
    m3 = m1;
    cout << "cout m1 m2 m3 \n";
    cout << m1 << "\n" << m2 << "\n" << m3 << endl;
    cout << "Indexing m1(1,2) += " << value<< "\n";
    m1(1,2) += value;
    cout << m1 << endl;
    cout << "Permute clockwise\n";
    permute_fields( m1, m2, m3);
    cout << m1 << "\n" << m2 << "\n" << m3 << endl;

    cout << "Construct m4( 2,8, value)\n";
    DoubMat m4(2,8,42.);
    cout << m4 <<endl;

    return 0;
}
