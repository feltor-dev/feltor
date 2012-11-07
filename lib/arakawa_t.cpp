#include <iostream>
#include <iomanip>
#include "arakawa.h"
#include "matrix.h"

using namespace std;
using namespace toefl;

const double h = 1.;
const double c = 1./(12.*h*h);
int main()
{
    int rows = 5, cols = 5;
    GhostMatrix<double> lhs( rows, cols), rhs( rows, cols);
    Matrix<double> jac( rows, cols);
    jac.zero(), lhs.zero(), rhs.zero();
    Arakawa arakawa(h);

    for( int i=0; i<rows; i++)
        for( int j=0; j<cols; j++)
        {
            lhs( i, j) = i + 7*j;
            rhs( i, j) = 2*i*i*i +3*j*j;
        }
    //Make Dirichlet BC
    for( int j = 0; j < cols + 1; j++)
    {
        lhs.at(-1, j) = 0;
        lhs.at(rows,j)= 0;
    }
    for( int i = 0; i < cols + 1; i++)
    {
        lhs.at(i, -1) = 0;
        lhs.at(i,cols)= 0;
    }
    cout << setprecision(2) << fixed;
    cout << lhs << endl << rhs <<endl;
    cout << "The Matrices including ghost Cells\n";
    lhs.display(cout);
    cout << endl;
    rhs.display(cout);
    cout << endl;
    arakawa( lhs, rhs, jac);
    cout << jac <<endl;
    return 0;
}
