#include <iostream>
#include <array>
#include "ghostmatrix.h"


using namespace std;
using namespace toefl;

int main()
{
    cout << "Test of GhostMatrix\n";
    cout << "Allocate a Ghostmatrix and assign the ghostcells:\n";
    GhostMatrix< double, TL_NONE> gm( 4,3);
    gm.zero();
    for( int j = -1; j < 4; j++)
    {
        gm.at( -1, j) = 10;
        gm.at( 4, j) = 15;
    }
    gm(1,1) = 9;
    for( int i = 0; i < 4; i++)
    {
        gm.at( i, -1) = 4;
        gm.at( i, 3) = 7;
    }

    gm.display(cout );
    cout << "Test of copy and assign\n";
    GhostMatrix<double, TL_NONE> gm2(gm);
    cout << "Copied matrix: \n";
    gm2.display();
    gm2 = gm;
    cout << "Assigned matrix: \n";
    gm2.display();

    cout << "Test of void GhostMatrices and boundary init.\n";
    gm(0,0) = 3;
    gm(0,1) = 1;
    gm(0,2) = 7;
    gm(1,0) = 2;
    gm(1,1) = 9;
    gm(1,2) = 3;
    gm(2,0) = 8;
    gm(2,1) = 9;
    gm(2,2) = 5;
    gm(3,0) = 7;
    gm(3,1) = 2;
    gm(3,2) = 1;
    GhostMatrix<double, TL_NONE> ghost(4,3,TL_PERIODIC, TL_PERIODIC, TL_VOID);
    ghost.allocate( );
    swap_fields( ghost, gm);//OK!
    ghost.initGhostCells( );

    ghost.display( cout);
    
    array<GhostMatrix<double>, 3> a{{gm,gm,gm}};
    array<Matrix<double>, 3> b{{ghost,ghost,ghost}};


    return 0;
}
