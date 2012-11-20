#include <iostream>
#include "ghostmatrix.h"


using namespace std;
using namespace toefl;

int main()
{
    cout << "Test of GhostMatrix\n";
    cout << "Allocate a Ghostmatrix and assign the ghostcells:\n";
    GhostMatrix< double, TL_NONE> gm( 3,3);
    gm.zero();
    for( int j = -1; j < 4; j++)
    {
        gm.at( -1, j) = 10;
        gm.at( 3, j) = 15;
    }
    gm(1,1) = 9;
    for( int i = 0; i < 3; i++)
    {
        gm.at( i, -1) = 4;
        gm.at( i, 3) = 7;
    }

    gm.display(cout );
    cout << "Test of copy and assign\n";
    GhostMatrix<double, TL_NONE> gm2(gm);
    gm2 = gm;
    gm2.display();

    cout << "Test of void GhostMatrices and boundary init.\n";
    cout << "Change interior:\n";
    gm(0,0) = 3;
    gm(2,2) = 5;
    gm(0,2) = 7;
    gm(2,0) = 8;
    gm(0,1) = 1;
    gm(1,0) = 2;
    gm(1,2) = 3;
    gm(2,1) = 9;
    GhostMatrix<double, TL_NONE> ghost;
    //ghost.resize(3,4);
    ghost.allocate( 3,3);
    swap_fields( ghost, gm);//OK!
    ghost.initGhostCells( TL_DST10, TL_PERIODIC);

    ghost.display( cout);
    




    return 0;
}
