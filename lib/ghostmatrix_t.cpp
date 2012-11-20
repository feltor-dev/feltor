#include <iostream>
#include "ghostmatrix.h"


using namespace std;
using namespace toefl;

int main()
{
    cout << "Test of GhostMatrix\n";
    cout << "Allocate a Ghostmatrix and assign the ghostcells:\n";
    GhostMatrix< double, TL_NONE> gm( 3,4);
    gm.zero();
    for( int j = -1; j < 5; j++)
    {
        gm.at( -1, j) = 10;
        gm.at( 3, j) = 15;
    }
    gm(1,1) = 9;
    for( int i = 0; i < 3; i++)
    {
        gm.at( i, -1) = 4;
        gm.at( i, 4) = 7;
    }

    gm.display(cout );
    cout << "Test of copy and assign\n";
    GhostMatrix<double, TL_NONE> gm2(gm);
    gm2 = gm;
    gm2.display();

    cout << "Test of void GhostMatrices.\n";
    cout << "Should only show the interior of previous example:\n";
    GhostMatrix<double, TL_NONE> ghost;
    //ghost.resize(3,4);
    ghost.allocate( 3,4);
    swap_fields( ghost, gm);//OK!
    try{
    ghost.display( cout);
    }catch( Message& m){m.display();}
    




    return 0;
}
