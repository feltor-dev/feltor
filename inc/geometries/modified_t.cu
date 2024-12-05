#include <iostream>


#include "modified.h"


int main()
{
    std::cout << "Test predicate modifier functions\n";

    assert( dg::geo::mod::everywhere( 1,2) == true);
    assert( dg::geo::mod::nowhere( 1,2) == false);
    dg::geo::mod::HeavisideZ heavy( 0.8, +1);
    assert( heavy( 1,2) == true);
    assert( heavy( 1,0.79) == false);
    dg::geo::mod::HeavisideZ heavyM( 0.8, -1);
    assert( heavyM( 1,2) == false);
    assert( heavyM( 1,0.79) == true);

    dg::geo::mod::RightSideOf twoP( {-1,-2}, {3,0});
    assert( twoP( 0,0) == false);
    assert( twoP( 0,3) == false);
    assert( twoP( 0,-2) == true);

    dg::geo::mod::RightSideOf threeP( {-1,-2}, {3,0}, {0,3});
    assert( threeP( 0,0) == false);
    assert( threeP( 0,4) == true);
    assert( threeP( 0,-2) == true);

    dg::geo::mod::RightSideOf twoM( {3,0}, {-1,-2});
    assert( twoM( 0,0) == true);
    assert( twoM( 0,3) == true);
    assert( twoM( 0,-2) == false);

    dg::geo::mod::RightSideOf threeM( {0,3}, {3,0}, {-1,-2});
    assert( threeM( 0,0) == true);
    assert( threeM( 0,4) == false);
    assert( threeM( 0,-2) == false);

    dg::geo::mod::Above above( {0,0.8}, {0,1.8}); //  same as HeavisideZ
    assert( above( 1,2) == true);
    assert( above( 1,0.79) == false);
    dg::geo::mod::Above aboveM( {0,0.8}, {0,1.8}, false); //  same as HeavisideZ
    assert( aboveM( 1,2) == false);
    assert( aboveM( 1,0.79) == true);
    std::cout << "ALL TESTS PASSED\n";
    return 0;


};
