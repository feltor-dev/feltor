#include <iostream>


#include "dg/file/file.h"
#include "make_field.h"
#include "modified.h"

/* Input-File for COMPASS axisymmetric solovev equilibrium *
  I const with X-point, Te_50eV,B0_1T,deuterium, Cref~2e-5
  ----------------------------------------------------------- */

const std::string geometry_params_toroidal = R"asdf({
    "R_0": 100,
    "equilibrium" : "toroidal"
})asdf";

const std::string geometry_params_Xpoint = R"asdf({
    "A" : 0.0,
    "PP": 1,
    "PI": 1,
    "c" :[  0.07350114445500399706283007092406934834526,
           -0.08662417436317227513877947632069712210813,
           -0.1463931543401102620740934776490506239925,
           -0.07631237100536276213126232216649739043965,
            0.09031790113794227394476271394334515457567,
           -0.09157541239018724584036670247895160625891,
           -0.003892282979837564486424586266476650443202,
            0.04271891225076417603805495295590637082745,
            0.2275545646002791311716859154040182853650,
           -0.1304724136017769544849838714185080700328,
           -0.03006974108476955225335835678782676287818,
            0.004212671892103931178531621087067962015783 ],
    "R_0"                : 547.891714877869,
    "inverseaspectratio" : 0.41071428571428575,
    "elongation"         : 1.75,
    "triangularity"      : 0.47,
    "equilibrium"  : "solovev",
    "description" : "standardX"
})asdf";
const std::string geometry_params_2Xpoint = R"asdf({
    "A": 0,
    "PP": 1,
    "PI": 1,
    "c":
    [
        0.085858385692821789,
        0.25157608973291837,
        -0.38061591778794789,
        -0.10218614802095409,
        0.36709261273307433,
        -0.39303943992277839,
        -0.016931807751497822,
        0,
        0,
        0,
        0,
        0
    ],
    "R_0" : 545,
    "inverseaspectratio": 0.3211009174311926,
    "elongation": 1.44,
    "triangularity": 0.3,
    "equilibrium": "solovev",
    "description" : "doubleX"

})asdf";

// const std::array<double,2> xpoint = {431.552779491927, -433.176887075315};

//@ ----------------------------------------------------------

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

    std::cout << "Check ClosedFieldline region toroidal\n";
    {
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() = dg::file::string2Json( geometry_params_toroidal);
    dg::geo::TokamakMagneticField mag = dg::geo::createMagneticField(
            js);
    dg::geo::mod::ClosedFieldlineRegion closed(mag);
    assert( not closed( 100, 0));
    assert( not closed( 100, 10));
    dg::geo::mod::ClosedFieldlineRegion open(mag, false);
    assert( open( 100, 0));
    assert( open( 100, 10));

    }

    std::cout << "Check ClosedFieldline region 1 X-point\n";
    {
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() = dg::file::string2Json( geometry_params_Xpoint);
    dg::geo::TokamakMagneticField neg_mag = dg::geo::createMagneticField(
            js);
    assert( neg_mag.psip()( 550, 0) < 0);
    js.asJson()["PP"] = -1;
    dg::geo::TokamakMagneticField pos_mag = dg::geo::createMagneticField(
            js);
    assert( pos_mag.psip()( 550, 0) > 0);
    for( auto mag : {neg_mag, pos_mag})
    {
        dg::geo::mod::ClosedFieldlineRegion closed(mag);
        assert(     closed( 550, 0)); // O-point
        assert(     closed( 450, -400)); // above X-point
        assert( not closed( 350, -360)); // SOL
        assert( not closed( 426, -460)); // PFR
        assert( not closed( 600, -360)); // SOL
        assert( not closed( 700,  360)); // SOL
        dg::geo::mod::ClosedFieldlineRegion open(mag, false);
        assert( not open( 550, 0)); // O-point
        assert( not open( 450, -400)); // above X-point
        assert(     open( 350, -360)); // SOL
        assert(     open( 426, -460)); // PFR
        assert(     open( 600, -360)); // SOL
        assert(     open( 700,  360)); // SOL
    }
    }
    std::cout << "Check ClosedFieldline region 2 X-point\n";
    {
    dg::file::WrappedJsonValue js( dg::file::error::is_warning);
    js.asJson() = dg::file::string2Json( geometry_params_2Xpoint);
    dg::geo::TokamakMagneticField neg_mag = dg::geo::createMagneticField(
            js);
    assert( neg_mag.psip()( 550, 0) < 0);
    js.asJson()["PP"] = -1;
    dg::geo::TokamakMagneticField pos_mag = dg::geo::createMagneticField(
            js);
    assert( pos_mag.psip()( 550, 0) > 0);
    for( auto mag : {neg_mag, pos_mag})
    {
        dg::geo::mod::ClosedFieldlineRegion closed(mag);
        assert(     closed( 550, 0)); // O-point
        assert(     closed( 460, -320)); // above 1st X-point
        assert(     closed( 460, +320)); // below 2nd X-point
        assert( not closed( 350, -360)); // SOL
        assert( not closed( 450, -400)); // PFR 1
        assert( not closed( 450, +400)); // PFR 2
        assert( not closed( 600, -360)); // SOL
        assert( not closed( 700,  360)); // SOL
        dg::geo::mod::ClosedFieldlineRegion open(mag, false);
        assert( not open( 550, 0)); // O-point
        assert( not open( 460, -320)); // above 1st X-point
        assert( not open( 460, +320)); // below 2nd X-point
        assert(     open( 350, -360)); // SOL
        assert(     open( 450, -400)); // PFR 1
        assert(     open( 450, +400)); // PFR 2
        assert(     open( 600, -360)); // SOL
        assert(     open( 700,  360)); // SOL
    }
    }

    std::cout << "ALL TESTS PASSED\n";
    return 0;


};
