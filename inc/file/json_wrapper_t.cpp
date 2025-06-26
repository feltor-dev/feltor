#include <iostream>
#include <cassert>
#include <filesystem>

#include "catch2/catch_all.hpp"
#include "json_wrapper.h"

// TODO Needs to be more precise
#ifdef DG_USE_JSONHPP
TEST_CASE("Json HPP wrapper")
#else
TEST_CASE("Json CPP wrapper")
#endif
{
    std::ofstream file( "file.json");
    file << R"unique({
        "hello": "world",
        "array": [ 42, 73],
        "nested":
        {
            "hi" : 38.0,
            "bool" : true,
            "another_nest":
            {
                "number" : 7
            }
        }
    })unique";
    file.close();
    REQUIRE( std::filesystem::exists( "file.json"));
    auto js = dg::file::file2Json( "file.json");
    std::string hello0, hello1;
    int idx0, idx1, idx2, idx3;
    unsigned uint0, uint1, uint2;
    double hi;
    bool test;
    SECTION( "Silent")
    {
        dg::file::WrappedJsonValue ws( js, dg::file::error::is_silent);
        std::stringstream ss;
        // catch stream buffer
        auto cerr_buf = std::cerr.rdbuf( ss.rdbuf() );
        hello0 = ws[ "hell"].asString( "default" );
        hello1 = ws.get( "hell", "default").asString( );
        idx0 = ws[ "array"][2].asInt();
        idx1 = ws[ "array"].get(2,0).asInt();
        idx2 = ws["hello"].asInt();
        idx3 = ws[ "arr"][1].asInt( );
        hi = ws[ "neted"]["hi"].asDouble( );
        test = ws[ "nested"]["bol"].asBool( true);
        uint0 = ws["nested"]["another_net"]["number"].asUInt();
        uint1 = ws["hello"]["another_net"]["number"].asUInt();
        uint2 = ws[0][2]["number"].asUInt();
        // reset buffer
        std::cerr.rdbuf( cerr_buf );
        CHECK( ss.str().empty());
        //Avoid compiler warnings
        ss << idx0<<idx1<<idx2<<idx3<<hi<<test<<uint0<<uint1<<uint2;
    }
    SECTION( "Warning")
    {
        dg::file::WrappedJsonValue ws( js, dg::file::error::is_warning);
        std::stringstream ss;
        // catch stream buffer
        auto cerr_buf = std::cerr.rdbuf( ss.rdbuf() );
        hello0 = ws[ "hell"].asString( "default" );
        CHECK( not ss.str().empty());
        ss.str( "");
        hello1 = ws.get( "hell", "default").asString( );
        CHECK( not ss.str().empty());
        ss.str( "");
        idx0 = ws[ "array"][2].asInt();
        CHECK( not ss.str().empty());
        ss.str( "");
        idx1 = ws[ "array"].get(2,0).asInt();
        CHECK( not ss.str().empty());
        ss.str( "");
        idx2 = ws["hello"].asInt();
        CHECK( not ss.str().empty());
        ss.str( "");
        idx3 = ws[ "arr"][1].asInt( );
        CHECK( not ss.str().empty());
        ss.str( "");
        hi = ws[ "neted"]["hi"].asDouble( );
        CHECK( not ss.str().empty());
        ss.str( "");
        test = ws[ "nested"]["bol"].asBool( true);
        CHECK( not ss.str().empty());
        ss.str( "");
        uint0 = ws["nested"]["another_net"]["number"].asUInt();
        CHECK( not ss.str().empty());
        ss.str( "");
        uint1 = ws["hello"]["another_net"]["number"].asUInt();
        CHECK( not ss.str().empty());
        ss.str( "");
        uint2 = ws[0][2]["number"].asUInt();
        CHECK( not ss.str().empty());
        ss.str( "");
        // reset buffer
        std::cerr.rdbuf( cerr_buf );
        //Avoid compiler warnings
        ss << idx0<<idx1<<idx2<<idx3<<hi<<test<<uint0<<uint1<<uint2;
    }
    SECTION( "Error throws")
    {
        // We just test if the message contains certain key words
        dg::file::WrappedJsonValue ws( js, dg::file::error::is_throw);
        try{
            hello0 = ws[ "hell"].asString( "default" );
        }catch( std::exception& e)
        {
            std::string s = e.what();
            CHECK( s.find("\"hell\"") != std::string::npos);
        }
        try{
            hello1 = ws.get( "hell", "default").asString( );
        }catch( std::exception& e)
        {
            std::string s = e.what();
            CHECK( s.find("\"hell\"") != std::string::npos);
        }
        try{
            idx0 = ws[ "array"][2].asInt();
        }catch( std::exception& e)
        {
            std::string s = e.what();
            CHECK( s.find("\"array\"") != std::string::npos);
            CHECK( s.find("2") != std::string::npos);
        }
        try{
            idx0 = ws[ "array"].get(2,0).asInt();
        }catch( std::exception& e)
        {
            std::string s = e.what();
            CHECK( s.find("\"array\"") != std::string::npos);
            CHECK( s.find("2") != std::string::npos);
        }
        try{
            idx0 = ws["hello"].asInt();
        }catch( std::exception& e)
        {
            std::string s = e.what();
            CHECK( s.find("\"hello\"") != std::string::npos);
            CHECK( s.find("\"world\"") != std::string::npos);
            CHECK( s.find("Int") != std::string::npos);
        }
        try{
            idx1 = ws[ "arr"][1].asInt( );
        }catch( std::exception& e)
        {
            std::string s = e.what();
            CHECK( s.find("\"arr\"") != std::string::npos);
        }
        try{
            hi = ws[ "neted"]["hi"].asDouble( );
        }catch( std::exception& e)
        {
            std::string s = e.what();
            CHECK( s.find("\"neted\"") != std::string::npos);
        }
        try{
            test = ws[ "nested"]["bol"].asBool( true);
        }catch( std::exception& e)
        {
            std::string s = e.what();
            CHECK( s.find("\"nested\"") != std::string::npos);
            CHECK( s.find("\"bol\"") != std::string::npos);
        }
        try{
            uint0 = ws["nested"]["another_net"]["number"].asUInt();
        }catch( std::exception& e)
        {
            std::string s = e.what();
            CHECK( s.find("\"nested\"") != std::string::npos);
            CHECK( s.find("\"another_net\"") != std::string::npos);
        }
        try{
            uint1 = ws["hello"]["another_net"]["number"].asUInt();
        }catch( std::exception& e)
        {
            std::string s = e.what();
            CHECK( s.find("\"hello\"") != std::string::npos);
            CHECK( s.find("\"another_net\"") != std::string::npos);
        }
        try{
            uint2 = ws[0][2]["number"].asUInt();
        }catch( std::exception& e)
        {
            std::string s = e.what();
            CHECK( s.find("0") != std::string::npos);
        }
    }
    SECTION("Access")
    {
        dg::file::WrappedJsonValue ws( js);
        std::string hello = ws["hello"].asString();
        CHECK( hello == "world");
        idx0 = ws[ "array"][0].asInt();
        CHECK( ws["array"].size() == 2);
        CHECK( idx0 == 42);
        idx0 = ws[ "array"].get(0,0).asInt();
        CHECK( idx0 == 42);
        idx1 = ws["array"][1].asInt();
        CHECK( idx1 == 73);
        hi = ws[ "nested"][ "hi"].asDouble();
        CHECK( hi == 38);
        test = ws[ "nested"]["bool"].asBool(true);
        CHECK( test == true);
        unsigned uint = ws["nested"]["another_nest"]["number"].asUInt();
        CHECK( uint == 7);
        INFO( "Test access string");
        std::string acc = ws["nested"]["another_nest"]["number"].access_string();
        INFO( acc);
        CHECK( acc.find("\"nested\"") != std::string::npos);
        CHECK( acc.find("\"another_nest\"") != std::string::npos);
        CHECK( acc.find("\"number\"") != std::string::npos);
    }

    SECTION( "Test Vector 2 Json")
    {
        dg::file::WrappedJsonValue ws( dg::file::vec2json( {42, 73}));
        CHECK( ws.get( 0,0).asInt() == 42);
        CHECK( ws.get( 1,0).asInt() == 73);
    }
    std::filesystem::remove( "file.json");

}
