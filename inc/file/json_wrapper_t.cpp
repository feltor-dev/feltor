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
    auto js = dg::file::file2Json( "file.json");
    std::cout << "\n\n";
    using dg::file::error;

    for( auto mode : {error::is_silent, error::is_warning, error::is_throw})
    {
        std::cerr << "\nDisplay possible error messages for ";
        if( mode == error::is_silent)
            std::cerr << "**Silent**\n";
        else if( mode == error::is_warning)
            std::cerr << "**Warning**\n";
        else if( mode == error::is_throw)
            std::cerr << "**Throw**\n";
        dg::file::WrappedJsonValue ws( js, mode);
        try{
            std::string hello = ws[ "hell"].asString( "default" );
            std::cerr << "Hello "<<hello<<"\n";
        }catch( std::exception& e){ std::cerr << e.what() <<std::endl; }
        try{
            std::string hello = ws.get( "hell", "default").asString( );
            std::cerr << "Hello "<<hello<<"\n";
        }catch( std::exception& e){ std::cerr << e.what() <<std::endl; }
        try{
            int idx0 = ws[ "array"][2].asInt();
            std::cerr << "array 0 "<<idx0<<"\n";
        }catch( std::exception& e){ std::cerr << e.what() <<std::endl; }
        try{
            int idx0 = ws[ "array"].get(2,0).asInt();
            std::cerr << "array 0 "<<idx0<<"\n";
        }catch( std::exception& e){ std::cerr << e.what() <<std::endl; }
        try{
            int idx0 = ws["hello"].asInt();
            std::cerr << "array 0 "<<idx0<<"\n";
        }catch( std::exception& e){ std::cerr << e.what() <<std::endl; }
        try{
            int idx1 = ws[ "arr"][1].asInt( );
            std::cerr << "array 1 "<<idx1<<"\n";
        }catch( std::exception& e){ std::cerr << e.what() <<std::endl; }
        try{
            double hi = ws[ "neted"]["hi"].asDouble( );
            std::cerr << "hi "<<hi<<"\n";
        }catch( std::exception& e){ std::cout << e.what() <<std::endl; }
        try{
            bool test = ws[ "nested"]["bol"].asBool( true);
            std::cerr << "bool "<<test<<"\n";
        }catch( std::exception& e){ std::cout << e.what() <<std::endl; }
        try{
            unsigned uint = ws["nested"]["another_net"]["number"].asUInt();
            std::cerr << "number "<<uint<<"\n";
        }catch( std::exception& e){ std::cout << e.what() <<std::endl; }
        try{
            unsigned uint = ws["hello"]["another_net"]["number"].asUInt();
            std::cerr << "number "<<uint<<"\n";
        }catch( std::exception& e){ std::cout << e.what() <<std::endl; }
        try{
            unsigned uint = ws[0][2]["number"].asUInt();
            std::cerr << "number "<<uint<<"\n";
        }catch( std::exception& e){ std::cout << e.what() <<std::endl; }
    }
    std::cout << "\n\n";
    {
        std::cout << "Test correct access:\n";
        dg::file::WrappedJsonValue ws( js);
        std::string hello = ws["hello"].asString();
        CHECK( hello == "world");
        int idx0 = ws[ "array"][0].asInt();
        CHECK( ws["array"].size() == 2);
        CHECK( idx0 == 42);
        idx0 = ws[ "array"].get(0,0).asInt();
        CHECK( idx0 == 42);
        int idx1 = ws["array"][1].asInt();
        CHECK( idx1 == 73);
        double hi = ws[ "nested"][ "hi"].asDouble();
        CHECK( hi == 38);
        bool test = ws[ "nested"]["bool"].asBool(true);
        CHECK( test == true);
        unsigned uint = ws["nested"]["another_nest"]["number"].asUInt();
        CHECK( uint == 7);
        std::cout << "Test access string\n";
        std::cout << ws["nested"]["another_nest"]["number"].access_string()
                  <<" (nested: another_nest: number)\n";
    }

    SECTION( "Test Vector 2 Json");
    {
        dg::file::WrappedJsonValue ws( dg::file::vec2json( {42, 73}));
        CHECK( ws.get( 0,0).asInt() == 42);
        CHECK( ws.get( 1,0).asInt() == 73);
    }
    std::filesystem::remove( "file.json");

}
