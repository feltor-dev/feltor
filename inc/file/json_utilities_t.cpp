#include <iostream>
#include <cassert>

#include "json_utilities.h"


int main()
{

    Json::Value js;
    dg::file::file2Json( "test.json", js);
    std::cout << "\n\n";
    using dg::file::error;

    for( auto mode : {error::is_silent, error::is_warning, error::is_throw})
    {
        std::cerr << "Display possible error messages for ";
        if( mode == error::is_silent)
            std::cerr << "**Silent**\n";
        else if( mode == error::is_warning)
            std::cerr << "**Warning**\n";
        else if( mode == error::is_throw)
            std::cerr << "**Throw**\n";
        dg::file::WrappedJsonValue ws( js, mode);
        try{
            std::string hello = ws.get_string( "hell", "default");
            std::cerr << "Hello "<<hello<<"\n";
        }catch( std::exception& e){ std::cerr << e.what() <<std::endl; }
        try{
            int idx0 = ws.get( "array").get_int( 2, 0);
            std::cerr << "array 0 "<<idx0<<"\n";
        }catch( std::exception& e){ std::cerr << e.what() <<std::endl; }
        try{
            int idx0 = ws.get_int( "hello", 0);
            std::cerr << "array 0 "<<idx0<<"\n";
        }catch( std::exception& e){ std::cerr << e.what() <<std::endl; }
        try{
            int idx1 = ws.get( "arr").get_int( 1, 0);
            std::cerr << "array 1 "<<idx1<<"\n";
        }catch( std::exception& e){ std::cerr << e.what() <<std::endl; }
        try{
            double hi = ws.get( "neted").get_double( "hi", 0);
            std::cerr << "hi "<<hi<<"\n";
        }catch( std::exception& e){ std::cout << e.what() <<std::endl; }
        try{
            bool test = ws.get( "nested").get_bool( "bol", true);
            std::cerr << "bool "<<test<<"\n";
        }catch( std::exception& e){ std::cout << e.what() <<std::endl; }
        try{
            unsigned uint = ws.get("nested").get("another_net").get_unsigned("number",0);
            std::cerr << "number "<<uint<<"\n";
        }catch( std::exception& e){ std::cout << e.what() <<std::endl; }
        try{
            unsigned uint = ws.get("hello").get("another_net").get_unsigned("number",0);
            std::cerr << "number "<<uint<<"\n";
        }catch( std::exception& e){ std::cout << e.what() <<std::endl; }
        try{
            unsigned uint = ws.get(0).get(2).get_unsigned("number",0);
            std::cerr << "number "<<uint<<"\n";
        }catch( std::exception& e){ std::cout << e.what() <<std::endl; }
    }
    std::cout << "\n\n";
    {
        std::cout << "Test correct access:\n";
        dg::file::WrappedJsonValue ws( js);
        std::string hello = ws.get_string( "hello", "");
        assert( hello == "world");
        int idx0 = ws.get( "array").get_int( 0, 0);
        assert( idx0 == 42);
        int idx1 = ws.get( "array").get_int( 1, 0);
        assert( idx1 == 73);
        double hi = ws.get( "nested").get_double( "hi", 0);
        assert( hi == 38);
        bool test = ws.get( "nested").get_bool( "bool", true);
        assert( test == true);
        unsigned uint = ws.get("nested").get("another_nest").get_unsigned("number",0);
        assert( uint == 7);
        std::cout << "TEST PASSED!\n";
    }

    return 0;
}
