//#define CATCH_CONFIG_MAIN
//#include "catch2/catch.hpp"
// No longer needed in catch2 v3

#include <catch2/catch_session.hpp>

int main( int argc, char* argv[] ) {
  // your setup ...

  int result = Catch::Session().run( argc, argv );

  // your clean-up...

  return result;
}
