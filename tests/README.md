# Unit tests for Feltor


We use [catch2](https://github.com/catchorg/Catch2) v3 as a test framework for our unit tests.
(On linux can be installed through package manager).
We need catch version > 3.0

## General

What unit tests do is show compilation and correct behaviour of individual functions or class members.
Tests are supposed to be written such that
 -# a test can run automatically and **does not need any user input**
 -# there is **no need to interpret some output** to figure out whether or not the test was passed or not

We use "Black box testing" i.e. we test only through the use of the public interface.

Tests should be small and follow the "AAA" design "Arange" "Act" "Assert"

The messages logged in INFO are printed only when using the `-s` command line option or when a test fails


All functions and global variables in t.cpp programs need to be static.

## Convert a program to catch

 -# Include `#include "catch2/catch_test_macros.hpp"`
 -# Replace main with `TEST_CASE( "test name")` and remove the return 0;
 -# In MPI programs remove `MPI_Init` and  `MPI_Finalize` (catch calls it for us)
 -# Replace any `assert` with `CHECK`
 -# Replace any `std::cout` with `INFO`
 -# Make global functions in cpp file static
 -# Link to `-lCatch2` in Makefile

## Some notable features
`GENERATE` A "for" loop for tests

`TEST_CASE_TEMPLATE` A static for loop for tests for various types
