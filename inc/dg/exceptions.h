#pragma once


#include <exception>


/*!@file 
 *
 * Error class to thow. Derived from std::exception
 */
namespace dg
{

/**
 * @brief Class you might want to throw in case of a non convergence
 *
 * @ingroup utilities
 */
struct Fail : public std::exception
{

    /**
     * @brief Construct from error limit
     *
     * @param eps error not reached
     */
    Fail( double eps): eps( eps) {}
    /**
     * @brief Return error limit
     *
     * @return eps
     */
    double epsilon() const { return eps;}
    /**
     * @brief What string
     *
     * @return string "Failed to converge"
     */
    char const* what() const throw(){ return "Failed to converge";}
  private:
    double eps;
};

}//namespace dg
