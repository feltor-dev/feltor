#include <iostream>
#include <any>
#include <vector>
#include <cassert>

#include "memory.h"

#include "catch2/catch_all.hpp"

struct aAnimal
{
    virtual void speak()const = 0;
    virtual aAnimal* clone() const=0;
    virtual ~aAnimal(){
        std::cout << "Animal Destructor\n";
    }
};

//Yes pure virtual functions can have a definition
void aAnimal::speak()const{ std::cout << "I am ";}

struct Mouse: public aAnimal
{
    virtual void speak()const{
        aAnimal::speak();
        std::cout << " a mouse!\n";
    }
    virtual Mouse* clone()const{
        std::cout << "Mouse is cloned!\n";
        return new Mouse(*this);}
};

struct Cat : public aAnimal
{
    virtual void speak()const {
        aAnimal::speak();
        std::cout << " a cat!\n";
    }
    virtual Cat* clone()const{
        std::cout << "Cat is cloned!\n";
        return new Cat(*this);}
};


template<class T>
struct Verbose
{
    Verbose() : a(0){
        std::cout << "Construct without value\n";
    }
    Verbose(T v) : a(v) {
        std::cout << "Construct from value\n";
    }
    Verbose(const Verbose& v) {
        a = v.a;
        std::cout << "Copy\n";
    }
    Verbose(Verbose&& v) {
        a = v.a;
        std::cout << "Move\n";
    }
    Verbose& operator=(Verbose&& v) {
        a = v.a;
        std::cout << "Move assignment\n";
        return *this;
    }
    Verbose& operator=(const Verbose& v) {
        a = v.a;
        std::cout << "Assignment\n";
        return *this;
    }
    T a;
};
template<class T>
using verbose_vec = std::vector<Verbose<T>>;

// This program should also be tested with valgrind
// valgrind ./memory_t

TEST_CASE( "ClonePtr")
{
    INFO("Test correct behaviour of handle: cat and mouse\n");
    dg::ClonePtr<aAnimal> h0, h1(new Mouse()); //default and pointer constructor
    dg::ClonePtr<aAnimal> h2(*h1); //reference constructor
    dg::ClonePtr<aAnimal> h3(h0); // copy an empty object
    aAnimal* ptr = new Cat();
    h0.reset(ptr); //pointer reset
    h1.reset( *h2); //reference reset
    *h1=*h2;//reference test
    using std::swap;
    swap(h1,h0); //swap test
    h0->speak();
    h1->speak();
    h2->speak();
    h1 = h0;
    h1->speak();
    h0.reset(nullptr);
    {
        Cat cat;
        h0 = cat;
    }
    std::cout<< "Are you a cat?\n";
    h0->speak();
}
TEST_CASE( "AnyVector")
{
    INFO("\nTest AnyVector behaviour \n");
    dg::detail::AnyVector<verbose_vec> buffer;
    std::cout << "### Construct Verbose (Construct)\n";
    buffer.set<double>( 10);

    std::cout << "### auto reference cast Verbose (no copy!)\n";
    // REMEMBER TO WRITE auto& NOT JUST auto!!
    // T& ref = a;
    // auto vv = ref; deduces auto == T NOT T&
    auto& vv = buffer.get<double>();
    // auto == std::vector<Verbose<double>>
    vv[7].a = 1.7;
    std::cout << buffer.get<double>()[7].a<<" (1.7)\n";
    std::cout << "### Resize with same size shouldn't do anything\n";
    buffer.set<double>( 10);

    std::cout << "### Reset buffer (Construct)\n";
    buffer.set<float>( 10); // Construct float
    std::cout << "### Cast buffer (check that no copy or move is displayed)\n";
    auto& w = buffer.get<float>();
    w[7].a = 42.f;
    std::cout << buffer.get<float>()[7].a<<" (42)\n";
    std::cout << "### Test typeid\n";
    try{
        buffer.get<double>();
    }catch ( std::bad_any_cast& e)
    {
        std::cerr<< "### Expected error: "<<e.what()<<"\n";
    }
    std::cout << "### Construct 2nd buffer\n";
    dg::detail::AnyVector<verbose_vec> buffer2;
    buffer2.set<double>(4);
    std::cout << "### Test swap (no copies or moves) \n";
    std::swap ( buffer, buffer2);
    std::cout << buffer2.get<float>()[7].a<<" (42)\n";
    std::cout << "### Construct verbose vector \n";
    verbose_vec<float> to_swap( 4);
    std::cout << "### Swap in verbose vector of same type (no copy or move) \n";
    buffer2.swap( to_swap);
    assert( to_swap.size() == 10);
    std::cout << "### Construct verbose vector \n";
    verbose_vec<double> to_swap_d( 3, 7.);
    std::cout << "### Swap in verbose vector of another type (no copy or move) \n";
    buffer2.swap( to_swap_d);
    assert( to_swap_d.size() == 0);
    std::cout << buffer2.get<double>()[2].a<<" (7)\n";
}
