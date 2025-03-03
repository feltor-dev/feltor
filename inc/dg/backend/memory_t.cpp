#include <iostream>
#include <any>
#include <vector>

#include "memory.h"

#include "catch2/catch_all.hpp"

struct aAnimal
{
    virtual std::string speak()const = 0;
    virtual aAnimal* clone() const=0;
    virtual int id() const = 0;
    virtual ~aAnimal(){
        //std::cout << "Animal Destructor\n";
    }
};

std::string aAnimal::speak()const { return "Animal";}

struct Mouse: public aAnimal
{
    virtual std::string speak()const{
        return  "Mouse";
    }
    virtual Mouse* clone()const{
        m_id --;
        return new Mouse(*this);}
    virtual int id() const { return m_id;}
    private:
    mutable int m_id = -10;
};

struct Cat : public aAnimal
{
    virtual std::string speak()const {
        return "Cat";
    }
    virtual Cat* clone()const{
        m_id ++;
        return new Cat(*this);}
    virtual int id() const { return m_id;}
    private:
    mutable int m_id = +10;
};


template<class T>
struct Verbose
{
    Verbose() : a(0){
        //std::cout << "Construct without value\n";
    }
    Verbose(T v) : a(v) {
        //std::cout << "Construct from value\n";
    }
    Verbose(const Verbose& v) {
        a = v.a;
        //std::cout << "Copy\n";
    }
    Verbose(Verbose&& v) {
        a = v.a;
        //std::cout << "Move\n";
    }
    Verbose& operator=(Verbose&& v) {
        a = v.a;
        //std::cout << "Move assignment\n";
        return *this;
    }
    Verbose& operator=(const Verbose& v) {
        a = v.a;
        //std::cout << "Assignment\n";
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
    CHECK( h1->id() == -11);
    CHECK( h2->id() == -11);
    aAnimal* ptr = new Cat();
    h0.reset(ptr); //pointer reset
    h1.reset( *h2); //reference reset
    *h1=*h2;//reference test
    CHECK( h2->id() == -12);
    CHECK( h1->id() == -12);
    using std::swap;
    swap(h1,h0); //swap test
    CHECK( h0->id() == -12);
    CHECK( h0->speak() == "Mouse");
    CHECK( h1->speak() == "Cat");
    CHECK( h2->speak() == "Mouse");
    h1 = h0;
    CHECK( h0->id() == -13);
    CHECK( h1->id() == -13);
    CHECK( h1->speak() == "Mouse");
    h0.reset(nullptr);
    {
        Cat cat;
        h0 = cat;
    }
    CHECK( h0->speak() == "Cat");
}
TEST_CASE( "AnyVector")
{
    INFO("\nTest AnyVector behaviour \n");
    dg::detail::AnyVector<verbose_vec> buffer;
    INFO( "### Construct Verbose (Construct)\n");
    buffer.set<double>( 10);

    INFO( "### auto reference cast Verbose (no copy!)\n");
    // REMEMBER TO WRITE auto& NOT JUST auto!!
    // T& ref = a;
    // auto vv = ref; deduces auto == T NOT T&
    auto& vv = buffer.get<double>();
    // auto == std::vector<Verbose<double>>
    vv[7].a = 1.7;
    INFO( buffer.get<double>()[7].a<<" (1.7)\n");
    INFO( "### Resize with same size shouldn't do anything\n");
    buffer.set<double>( 10);

    INFO( "### Reset buffer (Construct)\n");
    buffer.set<float>( 10); // Construct float
    INFO( "### Cast buffer (check that no copy or move is displayed)\n");
    auto& w = buffer.get<float>();
    w[7].a = 42.f;
    INFO( buffer.get<float>()[7].a<<" (42)\n");
    INFO( "### Test typeid\n");
    CHECK_THROWS_AS( buffer.get<double>(), std::bad_any_cast);
    INFO( "### Construct 2nd buffer\n");
    dg::detail::AnyVector<verbose_vec> buffer2;
    buffer2.set<double>(4);
    INFO( "### Test swap (no copies or moves) \n");
    std::swap ( buffer, buffer2);
    INFO( buffer2.get<float>()[7].a<<" (42)\n");
    INFO( "### Construct verbose vector \n");
    verbose_vec<float> to_swap( 4);
    INFO( "### Swap in verbose vector of same type (no copy or move) \n");
    buffer2.swap( to_swap);
    CHECK( to_swap.size() == 10);
    INFO( "### Construct verbose vector \n");
    verbose_vec<double> to_swap_d( 3, 7.);
    INFO( "### Swap in verbose vector of another type (no copy or move) \n");
    buffer2.swap( to_swap_d);
    CHECK( to_swap_d.size() == 0);
    INFO( buffer2.get<double>()[2].a<<" (7)\n");
    CHECK( buffer2.get<double>()[2].a == 7);
}
