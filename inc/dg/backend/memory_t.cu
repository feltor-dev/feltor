#include <iostream>

#include "memory.h"

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


int main()
{
    {
        std::cout << "Test correct behaviour of handle: cat and mouse\n";
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
    {
        std::cout << "Test correct behaviour of buffer class with mouse\n";
        dg::Buffer<Mouse> buffer;
        buffer.data().speak();
        dg::Buffer<Mouse> buffer2 = buffer;
        buffer2.data().speak();
        std::swap( buffer, buffer2);
    }

    return 0;
}
