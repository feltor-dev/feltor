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
        dg::Handle<aAnimal> h0, h1(new Mouse()); //default and pointer constructor
        dg::Handle<aAnimal> h2(h1.get()); //reference constructor
        aAnimal* ptr = new Cat();
        h0.reset(ptr); //pointer reset
        h1.reset( h2.get()); //reference reset
        h1.get()=h2.get();//reference test
        h1.swap(h0); //swap test
        h0.get().speak();
        h1.get().speak();
        h2.get().speak();
        h1 = h0;
        h1.get().speak();
        h1.clear();
    }
    {
        std::cout << "Test correct behaviour of buffer class with mouse\n";
        dg::Buffer<Mouse> buffer;
        buffer.data().speak();
        dg::Buffer<Mouse> buffer2 = buffer;
        buffer2.data().speak();
    }

    return 0;
}
