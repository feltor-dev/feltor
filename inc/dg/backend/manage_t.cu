#include <iostream>

#include "manage.h"

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

struct Dog: public aAnimal
{
    virtual void speak()const{
        aAnimal::speak();
        std::cout << " a dog!\n";
    }
    virtual Dog* clone()const{return new Dog(*this);}
};

struct Cat : public aAnimal
{
    virtual void speak()const { 
        aAnimal::speak();
        std::cout << " a cat!\n";
    }
    virtual Cat* clone()const{return new Cat(*this);}
};


int main()
{
    {
        dg::Handle<aAnimal> h0, h1(new Dog());
        aAnimal* ptr = new Cat();
        h0.reset(ptr);
        std::cout << "Test correct behaviour of handle: cat and dog\n";
        h0.get().speak();
        h1.get().speak();
        std::cout << "Now copy handle: cat \n";
        h1 = h0;
        h1.get().speak();
    }
    {
        dg::Buffer<Dog> buffer;
        std::cout << "Test correct behaviour of buffer class with dog\n";
        buffer.data().speak();
        dg::Buffer<Dog> buffer2 = buffer;
        buffer2.data().speak();
    }




    return 0;
}
