## Contributions are welcome

We are always happy when you are interested in the code or want to
contribute. Currently we use the
[fork and pull](https://help.github.com/articles/about-collaborative-development-models/) type of collaboration. Contributions are possible by making a
[pull-request](https://help.github.com/articles/about-pull-requests/) to
the official [feltor-dev/feltor](https://www.github.com/feltor-dev/feltor)
master branch. This means you have to fork the feltor-dev/feltor
repository. When you make a pull-request we will review the code and may
require changes before your code is merged into the official branch. If
you don't feel comfortable, you can always create a pull-request to
[mwiesenberger/feltor](https://www.github.com/mwiesenberger/feltor) or
[mrheld/feltor](https://www.github.com/mrheld/feltor) to get some advice
or help on your code before going for the official branch. We only have
one rule:

* *The master branch on feltor-dev/feltor must always be as stable and
correct as possible*

You can do whatever you want in your own repository but a suggestion
would be to use something similar to the
[github flow](https://guides.github.com/introduction/flow/). Some time ago
we also tried the
[GitFlow](http://nvie.com/posts/a-successful-git-branching-model/) but we
find that it is unnecessarily complex and causes much confusion.

## Readability and Documentation

When you contribute code it is important to us that your code is
readable. As a rule of thumb you should be able to easily figure out
what the code is doing even after you haven't looked at it for a month
or two. Simple code is less error-prone and easy to use.

When you contribute new dg library functions or clases we require you to
document them with Doxygen, for others to know what the function/class
is supposed to do. When you work on a new physical project or
diagnostics you should write a technical documentation on the program(s)
with LateX and upload it if you decide to make your project available in
the official repository. Please find the workflow associated with the
documentation in the following graph:

![contributions](https://feltor-dev.github.io/_includes/FELTOR_contribute_workflow.jpg)





## Error handling

Reliability and credibility are two important concepts for a scientific
software package such as FELTOR. In order to be as correct as possible we 
write at least one test program to every class and function we
write and publish (unit testing). These test programs are designed to
e.g. test convergence of algorithms or check the validity of output
under various circumstances. In this context, we place a high value on documentation:

---

The documentation is **binding**: if the input paramters meet the stated precondtions the result is what the documentation says it should be

---

_We define a **bug** in a code as a situation in which either the input paramters passed to a_
_function do not match the preconditions stated in the documentation, or_
_the preconditions are met but the result is not what the documentation says it is._



Unfortunately, we cannot explicitly check all (pre- and post-)conditions inside a
function because sometimes there is simply too much overhead to do so. 
(Consider the case where a function requires a sorted array. 
Checking that the given array is sorted may take longer than the whole algorithm performed on the array.)
This means that in our library we check 
conditions only if it does not hurt performance (after all it is a high performance library). 

If the conditions on input or output parameters are not met, 
the behaviour of a function shall be **undefined**. 
That means the function can do whatever it pleases 
including throwing an exception, aborting the program or execute normally as if nothing happened. 

**assert** is a C-macro that writes a message to std::cerr and then calls abort when the input expression is false.
The macro is disabled if a macro with the name NDEBUG is defined. The message  includes the expression, the filename and the line-number. 

**throw** is a mechanism in C++ to initiate stack-unwinding. 
If it is not caught in a try and catch block std::terminate (in g++ it tries to catch a std::exception, 
print the what message and then calls abort) is called. The throw mechanism can do more than the C-assert. 
Note that if the documentation says that a function will throw an exception for certain input paramters, 
this does not indicate a bug. It is the predicted behaviour of the function. 
The bug would be if the documentation says the function throws for certain 
input parameter and then the function does not throw for these parameters. 

A bug is detected by checking that pre- and postconditions of a function 
meet the documentation. Unfortunately, we can never guarantee that a 
function always works correctly we can only proof that it works incorrectly. 
However, we do write **a lot** of test programs to try to debug as good as we can.

---

**Whoever finds a bug is responsible for it.**

---

## Design goals

We try to realize the following design goals:

### Code readability

Numerical algorithms can be formulated clearly and concisesly. Specific parallelization strategies or optimization (vectorization) details are absent in application codes.

### Speed

Feltor provides specialized versions of performance critical functions for various target hardware architectures including for example GPUs and Intel Xeon Phis. Writing parallel code is the default in Feltor.

### Reliable

As users of our own library we know how important it is that functions
and classes perform tasks the way it is documented.
Only thus can we rely on the correctness of results that we possibly
want to publish in a scientific journal.
Therefore, we test every class and function we
write and publish from as many different angles as we can think of
(unit testing, see [error handling](errorhandling.md) for more details).

### Ease of use

We try to make our interfaces as intuitive and simple as possible. You don't have to be specialist in C++ to write useful, fast and reliable code with Feltor. This feature is enhanced by an exhaustive [documentation](https://feltor-dev.github.io/doc/dg/html/modules.html) and [tutorials](tutorial_lvl1.md).

### Platform independent

Application code runs unchanged on a
large variety of hardware ranging from a desktop environment
to mid-sized compute clusters with dedicated accelerator cards.
The library adapts to the resources present in
the system and chooses correct implementation of functions
at compile time.

### Extensibility

The library is open for extensions to future hardware, new numerical algorithms and physical model equations.

### Fast development

As a user it should be easy for you to get a working simulation of your brand new model equations in a minimum amount of time. On both Linux and Windows.

### Adhere to good practice

When it comes to design questions we try to follow established coding practices. Here, we mark the works of Scott Meyers, Herb Sutter, Andrei Alexandrescu and others.

### Defined scope

Our specialty are "discontinuous Galerkin methods on structured grids" and their application to drift- and gyro-fluid equations in two and three dimensions. For anything else (particle in cell methods, kinetic equations, unstructured grids or something as "trivial" as parsing input files ) there are better libraries out there.

## Drawbacks

Unfortunately, the above goals made it necessary to sometimes make concessions 

* Each time the hardware is changed the code ultimately has to be recompiled 
* Compilation takes a long time (but not prohibitively long) due to the use of template metaprogramming. 
* The set of defined primitive functions restricts the number of possible numerical algorithms ( e.g. direct solvers cannot be implemented with this library)
* The use of primitive functions limits the optimization possibilities to the scope of each of these functions. 

## Legal stuff

When you make a pull-request to the feltor-dev/feltor repository you
agree to put your code under the MIT license. You will be automatically
listed as a contributor to the code on the website. If you want to use a
different license for your own contributions, please contact us. In your
own fork/copy of the code the MIT License gives you great freedom in
what you do. In fact, you can practically do whatever you please. The
only thing you must do is to name us (Matthias Wiesenberger and Markus
Held) as creators.
