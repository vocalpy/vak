(reference-models)=

# Declaring models in vak

This section of the reference explains the design 
of the abstractions in vak for representing 
deep learning and neural network models, 
and the rationale behind that design.

Goals for the design include:  
- make it easy to test a particular model 
that was developed for a specified task, 
- make it easy to instantiate 
and work with a model interactively,
e.g. by feeding in a single input 
and then visualizing the output 
to directly inspect performance
- to rely on a "backend" 
that allows us to achieve these goals 
and at the same time 
provide more low-level, fine-grained control 
when needed

Since that last goal permits the first two,
we discuss how we achieved it first.
We have chosen to rely on the lightning framework.

## Declaring a model

To make it easy to declare a model we provide the following abstractions:
- A model definition
- Classes that represent a family of models, all developed for a specific task
- A base model class, that knows how to make an isntance of a model given a definition

## Instatiating a model
  