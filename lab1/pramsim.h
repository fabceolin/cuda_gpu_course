#ifndef PRAMSIM_H
#define PRAMSIM_H

// Number of processors
#define N 64

// Size of the memory
#define MEM (N * 5)


// Simulator "instructions"
int Read(int address);
void Write(int address, int value);
void Halt();
void JumpIf(int target, int value);

// Programmer-defined functions
void init(int memory[]);
void display(int memory[]);
void step(int pc, int i, int * x, int * y, int * z);

#endif
