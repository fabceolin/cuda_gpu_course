#include <stdio.h>
#include "pramsim.h"

// Initialization of inputs, before the program starts
void init(int memory[])
{
	for(int i = 0; i != MEM; ++i)
	{
		memory[i] = i;
	}
}

// Display the result after the program finishes
void display(int memory[])
{
	printf("Memory contents:\n");
	for(int i = 0; i != N; ++i)
	{
		printf("%d ", memory[i]);
	}
	printf("\n");
}

// This is our PRAM program
// After each case x label, you can put an instruction
void step(int pc, int i, int * x, int * y, int * z)
{
	// i is the processor number
	// *x and *y are temporary registers
	switch(pc)
	{
	case 0: *x = Read(i);
	break;
	case 1: *y = Read(N + i);
	break;
	case 2: *x = *x + *y;
	break;
	case 3: Write(i, *x);
	break;
	case 4: Halt();
	break;
	default: Halt();
	}
}
