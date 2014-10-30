#include <stdio.h>
#include "pramsim.h"

// Initialization of inputs, before the program starts
void init(int memory[])
{
	for(int i = 0; i != 2 * N; ++i)
	{
		memory[i] = i;
	}
}

// Display the result after the program finishes
void display(int memory[])
{
	printf("Result:\n");
	printf("%d\n", memory[0]);
}

// This is our PRAM program
// After each case x label, you can put an instruction
void step(int pc, int i, int * x, int * y, int * z)
{
	// i is the processor number
	// *x, *y and *z are temporary registers
	switch(pc)
	{
	case 0: *z = N;
	break;
	case 1: *x = Read(i);
	break;
	case 2: *y = Read(i+*z);
	break;
	case 3: *x = *x + *y;
	break;
	case 4: Write(i,*x);
	break;
	case 5: *z = *z / 2;
	break;
	case 6: JumpIf(1,(*z>=1));
	break;
	case 7: Halt();
	break;
	// Add other cases as necessary
	default: Halt();
	}
}
