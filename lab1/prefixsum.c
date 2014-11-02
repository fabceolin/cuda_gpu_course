#include <stdio.h>
#include "pramsim.h"
#include <math.h>
// Initialization of inputs, before the program starts
void init(int memory[])
{
	for(int i = 0; i != N; ++i)
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

#define M_LOG2E 1.44269504088896340736 //log2(e)

/*inline long double log2(const long double x){
        return  log(x) * M_LOG2E;
}*/

// This is our PRAM program
// After each case x label, you can put an instruction
void step(int pc, int i, int * x, int * y, int * z)
{
	// i is the processor number
	// *x, *y and *z are temporary registers
	switch(pc)
	{
	case 0: *z = 0;
	break;
	case 1: JumpIf(6,i<(int)pow(2,*z));
	break;
	case 2: *x = Read(i-(int)pow(2,*z));
	break;
	case 3: *y = Read(i);
	break;
	case 4: *x = *x + *y;
	break;
	case 5: Write(i,*x);
	break;
	case 6: *z = *z + 1;
	break;
	case 7: JumpIf(1,*z<(int)log2(N));
    break;
	case 8: Halt();
	break;
	// Add other cases as necessary
	default: Halt();
	}
}
