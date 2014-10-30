#include <stdlib.h>
#include <stdio.h>
#include "pramsim.h"
#include "math.h"

// Initialization of inputs, before the program starts
void init(int memory[])
{
	// Print digits from high-order digit N-1 to low-order digit 0
	printf("a=");
	for(int i = N - 1; i >= 0; --i)
	{
		// Create numbers with digits from 0 and 9
		//memory[i] = 4;
		memory[i] = rand() % 10;
		printf("%d", memory[i]);
	}
	printf("\nb=");
	for(int i = 2 * N - 1; i >= N; --i)
	{
		// Create numbers with digits from 0 and 9
		//memory[i] = (rand() % 2)+5;
		memory[i] = rand() % 10;
		printf("%d", memory[i]);
	}
	printf("\np=");
	for(int i = 3 * N - 1; i >= 2 * N; --i)
	{
		// Create numbers with digits from 0 and 9
		//memory[i] = (rand() % 2)+5;
		memory[i] = 0;
		printf("%d", memory[i]);
	}
	printf("\n");
}

// Display the result after the program finishes
void display(int memory[])
{
	printf("r=");
	for(int i = N - 1; i >= 0; --i)
	{
		printf("%d", memory[i]);
	}
	printf("\n");

	printf("g=");
	for(int i = 2 * N - 1; i >= N; --i)
	{
		printf("%d", memory[i]);
	}
	printf("\n");

	printf("p=");
	for(int i = 3 * N - 1; i >= 2 * N; --i)
	{
		printf("%d", memory[i]);
	}
	printf("\n");
}

// This is our PRAM program
// After each case x label, you can put an instruction
void step(int pc, int i, int * x, int * y, int * z)
{
    int debug=0;
    int writecount=0;

//    printf("%d %d %d %d %d\n",pc, i, *x, *y, *z);
	// i is the processor number
	// *x, *y and *z are temporary registers
	switch(pc)
	{
	case 0: *z = 0;
	break;
    // Init N <= g < 2N-1 and 2N <= p < 3N - 1 and 3N <= z <= 4N-1
	case 1: *y = Read(i+N);
	break;
	case 2: *x = Read(i);
	break;
	case 3: *x = *x + *y;
	break;
	case 4: Write(i,((*x)%10));
	break;
	case 5: Write(i+N,((*x)>9));
	break;
	case 6: Write(i+2*N,(*x==9)); // p
	break;
	case 7: Write(3*N+i, 0);
	break;
    // Prefix sum the g
	case 8: JumpIf(13,i<(int)pow(2,Read(3*N+i)));
	break;
	case 9: *x = Read(i+N) /* g */ || ( Read(i+2*N) /* p */ && Read(i+N-(int)pow(2,Read(3*N+i))) /*g' */ );
	break;
	case 10: Write(i+N,*x); //g
	break;
	case 11: *x = Read(i+2*N) && Read(i+2*N-(int)pow(2,Read(3*N+i)));
	break;
	case 12: Write(i+2*N,*x);
	break;
	case 13: Write(3*N+i,Read(3*N+i)+1); // step increment
	break;
	case 14: JumpIf(8,Read(3*N+i)<(int)log2(N)) ;
	break;
    // Sum
	case 15: JumpIf(19,i==0);
	break;
	case 16: *x = Read(i);
	break;
	case 17: *y = Read(i+N-1) || Read(i+2*N-1);
    break;
	case 18: Write(i,(*x + *y)%10);
	break;
	case 19: Halt();
	break;
	default: Halt();
	}
}

