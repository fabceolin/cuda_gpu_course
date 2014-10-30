#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "pramsim.h"

int memory[MEM] = {0}, shadow[MEM] = {0};
bool active[N] = {true};
int regx[N] = {0}, regy[N] = {0}, regz[N] = {0};
int pc[N] = {0};
int npc;

// Implementation of the instructions
int Read(int address)
{
	// Read from the shadow memory
	return shadow[address];
}

void Write(int address, int value)
{
	// Write to the actual memory
	memory[address] = value;
}

void Halt()
{
	// Halted processors have negative PC
	npc = -1;
}

void JumpIf(int target, int value)
{
	if(value) {
		npc = target;
	}
}

void simloop()
{
	int stepcount = 0;
	bool is_running;
	do
	{
		is_running = false;
		// Take a snapshot of memory
		memcpy(shadow, memory, MEM * sizeof(int));
		for(int i = 0; i != N; ++i)
		{
			if(pc[i] >= 0)	// Not halted
			{
				// Run one instruction on each active processor
				npc = pc[i] + 1;	// By default, next PC is following instruction
				step(pc[i], i, &regx[i], &regy[i], &regz[i]);
	            //display(memory);
				pc[i] = npc;	// Go to next instruction

				if(npc > 0) is_running = true;
			}
		}
		++stepcount;
	}
	while(is_running);	// At least one processor active
	printf("Program finished after %d steps\n", stepcount);
}

int main()
{
	init(memory);
	simloop();
	display(memory);
	return 0;
}

