CFLAGS=-std=c99 -g
LDFLAGS=-lm -g

all: vectoradd reduction prefixsum longaddition

vectoradd: pramsim.o vectoradd.o
	gcc $(LDFLAGS) -o $@ pramsim.o vectoradd.o

reduction: pramsim.o reduction.o
	gcc $(LDFLAGS) -o $@ pramsim.o reduction.o

prefixsum: pramsim.o prefixsum.o
	gcc $(LDFLAGS) -lm -o $@ pramsim.o prefixsum.o -lm

longaddition: pramsim.o longaddition.o
	gcc $(LDFLAGS) -o $@ pramsim.o longaddition.o -lm

%.o: %.c
	gcc $(CFLAGS) -c $<

clean:
	rm -f vectoradd reduction prefixsum longaddition pramsim.o vectoradd.o reduction.o prefixsum.o longaddition.o

# Dependencies
pramsim.o: pramsim.h
vectoradd.o: pramsim.h
reduction.o: pramsim.h
prefixsum.o: pramsim.h
longaddition.o: pramsim.h
