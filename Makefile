.PHONY = all 

all: TARGET

TARGET: objects/utility.o objects/tensorFlopKernel.o objects/tensorFlopCaput.o objects/kras.o objects/a64.o
	ld objects/a64.o objects/kras.o objects/tensorFlopCaput.o objects/tensorFlopKernel.o objects/utility.o -o a64
	
objects/utility.o: utility.asm 
	nasm -f elf64 utility.asm -o objects/utility.o

objects/tensorFlopKernel.o: tensorFlopKernel.asm 
	nasm -f elf64 tensorFlopKernel.asm -o objects/tensorFlopKernel.o

objects/tensorFlopCaput.o: tensorFlopCaput.asm 
	nasm -f elf64 tensorFlopCaput.asm -o objects/tensorFlopCaput.o

objects/kras.o: kras.asm 
	nasm -f elf64 kras.asm -o objects/kras.o

objects/a64.o: a64.asm
	nasm -f elf64 a64.asm -o objects/a64.o
	
clean:
	rm a64.o