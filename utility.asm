bits 64
%macro saveRegisters 0
    push rax
    push rbx 
    push rcx 
    push rdx
%endmacro
%macro getBackRegisters 0
    pop rdx
    pop rcx 
    pop rbx 
    pop rax
%endmacro
section .data
    numberPrint db 0,0,0,0,0,0,0,0,0,10
    NUMBER_LENGTH equ $-numberPrint

    sepator db "=--=", 10
    SEPARATOR_LENGTH equ $-sepator

    hugeSeparator db "-=================================-",10
    HUGE_SEPARATOR_LENGTH equ $-hugeSeparator

    FLOAT_DISPLAY_DECIMALS equ 4
    
    ;# Random weights #

    randomHalfPart dq 0.1
    tweakedSoftMaxRoot dq 0.1 
    randomOddPart dq 1.793

    ;# Data Shortkey #;
    DOUBLE_SIZE equ 8
    INT_SIZE equ 1
    
section .bss 
; CAF ELEMENTS

    cafPointer: resq 1
    cafElements: resq 1
    cafSize: resq 1

    ; PRINTING ELEMENTS

    printedFloatPointer: resq 1
    printedIntPointer: resq 1
    auxiliaryIntegerPointer: resq 1

    ; RANDOM ELEMENTS

    randomFloatPointer: resq 1
    randomIntPointer: resq 1
    randomRangePointer: resq 1
    randomRange: resq 1 

    ; PSEUDO RANDOM NUMBER GENERATION

    randomSeed: resb 1 
    randomTimePointer: resb 1
section .text 
    quit:
        mov rax, 1
        mov rbx,0
        int 0x80
        ret
    print:
        mov rax, 4
        mov rbx, 1
        int 0x80
        ret    
    displayFloat:
       saveRegisters
        mov qword [printedFloatPointer], rax; move the float to rdx
        mov rax, 1
        mov rbx, 10
        xor rcx, rcx 
        xor rdx, rdx
        offset:
            mul rbx
            
            inc rcx
            cmp rcx, FLOAT_DISPLAY_DECIMALS
            jne offset
        
        mov qword [auxiliaryIntegerPointer], rax
        fld qword [printedFloatPointer]
        fimul dword [auxiliaryIntegerPointer]
        fistp qword [auxiliaryIntegerPointer]

        mov rax, [auxiliaryIntegerPointer]

        call displayInteger
        
        getBackRegisters
        ret
    displayInteger:
        saveRegisters

        xor rdx, rdx 
        xor rcx, rcx
        cmp rax, 0
        jge .unsigned

        mov [numberPrint], byte "-"

        .unsigned:
            mov qword rcx, rax
            sar rcx, 63
            add rax, rcx 
            xor rax, rcx 

        mov rcx, NUMBER_LENGTH
        sub rcx, 2
        mov rbx, 10

        .displayDigit:
            xor rdx, rdx

            cmp rcx,FLOAT_DISPLAY_DECIMALS
            jne .addDigit

            mov [numberPrint+rcx], byte "."
            jmp .done
        
        .addDigit:
            div rbx
            add rdx, 48
            mov [numberPrint+rcx], byte dl
            jmp .done

        .done:
            dec rcx
            cmp rax, 0
            jnz .displayDigit

            cmp rcx, FLOAT_DISPLAY_DECIMALS
            jge .displayDigit

        mov rcx, numberPrint
        mov rdx, NUMBER_LENGTH
        call print

        call resetPrintedString

        getBackRegisters
        ret
    displayUnity:
        saveRegisters

        xor rdx, rdx 
        xor rcx, rcx
        mov rcx, NUMBER_LENGTH
        sub rcx, 2

        mov rbx, 10

        .displayUnityDigit:
            xor rdx, rdx

            div rbx
            add rdx, 48
            mov [numberPrint+rcx], byte dl

            
            dec rcx
            cmp rax, 0
            jnz .displayUnityDigit


        mov rcx, numberPrint
        mov rdx, NUMBER_LENGTH
        call print

        call resetPrintedString

        getBackRegisters
        ret
    displaySeperator:
        saveRegisters
        mov rcx, sepator
        mov rdx, SEPARATOR_LENGTH
        call print
        getBackRegisters
        ret
    displayHugeSeparator:
        saveRegisters
        mov rcx, hugeSeparator
        mov rdx, HUGE_SEPARATOR_LENGTH
        call print
        getBackRegisters
        ret
    resetPrintedString:
        mov rax, 8
        resetPrintedStringLOOP:
            mov byte [numberPrint+rax], 0
            dec rax
            cmp rax, 0
            jge resetPrintedStringLOOP
        mov byte [numberPrint+9], 10
        ret 
    caf:
        mov	rax, 45 ; sys_brk
        xor	rbx, rbx
        int	0x80


        mov rcx, rax ; save the initial break


        mov [cafPointer], rax ; Move pointer to break
        mov rbx, [cafSize]
        mov rax, [cafElements]
        mul rbx ; Set rax as the amount of bytes we need

        add [cafPointer], rax ; Move again pointer

        mov	rbx, [cafPointer]
        mov	rax, 45
        int	0x80

        cmp rax, 0
        jge noissue

        noissue:
        
        ;mov rdx, rax ; save the final break
        mov rax, rcx
        ret

    generateRandomFloat:
        saveRegisters

        xor rax,rax 
        xor rbx,rbx 
        xor rdx,rdx
        xor rcx,rcx

        rdrand rbx 
        mov byte al, bl

        
        xor rcx, rcx 
        add rcx, [randomRange]
        div rcx
        mov rax, rdx
        mov rcx, [randomRange]

        mov qword [randomIntPointer], rax
        mov qword [randomRangePointer], rcx
        fild qword [randomIntPointer] 
        fidiv dword [randomRangePointer] 
        fstp qword [randomFloatPointer]
        
        getBackRegisters

        ret
    insertListAtIndex:
        push rax ; src-list length
        push rbx ; src-list pointer
        push rcx ; dst-list index
        push r8 ; dst-list pointer

        ; Get the index in bytes
        push rax ; save the rax register
        mov rax, rcx 
        mul r9 ; multiply the index by the size of a list element

        ; Go to index
        
        add qword r8, rax 
        pop rax ; load back the rax register

        ; Loading and setting loop
        insertListAtIndexLOOP:
            mov qword rdi, [rbx] ; load
            mov qword [r8], rdi  ; set

            ; Go to next index
            add qword rbx, 8 
            add qword r8, 8

            ; looping stuff
            dec rax 
            cmp rax,0
            jnz insertListAtIndexLOOP
        
        pop r8 
        pop rcx 
        pop rbx 
        pop rax
        ret

