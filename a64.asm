bits 64

section .data
    message db "Allocation worked !",10
    MESSAGE_LENGTH equ $-message

    problem db "Sadly, allocation didn't worked :(",10
    PROBLEM_LENGTH equ $-problem

    neuronInfos db "Weights count :  ",10
    INFOS_LENGTH equ $-neuronInfos

    numberPrint db 0,0,0,0,0,0,0,0,0,10
    NUMBER_LENGTH equ $-numberPrint

    sepator db "=--=", 10
    SEPARATOR_LENGTH equ $-sepator

    hugeSeparator db "-=================================-",10
    HUGE_SEPARATOR_LENGTH equ $-hugeSeparator
    
    FLOAT_DISPLAY_DECIMALS equ 4

    DOUBLE_PRECISION_SIZE equ 8
    INT_SIZE equ 1
    
    LAYER_INFORMATIONS_SIZE equ 19 ; 1 -> Layer Id, 1-> NeurCount, 1-> function, 8-> weights (pointer), 8 -> bias (dp float)
    
    MAX_LAYERS_COUNT equ 10
    MAX_NETWORKS_COUNT equ 2
    MAX_LOSS_HISTORY equ 10
    MAX_IDEALS_COUNT equ 10
    

    NULL dq 0

    LEAKY_RELU_COEFFICIENT dq 0.1
    randomHalfPart dq 0.1
    randomOddPart dq 1.793

    userInputLayerSizes db 2, 3, 4, 2, 0
    userInputIdealArrays dq 1.0, 0.0

    floatTest dq 1.0,1.0, 1.0, 0.0, 0.0, 0.0
    floatTest2 dq 1.0, 3.0, 2.5, 5.0, 4.0, 2.0



section .bss

;==- UTILITY ELEMENTS -==;

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

;==- ACTUAL TENSORFLOP -==;

    ; TENSORFLOP HANDLING   

    tensorflopHandles: resq MAX_NETWORKS_COUNT ; Pointer to my CNN-Handles
    tensorflopHandleCreated : resb 1 ; Amount of CNN and-so Cnn-Handles created

    ; CNN-global
    CnnLayersCount: resq 1 ; The number of layer of the current CNN
    CnnWeightsCount: resq 1 ; The number of weights of the current CNN
    CnnNeuronsCount: resq 1 ; The number of neurons of the current CNN

    CnnDataInputSize: resq 1 ; Size (vector size) of what enters in CNN 
    CnnDataOutputSize: resq 1 ; Size (vector size) of what comes out of CNN 

    CnnActivatedMatrixPointer: resq 1 ; A pseudo matrix of all my activated layer's neurons | Size DOUBLE_PRECISION_SIZE * CnnNeuronsCount
    CnnUnactivatedMatrixPointer: resq 1 ; A pseudo matrix of all my unactivated layer's neurons | Size DOUBLE_PRECISION_SIZE * CnnNeuronsCount
    CnnCostMatrixPointer: resq 1 ; 
    CnnBackpropagationWeightsMatrixPointer: resq 1;
    CnnBackpropagationBiasesMatrixPointer: resq 1;
    
    CnnLayersFunctions : resb MAX_LAYERS_COUNT ; Array of activation functions of layers created 
    CnnLayersNeurons : resb MAX_LAYERS_COUNT ; Array of sizes of my layers (in neurons) 
    CnnFirstLayerOffset : resq 1 ; Offset of Neurons needed to propagate (skipping input neurons)

    ; CNN-Computation 

    computeInputLayerSize: resq 1 ; Size of the input layer
    computeOutputLayerSize: resq 1 ; Size of the output layer
    computeOutputNeuronPointer: resq 1 ; Pointer to the neuron, used in dot product to store the result
    computeTempOutputPointer: resq 1 ; POint to the result of a single multiplication in a dot product, temporary

    computedFloatPointer: resq 1 ; Commonly used for activation function, as the result and the parameter of these
    computedSoftMaxSum: resq 1 ; Summation of e^Xi of an array X, used in softMax

    ; CNN-Backpropagation 

    backprop: resq 1

    ; CNN-Loss/Ideals 
    
    lossHistory: resq MAX_LOSS_HISTORY
    lossHistoryIndex: resq 1
    idealsArray: resq MAX_IDEALS_COUNT
    idealsIndex: resq 1

section .text
    global _start
    global .issue

    _start:
        mov qword [CnnLayersCount],  4
        
        call createCnn

    end:
        mov rax, 1
        mov rbx,0
        int 0x80
    issue:
        mov rcx,problem
        mov rdx,PROBLEM_LENGTH
        call print

        jmp end

    
;==- PROCEDURES -==;
;====================;
;/\- User-utility -/\;
;====================;
    print:
        mov rax, 4
        mov rbx, 1
        int 0x80
        ret    
    displayFloat:
        push rax
        push rbx 
        push rcx 
        push rdx 
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
        pop rdx 
        pop rcx 
        pop rbx 
        pop rax
        ret

    displayInteger:
        push rax
        push rbx 
        push rcx 
        push rdx 

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

        pop rdx 
        pop rcx 
        pop rbx 
        pop rax

        ret
    displayUnity:
        push rax
        push rbx 
        push rcx 
        push rdx 

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

        pop rdx 
        pop rcx 
        pop rbx 
        pop rax

        ret
    displaySeperator:
        push rax
        push rbx 
        push rcx 
        push rdx 
        mov rcx, sepator
        mov rdx, SEPARATOR_LENGTH
        call print
        pop rdx 
        pop rcx 
        pop rbx 
        pop rax
        ret
    displayHugeSeperator:
        push rax
        push rbx 
        push rcx 
        push rdx 
        mov rcx, hugeSeparator
        mov rdx, HUGE_SEPARATOR_LENGTH
        call print
        pop rdx 
        pop rcx 
        pop rbx 
        pop rax
        ret
    resetPrintedString:
        mov byte [numberPrint], 0
        mov byte [numberPrint+1], 0
        mov byte [numberPrint+2], 0
        mov byte [numberPrint+3], 0
        mov byte [numberPrint+4], 0
        mov byte [numberPrint+5], 0
        mov byte [numberPrint+6], 0
        mov byte [numberPrint+7], 0
        mov byte [numberPrint+8], 0
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
        jl issue
        
        ;mov rdx, rax ; save the final break
        mov rax, rcx
        ret

    generateRandomFloat:
        push rax
        push rbx
        push rcx
        push rdx

        xor rax, rax 
        xor rbx, rbx 
        xor rdx,rdx
        xor rcx, rcx

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
        
        pop rdx 
        pop rcx
        pop rbx 
        pop rax

        ret
    insertListAtIndex:
        push rax ; src-list length
        push rbx ; src-list pointer
        push rcx ; dst-list index
        push r8 ; dst-list pointer

        ; Get the index in bytes
        mov rdi, rax ; save the rax register
        mov rax, rcx 
        mul r9 ; multiply the index by the size of a list element

        ; Go to index

        add qword r8, rax 
        mov rax, rdi ; load back the rax register

        ; Loading and setting loop
        insertListAtIndexLOOP:
            mov qword rdi, [rbx] ; load
            mov qword [r8], rdi  ; set

            ; Go to next index
            add qword rbx, DOUBLE_PRECISION_SIZE 
            add qword r8, DOUBLE_PRECISION_SIZE

            ; looping stuff
            dec rax 
            cmp rax,0
            jnz insertListAtIndexLOOP
        
        pop r8 
        pop rcx 
        pop rbx 
        pop rax
        ret
;==================;
;/\- TensorFlop -/\;
;==================;
    ; -CNN-Handling- ;
    createCnn:
        ; INIT THE WHOLE CNN ;
        call initCNN

        ; CREATE THE CNN HANDLE ;
        call createCnnHandle
        mov [tensorflopHandles], qword rax ; store my handle

        ; ALLOCATE PARAMETERS MEMORY AND STORE THEM INTO THE HANDLE ;
        call createCnnWeights 

        ; FILL THE WEIGHTS WITH RANDOM VALUES [0,1] ;   
        call fillCnnParameters

        ; CREATE ACTIVED NEURON MATRIX ;
        call createStandardMatrix
        mov qword [CnnActivatedMatrixPointer], rax

        ; CREATE UNACTIVED NEURON MATRIX ;
        call createStandardMatrix
        mov qword [CnnUnactivatedMatrixPointer], rax

        ; CREATE COSTS NEURON MATRIX ;
        call createStandardMatrix
        mov qword [CnnCostMatrixPointer], rax
        
        ; STORE THE INPUT ;
        mov rbx, floatTest
        call initInput

        ; COMPUTE A PROPAGATION ;
        
        call computeCnn

        ; MISCELLANEOUS AND DEBUGGING ;

        ; Display the neurons (activated)
        xor rcx, rcx
        mov rbx, [CnnActivatedMatrixPointer]
        mov byte cl, [CnnNeuronsCount]
        call showPseudoMatrix

        call displaySeperator

        ; Debugging loss calculation
        call calculateGlobalLoss
        call displayFloat
        ret
    createCnnHandle: ; Create a CnnHandle and fill it with the data available
        mov rax, [CnnLayersCount] 
        mov qword [cafElements], rax ; number of layers
        mov qword [cafSize], LAYER_INFORMATIONS_SIZE ; one layer needs 19 bytes
        call caf ; allocate it, so we have our CnnHandles

        mov rbx, rax ; save the handle base location

        xor rcx, rcx; set my for loop as 0
        xor r8, r8
        inc r8
        fillHandleLOOP:            
            ; Store layer index (1 byte)
            mov byte [rax], cl
            inc rax

            ; Store neurons count, (1 byte )
            mov dl, byte [CnnLayersNeurons+rcx]
            mov byte [rax], dl
            inc rax

            ; Store activation function ID, (1 byte)
            mov dl, byte [CnnLayersFunctions+rcx]
            mov byte [rax], dl
            inc rax

            ; Let weigths-pointer and bias (16 bytes) as 0
            ; But still increment to next_information 
            add rax, 16

            inc rcx
            inc r8
            cmp rcx, [CnnLayersCount]
            jne fillHandleLOOP



        mov rax, rbx ; return location of handle
        ret
    createCnnWeights: ; Allocate the memory for the weights
        ; Set the number of iterations as $Layers minus 1, cause last layer doesn't have weights
        mov qword r11, [CnnLayersCount]
        dec r11

        call getWorkingCnn ; get access to current used cnn handle -> set rax as &cnnHandle
        mov r10, rax ; change to r10 register <-> r10 register is now the pointer to my CnnHandle
        add r10, 1 ; set rax as &cnnHandle + 1 <=> &NeuronsCounts[0]

        xor rcx, rcx
        weigthsCreationLOOP:      
            ;== Get number of weights ==;

            xor r8,r8
            xor rax, rax
            mov byte r8b, [r10] ; store the nth-layer neurons-count in rb8
            add r10, LAYER_INFORMATIONS_SIZE ; go to layer n + 1
            mov byte al, [r10] ; store the (n+1)th-layer neurons-count in rax
            mul r8 ; get the number of weights by multiplying r8 ($L) and rax ($L+1), -> rax AND rdx (unused)
            
            
            ;== Allocate weights ==;

            push rcx ; push my loop index to save it
            mov qword [cafElements], rax ; number of weights
            mov qword [cafSize], DOUBLE_PRECISION_SIZE ; a weight is a db float -> 8 bytes 
            call caf
            pop rcx ; get back loop index

            ;== Store my weights pointer ==;

            sub r10, LAYER_INFORMATIONS_SIZE ; go to nth-layer -> neurons-count 
            add r10, 2 ; go to Weight pointer
            mov [r10], rax ; store the pointer to the allocated memory in nth-layer Weight Pointer part
            mov qword [rax], rcx

            ;== Prepare handle pointer for next iteration ==;

            add r10, LAYER_INFORMATIONS_SIZE
            sub r10, 2

            ;= loop stuff =;

            inc rcx



            cmp rcx, r11
            jne weigthsCreationLOOP

        
        

        ret 
    fillCnnParameters: ; Fill weights and biases with random values
        mov qword r11, [CnnLayersCount]
        dec r11

        call getWorkingCnn
        mov r10, rax
        

        layerLOOP:
            mov qword r12, [r10+3] ; store in r12 the weights' layer pointer

            mov rax, r10 ; pass as argument the layer handle
            call getLocalWeightsCount ; get the amount of weights in rax

            mov r13, rax


            weightLOOP:
                mov qword [randomRange], 100
                call generateRandomFloat
                
                fld qword [randomFloatPointer]
                fld qword [randomHalfPart]
                fsubp
                fld qword [randomOddPart]
                fmulp
                fstp qword [randomFloatPointer]
                mov qword rax, [randomFloatPointer]
                mov qword [r12], rax


                add r12, DOUBLE_PRECISION_SIZE
                dec r13 
                cmp r13, 0
                jne weightLOOP

            add r10, LAYER_INFORMATIONS_SIZE


            dec r11 
            cmp r11, 0
            jne layerLOOP

        ret
    createStandardMatrix: ; Create a memory empty space for my activated, unactivated and loss
        mov qword rdx, [CnnNeuronsCount]
        mov qword [cafElements], rdx ; number of neurons
        mov qword [cafSize], DOUBLE_PRECISION_SIZE ; a weight is a db float -> 8 bytes 
        call caf
        ret
    ; -Compute Convolutional Neural Network- ;   
    computeCnn: ; Compute a full propagation
        ; rax -> volatile 
        ; rbx -> weight pointer (input)
        ; rcx -> unactivated neuron pointer (output)
        ; rdx -> activated neuron pointer (input)
        ; rdi -> activated neuron pointer (output)
        ; r8 -> layer offset { Let L, L+1 : L*L+1*Double_Size
        ; r10 -> handle index (used to get rbx)
        ; r11 -> the layer count and loop index

        ; Get the current-CNN Handle
        call getWorkingCnn
        mov r10, rax

        ; Get the informations we need
        ;       - first layer's neuron-count 
        ;       - a pointer to my weights 
        add qword r10, 3 ; move to weights Pointer
        mov qword rbx, [r10] ; store in rbx the weight pointer
        sub qword r10, 2 ; move back to layer's neuron-count


        ; Convulational layer count
        mov r11, [CnnLayersCount]
        dec qword r11 


        ; Get the activated input neuron 

        mov rdx, [CnnActivatedMatrixPointer]

        ; We'll need to increment the pointer without modifying my raw matricies
        mov rcx, [CnnUnactivatedMatrixPointer] 
        mov rdi, [CnnActivatedMatrixPointer]

        add rcx, [CnnFirstLayerOffset]
        add rdi, [CnnFirstLayerOffset]

        
        ; For each layer, proceed to the dots products in each layer
        computeCnnLOOP:
            ; Store the input layer size
            xor rax, rax
            mov byte al, [r10]
            mov qword [computeInputLayerSize], rax

            ; Store the output layer size
            add r10, LAYER_INFORMATIONS_SIZE ; Offset of LAYER_INFORMATIONS_SIZE in my handle
            mov byte al, [r10]
            mov qword [computeOutputLayerSize], rax

            ; Eventually compute the layer
            call computeSingleLayer

            ; Looping stuff
            dec r11
            cmp r11, 0
            jnz computeCnnLOOP
            
        ; Calculate last layer offset and length
        xor rax, rax
        mov byte al, [r10]
        mov qword [computeOutputLayerSize], rax

        push rdx 
        mov rbx, DOUBLE_PRECISION_SIZE
        mul rbx 
        sub rcx, rax 
        pop rdx 

        mov qword rbx, [computeOutputLayerSize]
        call softMax

        ret 
    computeSingleLayer: ; Compute the propagation in a single layer
        push rax

        ; Calculate the offset of each set of weights 
        push rdx
        mov rax, [computeInputLayerSize]
        mov r9, DOUBLE_PRECISION_SIZE
        mul r9
        mov r8, rax
        pop rdx


        computeSingleLayerLOOP:
            ; Compute the dot product
            call computeDotProduct
            mov rax, [computeOutputNeuronPointer]

            mov [rcx], rax ; Save in unactivated (Zs) matrix 
            
            ; If it's the last layer (r11==1) don't activate 
            cmp r11, 1
            jle hasNotToActivate

            ; Call the activation function
            mov [computedFloatPointer], rax
            call leakyReLU
            mov qword rax, [computedFloatPointer]

            hasNotToActivate:

            mov [rdi], rax ; Save in activated (As) matrix

            ; Update my pointers

            add rcx, DOUBLE_PRECISION_SIZE
            add rdi, DOUBLE_PRECISION_SIZE

            add rbx, r8 ; add to my weights pointer, the input-layer's neuron-count times the size of an element

            dec qword [computeOutputLayerSize]
            cmp qword [computeOutputLayerSize],0
            jnz computeSingleLayerLOOP

        ; Add to my neurons pointer, the input-layer's neuron-count times the size of an element
        add rdx, r8 
        
        pop rax
        ret 
    computeDotProduct: ; Dot product of two pointer, vector-size specified in rbx
        push rbx
        push rdx

        xor rax, rax 
        mov byte al, [computeInputLayerSize]
        mov qword [computeOutputNeuronPointer],0 ; reset the output neuron 

        computeDotProductLOOP:
            movhpd xmm0, [NULL] ; reset upper part of the xmm0 in case it's a single dot product
            movhpd xmm1, [NULL] ; reset upper part of the xmm1 in case it's a single dot product
            movq xmm0, [rbx] ; store first weight  
            movq xmm1, [rdx] ; store first input 

            cmp qword [computeInputLayerSize], 2 
            jl computeSingleDotProduct

            movhpd xmm0, [rbx+8] ; store second weight 
            movhpd xmm1, [rdx+8] ; store second input

            computeSingleDotProduct:

            mulpd xmm1,xmm0 ; process the dot product : (W{L,I,K},W{L,I+1,K}) . (A{L,I},A(L,I+1)) 
                            ;                      or : (W{L,I,K},0) . (A{L,I},0) 
                            ; and store (add) in A(L+1,K)

            movlpd [computeTempOutputPointer], xmm1 ; get the lower part of the dot product
            fld qword [computeTempOutputPointer] ; push it on the stack

            movhpd [computeTempOutputPointer], xmm1 ; get the higher part of the dot product
            fld qword [computeTempOutputPointer] ; push it on the stack

            faddp ; add them together 


            fld qword [computeOutputNeuronPointer] ; Load the summation
            faddp ; add result to neuron 
            fstp qword [computeOutputNeuronPointer] ; store result to neuron

            add qword rbx, 16
            add qword rdx, 16
            sub qword [computeInputLayerSize], 2 
            cmp qword [computeInputLayerSize], 0 
            jg computeDotProductLOOP

        mov byte [computeInputLayerSize], al
        pop rdx
        pop rbx
        ret
    
    ; -Loss And Ideals- ;
    calculateGlobalLoss:
        mov rcx, [CnnDataOutputSize] ; Last layer size 
        mov rax, [CnnNeuronsCount] ; Amount of Neurons 
        sub rax, rcx ; Get the index of last layer

        call getOffsetFromIndex ; get the offset in bytes

        add rax, [CnnActivatedMatrixPointer] ; go to last layer 
        mov rax, [rax] ; get the last layer's first neuron's value
        calculateGlobalLossLOOP:

            dec rcx 
            test rcx, rcx 
            jnz calculateGlobalLossLOOP
        ret 
    ; -Backpropagate Convolutional Neural Network- ;
    backpropagateSingleNeuron:
        ret
;=============================;
;/\- TensorFlopish-utility -/\;
;=============================;
    ; -Activation- ;
    ReLU:
        push rax ; save registers
        xor rax, rax ; prepare it 

        mov byte al, [computedFloatPointer+7] ; get the byte containing the sign bit 
        test al, 0d128 ; test the sign 
        jns ReLUDone

        mov qword [computedFloatPointer], 0 ; if negative, set pointer to 0

        ReLUDone:

        pop rax ; get back registers
        ret
    leakyReLU:
        push rax ; save registers
        xor rax, rax ; prepare it 

        mov byte al, [computedFloatPointer+7] ; get the byte containing the sign bit 
        test al, 0d128 ; test the sign 
        jns leakyReLUDone

        ; If the float is negative :

        fld qword [computedFloatPointer] ; Load the float
        fld qword [LEAKY_RELU_COEFFICIENT] ; Load the coefficient of the leeaky relu
        fmulp ; Multiply both and pop the FPU stack
        fstp qword [computedFloatPointer] ; Store it back in float pointer

        leakyReLUDone:
        pop rax ; get back registers
        ret
    HeavySide:
        push rax ; save registers
        xor rax, rax ; prepare it 

        mov byte al, [computedFloatPointer+7] ; get the byte containing the sign bit

        fld1 ; load one 
        fstp qword [computedFloatPointer] ; store one in the pointer

        test al, 0d128 ; test the sign 
        jns HeavySideDone

        mov qword [computedFloatPointer], 0 ; if negative, set pointer to 0

        HeavySideDone:

        pop rax ; get back registers
        ret
    LeakyHeavySide:
        push rax ; save registers
        xor rax, rax ; prepare it 

        mov byte al, [computedFloatPointer+7] ; get the byte containing the sign bit

        fld1 ; load one 
        fstp qword [computedFloatPointer] ; store +1.0 in float pointer

        test al, 0d128 ; test the sign 
        jns LeakyHeavySideDone

        fld qword [LEAKY_RELU_COEFFICIENT] ; if negative, set pointer to the leaky relu coefficient
        fstp qword [computedFloatPointer] ; store back in float pointer

        LeakyHeavySideDone:

        pop rax ; get back registers
        ret
    softMax:
        push rax ; summation sum
        push rbx ; src/dst list length
        push rcx ; src list pointer
        push rdx ; dst list pointer
        xor rax, rax
        
        fldz ; Load 0.0
        fstp qword [computedSoftMaxSum] ; Set the summation as 0.0

        push rbx ; save the initial count
        push rcx ; save the initial src-list-pointer
        push rdx ; save the initial dst-list-pointer

        softMaxSummationLOOP:
            ; Compute the exponential of the element 
            mov qword rax, [rcx]
            mov [computedFloatPointer], rax
            call exponential

            ; Store it in the dst-list, in order to not calculate it again
            mov qword rax, [computedFloatPointer]
            mov qword [rdx], rax

            ; Add it to the summation
            fld qword [computedFloatPointer]
            fld qword [computedSoftMaxSum]
            faddp
            fstp qword [computedSoftMaxSum]

            ; Looping stuff
            add rcx, DOUBLE_PRECISION_SIZE
            add rdx, DOUBLE_PRECISION_SIZE
            dec rbx 
            cmp rbx, 0 
            jnz softMaxSummationLOOP

        pop rdx ; get back initial dst-list-pointer
        pop rcx ; get back initial src-list-pointer
        pop rbx ; get back the initial count

        softMaxDivisionLOOP:
            fld qword [rdx] ; Load e^Xi stored 
            fld qword [computedSoftMaxSum] ; Load Sum(e^Xk)
            fdivp ; Divide as e^Xi / Sum(e^Xk) and pop the FPU stack
            fstp qword [rdx] ; store in the dst-list-pointer 
            
            ; Looping stuff
            add rdx, DOUBLE_PRECISION_SIZE
            dec rbx 
            cmp rbx, 0
            jnz softMaxDivisionLOOP

        pop rdx 
        pop rcx 
        pop rbx 
        pop rax
        ret
    exponential:
        ; Let x a double precision float : returns e^x
        ; Actually computes (2^intX * 2^((X-intX)*log2(e)) - 1 + 1)
        fld qword [computedFloatPointer] ; load in ST0 the float
        fldl2e ; load in ST1 log2(e)
        fmulp ; mul, pop and store in ST0
        fld st0 ; load ST0 in ST1
        frndint ; round ST0
        fxch ; swap
        fsub st0, st1 ; get the part -1< <1 in ST0 and the round part in ST1
        f2xm1 ; compute 2^(ST0) - 1 and store in ST0
        fld1
        faddp ; add 1 to get back 2^(x*log2(e)), pop loaded-1
        fscale ; multiply by 2^ST1 

        fstp qword [computedFloatPointer] ; returns e^x
        ret
    ; -Variable creation and access- ;
    getOffsetFromIndex:
        ; Assuming it's all db float type
        push rdx 
        xor rdx, rdx 
        mov rdx, DOUBLE_PRECISION_SIZE
        mul rdx
        pop rdx
        ret
    getWorkingCnn:
        mov rax, [tensorflopHandles] ; return a pointer to a cnnHandle
        ret
    getLocalWeightsCount:
        ; IN RAX -> a pointer to the Layer handle
        push rbx
        push rcx
        push rdx 

        mov rcx, rax
        xor rax, rax
        xor rbx, rbx
        mov byte bl, [rcx+1] ; store the nth-layer neurons-count in rb8
        add rcx, LAYER_INFORMATIONS_SIZE ; go to layer n + 1
        mov byte al, [rcx+1] ; store the (n+1)th-layer neurons-count in rax
        mul rbx ; get the number of weights by multiplying r8 ($L) and rax ($L+1), -> rax AND rdx (unused)

        pop rdx
        pop rcx 
        pop rbx

        ret
    getWeightsCount:
        xor rcx, rcx ; index of loop
        mov rdi, [CnnLayersCount] ; loop bound 
        dec rdi ; no weights for final layer

        xor rax, rax ; mul result and first argument of mul 
        xor rdx, rdx ; second argument of mul 
        xor rsi, rsi ; summation stored there 

        mov rbx, userInputLayerSizes ; pointer to my neurons count 
        getWeightsCountLoop:
            xor rax, rax
            xor rdx, rdx
            mov byte al, [rbx] ; store in rax (1 low byte) the neuron count of nth layer
            mov byte dl, [rbx+1] ; store in ------------------------------ of n+1th layer
            mul rdx ; multiply and store in rax 

            add rsi, rax ; add result of the multiplication (weights) in rsi 

            ; Looping stuff    
            inc rbx
            inc rcx 
            cmp rcx, rdi
            jne getWeightsCountLoop
        
        mov qword [CnnWeightsCount], rsi ; store the amount of weights in the right variable
        ret 
    getNeuronsCount:
        xor rcx, rcx ; index of loop
        xor rax, rax
        mov rbx, userInputLayerSizes

        getNeuronsCountLoop:
            xor rdx, rdx 
            mov byte dl, [rbx] ; Store in rdx (1 low byte) the layer size (neurons)
            add al, dl ; Add it to rax
            
            ; Looping stuff
            inc rbx
            inc rcx 
            cmp rcx, [CnnLayersCount]
            jne getNeuronsCountLoop
        
        mov qword [CnnNeuronsCount], rax ; Store rax in the right variable
        ret 
    getFirstLayerOffset:
        xor rax, rax
        mov byte al, [userInputLayerSizes] ; store the first layer size in al
        mov qword rbx, DOUBLE_PRECISION_SIZE ; store the data size (db float) in the second operand
        mul rbx ; multiply and store in rax
        mov qword [CnnFirstLayerOffset], rax ; store the first layer offset in the right variable
        ret
    getDataInputAndOutputSizes:
        mov rax, CnnLayersNeurons
        xor rbx, rbx
        mov byte bl, [rax]
        mov byte [CnnDataInputSize], bl

        add rax, [CnnLayersCount]
        dec rax
        xor rbx, rbx
        mov byte bl, [rax]
        mov [CnnDataOutputSize], bl
        ret
    ; -Initialisation of the CNN- ;
    initIndices:
        mov qword [idealsIndex], 0 
        mov qword [lossHistoryIndex], 0
        ret
    initInput:
        mov qword rax, [CnnDataInputSize]
        mov qword rcx, 0
        mov qword r8, [CnnActivatedMatrixPointer]
        mov qword r9, DOUBLE_PRECISION_SIZE
        call insertListAtIndex
        ret
    initLayerSizes:
        mov qword rax, [CnnLayersCount]
        mov qword rbx, userInputLayerSizes
        mov qword rcx, 0
        mov qword r8, CnnLayersNeurons
        mov qword r9, INT_SIZE
        call insertListAtIndex
        ret
    initCNN:
        call initLayerSizes

        call getWeightsCount 
        call getNeuronsCount
        call getFirstLayerOffset
        call getDataInputAndOutputSizes

        call initIndices
        ret
    ; -Display the neural network
    showWeights:
        push rax
        push rcx 
        push rdx 
        push r10 
        push r11 
        push r12
        push r13 
        mov qword r11, [CnnLayersCount]
        dec r11

        call getWorkingCnn
        mov r10, rax
        

        showWeightsLayerLOOP:
            mov qword r12, [r10+3] ; store in r12 the weights' layer pointer

            mov rax, r10 ; pass as argument the layer handle
            call getLocalWeightsCount ; get the amount of weights in rax

            mov r13, rax
            call displayUnity

            showWeightsWeightLOOP:
                mov qword rax, [r12]
                call displayFloat

                add r12, 8
                dec r13 
                cmp r13, 0
                jne showWeightsWeightLOOP

            add r10, LAYER_INFORMATIONS_SIZE

            mov rcx, sepator
            mov rdx, SEPARATOR_LENGTH
            call print
            
            dec r11 
            cmp r11, 0
            jne showWeightsLayerLOOP
    
        pop r13
        pop r12
        pop r11 
        pop r10 
        pop rdx 
        pop rcx
        pop rax 
        ret
    showPseudoMatrix:
        push rax 
        push rbx
        push rcx


        showPseudoMatrixLOOP:
            mov qword rax, [rbx]
            call displayFloat
            
            add rbx, 8
            dec rcx
            cmp rcx, 0
            jnz showPseudoMatrixLOOP
        pop rcx
        pop rbx 
        pop rax
        ret
    
    