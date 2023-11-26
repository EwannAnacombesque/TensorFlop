bits 64 
%include "utility.asm"
section .data 
    ;===- TensorFlop Constants -===;
    ;==============================;
    
    ;# Limitations #
    MAX_LAYERS_COUNT equ 10
    MAX_NETWORKS_COUNT equ 2
    MAX_INDEX_LOSS_HISTORY equ 500
    MAX_SIZE_LOSS_HISTORY equ 4000
    MAX_IDEALS_COUNT equ 10

    ;# Initial constants #
    INITIAL_LEAKY_RELU_COEFFICIENT dq 0.1
    INITIAL_LEARNING_RATE dq -0.001
        
    ;# Constants #
    TF_ZERO equ 0
    TF_NO_BIASES equ 1
    TF_USE_BIASES equ 2
    TF_RANDOM equ 3
    TF_LOAD_DATA equ 4
    TF_LINEAR equ 5
    TF_RELU equ 6
    TF_LEAKY_RELU equ 7
    TF_SIGMOID equ 8
    TF_TANH equ 9
    TF_SOFTMAX equ 10
    TF_NORMALIZE equ 11
    TF_CLASSIFICATION_RADIUS equ 12 
    TF_CLASSIFICATION_CHESSBOARD equ 13
    TF_CLASSIFICATION_HALF equ 14

section .bss 
    ;===- TENSORFLOP VARIABLES -===;
    ;==============================;

    ; TENSORFLOP HANDLING   

    tensorflopHandles: resq MAX_NETWORKS_COUNT ; Pointer to my CNN-Handles
    tensorflopHandleCreated : resb 1 ; Amount of CNN and-so Cnn-Handles created

    ; CNN-global
    CnnLayersCount: resq 1 ; The number of layer of the current CNN
    CnnWeightsCount: resq 1 ; The number of weights of the current CNN
    CnnNeuronsCount: resq 1 ; The number of neurons of the current CNN

    CnnDataInputSize: resq 1 ; Size (vector size) of what enters in CNN 
    CnnDataOutputSize: resq 1 ; Size (vector size) of what comes out of CNN 

    CnnActivatedMatrixPointer: resq 1 ; A pseudo matrix of all my activated layer's neurons | Size DOUBLE_SIZE * CnnNeuronsCount
    CnnUnactivatedMatrixPointer: resq 1 ; A pseudo matrix of all my unactivated layer's neurons | Size DOUBLE_SIZE * CnnNeuronsCount
    CnnWeightsMatrixPointer: resq 1 ; A pseudo matrix of all my weights 

    CnnCostMatrixPointer: resq 1 ; 
    CnnBackpropagationWeightsMatrixPointer: resq 1;
    CnnBackpropagationBiasesMatrixPointer: resq 1;
    
    CnnActivationFunctions : resq MAX_LAYERS_COUNT ; Array of activation functions of layers created 
    CnnLayersSizes : resq MAX_LAYERS_COUNT ; Array of sizes of my layers (in neurons)

    CnnFirstLayerOffset : resq 1 ; Offset of Neurons needed to propagate (skipping input neurons)
    CnnLastLayerOffset : resq 1 ; Offset of Neurons needed to backpropagate (skipping output neurons)
    
    CnnLearningRate: resq 1
    CnnEpochs : resq 1
    CnnBatchSize : resq 1

    ; CNN-Computation 

    computeInputLayerSize: resq 1 ; Size of the input layer
    computeOutputLayerSize: resq 1 ; Size of the output layer
    computeUnchangedOutputLayerSize: resq 1
    computeActivationFunction: resq 1 ; Activation function of the layer
    computeTempUnactivatedPointer: resq 1
    computeOutputNeuronPointer: resq 1 ; Pointer to the neuron, used in dot product to store the result
    computeTempOutputPointer: resq 1 ; POint to the result of a single multiplication in a dot product, temporary

    computedFloatPointer: resq 1 ; Commonly used for activation function, as the result and the parameter of these
    computedSoftMaxSum: resq 1 ; Summation of e^Xi of an array X, used in softMax

    ; CNN-Backpropagation 

    backproptemp: resq 1
    backpropTempFloatPointer: resq 1
    backpropTempFloatPointerTwo: resq 1

    backpropLayerIndex: resq 1
    backpropFormerLayerIndex: resq 1
    backpropLayerSize: resq 1
    backpropLayerBytesSize: resq 1
    backpropFormerLayerSize: resq 1
    backpropFormerLayerBytesSize: resq 1

    backpropNeuronsLayerByteOffset: resq 1
    backpropReverseNeuronsLayerByteOffset: resq 1
    backpropReverseNeuronsFormerLayerByteOffset: resq 1

    backpropInputLayerNeuronsPointer: resq 1 
    backpropInputLayerWeightsPointer: resq 1

    ; CNN-Loss 
    
    lossHistory: resq MAX_INDEX_LOSS_HISTORY
    lossHistoryIndex: resq 1

    ; CNN-Ideals / Inputs / Samples 

    trainSample: resq 1
    validationSample: resq 1
    idealsArray: resq MAX_IDEALS_COUNT
    idealsIndex: resq 1
section .text 
;============================;
;/\- Activation functions -/\;
;============================;
    selectActivationFunction:
        ; Proceed to check if it's linear
        cmp qword [computeActivationFunction], TF_LINEAR
        jne selectActivationFunctionNotLinear

        call Linear ; call the right function

        jmp selectActivationDone
        selectActivationFunctionNotLinear:

        ;============================================;

        ; Proceed to check if it's ReLU
        cmp qword [computeActivationFunction], TF_RELU
        jne selectActivationFunctionNotReLU

        call ReLU ; call the right function

        jmp selectActivationDone
        selectActivationFunctionNotReLU:

        ;============================================;
        
        ; Proceed to check if it's LeakyReLU
        cmp qword [computeActivationFunction], TF_LEAKY_RELU
        jne selectActivationFunctionNotLeakyReLU

        call LeakyReLU

        jmp selectActivationDone
        selectActivationFunctionNotLeakyReLU:

        ;============================================;
        
        ; Proceed to check if it's Sigmoid

        cmp qword [computeActivationFunction], TF_SIGMOID
        jne selectActivationFunctionNotSigmoid

        call Sigmoid

        jmp selectActivationDone
        selectActivationFunctionNotSigmoid:

        ;============================================;
        
        ; Proceed to check if it's Tanh
        
        cmp qword [computeActivationFunction], TF_TANH
        jne selectActivationFunctionNotTanh

        call Tanh

        jmp selectActivationDone
        selectActivationFunctionNotTanh:

        selectActivationDone:
        ret
    Linear:
        ret
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
    LeakyReLU:
        push rax ; save registers
        xor rax, rax ; prepare it 

        mov byte al, [computedFloatPointer+7] ; get the byte containing the sign bit 
        test al, 0d128 ; test the sign 
        jns LeakyReLUDone

        ; If the float is negative :

        fld qword [computedFloatPointer] ; Load the float
        fld qword [INITIAL_LEAKY_RELU_COEFFICIENT] ; Load the coefficient of the leeaky relu
        fmulp ; Multiply both and pop the FPU stack
        fstp qword [computedFloatPointer] ; Store it back in float pointer

        LeakyReLUDone:
        pop rax ; get back registers
        ret
    Sigmoid:
        ret 
    Tanh:
        ret
    selectDerivativeActivationFunction:
        ; Proceed to check if it's linear
        cmp qword [computeActivationFunction], TF_LINEAR
        jne selectDerivativeActivationFunctionNotLinear

        call ConstantFunction ; call the right function

        jmp selectDerivativeActivationDone
        selectDerivativeActivationFunctionNotLinear:

        ;============================================;

        ; Proceed to check if it's ReLU
        cmp qword [computeActivationFunction], TF_RELU
        jne selectDerivativeActivationFunctionNotReLU

        call Heaviside ; call the right function

        jmp selectDerivativeActivationDone
        selectDerivativeActivationFunctionNotReLU:

        ;============================================;
        
        ; Proceed to check if it's LeakyReLU
        cmp qword [computeActivationFunction], TF_LEAKY_RELU
        jne selectDerivativeActivationFunctionNotLeakyReLU

        call LeakyHeaviside

        jmp selectDerivativeActivationDone
        selectDerivativeActivationFunctionNotLeakyReLU:

        ;============================================;
        
        ; Proceed to check if it's Sigmoid

        cmp qword [computeActivationFunction], TF_SIGMOID
        jne selectDerivativeActivationFunctionNotSigmoid

        call DerivativeSigmoid

        jmp selectDerivativeActivationDone
        selectDerivativeActivationFunctionNotSigmoid:

        ;============================================;
        
        ; Proceed to check if it's Tanh
        
        cmp qword [computeActivationFunction], TF_TANH
        jne selectDerivativeActivationFunctionNotTanh

        call DerivativeTanh

        jmp selectDerivativeActivationDone
        selectDerivativeActivationFunctionNotTanh:

        selectDerivativeActivationDone:
        ret
    ConstantFunction:
        ret
    Heaviside:
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
    LeakyHeaviside:
        push rax ; save registers
        xor rax, rax ; prepare it 

        mov byte al, [computedFloatPointer+7] ; get the byte containing the sign bit

        fld1 ; load one 
        fstp qword [computedFloatPointer] ; store +1.0 in float pointer

        test al, 0d128 ; test the sign 
        jns LeakyHeavySideDone


        fld qword [INITIAL_LEAKY_RELU_COEFFICIENT] ; if negative, set pointer to the leaky relu coefficient
        fstp qword [computedFloatPointer] ; store back in float pointer

        LeakyHeavySideDone:

        pop rax ; get back registers
        ret
    DerivativeSigmoid:
        ret
    DerivativeTanh:
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
            add rcx, DOUBLE_SIZE
            add rdx, DOUBLE_SIZE
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
            add rdx, DOUBLE_SIZE
            dec rbx 
            cmp rbx, 0
            jnz softMaxDivisionLOOP

        pop rdx 
        pop rcx 
        pop rbx 
        pop rax
        ret
    DerivativeSoftmax:
        ; 1.1-Ai
        fld1
        fld qword [tweakedSoftMaxRoot]
        faddp
        fld qword [computedFloatPointer]
        fsubp
        fstp qword [backpropTempFloatPointerTwo]

        ; 0.1 + Ai

        fld qword [tweakedSoftMaxRoot]
        fld qword [computedFloatPointer]
        faddp 

        ; Ai*(1-Ai)
        fld qword [backpropTempFloatPointerTwo]
        fmulp
        fstp qword [computedFloatPointer]
        ret
    
    exponential:
        fcomp st0
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
;==========================;
;/\- Utility procedures -/\;
;==========================;
    getOffsetFromIndex:
        ; Assuming it's all db float type
        push rdx 
        xor rdx, rdx 
        mov rdx, DOUBLE_SIZE
        mul rdx
        pop rdx
        ret
    getWorkingCnn:
        mov rax, [tensorflopHandles] ; return a pointer to a cnnHandle
        ret
    getLocalWeightsCount:
        ; IN RAX -> a layer index (offset * 8)
        push rbx
        push rcx
        push rdx 

        mov rcx, CnnLayersSizes
        add rcx, rax
        mov rbx, [rcx] ; store the nth-layer neurons-count in rb8
        mov rax, [rcx+DOUBLE_SIZE] ; store the (n+1)th-layer neurons-count in rax
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

        mov rbx, CnnLayersSizes ; pointer to my neurons count 
        getWeightsCountLoop:
            xor rax, rax
            xor rdx, rdx
            mov rax, [rbx] ; store in rax (1 low byte) the neuron count of nth layer
            mov rdx, [rbx+8] ; store in ------------------------------ of n+1th layer
            mul rdx ; multiply and store in rax 

            add rsi, rax ; add result of the multiplication (weights) in rsi 

            ; Looping stuff    
            add rbx, DOUBLE_SIZE
            inc rcx 
            cmp rcx, rdi
            jne getWeightsCountLoop
        
        mov qword [CnnWeightsCount], rsi ; store the amount of weights in the right variable
        ret 
    getNeuronsCount:
        xor rcx, rcx ; index of loop
        xor rax, rax
        mov rbx, CnnLayersSizes

        getNeuronsCountLoop:
            xor rdx, rdx 
            mov rdx, [rbx] ; Store in rdx (1 low byte) the layer size (neurons)
            add rax, rdx ; Add it to rax
            
            ; Looping stuff
            add rbx, DOUBLE_SIZE
            inc rcx 
            cmp rcx, [CnnLayersCount]
            jne getNeuronsCountLoop
        
        mov qword [CnnNeuronsCount], rax ; Store rax in the right variable
        ret
    getLayerSize:
        push rbx

        cmp rax, 0
        jge getLayerSizePositive

        add rax, [CnnLayersCount]

        getLayerSizePositive:
        mov rbx, DOUBLE_SIZE
        mul rbx 
        mov rbx, rax 
        add rbx, CnnLayersSizes

        mov rax, [rbx]

        pop rbx
        ret 
    getFirstLayerOffset:
        mov rax, [CnnLayersSizes] ; store the first layer size in al
        mov qword rbx, DOUBLE_SIZE ; store the data size (db float) in the second operand
        mul rbx ; multiply and store in rax
        mov qword [CnnFirstLayerOffset], rax ; store the first layer offset in the right variable
        ret
    getLastLayerOffset:
        mov rax, [CnnNeuronsCount] ; Amount of Neurons 
        sub rax, [CnnDataOutputSize] ; Get the index of last layer
        call getOffsetFromIndex ; get the offset in bytes
        mov qword [CnnLastLayerOffset], rax
        ret
    getDataInputAndOutputSizes:
        mov rax, [CnnLayersSizes]
        mov [CnnDataInputSize], rax

        mov rax, [CnnLayersCount]
        dec rax
        mov rdx, DOUBLE_SIZE 
        mul rdx 
        add rax, CnnLayersSizes
        mov rbx, [rax]
        mov [CnnDataOutputSize], rbx
        ret
    getNeuronsLayerByteOffset:
        push rdx

        ; Case layer -n is asked
        cmp rax, 0
        jge getNeuronsLayerByteOffsetPositive

        ; If the layer is negative, just add the layer count
        add rax, [CnnLayersCount]

        getNeuronsLayerByteOffsetPositive: 
        dec rax

        ; Get the real layer offset in bytes (x8)
        mov rdx, DOUBLE_SIZE
        mul rdx 

        
        xor rdx, rdx ; store here the neurons count
        getNeuronsLayerByteOffsetLOOP:
            add rdx, [CnnLayersSizes+rax] ; add the layer's neurons
            sub rax, DOUBLE_SIZE
            cmp rax, 0
            jge getNeuronsLayerByteOffsetLOOP

        mov rax, DOUBLE_SIZE
        mul rdx 
        pop rdx
        ret
    getReverseNeuronsLayerByteOffset:
        push rdx
        push rdi

        ; Case layer -n is asked
        cmp rax, 0
        jge getReverseNeuronsLayerByteOffsetPositive

        ; If the layer is negative, just add the layer count
        add rax, [CnnLayersCount]

        getReverseNeuronsLayerByteOffsetPositive: 
        inc rax

        ; Get the real layer offset in bytes (x8)
        push rax 
        mov rax, DOUBLE_SIZE
        mov rdi, [CnnLayersCount]
        mul rdi 
        mov rdi ,rax

        pop rax

        mov rdx, DOUBLE_SIZE
        mul rdx 
        
        xor rdx, rdx ; store here the neurons count
        getReverseNeuronsLayerByteOffsetLOOP:
            add rdx, [CnnLayersSizes+rax] ; add the layer's neurons
            add rax, DOUBLE_SIZE
            cmp rax,  rdi
            jl getReverseNeuronsLayerByteOffsetLOOP
        
        mov rax, rdx
        mov rax, DOUBLE_SIZE
        mul rdx


        pop rdi
        pop rdx
        ret
    getWeightsLayerByteOffset:
        push rbx 
        push rcx
        push rdx

        ; Case layer -n is asked
        cmp rax, 0
        jge getWeightsLayerByteOffsetPositive

        ; If the layer is negative, just add the layer count
        add rax, [CnnLayersCount]
        dec rax

        getWeightsLayerByteOffsetPositive:

        ; Get the real layer offset in bytes (x8)
        mov rdx, DOUBLE_SIZE
        mul rdx
        mov rcx, rax
        
        xor rbx, rbx        
        getWeightsLayerByteOffsetLOOP:
            xor rdx, rdx 
            xor rax, rax 
            
            mov rax, [CnnLayersSizes+rcx] ; move the layer's neurons
            sub rcx, DOUBLE_SIZE 
            mov rdx, [CnnLayersSizes+rcx] ; move the precedent layer's neurons
            mul rdx

            add rbx, rax 

            cmp rcx, 0
            jg getWeightsLayerByteOffsetLOOP
        
        mov rdx, rbx
        mov rax, DOUBLE_SIZE
        mul rdx

        pop rdx
        pop rcx 
        pop rbx
        ret
    getReverseWeightsLayerByteOffset:
        push rdx 
        push rax 
        mov qword rdx, [CnnWeightsCount]
        mov rax, DOUBLE_SIZE
        mul rdx
        mov rdx, rax
        pop rax

        call getWeightsLayerByteOffset
        sub rdx, rax
        mov rax, rdx
        pop rdx
        ret
    