bits 64 
%include "tensorFlopKernel.asm"
section .data 
    ;===-  TensorFlop text   -===;
    ;============================;
    presentationText db "==============================",10,"======--- TENSORFLOP ---======",10,"==============================",10,"==== First  ever assembly ====",10,"=== made CNN engine (flop) ===",10,"==============================",10,10
    PRESENTATION_TEXT_LENGTH equ $-presentationText

    ;===-  Test Purpose   -===;
    ;=========================;
    userInputIdealArrays dq 1.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0
    floatTest dq 1.0,0.7, 0.5, 0.0, 0.0, 0.0
    floatTest2 dq 2.0, -3.0, 2.5, 5.0, 4.0, 2.0

section .text
;============================;
;/\-  General TensorFlop  -/\;
;============================;
    initialiseTensorFlop:
        mov rcx, presentationText
        mov rdx, PRESENTATION_TEXT_LENGTH
        call print 
        ret
;=============================;
;/\- TensorFlop Primitives -/\;
;=============================;
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

        ret
    predictCnn:
        ; STORE THE INPUT ;
        call initInput
        
        ; COMPUTE A PROPAGATION
        call computeCnn
        ret
    fitCnn:
        mov qword r11, [CnnEpochs]
        epochLoop:
            push r11 
            call resetWeightGradients

            call batchCnn 
            call applyGradient

            pop r11
            dec r11 
            cmp r11, 0
            jnz epochLoop
        ret
    batchCnn:
        ; LOOP THE AMOUNT OF BATCHSIZE NEEDED ;
        mov qword r11, [CnnBatchSize]
        backpropagationLOOP:
            push r11
            
            call initInput
        
            ; Compute a single propagation
            call computeCnn
            call calculateGlobalLoss
            
            ; Compute the propagation
            call backpropagateCNN

            call shiftInputAndOutput
                        
            pop r11
            dec r11 
            cmp r11, 0
            jnz backpropagationLOOP

        ret
;========================;
;/\-   CNN Creation   -/\;
;========================;
    initCNN: ; Brainstorm
        call getWeightsCount 
        call getNeuronsCount
        call getDataInputAndOutputSizes
        
        call getFirstLayerOffset
        call getLastLayerOffset
        

        call initIndices
        call createBackpropagationMatrix
        ret
    initIndices: ; Set all the indices as 0
        mov qword [idealsIndex], 0 
        mov qword [lossHistoryIndex], 0
        ret
    initInput: ; Load my input (experimentals for now)
        mov rbx, [CnnDataInputSample]

        mov qword rax, [CnnDataInputDim]
        mov qword rcx, 0
        mov qword r8, [CnnActivatedMatrixPointer]
        mov qword r9, DOUBLE_SIZE
        call insertListAtIndex

        mov qword rax, [CnnDataInputDim]
        mov qword rcx, 0
        mov qword r8, [CnnUnactivatedMatrixPointer]
        mov qword r9, DOUBLE_SIZE
        call insertListAtIndex

        ret
    shiftInputAndOutput:
        cmp qword [CnnStaticInput], TF_FALSE
        jne shiftInputAndOutputStatic

        ;mov qword rbx, [CnnDataInputSize]
        ;add qword [CnnDataInputSample], rbx
        ;mov qword rbx, [CnnDataOutputSize]
        ;add qword [CnnDataOutputSample], rbx
        
        shiftInputAndOutputStatic:
        ret 
    createCnnHandle: ; Create a CnnHandle and fill it with the data available
        mov rax, [CnnLayersCount] 
        mov qword [cafElements], rax ; number of layers
        mov qword [cafSize], 19 ; one layer needs 19 bytes
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
            ;mov dl, byte [CnnOBSOLETELayersOBSOLETENeurons+rcx]
            ;mov byte [rax], dl
            inc rax

            ; Store activation function ID, (1 byte)
            mov dl, byte [CnnActivationFunctions+rcx]
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
        ;== Allocate weights ==;

        mov rax, [CnnWeightsCount]
        mov qword [cafElements], rax ; number of weights
        mov qword [cafSize], DOUBLE_SIZE ; a weight is a db float -> 8 bytes 
        call caf
        
        mov qword [CnnWeightsMatrixPointer], rax
        

        ret 
    fillCnnParameters: ; Fill weights and biases with random values
        mov qword r11, [CnnLayersCount]
        dec r11

        call getWorkingCnn
        mov r10, rax
        
        mov qword r12, [CnnWeightsMatrixPointer]
        xor rbx, rbx 
        layerLOOP:
            mov rax, rbx
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

                add r12, DOUBLE_SIZE
                dec r13 
                cmp r13, 0
                jne weightLOOP

            
            add rbx, DOUBLE_SIZE
            dec r11 
            cmp r11, 0
            jne layerLOOP

        ret
    createStandardMatrix: ; Create a memory empty space for my activated, unactivated and loss
        mov qword rdx, [CnnNeuronsCount]
        mov qword [cafElements], rdx ; number of neurons
        mov qword [cafSize], DOUBLE_SIZE ; a neuron is a db float -> 8 bytes 
        call caf
        ret
;===========================;
;/\-   CNN Propagation   -/\;
;===========================;  
    computeCnn: ; Compute a full propagation
        ; rax -> volatile 
        ; rbx -> weight pointer (input)
        ; rcx -> unactivated neuron pointer (output)
        ; rdx -> activated neuron pointer (input)
        ; rdi -> activated neuron pointer (output)
        ; r8 -> layer offset { Let L, L+1 : L*L+1*Double_Size
        ; r10 -> handle index (used to get rbx)
        ; r11 -> the layer count and loop index

        ; Get the informations we need
        ;       - first layer's neuron-count 
        ;       - a pointer to my weights

        mov qword rbx, [CnnWeightsMatrixPointer]; store in rbx the weight pointer


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

        xor r10, r10
        ; For each layer, proceed to the dots products in each layer
        computeCnnLOOP:
            ; Store the input layer size
            mov rax, [CnnLayersSizes+r10]
            mov qword [computeInputLayerSize], rax

            ; Store the output layer size
            add r10, DOUBLE_SIZE
            mov rax, [CnnLayersSizes+r10]
            mov qword [computeOutputLayerSize], rax
            mov qword [computeUnchangedOutputLayerSize], rax

            ; Store the output layer activation function 
            mov rax, [CnnActivationFunctions+r10]
            mov qword [computeActivationFunction], rax
 
            ; Eventually compute the layer
            call computeSingleLayer

            ; Looping stuff
            dec r11
            cmp r11, 0
            jnz computeCnnLOOP
            

        ret 
    computeSingleLayer: ; Compute the propagation in a single layer
        push rax

        ; Calculate the offset of each set of weights 
        push rdx
        mov rax, [computeInputLayerSize]
        mov r9, DOUBLE_SIZE
        mul r9
        mov r8, rax
        pop rdx
        
        mov [computeTempUnactivatedPointer], rcx

        computeSingleLayerLOOP:
            ; Compute the dot product
            call computeDotProduct
            mov rax, [computeOutputNeuronPointer]

            mov [rcx], rax ; Save in unactivated (Zs) matrix 
            
            add rcx, DOUBLE_SIZE

            add rbx, r8 ; add to my weights pointer, the input-layer's neuron-count times the size of an element

            dec qword [computeOutputLayerSize]
            cmp qword [computeOutputLayerSize],0
            jnz computeSingleLayerLOOP

        call computeBiases
        call computeScales
        call computeActivation

        ; Add to my neurons pointer, the input-layer's neuron-count times the size of an element
        add rdx, r8 
        
        pop rax
        ret 

    computeDotProduct: ; Dot product of two pointer, vector-size specified in rbx
        push rbx
        push rdx


        mov qword rax, [computeInputLayerSize]
        mov qword [computeOutputNeuronPointer],0 ; reset the output neuron 
        push rax 
        computeDotProductLOOP:
            
            pxor xmm0, xmm0 ; reset upper part of the xmm0 in case it's a single dot product
            pxor xmm1, xmm1 ; reset upper part of the xmm1 in case it's a single dot product
            movq xmm0, [rbx] ; store first weight  
            movq xmm1, [rdx] ; store first input 

            cmp byte [computeInputLayerSize], 2 
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

        pop rax
        mov qword [computeInputLayerSize], rax
        pop rdx
        pop rbx
        ret
    computeActivation: ; Activate my neurons
        push rbx 
        push rcx
        push rdx 
        
        cmp qword [computeActivationFunction], TF_SOFTMAX
        je wholeLayerActivation

        ;======================================================;
        ;=============== Per Neuron Activation ================:
        computeActivationLoop:
            ; Call the activation function
            mov rdx, [computeTempUnactivatedPointer]
            mov rax, [rdx]
            mov [computedFloatPointer], rax
            call selectActivationFunction
            mov qword rax, [computedFloatPointer]

            mov [rdi], rax ; Save in activated (As) matrix

            ; Update my pointers
            add rdi, DOUBLE_SIZE
            add qword [computeTempUnactivatedPointer], DOUBLE_SIZE

            cmp [computeTempUnactivatedPointer], rcx
            jne computeActivationLoop 
        
        jmp computeActivationDone

        wholeLayerActivation:
        ;======================================================;
        ;================ Per Layer Activation ================:

        mov rcx, [computeTempUnactivatedPointer]
        mov rdx, rdi
        mov rbx, [computeUnchangedOutputLayerSize]
        call softMax

        mov rdx, [computeUnchangedOutputLayerSize]
        mov rax, DOUBLE_SIZE
        mul rdx 
        add rdi, rax

        computeActivationDone:
        pop rdx
        pop rcx 
        pop rbx
        ret
    computeBiases:
        ret 
    computeScales:
        ret 
;===========================;
;/\- CNN Backpropagation -/\;
;===========================;
    createBackpropagationMatrix:
        mov qword rdx, [CnnWeightsCount]
        mov qword [cafElements], rdx ; number of weights
        mov qword [cafSize], DOUBLE_SIZE ; a weight is a db float -> 8 bytes 
        call caf
        mov qword [CnnBackpropagationWeightsMatrixPointer], rax
        ret
    backpropagateFirstLayer:
        ;Proceed to calculate all sigmas 
        ; sigma(i) = Act^-1(Zi)*2(Ai-I)
        ; and so (0.1+Ai)*(1.1-Ai)*2*(Ai-Ii)

        mov rax, [CnnLayersCount]
        dec rax 
        mov rdx, DOUBLE_SIZE 
        mul rdx 
        mov rdx, [CnnActivationFunctions+rax]
        mov qword [computeActivationFunction], rdx

        xor rdx, rdx ; loop index 
        mov qword rdi, [CnnCostMatrixPointer]
        mov rbx, [CnnDataOutputSample] 
        mov qword rax, [CnnLastLayerOffset]
        add qword rax, [CnnActivatedMatrixPointer] ; go to last layer
        mov qword rcx, [CnnLastLayerOffset]
        

        xor rdx, rdx 
        backpropagateFirstLayerSigmasLOOP:            
            fcomp st0
            ; Ai-Ii
            fld qword [rax]
            fld qword [rbx]
            fsubp

            ; 2(Ai-Ii)
            fld st0 
            faddp
            fstp qword [backpropTempFloatPointer]

            call backpropagateActivation

            ; Ai*(1-Ai) * 2*(Ai-Ii)
            fld qword [computedFloatPointer]
            fld qword [backpropTempFloatPointer]
            fmulp
            fstp qword [rdi]

            ; LOOP STUFF
            add rdi, DOUBLE_SIZE
            add rbx, DOUBLE_SIZE
            add rax, DOUBLE_SIZE
            add rcx, DOUBLE_SIZE
            inc rdx
            cmp rdx, [CnnDataOutputDim]
            jne backpropagateFirstLayerSigmasLOOP
        

        mov rdi, [CnnCostMatrixPointer] ; get dC/dAi

        mov rsi, [CnnDataOutputDim] ; get dC/dAi
        mov rax, DOUBLE_SIZE
        mul rsi 
        mov rsi, rax 
        add rsi, [CnnCostMatrixPointer]
        
        mov rax, -2
        call getNeuronsLayerByteOffset
        mov r8, rax
        add r8, [CnnActivatedMatrixPointer]

        mov rax, -2
        call getLayerSize
        mov rdx, DOUBLE_SIZE
        mul rdx
        mov qword [backpropLayerBytesSize], rax

        mov rax, -1 
        call getWeightsLayerByteOffset
        mov rdx, rax
        add rdx, [CnnWeightsMatrixPointer]
        
        mov rax, [CnnBackpropagationWeightsMatrixPointer]

        xor rbx, rbx ; neuron-local index
        xor rcx, rcx ; layer-local index
        backpropagateFirstLayerNeuronsLOOP:
            fcomp st0
            
            fld qword [r8+rbx]
            fld qword [rdi]
            fmulp
            fld qword [rax] ;
            faddp ;
            fstp qword [rax]

            fld qword [rdx]
            fld qword [rdi]
            fmulp
            fld qword [rsi+rbx]
            faddp
            fstp qword [rsi+rbx]

            add rax, DOUBLE_SIZE
            add rdx, DOUBLE_SIZE
            add rbx, DOUBLE_SIZE 
            cmp rbx, [backpropLayerBytesSize] 
            jne backpropagateFirstLayerNeuronsLOOP

        ; Layer-locality 

            xor rbx, rbx
            add rdi, DOUBLE_SIZE
            
            inc rcx 
            cmp rcx, [CnnDataOutputDim]
            jne backpropagateFirstLayerNeuronsLOOP
            
        ret
    backpropagateSingleLayerPrepare:
        ; Get the layer activation functon
        mov qword rax, [backpropLayerIndex]
        mov rdx, DOUBLE_SIZE
        mul rdx
        add rax, CnnActivationFunctions
        mov r8, [rax]
        mov [computeActivationFunction], r8

        ; Get the layer size 
        mov rdx, qword [backpropLayerIndex]
        mov rax, DOUBLE_SIZE
        mul rdx
        mov rbx, rax
        mov rax, [CnnLayersSizes+rbx]
        ;Actual offset
        mov qword [backpropLayerSize], rax
        
        ; Get the layer offset (unactivated)
        mov rax, qword [backpropLayerIndex]
        call getNeuronsLayerByteOffset
        mov qword [backpropNeuronsLayerByteOffset], rax
        
        ; Get the reverse layer offset (cost)
        mov rax, qword [backpropLayerIndex]
        call getReverseNeuronsLayerByteOffset
        mov qword [backpropReverseNeuronsLayerByteOffset], rax
        
        ; Get the reverse former layer offset (cost)
        mov rax, qword [backpropFormerLayerIndex]
        call getReverseNeuronsLayerByteOffset
        mov qword [backpropReverseNeuronsFormerLayerByteOffset], rax

        ; Get the former layer size in bytes
        mov rdx, qword [backpropFormerLayerIndex]
        mov rax, DOUBLE_SIZE
        mul rdx
        mov rbx, rax
        mov byte al, [CnnLayersSizes+rax] 
        ;Actual offset
        mov rdx, DOUBLE_SIZE
        mul rdx 
        mov qword [backpropFormerLayerBytesSize], rax 
        ret
    backpropagateSingleLayerInitialise:


        ; Get the former layer offset -> Ai-1,k'
        mov qword rax, [backpropFormerLayerIndex]
        call getNeuronsLayerByteOffset
        add rax, [CnnActivatedMatrixPointer]
        mov r8, rax 

        ; Get the former weights offset -> Wi-1,k
        mov rax, [backpropFormerLayerIndex]
        call getWeightsLayerByteOffset
        add rax, [CnnWeightsMatrixPointer]
        mov rdx, rax 

        ; Get the former layer costs offset -> store dC/Ai-1,k'
        mov rdi, [CnnCostMatrixPointer]
        add rdi, [backpropReverseNeuronsLayerByteOffset]

        ; Get the former layer costs offset -> store dC/Ai-1,k'
        mov rsi, [CnnCostMatrixPointer]
        add rsi, [backpropReverseNeuronsFormerLayerByteOffset]

        ; Get the weight gradient offset 
        mov rax, [backpropLayerIndex]
        call getReverseWeightsLayerByteOffset
        add rax, [CnnBackpropagationWeightsMatrixPointer]

        xor rbx, rbx ; neuron-local index
        xor rcx, rcx ; layer-local index
        ret
    backpropagateSingleLayerSigmas:
        mov rbx, [CnnCostMatrixPointer] ; Get the {dC/dAi | H(Zi)*dC/dAi} matrix 
        add rbx, [backpropReverseNeuronsLayerByteOffset] ; Go to the right layer

        
        mov rcx, [backpropNeuronsLayerByteOffset] ; Go to the right layer

        xor rax, rax ; loop index -> stop at layer size
        backpropagateSingleLayerSigmasLOOP:
            fcomp st0 ; Make sure FPU stack is cleared

            ; Inverse activation function -> ReLU : Heaviside 
            ;                             -> LeakyReLU : LeakyHeaviside
            call backpropagateActivation

            ; Multiply by dC/dAi to get sigma H(Zi)*dC/dAi
            fld qword [computedFloatPointer]
            fld qword [rbx]
            fmulp 
            
            ; Store the sigma
            fstp qword [rbx]
            
            ; looping stuff 
            add rbx, DOUBLE_SIZE
            add rcx, DOUBLE_SIZE

            inc rax 
            cmp rax, [backpropLayerSize]
            jne backpropagateSingleLayerSigmasLOOP
        ret
    backpropagateSingleLayer:
        

        ; Get the layer and former layer sizes
        call backpropagateSingleLayerPrepare
        
        ; Transform layer dC/dAi into H(Zi)*dC/dAi
        call backpropagateSingleLayerSigmas

        ; Store all pointers in needed registers
        call backpropagateSingleLayerInitialise


        
        backpropagateSingleLayerNeuronsLOOP:
            fcomp st0
            fld qword [r8+rbx]
            fld qword [rdi]
            fmulp
            fld qword [rax] ;
            faddp ;
            fstp qword [rax]

            fld qword [rdx]
            fld qword [rdi]
            fmulp
            fld qword [rsi+rbx]
            faddp 
            fstp qword [rsi+rbx]

            add rax, DOUBLE_SIZE
            add rdx, DOUBLE_SIZE
            add rbx, DOUBLE_SIZE 
            cmp rbx, [backpropFormerLayerBytesSize] 
            jne backpropagateSingleLayerNeuronsLOOP

        ; Layer-locality 
            xor rbx, rbx
            add rdi, DOUBLE_SIZE 

            inc rcx 
            cmp rcx, [backpropLayerSize]
            jne backpropagateSingleLayerNeuronsLOOP
        ret
    backpropagateCNN:
        ; Reset my neurons gradient 
        call resetNeuronsCosts

        ; Get the first layer to back propagate
        mov qword rax, [CnnLayersCount]
        mov qword [backpropLayerIndex], rax
        sub qword [backpropLayerIndex], 2
        mov qword [backpropFormerLayerIndex], rax
        sub qword [backpropFormerLayerIndex],3
        
        ; Backpropagate first layer 
        call backpropagateFirstLayer

        
        ; Backpropagate other layers
        backpropagateCNNLOOP:
            call backpropagateSingleLayer

            dec qword [backpropLayerIndex]
            dec qword [backpropFormerLayerIndex]

            cmp qword [backpropLayerIndex],0
            jnz backpropagateCNNLOOP

        ret
    backpropagateActivation:
        push rcx
        push rdx
        cmp qword [computeActivationFunction], TF_SOFTMAX
        je backpropagateActivationUseActivated

        ;=======================================================================;
        ;=============== Derivative Activation With Unactivated ================:
        add rcx, [CnnUnactivatedMatrixPointer] ; Get the Zi matrix

        mov rdx, [rcx]
        mov [computedFloatPointer], rdx
        call selectDerivativeActivationFunction

        jmp backpropagateActivationDone 

        backpropagateActivationUseActivated:

        ;=====================================================================;
        ;=============== Derivative Activation With Activated ================:
        
        add rcx, [CnnActivatedMatrixPointer]

        mov rdx, [rcx]
        mov [computedFloatPointer], rdx
        call DerivativeSoftmax


        backpropagateActivationDone:
        pop rdx 
        pop rcx
        ret
    resetNeuronsCosts:
        mov rax, [CnnNeuronsCount]
        mov rbx, [CnnCostMatrixPointer]
        resetNeuronsCostsLOOP:
            mov qword [rbx], 0
            add rbx, DOUBLE_SIZE
            dec rax 
            cmp rax, 0
            jnz resetNeuronsCostsLOOP
        ret
    resetWeightGradients:
        mov rax, [CnnWeightsCount]
        mov rbx, [CnnBackpropagationWeightsMatrixPointer]
        resetWeightGradientsLOOP:
            mov qword [rbx], 0
            add rbx, DOUBLE_SIZE
            dec rax 
            cmp rax, 0
            jnz resetWeightGradientsLOOP
        ret
    applyGradient:
        mov rcx, [CnnBackpropagationWeightsMatrixPointer]; Weights gradient pointer 
        mov qword rsi, [CnnLayersCount]
        dec rsi 
        xor rbx, rbx
        applyGradientLayerLOOP:
            mov rdi, [CnnWeightsMatrixPointer] ; Weights pointer
            
            mov qword rax, -1
            sub rax, rbx ; ask for layer -(rbx+1)
            call getWeightsLayerByteOffset
            add rdi, rax
            
            ; Get layer index and offset
            mov qword r8, [CnnLayersCount]
            sub r8, rbx
            mov rax, DOUBLE_SIZE
            mul r8
            mov r8, rax
            ; Get layer size
            sub r8, DOUBLE_SIZE
            mov rdx, [CnnLayersSizes+r8]
            ; Get former layer size
            sub r8, DOUBLE_SIZE
            mov rax, [CnnLayersSizes+r8]
            mul rdx 

            applyGradientWeightsLOOP:
                fcomp st0 

                ; dC/dW * learning rate
                fld qword [CnnLearningRate]
                fld qword [rcx]
                fmulp
                ; W = W-dC/dW
                fld qword [rdi]
                faddp
                fstp qword [rdi]
                

                add rdi, DOUBLE_SIZE ; jump to next weight 
                add rcx, DOUBLE_SIZE ; jump to next weight gradient
                dec rax 
                jnz applyGradientWeightsLOOP

            inc rbx
            cmp rbx, rsi
            jne applyGradientLayerLOOP
        ret
    calculateGlobalLoss:
        
        mov rcx, [CnnDataOutputDim] ; Last layer size 
        mov r8, 0

        mov rax, [CnnNeuronsCount] ; Amount of Neurons 
        sub rax, rcx ; Get the index of last layer

        call getOffsetFromIndex ; get the offset in bytes

        add rax, [CnnActivatedMatrixPointer] ; go to last layer
        
        xor rcx, rcx
        cmp qword [lossHistoryIndex], MAX_SIZE_LOSS_HISTORY
        jge lossHistoryOverflow

        mov rcx, [lossHistoryIndex]

        lossHistoryOverflow:
        
        fldz
        fstp qword [lossHistory+rcx] ; set my loss as 0

        xor rdx, rdx 
        xor r8, r8
        mov rbx, [CnnDataOutputSample]
        calculateGlobalLossLOOP:
            fcomp st0 

            fld qword [rax+rdx]
            fld qword [rbx+rdx]
            fsubp 
            fld st0
            fmulp

            fld qword [lossHistory+rcx]
            faddp
            fstp qword [lossHistory+rcx]

            add rdx, DOUBLE_SIZE
            inc r8 
            cmp r8, [CnnDataOutputDim]
            jne calculateGlobalLossLOOP
        
        add qword [lossHistoryIndex], DOUBLE_SIZE
        mov rax, [lossHistory+rcx]
        ret 