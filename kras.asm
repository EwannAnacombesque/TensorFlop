bits 64
%include "tensorFlopCaput.asm"
section .data
    ;===- Verbosity Texts -===;
    ;=========================;

    showWeightsText db "Weights of the CNN :",10
    SHOW_WEIGHTS_TEXT_LENGTH equ $-showWeightsText

    showNeuronsText db "Activated neurons :",10
    SHOW_NEURONS_TEXT_LENGTH equ $-showNeuronsText

    showUnactivatedNeuronsText db "Unactivated neurons :",30
    SHOW_UNACTIVATED_NEURONS_TEXT_LENGTH equ $-showUnactivatedNeuronsText

    showLossHistoryText db "Losses :",10
    SHOW_LOSS_HISTORY_TEXT_LENGTH equ $-showLossHistoryText
    
    showNeuronsCostsText db "Backpropagation neurons' costs :",10
    SHOW_NEURONS_COSTS_TEXT_LENGTH equ $-showNeuronsCostsText
    
    showOutputText db "Last output :",10
    SHOW_OUTPUT_TEXT_LENGTH equ $-showOutputText

    showWeightsGradientText db "Weights gradient :",10
    SHOW_WEIGHTS_GRADIENT_TEXT_LENGTH equ $-showWeightsGradientText

section .bss
    ;===- API Userish Variables -===;
    ;===============================;

    ;# Add layer #;
    krasLayersCount: resq 1
    krasLayersOffset: resq 1

    krasLayersSizes: resq MAX_LAYERS_COUNT 
    krasActivationFunctions: resq MAX_LAYERS_COUNT
    krasBiases: resq MAX_LAYERS_COUNT
    krasBiasesInitialiser: resq MAX_LAYERS_COUNT

    ;# Fit #;
    krasEpochs: resq 1
    krasBatchSize: resq 1

section .text 
;===========================;
;/\- Kras-TensorFlop API -/\;
;===========================;
    krasInitialise:
        call krasPrepare

        call initialiseTensorFlop
        ret
    krasPrepare:
        mov qword [krasLayersCount], 0
        mov qword [krasLayersOffset], 0
        ret
    krasAddLayer:
        mov qword rcx, [krasLayersOffset]

        ; Store the layer size, and go back to offset
        add qword rcx, krasLayersSizes
        mov [rcx], rax
        sub qword rcx, krasLayersSizes

        ; Store the layer activation function, and go back to offset 
        add qword rcx, krasActivationFunctions
        mov [rcx], rbx
        sub qword rcx, krasActivationFunctions

        ; Store if the layer has biases or not, and go back to offset 
        add qword rcx, krasBiases
        mov [rcx], rdx
        sub qword rcx, krasBiases

        ; Store biases initialiser , and go back to offset 
        add qword rcx, krasBiasesInitialiser
        mov [rcx], rdi
        sub qword rcx, krasBiasesInitialiser

        inc qword [krasLayersCount]
        add qword [krasLayersOffset], DOUBLE_SIZE
        ret
    krasCompile:
        call krasInitialiseLayers
        call createCnn
        ret
    krasInitialiseLayers:
        mov qword rax, [krasLayersCount]
        mov qword [CnnLayersCount], rax

        mov qword rax, [CnnLayersCount]
        mov qword rbx, krasLayersSizes
        mov qword rcx, 0
        mov qword r8, CnnLayersSizes
        mov qword r9, DOUBLE_SIZE
        call insertListAtIndex

        mov qword rax, [CnnLayersCount]
        mov qword rbx, krasActivationFunctions
        mov qword rcx, 0
        mov qword r8, CnnActivationFunctions
        mov qword r9, DOUBLE_SIZE
        call insertListAtIndex
        ret
    krasPredict:
        call predictCnn
        ret
    krasFit:
        mov [krasEpochs], rax 
        mov [CnnEpochs], rax 
        call fitCnn
        ret 
;===========================;
;/\- Kras- Dev Verbosity -/\;
;===========================;
    krasShowWeights:
        push rax
        push rcx 
        push rdx 
        push r10 
        push r11 
        push r12
        push r13
        mov rcx, showWeightsText
        mov rdx, SHOW_WEIGHTS_TEXT_LENGTH
        call print

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

            add r10, 19

            call displaySeperator

            dec r11 
            cmp r11, 0
            jne showWeightsLayerLOOP
        
        call displayHugeSeparator

        pop r13
        pop r12
        pop r11 
        pop r10 
        pop rdx 
        pop rcx
        pop rax 
        ret
    krasShowNeurons:
        mov rcx, showNeuronsText
        mov rdx, SHOW_NEURONS_TEXT_LENGTH
        call print

        mov qword rbx, [CnnActivatedMatrixPointer]
        mov qword rcx, [CnnNeuronsCount]
        call showPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowUnactivatedNeurons:
        mov rcx, showUnactivatedNeuronsText
        mov rdx, SHOW_UNACTIVATED_NEURONS_TEXT_LENGTH
        call print

        mov qword rbx, [CnnUnactivatedMatrixPointer]
        mov qword rcx, [CnnNeuronsCount]
        call showPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowNeuronsCosts:
        mov rcx, showNeuronsCostsText
        mov rdx, SHOW_NEURONS_COSTS_TEXT_LENGTH
        call print

        mov qword rbx, [CnnCostMatrixPointer]
        mov qword rcx, [CnnNeuronsCount]
        call showPseudoMatrix

        call displaySeperator

        ret
    krasShowWeightsGradient:
        mov rcx, showWeightsGradientText
        mov rdx, SHOW_WEIGHTS_GRADIENT_TEXT_LENGTH
        call print

        mov qword rbx, [CnnBackpropagationWeightsMatrixPointer]
        mov qword rcx,[CnnWeightsCount]
        call showPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowLossHistory:
        mov rcx, showLossHistoryText
        mov rdx, SHOW_LOSS_HISTORY_TEXT_LENGTH
        call print

        mov qword rbx, lossHistory
        mov qword rcx, [CnnEpochs]
        call showPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowOutput:
        mov rcx, showOutputText
        mov rdx, SHOW_OUTPUT_TEXT_LENGTH
        call print

        mov qword rbx, [CnnActivatedMatrixPointer]
        add qword rbx, [CnnLastLayerOffset]
        mov qword rcx, [CnnDataOutputSize]
        call showPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowUnactivatedOutput:
        mov rcx, showOutputText
        mov rdx, SHOW_OUTPUT_TEXT_LENGTH
        call print

        mov qword rbx, [CnnUnactivatedMatrixPointer]
        add qword rbx, [CnnLastLayerOffset]
        mov qword rcx, [CnnDataOutputSize]
        call showPseudoMatrix

        call displaySeperator
        ret
    krasShowPseudoMatrix:
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
;===========================;
;/\- Kras- Calling Macro -/\;
;===========================;
%macro addLayerParameters 4 
    mov rax, %1
    mov rbx, %2
    mov rdx, %3
    mov rdi, %4
%endmacro
%macro fitParameters 1
    mov rax, %1
%endmacro