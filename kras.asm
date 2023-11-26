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
    krasLearningRate: resq 1
    krasEpochs: resq 1
    krasBatchSize: resq 1

    ;# Export #;

    krasExportFileName: resq 1

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

        mov qword [krasLearningRate], rbx
        mov qword [CnnLearningRate], rbx
        call fitCnn
        ret 
;===========================;
;/\- Kras- Dev Verbosity -/\;
;===========================;
    krasShowWeights:
        mov qword rbx, [CnnWeightsMatrixPointer]
        mov qword rcx,[CnnWeightsCount]
        call krasShowPseudoMatrix
        ret
    krasShowNeurons:
        mov rcx, showNeuronsText
        mov rdx, SHOW_NEURONS_TEXT_LENGTH
        call print

        mov qword rbx, [CnnActivatedMatrixPointer]
        mov qword rcx, [CnnNeuronsCount]
        call krasShowPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowUnactivatedNeurons:
        mov rcx, showUnactivatedNeuronsText
        mov rdx, SHOW_UNACTIVATED_NEURONS_TEXT_LENGTH
        call print

        mov qword rbx, [CnnUnactivatedMatrixPointer]
        mov qword rcx, [CnnNeuronsCount]
        call krasShowPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowNeuronsCosts:
        mov rcx, showNeuronsCostsText
        mov rdx, SHOW_NEURONS_COSTS_TEXT_LENGTH
        call print

        mov qword rbx, [CnnCostMatrixPointer]
        mov qword rcx, [CnnNeuronsCount]
        call krasShowPseudoMatrix

        call displaySeperator

        ret
    krasShowWeightsGradient:
        mov rcx, showWeightsGradientText
        mov rdx, SHOW_WEIGHTS_GRADIENT_TEXT_LENGTH
        call print

        mov qword rbx, [CnnBackpropagationWeightsMatrixPointer]
        mov qword rcx,[CnnWeightsCount]
        call krasShowPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowLossHistory:
        mov rcx, showLossHistoryText
        mov rdx, SHOW_LOSS_HISTORY_TEXT_LENGTH
        call print

        mov qword rbx, lossHistory
        mov qword rcx, [CnnEpochs]
        call krasShowPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowOutput:
        mov rcx, showOutputText
        mov rdx, SHOW_OUTPUT_TEXT_LENGTH
        call print

        mov qword rbx, [CnnActivatedMatrixPointer]
        add qword rbx, [CnnLastLayerOffset]
        mov qword rcx, [CnnDataOutputSize]
        call krasShowPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowUnactivatedOutput:
        mov rcx, showOutputText
        mov rdx, SHOW_OUTPUT_TEXT_LENGTH
        call print

        mov qword rbx, [CnnUnactivatedMatrixPointer]
        add qword rbx, [CnnLastLayerOffset]
        mov qword rcx, [CnnDataOutputSize]
        call krasShowPseudoMatrix

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
;===============================;
;/\- Kras- Sample management -/\;
;===============================;
    krasCreateSample:
        ret 
    
;=============================;
;/\- Kras- File management -/\;
;=============================;
    krasLoadWeights:
        mov rax, 5
        mov rbx, [krasExportFileName]
        mov rcx, 0 
        mov rdx, 0o777
        int 0x80

        push rax
 
        mov rbx, rax 
        mov rax, 3
        mov rcx, [CnnWeightsMatrixPointer]
        mov rdx, 616
        
        int 0x80

        pop rbx 
        mov rax, 6
        int 0x80
        ret
    krasSaveLoss:
        mov rax, 5
        mov rbx, [krasExportFileName]
        mov rcx, 65 
        mov rdx, 0o777
        int 0x80

        push rax
 
        mov rbx, rax 
        mov rax, 4 
        mov rcx, lossHistory
        mov rdx, 1600
        
        int 0x80

        pop rbx 
        mov rax, 6
        int 0x80
        ret
    krasSaveWeights:
        mov rax, 5
        mov rbx, [krasExportFileName]
        mov rcx, 65 
        mov rdx, 0o777
        int 0x80

        push rax
 
        mov rbx, rax 
        mov rcx, [CnnWeightsMatrixPointer]
        mov rdx, [CnnWeightsCount]
        mov rax, DOUBLE_SIZE
        mul rdx 
        mov rdx, rax 
        mov rax, 4
        
        int 0x80

        pop rbx 
        mov rax, 6
        int 0x80
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
%macro fitParameters 2
    mov rax, %1
    mov rbx, %2
%endmacro
%macro saveParameters 1
    mov qword [krasExportFileName], %1
%endmacro
%macro loadParameters 1
    mov qword [krasExportFileName], %1
%endmacro