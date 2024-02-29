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

    showInputText db "Last input :",10
    SHOW_INPUT_TEXT_LENGTH equ $-showInputText

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

    ;# Generated sample #;

    krasSampleSize : resq 1
    krasSampleInputDim : resq 1
    krasSampleOutputDim : resq 1
    
    ;# Verification sample #;
    verificationInputs: resq 2
    krasVerificationSquareData: resq 1

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

        mov [krasBatchSize], rcx 
        mov [CnnBatchSize], rcx 

        call fitCnn
        ret 
;===========================;
;/\- Kras- Dev Verbosity -/\;
;===========================;
    krasShowWeights:
        mov qword rbx, [CnnWeightsMatrixPointer]
        mov qword rcx,[CnnWeightsCount]
        call showPseudoMatrix
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
        mov qword rcx, [krasEpochs]
        mov qword rax, [krasBatchSize]
        mul rcx 
        mov rcx, rax
        call showPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowOutput:
        mov rcx, showOutputText
        mov rdx, SHOW_OUTPUT_TEXT_LENGTH
        call print

        mov qword rbx, [CnnActivatedMatrixPointer]
        add qword rbx, [CnnLastLayerOffset]
        mov qword rcx, [CnnDataOutputDim]
        call showPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowInput:
        mov rcx, showInputText
        mov rdx, SHOW_INPUT_TEXT_LENGTH
        call print

        mov qword rbx, [CnnActivatedMatrixPointer]
        add qword rbx, 0
        mov qword rcx, [CnnDataInputDim]
        call showPseudoMatrix

        call displayHugeSeparator
        ret
    krasShowUnactivatedOutput:
        mov rcx, showOutputText
        mov rdx, SHOW_OUTPUT_TEXT_LENGTH
        call print

        mov qword rbx, [CnnUnactivatedMatrixPointer]
        add qword rbx, [CnnLastLayerOffset]
        mov qword rcx, [CnnDataOutputDim]
        call showPseudoMatrix

        call displaySeperator
        ret

;===============================;
;/\- Kras- Sample management -/\;
;===============================;
    krasPrepareSample:
        ;# Allocate memory for the input sample #;
        mov rax, [krasSampleSize]
        mov rdx, [krasSampleInputDim]
        mul rdx

        mov qword [cafElements], rax ; number of elements in my sample
        mov qword [cafSize], DOUBLE_SIZE ; size of a float
        call caf

        mov qword [CnnDataInputSample], rax

        ;# Allocate memory for the output sample #;
        mov rax, [krasSampleSize]
        mov rdx, [krasSampleOutputDim]
        mul rdx

        mov qword [cafElements], rax ; number of elements in my sample
        mov qword [cafSize], DOUBLE_SIZE ; size of a float
        call caf

        mov qword [CnnDataOutputSample], rax

        ;# Get input size #;
        mov rdx, [krasSampleInputDim]
        mov rax, DOUBLE_SIZE
        mul rdx 
        mov qword [CnnDataInputSize], rax
        
        ;# Get output size #;
        mov rdx, [krasSampleOutputDim]
        mov rax, DOUBLE_SIZE
        mul rdx 
        mov qword [CnnDataOutputSize], rax
        ret
    krasCreateSample:
        mov qword [krasSampleSize], rax
        mov qword [krasSampleInputDim], rbx
        mov qword [krasSampleOutputDim], rcx

        call krasPrepareSample
        call krasGenerateRadiusSample
        ret 

    krasGenerateRadiusSample:
        mov qword rdi, [krasSampleInputDim]
        
        mov rbx, [CnnDataInputSample]
        mov rcx, [CnnDataOutputSample]

        mov rdx, [krasSampleSize]
        
        fcomp st0
        krasGenerateRadiusSampleMainLOOP:
            call generateRandomSample
            call getDistance
            
            fld qword [INITIAL_CLASSIFICATION_RADIUS_VALUE]
            fld qword [computedFloatPointer]
            fcomip st0, st1 
            jb krasGenerateRadiusSampleInside

            krasGenerateRadiusSampleOutside:
            fldz  ; change 
            fstp qword [rcx]
            add rcx, DOUBLE_SIZE
            fld1 
            fstp qword [rcx]

            jmp krasGenerateRadiusSampleDone 

            krasGenerateRadiusSampleInside:
            fldz 
            fstp qword [rcx]
            add rcx, DOUBLE_SIZE
            fld1
            fstp qword [rcx]
            
            krasGenerateRadiusSampleDone:

            add rbx, [CnnDataInputSize]
            add rcx, DOUBLE_SIZE
            
            fcomp st0 
            
            dec rdx 
            cmp qword rdx, 0
            jnz krasGenerateRadiusSampleMainLOOP

        ret
;=================================;
;/\- Kras- Verification sample -/\;
;=================================;
    krasVerifySquare:
        mov qword [cafElements], 800 ; number of elements in my square, outputdim*side**2
        mov qword [cafSize], DOUBLE_SIZE ; size of a float
        call caf
        mov qword [krasVerificationSquareData], rax
        
        fldz
        fld1
        fsubp
        fst qword [verificationInputs]
        fstp qword [verificationInputs+DOUBLE_SIZE]

        xor rdx, rdx
        xor rcx, rcx
        mov rbx, [krasVerificationSquareData]
        mov rax, [CnnActivatedMatrixPointer]
        add rax, [CnnLastLayerOffset]
        krasVerifySquareXYLOOP:
                ; Set my inputs as my square values 
                mov rdi, [CnnDataInputSample]
                fld qword [verificationInputs]
                fstp qword [rdi]
                fld qword [verificationInputs+DOUBLE_SIZE]
                fstp qword [rdi+DOUBLE_SIZE]

                ; Actually predict my square value 
                saveRegisters
                call krasPredict
                getBackRegisters

                ; Store my values
                mov rdi, [rax]
                mov [rbx], rdi
                add rbx, DOUBLE_SIZE
                mov rdi, [rax+DOUBLE_SIZE]
                mov [rbx], rdi
                add rbx,DOUBLE_SIZE

                ; Increment my square dx 
                fld qword [INITIAL_VERIFICATION_SQUARE_DXY]
                fld qword [verificationInputs]
                faddp 
                fstp qword [verificationInputs]

                ; Looping back
                inc rdx
                cmp rdx, 20
                jne krasVerifySquareXYLOOP

             
            ; Set back my dx to zero
            fldz
            fld1
            fsubp
            fstp qword [verificationInputs]
            xor rdx, rdx

            ; Increment my square dy
            fld qword [INITIAL_VERIFICATION_SQUARE_DXY]
            fld qword [verificationInputs+DOUBLE_SIZE]
            faddp 
            fstp qword [verificationInputs+DOUBLE_SIZE]

            ; Looping back to dx=0
            inc rcx
            cmp rcx, 20
            jne krasVerifySquareXYLOOP
        ret
    krasSaveVerificationSquare:
        mov rax, 5
        mov rbx, [krasExportFileName]
        mov rcx, 65 
        mov rdx, 0o777
        int 0x80

        push rax
 
        mov rbx, rax 
        mov rax, 4 
        mov rcx, [krasVerificationSquareData]
        mov rdx, 6400

        
        int 0x80

        pop rbx 
        mov rax, 6
        int 0x80
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
        mov rdx, 32000

        
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
%macro fitParameters 3
    mov rax, %1
    mov rbx, %3
    mov rcx, %2

    mov qword [CnnStaticInput], TF_TRUE
%endmacro
%macro fitExtraParameters 1
    mov qword [CnnStaticInput], %1
%endmacro 
%macro saveParameters 1
    mov qword [krasExportFileName], %1
%endmacro
%macro loadParameters 1
    mov qword [krasExportFileName], %1
%endmacro
%macro sampleParameters 3
    mov rax, %1
    mov rbx, %2
    mov rcx, %3
%endmacro
