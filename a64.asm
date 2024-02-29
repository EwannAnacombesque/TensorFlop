bits 64

%include "kras.asm"
section .data
    customEpochs: dq 200
    customBatchSize : dq 2
    customLearningRate: dq -0.03
    customExportLossFileName: db "mixedLossHistory.bin",0
    customExportWeightsFileName: db "myCnnWeightsSaved.bin",0
    customImportWeightsFileName: db "myCnnWeightsLoaded.bin",0
    customExportVerificationSquareFileName: db "squareVerif20per20.bin",0

section .text
    global _start
    _start:
        call krasInitialise
 
        ; Add {input, dense and output} layers
        addLayerParameters 2, TF_LINEAR, TF_NO_BIASES, TF_ZERO
        call krasAddLayer

        addLayerParameters 4, TF_TANH, TF_NO_BIASES, TF_ZERO
        call krasAddLayer

        addLayerParameters 9, TF_TANH , TF_NO_BIASES, TF_ZERO
        call krasAddLayer

        addLayerParameters 2, TF_SOFTMAX, TF_NO_BIASES, TF_ZERO
        call krasAddLayer
        
        ; Set up the CNN, by compiling it 
        call krasCompile

        ; Possibly load existing weights 
        ;loadParameters qword customImportWeightsFileName
        ;call krasLoadWeights
        
        ; Get a reference value of the initial output of the CNN
        fitParameters 1, 1, [customLearningRate]
        fitExtraParameters TF_TRUE
        call krasFit
        call krasShowOutput

        ; Fit it 
        fitParameters [customEpochs], [customBatchSize], [customLearningRate]
        fitExtraParameters TF_TRUE
        call krasFit

        call krasShowOutput 

        ; Miscellaneous 

        call krasShowLossHistory

        ; Possibly export stuff
        ;saveParameters qword customExportLossFileName
        ;call krasSaveLoss

        ;saveParameters qword customExportWeightsFileName
        ;call krasSaveWeights

        ;saveParameters qword customExportVerificationSquareFileName
        ;call krasSaveVerificationSquare
        call quit
