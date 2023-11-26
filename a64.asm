bits 64

%include "kras.asm"
section .data 
    customLearningRate: dq -0.0001
    customExportLossFileName: db "lossHistory.bin",0
    customExportWeightsFileName: db "myCnnWeightsSaved.bin",0
    customImportWeightsFileName: db "myCnnWeightsLoaded.bin",0

section .text
    global _start
    _start:
        
        call krasInitialise
 
        ; Add {input, dense and output} layers
        addLayerParameters 3, TF_LINEAR, TF_NO_BIASES, TF_ZERO
        call krasAddLayer

        addLayerParameters 5, TF_LEAKY_RELU, TF_NO_BIASES, TF_ZERO
        call krasAddLayer

        addLayerParameters 8, TF_LEAKY_RELU, TF_NO_BIASES, TF_ZERO
        call krasAddLayer

        addLayerParameters 2, TF_LEAKY_RELU, TF_NO_BIASES, TF_ZERO
        call krasAddLayer

        addLayerParameters 3, TF_SOFTMAX, TF_NO_BIASES, TF_ZERO
        call krasAddLayer
 
        ; Set up the CNN, by compiling it 
        call krasCompile

        ; Possibly load existing weights 
        loadParameters qword customImportWeightsFileName
        call krasLoadWeights

        ; Fit it 
        fitParameters 150, [customLearningRate]
        call krasFit
        
        ; Miscellaneous 
        call krasShowLossHistory

        call krasShowOutput

        ; Possibly export stuff
        saveParameters qword customExportLossFileName
        call krasSaveLoss

        saveParameters qword customExportWeightsFileName
        call krasSaveWeights
        call quit