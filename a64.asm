bits 64

%include "kras.asm"

section .text
    global _start
    _start:
        
        call krasInitialise
        
        ; Add {input, dense and output} layers
        addLayerParameters 3, TF_LINEAR, TF_USE_BIASES, TF_ZERO
        call krasAddLayer

        addLayerParameters 5, TF_LEAKY_RELU, TF_USE_BIASES, TF_ZERO
        call krasAddLayer

        addLayerParameters 8, TF_LEAKY_RELU, TF_USE_BIASES, TF_ZERO
        call krasAddLayer

        addLayerParameters 2, TF_LEAKY_RELU, TF_USE_BIASES, TF_ZERO
        call krasAddLayer

        addLayerParameters 4, TF_LEAKY_RELU, TF_USE_BIASES, TF_ZERO
        call krasAddLayer

        addLayerParameters 3, TF_SOFTMAX, TF_USE_BIASES, TF_ZERO
        call krasAddLayer

        ; Set up the CNN, by compiling it 
        call krasCompile

        ; Fit it 
        fitParameters 100
        call krasFit
        
        call showOutput

        call quit
    
