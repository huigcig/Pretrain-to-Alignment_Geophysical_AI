We provide two separate components of the final model: the base model weights and the LoRA adapter weights. Both components are required for inference. 

1. First, load the base model, then load and apply the corresponding LoRA adapter using the inference code provided in this repository. 
2. The LoRA adapter contains only the adaptation parameters and cannot be used independently. Please download both components before running inference.