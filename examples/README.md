# examples

I provided some coding examples which 3D-Shampoo was used on.

"ds_pp.py" is the simplest example for any type of model with DeepSpeed pipeline and data parallelism active (if enough GPUs available).
It helps to be familiar with the DeepSpeed library.

"my_megatron_3D_parallelism" is an example created by DeepSpeed self and modified to work with the 3D-Shampoo optimizer to fully utilize itself in 3D parallelism.
To run this code, you have to install Megatron-LM. Section 6.6 in my MSc-Thesis describes the steps on how to do get 3D parallelism and which libraries have to be installed.

Both codes where part of my MSc-Thesis

(ATTENTION: Some code may not be working due to changes and updates to DeepSpeed, Megatron-LM, PyTorch)

