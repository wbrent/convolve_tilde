# convolve_tilde

[convolve~] is a partitioned impulse response (IR) convolution external for Pure Data. The only creation argument is the partition size, which also determines the amount of delay between the dry and wet signal (i.e., pre-delay). Partition size must be a multiple of Pd's default block size: 64 samples (1.4512ms @ 44100Hz).
