The example code includes a demo that processes a sequence of length 1024 with 8 attention heads. You'll see timing metrics and sample outputs printed to the console.

### Customizing Parameters

You can modify the attention parameters in the `main()` function:

- `batch_size`: Number of sequences to process in parallel
- `seq_length`: Length of input sequence
- `num_heads`: Number of attention heads
- `head_dim`: Dimension of each attention head
- `decay_rate`: Decay factor for attention weights (between 0 and 1)
- `block_size`: Size of processing blocks

## Want to Learn More?

Check out the my Lightning Attention explanation: ["Understanding Lightning Attention: A Breakthrough in Linear Attention Efficiency"](https://ighoshsubho.bearblog.dev/understanding-lightning-attention-a-breakthrough-in-linear-attention-efficiency/)
