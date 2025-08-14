# Dummy Algorithm

The Dummy Algorithm is a simple algorithm that does not perform any fusion operation. Instead, it returns a pretrained model if one is available in the model pool. If no pretrained model is available, it returns the first model in the model pool.
This algorithm is useful for testing and debugging purposes, as it allows you to quickly check if the model pool is set up correctly and the fusion process is working as expected.

## Usage

To use the Dummy Algorithm, you need to specify `"dummy"` as the algorithm name.

```bash
fusion_bench method=dummy ...
```

## Implementation Details

The implementation of the Dummy Algorithm is straightforward. Here is the main method of the `DummyAlgorithm` class:

- [fusion_bench.method.dummy.DummyAlgorithm][]
