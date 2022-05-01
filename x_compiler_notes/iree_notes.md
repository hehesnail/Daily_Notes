## IREE Compiler

This markdown tracks the notes for google iree compiler (https://github.com/google/iree).

First check the developer notes (https://github.com/google/iree/tree/main/docs/developers). Build with ccache to save the rebuild time, try all the developer tools to check the effect of transforms, see the benchmark doc for iree module and e2e benchmarking, however, e2e dispatch functions benckmarking seems exists some errors.

Useful tools:
* iree-opt / iree-compile
* iree-translate: iree-run-module / iree-check-module / iree-run-mlir / iree-dump-module

### Flow Dialect

Target: tensor program modeling and compute workload partition.

### Stream Dialect

Target: device placement and asynchronous schedueling.

### HAL Dialect

Target: hardware abstraction layer for buffer and execution management.
