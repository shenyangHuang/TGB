STHN method adopted from: https://github.com/celi52/STHN/tree/main

To run:

1. Install requirements. The two new additional requirements for STHN are `pybind11` and `torchmetrics==0.11.0`
2. Compile the sampler
```bash
python sthn_sampler_setup.py build_ext --inplace
```

3. Run the example code

```bash
python sthn.py
```

If the code runs correctly the output would end with

```
INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< 
        Test: mrr: X
        Test: Elapsed Time (s):  Y
```