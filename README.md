# PS-SNC

In this study, we propose a novel partially shared signed network clustering model (PS-SNC) for detecting protein complexes from multiple state-specific signed PPI networks jointly. PS-SNC can not only consider the signs of PPIs, but also identify the common and unique protein complexes in different states. 

## Usage

example input:
- net.txt: The PPI file (protein1, protein2, $\pm$ 1)
- protein.txt: The protein set of two networks

example command:
```
python run_train.py 
```

example output:
- complex.txt: The all predicted complexes
- complex1.txt: The predicted complexes of network1
- complex2.txt: The predicted complexes of network2
  
