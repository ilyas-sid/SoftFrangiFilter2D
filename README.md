# SoftFrangiFilter2D
Implementation of the soft 2D Frangi filter [1] on Pytorch.
This implementation was used in our work to generate tube-shaped objects on chest x-rays [2].

Example:
```python
import torch
from soft_frangi.soft_frangi_filter2d import SoftFrangiFilter2D

soft_frangi_filter = SoftFrangiFilter2D(1, 7, [2,4,8], 0.5, 0.5, 'cpu')
img = torch.randn((8,1,256,256))

soft_frangi_response = soft_frangi_filter(img)
```

[[1] Multiscale vessel enhancement filtering](https://link.springer.com/chapter/10.1007/BFb0056195)

[[2] Tubular Shape Aware Data Generation for Segmentation](https://arxiv.org/pdf/2010.00907.pdf)
