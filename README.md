# Cog-CoDet
Cog wrapper for [CoDet](https://arxiv.org/abs/2310.16667). This is an implementation of the original work's [GitHub repository](https://github.com/CVMI-Lab/CoDet), see Replicate [model page](https://replicate.com/adirik/codet) for the API and demo.


## Basic Usage
To run a prediction:

```bash
cog predict -i image=@examples/1.jpeg -i confidence=0.5
```

To start your own server:

```bash
cog run -p 5000 python -m cog.server.http
```

## References
```
@inproceedings{ma2023codet,
  title={CoDet: Co-Occurrence Guided Region-Word Alignment for Open-Vocabulary Object Detection},
  author={Ma, Chuofan and Jiang, Yi and Wen, Xin and Yuan, Zehuan and Qi, Xiaojuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```