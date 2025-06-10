<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Exploring Diffusion Transformer Designs via Grafting</h1>      
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://keshik6.github.io/" target="_blank" style="text-decoration: none;">Keshigeyan&nbsp;Chandrasegaran</a><sup>*1,2</sup>,&nbsp;
    <a href="https://zymrael.github.io" target="_blank" style="text-decoration: none;">Michael&nbsp;Poli</a><sup>*1,2</sup>,
              <a href="https://danfu.org" target="_blank" style="text-decoration: none;">Daniel&nbsp;Y.&nbsp;Fu</a><sup>3,4</sup>,
    <a href="https://sites.google.com/view/dongjun-kim"  target="_blank">Dongjun&nbsp;Kim</a><sup>1</sup>,<br/>
    <a href="https://lea-m-hadzic.github.io" target="_blank">Lea&nbsp;M.&nbsp;Hadzic</a><sup>1</sup>,
    <a href="https://limanling.github.io" target="_blank">Manling&nbsp;Li</a><sup>1,5</sup>,
    <a href="https://web.stanford.edu/~agrim/" target="_blank">Agrim&nbsp;Gupta</a><sup>6</sup>,
    <a href="https://jiajunwu.com"  target="_blank">Stefano&nbsp;Massaroli</a><sup>2</sup>,<br>
     <a href="http://azaliamirhoseini.com"  target="_blank">&nbsp;Azalia Mirhoseini</a><sup>1</sup>,
    <a href="https://www.niebles.net"  target="_blank">&nbsp;Juan Carlos Niebles</a><sup>&dagger;1,7</sup>,
    <a href="https://cs.stanford.edu/~ermon/"  target="_blank">&nbsp;Stefano Ermon</a><sup>&dagger;1</sup>,
    <a href="https://profiles.stanford.edu/fei-fei-li"  target="_blank">&nbsp;Li Fei-Fei</a><sup>&dagger;1</sup><a><br/>
<span class="author-block"><sup>1</sup>&nbsp;Stanford University&nbsp;&nbsp;</span>
<span class="author-block"><sup>2</sup>&nbsp;Liquid AI&nbsp;&nbsp;</span>
<span class="author-block"><sup>3</sup>&nbsp;Together AI&nbsp;&nbsp;</span>
<span class="author-block"><sup>4</sup>&nbsp;UC San Diego&nbsp;&nbsp;</span><br/>
<span class="author-block"><sup>5</sup>&nbsp;Northwestern University&nbsp;&nbsp;</span>
<span class="author-block"><sup>6</sup>&nbsp;Google DeepMind&nbsp;&nbsp;</span>
<span class="author-block"><sup>7</sup>&nbsp;Salesforce Research&nbsp;&nbsp;</span><br/>
<sup>*</sup>&nbsp;Equal contribution, <sup>&dagger;</sup>&nbsp;Equal senior authorship<br/>
<a href="https://grafting.stanford.edu" title="Website" target="_blank" rel="nofollow" style="text-decoration: none;">🌎Website</a> |
<a href="https://huggingface.co/grafting/" title="Grafted Models" target="_blank" rel="nofollow" style="text-decoration: none;">🤗 Grafted Models</a> |
<a href="https://arxiv.org/abs/2506.05340" title="arXiv" target="_blank" rel="nofollow" style="text-decoration: none;">📄 arXiv</a>
</p>


## 📣 News

- **[2025-06-10]: Grafting codebase released**


## Abstract
Designing model architectures requires decisions such as selecting operators (e.g., attention, convolution) and configurations (e.g., depth, width). However, evaluating the impact of these decisions on model quality requires costly pretraining, limiting architectural investigation. Inspired by how new software is built on existing code, we ask: can new architecture designs be studied using pretrained models? To this end, we present grafting, a simple approach for editing pretrained diffusion transformers (DiTs) to materialize new architectures under small compute budgets. Informed by our analysis of activation behavior and attention locality, we construct a testbed based on the DiT-XL/2 design to study the impact of grafting on model quality. Using this testbed, we develop a family of hybrid designs via grafting: replacing softmax attention with gated convolution, local attention, and linear attention, and replacing MLPs with variable expansion ratio and convolutional variants. Notably, many hybrid designs achieve good quality (FID: 2.38-2.64 vs. 2.27 for DiT-XL/2) using <2% pretraining compute. We then graft a text-to-image model (PixArt-Sigma), achieving a 1.43x speedup with less than a 2% drop in GenEval score. Finally, we present a case study that restructures DiT-XL/2 by converting every pair of sequential transformer blocks into parallel blocks via grafting. This reduces model depth by 2x and yields better quality (FID: 2.77) than other models of comparable depth. Together, we show that new diffusion model designs can be explored by grafting pretrained DiTs, with edits ranging from operator replacement to architecture restructuring.


## About this code
The Grafting codebase is written in Pytorch and provides a simple implementation for grafting Diffusion Transformers (DiTs).

## Getting Started
Start generating samples using our grafted models (See `demo_notebooks/grafting_demo.ipynb`)

## Contact
- Keshigeyan Chandrasegaran: keshik@stanford.edu
- Michael Poli: poli@stanford.edu

For issues, feedback, or contributions, please open an issue or submit a pull request.

## Citation

```bibtex
@article{chandrasegaran2024grafting,
      title={Exploring Diffusion Transformer Designs via Grafting},
      author={Chandrasegaran, Keshigeyan and Poli, Michael and Fu, Daniel Y. and Kim, Dongjun and 
      Hadzic, Lea M. and Li, Manling and Gupta, Agrim and Massaroli, Stefano and 
      Mirhoseini, Azalia and Niebles, Juan Carlos and Ermon, Stefano and Li, Fei-Fei},
      year={2025},
      url={https://arxiv.org/abs/2506.05340}, 
}
```
