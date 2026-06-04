# micrograd Explained

**An interactive, slide-by-slide walkthrough of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) — a tiny scalar-valued autograd engine and the neural-net library built on top of it.**

Roughly 150 lines of pure Python are explained — from a single differentiable `Value` to backpropagation and a fully trained MLP.

---

## [Launch Presentation](https://brendanjameslynskey.github.io/micrograd_presentation/)

---

## What's Covered

| Part | Topic |
|------|-------|
| 1 | **What & Why** — autograd engines, backprop as reverse-mode autodiff, the expression graph, and why scalars (not tensors) aid clarity |
| 2 | **The Value object** — wrapping a number with `.data`, `.grad`, `_prev`, `_op` and a `_backward` closure; `__add__`, `__mul__`, `__pow__`, `relu`, `tanh`, `exp` and their local derivatives |
| 3 | **backward()** — topological sort of the graph, seeding `grad=1`, reverse traversal, and why gradients accumulate with `+=` |
| 4 | **Neurons, Layers, MLP** — `nn.py`: a Neuron (`w·x+b` + nonlinearity), a Layer of Neurons, an MLP of Layers, and `parameters()` |
| 5 | **Training loop** — forward pass, mean-squared-error loss, `zero_grad()`, `loss.backward()`, the gradient-descent update, and the moons classification demo |
| 6 | **micrograd vs PyTorch** — the shared API shape, why scalars are slow but illuminating, and how this scales to tensors in real frameworks |

The presentation closes with a summary grid, a note on where micrograd fits in the *Neural Networks: Zero to Hero* series, and key takeaways.

## Format

Built with [Reveal.js](https://revealjs.com/). Use `→` to advance, `↓` for sub-sections, and `Esc` for the slide overview.

## Part of

This deck is part of [Karpathy: Neural Networks Zero to Hero](https://github.com/BrendanJamesLynskey/LLM_Hub_Karpathy_Zero_to_Hero), itself part of the [LLMs](https://github.com/BrendanJamesLynskey/LLMs) hub.

Upstream credit: Andrej Karpathy's [micrograd repository](https://github.com/karpathy/micrograd) and the video [*The spelled-out intro to neural networks and backpropagation: building micrograd*](https://www.youtube.com/watch?v=VMj-3S1tku0).
