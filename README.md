# gograd

![Test](https://github.com/lucaionescu/gograd/workflows/Test/badge.svg)

Gograd is a small automatic reverse-mode differentiation framework written in Go.

## Example
```go
a := NewVar(4)
b := NewVar(3)
c := a.Mul(a.Add(b))

c.Backward()

fmt.Println(c.value)
// prints 28

fmt.Println(a.grad)
// prints 11

fmt.Println(b.grad)
// prints 4
```
## Resources
- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [geohot/tinygrad](https://github.com/geohot/tinygrad)
- [Automatic Differentiation in Machine Learning: a Survey](https://www.jmlr.org/papers/volume18/17-468/17-468.pdf)
- [From scratch: reverse-mode automatic differentiation (in Python)](https://sidsite.com/posts/autodiff/)
- [Reverse-mode automatic differentiation: a tutorial](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)
- [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/)
