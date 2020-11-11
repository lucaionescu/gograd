# gograd

![Test](https://github.com/lucaionescu/gograd/workflows/Test/badge.svg)

Gograd is a small automatic differentiation (reverse-mode) framework written in Go.

### Example
```go
	a := NewVar(4)
	b := NewVar(3)
	c := a.Mul(a.Add(b))

	c.Backward()

	fmt.Println(c.value)
	// prints 28

	fmt.Println(a.grad
	// prints 11

	fmt.Println(b.grad)
	// prints 4
```
