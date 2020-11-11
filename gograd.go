package main

import (
	"fmt"
	"math"
)

// Var stores a single variable of type float64
type Var struct {
	value    float64
	grad     float64
	children []*Var
	backward func(float64, *Var, *Var)
}

// NewVar creates a new variable of type Var
func NewVar(value float64) (newVar *Var) {
	newVar = &Var{value: value}
	newVar.children = []*Var{}
	newVar.grad = 0
	return
}

// Add adds two variables
func (x *Var) Add(y *Var) *Var {
	return &Var{
		value:    x.value + y.value,
		children: []*Var{x, y},
		backward: func(v float64, x *Var, y *Var) {
			x.grad += v
			y.grad += v
		}}
}

// Mul multiplies two variables
func (x *Var) Mul(y *Var) *Var {
	return &Var{
		value:    x.value * y.value,
		children: []*Var{x, y},
		backward: func(v float64, x *Var, y *Var) {
			x.grad += y.value * v
			y.grad += x.value * v
		}}
}

// Sub subtracts two variables
func (x *Var) Sub(y *Var) *Var {
	return &Var{
		value:    x.value - y.value,
		children: []*Var{x, y},
		backward: func(v float64, x *Var, y *Var) {
			x.grad += v
			y.grad += (v * -1)
		}}
}

// Div divides two variables
func (x *Var) Div(y *Var) *Var {
	return &Var{
		value:    x.value / y.value,
		children: []*Var{x, y},
		backward: func(v float64, x *Var, y *Var) {
			x.grad += y.value * v
			y.grad += x.value * v
		}}
}

// Pow raises one variable to the power of the other
func (x *Var) Pow(y *Var) *Var {
	return &Var{
		value:    math.Pow(x.value, y.value),
		children: []*Var{x, y},
		backward: func(v float64, x *Var, y *Var) {
			x.grad += (y.value * math.Pow(x.value, y.value-1)) * v
		}}
}

// Backward reverse accumulates the gradient
func (x *Var) Backward() {
	stack := []*Var{}
	visited := map[*Var]bool{
		x: false,
	}

	var expand func(x *Var)

	expand = func(x *Var) {
		if !visited[x] {
			visited[x] = true
			if len(x.children) > 0 {
				for _, c := range x.children {
					expand(c)
				}
			}
			stack = append(stack, x)
		}
	}

	expand(x)

	x.grad = 1

	for i := len(stack) - 1; i >= 0; i-- {
		c := stack[i]
		if len(c.children) > 0 {
			c.backward(c.grad, c.children[0], c.children[1])
		}
	}
}

func main() {
	a := NewVar(4)
	b := NewVar(3)
	c := a.Add(b)
	d := a.Mul(c)

	d.Backward()

	fmt.Println(d.value)
	fmt.Println(a.grad)
	fmt.Println(b.grad)
	fmt.Println(c.grad)
}
