package main

import (
	"math"
	"testing"
)

func TestAdd(t *testing.T) {
	cases := []struct {
		x, y, want *Var
	}{
		{NewVar(4), NewVar(4), NewVar(8)},
		{NewVar(-2), NewVar(3), NewVar(1)},
		{NewVar(0), NewVar(0), NewVar(0)},
		{NewVar(5.2), NewVar(3.5), NewVar(8.7)},
	}

	for _, c := range cases {
		got := c.x.Add(c.y)
		if math.Abs(got.value-c.want.value) >= 1e-9 {
			t.Errorf("Add(%.2f, %.2f) == %.2f, want %.2f", c.x.value, c.y.value, got.value, c.want.value)
		}
	}
}

func TestMul(t *testing.T) {
	cases := []struct {
		x, y, want *Var
	}{
		{NewVar(4), NewVar(4), NewVar(16)},
		{NewVar(-2), NewVar(3), NewVar(-6)},
		{NewVar(0), NewVar(1), NewVar(0)},
		{NewVar(5.2), NewVar(3.5), NewVar(18.2)},
	}

	for _, c := range cases {
		got := c.x.Mul(c.y)
		if math.Abs(got.value-c.want.value) >= 1e-9 {
			t.Errorf("Mul(%.2f, %.2f) == %.2f, want %.2f", c.x.value, c.y.value, got.value, c.want.value)
		}
	}
}

func TestSub(t *testing.T) {
	cases := []struct {
		x, y, want *Var
	}{
		{NewVar(4), NewVar(4), NewVar(0)},
		{NewVar(-2), NewVar(3), NewVar(-5)},
		{NewVar(0), NewVar(1), NewVar(-1)},
		{NewVar(5.2), NewVar(3.5), NewVar(1.7)},
	}

	for _, c := range cases {
		got := c.x.Sub(c.y)
		if math.Abs(got.value-c.want.value) >= 1e-9 {
			t.Errorf("Sub(%f, %f) == %f, want %f", c.x.value, c.y.value, got.value, c.want.value)
		}
	}
}

func TestDiv(t *testing.T) {
	cases := []struct {
		x, y, want *Var
	}{
		{NewVar(4), NewVar(4), NewVar(1)},
		{NewVar(-2), NewVar(4), NewVar(-0.5)},
		{NewVar(0), NewVar(1), NewVar(0)},
		{NewVar(5.2), NewVar(3.5), NewVar(1.4857142857)},
	}

	for _, c := range cases {
		got := c.x.Div(c.y)
		if math.Abs(got.value-c.want.value) >= 1e-9 {
			t.Errorf("Div(%f, %f) == %f, want %f", c.x.value, c.y.value, got.value, c.want.value)
		}
	}
}

func TestPow(t *testing.T) {
	cases := []struct {
		x, y, want *Var
	}{
		{NewVar(4), NewVar(4), NewVar(256)},
		{NewVar(-2), NewVar(3), NewVar(-8)},
		{NewVar(0), NewVar(1), NewVar(0)},
		{NewVar(5.2), NewVar(3.5), NewVar(320.6355723447)},
	}

	for _, c := range cases {
		got := c.x.Pow(c.y)
		if math.Abs(got.value-c.want.value) >= 1e-9 {
			t.Errorf("Pow(%f, %f) == %f, want %f", c.x.value, c.y.value, got.value, c.want.value)
		}
	}
}

func TestOperation_1(t *testing.T) {
	a := NewVar(4)
	b := NewVar(3)
	c := a.Mul(a.Add(b))

	if c.value != 28 {
		t.Errorf("a.Mul(a.Add(b)) == %f, want 28.0", c.value)
	}
}

func TestDerivative_1(t *testing.T) {
	a := NewVar(4)
	b := NewVar(3)
	c := a.Mul(a.Sub(b))

	c.Backward()

	if math.Abs(a.grad-5.0) >= 1e-9 {
		t.Errorf("a.grad == %f, want 5.0", a.grad)
	}

	if math.Abs(b.grad+4.0) >= 1e-9 {
		t.Errorf("b.grad == %f, want -4.0", b.grad)
	}
}

func TestDerivative_2(t *testing.T) {
	a := NewVar(4)
	b := NewVar(3)
	c := a.Add(b)
	d := a.Mul(c)

	d.Backward()

	if math.Abs(a.grad-11.0) >= 1e-9 {
		t.Errorf("a.grad == %f, want 11.0", a.grad)
	}

	if math.Abs(b.grad-4.0) >= 1e-9 {
		t.Errorf("b.grad == %f, want 4.0", b.grad)
	}

	if math.Abs(b.grad-4.0) >= 1e-9 {
		t.Errorf("b.grad == %f, want 4.0", b.grad)
	}
}
