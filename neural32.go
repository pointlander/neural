// Copyright 2016 The Neural Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package neural

import (
	"fmt"
	"math"
	"math/rand"
)

type Function32 func(x float32) float32

type FunctionPair32 struct {
	F, DF Function32
}

type Neural32 struct {
	Layers    []int
	Weights   [][][]float32
	Changes   [][][]float32
	Functions []FunctionPair32
}

// http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
type WeightInitializer32 func(in, out int) float32

func WeightInitializer32Basic(in, out int) float32 {
	return random32(-1, 1)
}

func WeightInitializer32FanIn(in, out int) float32 {
	return random32(-1, 1) / float32(math.Sqrt(float64(in)))
}

func WeightInitializer32FanInFanOut(in, out int) float32 {
	return random32(-1, 1) * float32(4*math.Sqrt(6/float64(in+out)))
}

func (n *Neural32) Init(initializer WeightInitializer32, layers ...int) {
	depth := len(layers) - 1
	if depth < 1 {
		panic("there should be at least 2 layers")
	}

	n.Layers = layers
	for l := range layers[:depth] {
		layers[l]++
	}

	n.Weights = make([][][]float32, depth)
	for l := range layers[:depth] {
		weights := matrix32(layers[l+1], layers[l])
		for i := 0; i < layers[l]; i++ {
			for j := 0; j < layers[l+1]; j++ {
				weights[j][i] = initializer(layers[l], layers[l+1])
			}
		}
		n.Weights[l] = weights
	}

	n.Changes = make([][][]float32, depth)
	for l := range layers[:depth] {
		n.Changes[l] = matrix32(layers[l], layers[l+1])
	}

	n.Functions = make([]FunctionPair32, depth)
	for f := range n.Functions {
		n.Functions[f] = FunctionPair32{
			F:  sigmoid32,
			DF: dsigmoid32,
		}
	}
}

func (n *Neural32) EnableRegression() {
	output := len(n.Functions) - 1
	n.Functions[output].F = func(x float32) float32 {
		return x
	}
	n.Functions[output].DF = func(x float32) float32 {
		return 1
	}
}

// http://iamtrask.github.io/2015/07/28/dropout/
func (n *Neural32) EnableDropout(probability float32) {
	depth := len(n.Layers) - 1
	functions := make([]FunctionPair32, depth)
	copy(functions, n.Functions)

	for i := range n.Functions[:depth-1] {
		n.Functions[i].F = func(x float32) float32 {
			x = functions[i].F(x)
			if rand.Float32() > 1-probability {
				x = 0
			} else {
				x *= 1 / (1 - probability)
			}
			return x
		}
	}
}

func NewNeural32(config func(neural *Neural32)) *Neural32 {
	neural := &Neural32{}
	config(neural)
	return neural
}

type Context32 struct {
	*Neural32
	Activations [][]float32
}

func (c *Context32) SetInput(input []float32) {
	copy(c.Activations[0], input)
}

func (c *Context32) GetOutput() []float32 {
	return c.Activations[len(c.Activations)-1]
}

func (n *Neural32) NewContext() *Context32 {
	layers, depth := n.Layers, len(n.Layers)

	activations := make([][]float32, depth)
	for i, width := range layers {
		activations[i] = vector32(width, 1.0)
	}

	return &Context32{
		Neural32:    n,
		Activations: activations,
	}
}

func (c *Context32) Infer() {
	depth := len(c.Layers) - 1

	if depth > 1 {
		for i := range c.Activations[:depth-1] {
			activations, weights := c.Activations[i], c.Weights[i]
			for j := range weights[:len(weights)-1] {
				sum := dot32(activations, weights[j])
				c.Activations[i+1][j] = c.Functions[i].F(sum)
			}
		}
	}

	i := depth - 1
	activations, weights := c.Activations[i], c.Weights[i]
	for j := range weights[:len(weights)] {
		sum := dot32(activations, weights[j])
		c.Activations[i+1][j] = c.Functions[i].F(sum)
	}
}

func (c *Context32) BackPropagate(targets []float32, lRate, mFactor float32) float32 {
	depth, layers := len(c.Layers), c.Layers

	deltas := make([][]float32, depth-1)
	for i := range deltas {
		deltas[i] = vector32(layers[i+1], 0)
	}

	l := depth - 2
	for i := 0; i < layers[l+1]; i++ {
		activation := c.Activations[l+1][i]
		e := targets[i] - activation
		deltas[l][i] = c.Functions[l].DF(activation) * e
	}
	l--

	for l >= 0 {
		for i := 0; i < layers[l+1]; i++ {
			var e float32

			for j := 0; j < layers[l+2]; j++ {
				e += deltas[l+1][j] * c.Weights[l+1][j][i]
			}

			deltas[l][i] = c.Functions[l].DF(c.Activations[l+1][i]) * e
		}
		l--
	}

	for l := 0; l < depth-1; l++ {
		change := make([]float32, layers[l+1])
		for i := 0; i < layers[l]; i++ {
			copy(change, deltas[l])
			scal32(c.Activations[l][i], change)
			scal32(mFactor, c.Changes[l][i])
			axpy32(lRate, change, c.Changes[l][i])
			for j := 0; j < layers[l+1]; j++ {
				c.Weights[l][j][i] = c.Weights[l][j][i] + c.Changes[l][i][j]
			}
			copy(c.Changes[l][i], change)
		}
	}

	var e float32
	for i := 0; i < len(targets); i++ {
		f := targets[i] - c.Activations[depth-1][i]
		e += f * f
	}

	return e
}

func (n *Neural32) Train(source func(iteration int) [][][]float32, iterations int, lRate, mFactor float32) []float32 {
	context, errors := n.NewContext(), make([]float32, iterations)

	for i := 0; i < iterations; i++ {
		var (
			e float32
			n int
		)

		patterns := source(i)

		for _, p := range patterns {
			context.SetInput(p[0])
			context.Infer()
			e += context.BackPropagate(p[1], lRate, mFactor)
			n += len(p[1])
		}

		errors[i] = e / float32(n)
	}

	return errors
}

func (n *Neural32) test(patterns [][][]float32) {
	context := n.NewContext()
	for _, p := range patterns {
		context.SetInput(p[0])
		context.Infer()
		fmt.Println(p[0], "->", context.GetOutput(), " : ", p[1])
	}
}
