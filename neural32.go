// Copyright 2016 The Neural Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package neural

import (
	"fmt"
	"math"
	"math/rand"
)

// Function32 defines a function that takes a float32 and returns a float32
type Function32 func(x float32) float32

// FunctionPair32 represents a function, a derivative of the function, and a
// transform used for inference during training
type FunctionPair32 struct {
	F, T, DF Function32
}

// Neural32 is a 32 bit neural network
type Neural32 struct {
	Layers    []int
	Weights   [][][]float32
	Changes   [][][]float32
	Functions []FunctionPair32
}

// WeightInitializer32 is a function that initializes the neural network weights
// See: http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
type WeightInitializer32 func(in, out int) float32

// WeightInitializer32Basic basic weight initialization
func WeightInitializer32Basic(in, out int) float32 {
	return random32(-1, 1)
}

// WeightInitializer32FanIn fan in weight initialization
func WeightInitializer32FanIn(in, out int) float32 {
	return random32(-1, 1) / float32(math.Sqrt(float64(in)))
}

// WeightInitializer32FanInFanOut fan in/fan out weight initialization
func WeightInitializer32FanInFanOut(in, out int) float32 {
	return random32(-1, 1) * float32(4*math.Sqrt(6/float64(in+out)))
}

// Init initializes the neural network
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
			T:  identity,
			DF: dsigmoid32,
		}
	}
}

// UseTanh use tanh for the activation function
func (n *Neural32) UseTanh() {
	for f := range n.Functions {
		n.Functions[f].F = tanh32
		n.Functions[f].DF = dtanh32
	}
}

// EnableRegression removes the activation function from the last layer so
// that regression is performed
func (n *Neural32) EnableRegression() {
	output := len(n.Functions) - 1
	n.Functions[output].F = identity
	n.Functions[output].DF = one
}

// EnableDropout enables dropout based regularization
// See: http://iamtrask.github.io/2015/07/28/dropout/
func (n *Neural32) EnableDropout(probability float32) {
	depth := len(n.Layers) - 1
	for i := range n.Functions[:depth-1] {
		n.Functions[i].T = func(x float32) float32 {
			if rand.Float32() > 1-probability {
				x = 0
			} else {
				x *= 1 / (1 - probability)
			}
			return x
		}
	}
}

// NewNeural32 creates a neural network with the given configuration
func NewNeural32(config func(neural *Neural32)) *Neural32 {
	neural := &Neural32{}
	config(neural)
	return neural
}

// Context32 is an inference context
type Context32 struct {
	*Neural32
	Activations [][]float32
}

// SetInput sets the input to the neural network
func (c *Context32) SetInput(input []float32) {
	copy(c.Activations[0], input)
}

// GetOutput gets the output of the neural network
func (c *Context32) GetOutput() []float32 {
	return c.Activations[len(c.Activations)-1]
}

// NewContext creates a new inference context from the given neural network
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

// Infer runs inference
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

// InferWithT runs inference using a transform in between layers
func (c *Context32) InferWithT() {
	depth := len(c.Layers) - 1

	if depth > 1 {
		for i := range c.Activations[:depth-1] {
			activations, weights := c.Activations[i], c.Weights[i]
			for j := range weights[:len(weights)-1] {
				sum := dot32(activations, weights[j])
				c.Activations[i+1][j] = c.Functions[i].T(c.Functions[i].F(sum))
			}
		}
	}

	i := depth - 1
	activations, weights := c.Activations[i], c.Weights[i]
	for j := range weights[:len(weights)] {
		sum := dot32(activations, weights[j])
		c.Activations[i+1][j] = c.Functions[i].T(c.Functions[i].F(sum))
	}
}

// BackPropagate run the backpropagation algorithm
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

// Train trains a neural network using data from source
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
			context.InferWithT()
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
