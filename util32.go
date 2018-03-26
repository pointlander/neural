// Copyright 2016 The Neural Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package neural

import (
	"math"
	"math/rand"
)

func matrix32(I, J int) [][]float32 {
	m, dense, offset := make([][]float32, I), make([]float32, I*J), 0
	for i := 0; i < I; i++ {
		m[i] = dense[offset : offset+J]
		offset += J
	}
	return m
}

func vector32(I int, fill float32) []float32 {
	v := make([]float32, I)
	for i := 0; i < I; i++ {
		v[i] = fill
	}
	return v
}

func random32(a, b float32) float32 {
	return (b-a)*rand.Float32() + a
}

func sigmoid32(x float32) float32 {
	return 1 / (1 + float32(math.Exp(-float64(x))))
}

func dsigmoid32(y float32) float32 {
	return y * (1 - y)
}

func identity(x float32) float32 {
	return x
}

func one(x float32) float32 {
	return 1
}
