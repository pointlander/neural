// Copyright 2016 The Neural Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 arm

package neural

func dot32(X, Y []float32) float32 {
	var sum float32
	for i, x := range X {
		sum += x * Y[i]
	}
	return sum
}

func scal32(alpha float32, X []float32) {
	for i, x := range X {
		X[i] = alpha * x
	}
}

func axpy32(alpha float32, X []float32, Y []float32) {
	for i, y := range Y {
		Y[i] = alpha*X[i] + y
	}
}

func dot64(X, Y []float64) float64 {
	var sum float64
	for i, x := range X {
		sum += x * Y[i]
	}
	return sum
}

func scal64(alpha float64, X []float64) {
	for i, x := range X {
		X[i] = alpha * x
	}
}

func axpy64(alpha float64, X []float64, Y []float64) {
	for i, y := range Y {
		Y[i] = alpha*X[i] + y
	}
}
