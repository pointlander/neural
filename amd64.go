// Copyright 2016 The Neural Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64

package neural

import (
	"github.com/ziutek/blas"
)

func dot32(X, Y []float32) float32 {
	return blas.Sdot(len(X), X, 1, Y, 1)
}

func scal32(alpha float32, X []float32) {
	blas.Sscal(len(X), alpha, X, 1)
}

func axpy32(alpha float32, X []float32, Y []float32) {
	blas.Saxpy(len(X), alpha, X, 1, Y, 1)
}

func dot64(X, Y []float64) float64 {
	return blas.Ddot(len(X), X, 1, Y, 1)
}

func scal64(alpha float64, X []float64) {
	blas.Dscal(len(X), alpha, X, 1)
}

func axpy64(alpha float64, X []float64, Y []float64) {
	blas.Daxpy(len(X), alpha, X, 1, Y, 1)
}
