// Copyright 2016 The Neural Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package neural

import "math/rand"

var patterns = [][][]float32{
	{{0, 0}, {0}},
	{{0, 1}, {1}},
	{{1, 0}, {1}},
	{{1, 1}, {0}},
}

var source = func(iterations int) [][][]float32 {
	return patterns
}

func ExampleSimpleNeural32() {
	rand.Seed(0)

	config := func(neural *Neural32) {
		neural.Init(WeightInitializer32FanIn, 2, 2, 1)
	}
	n := NewNeural32(config)

	n.Train(source, 1000, 0.6, 0.4)

	n.test(patterns)

	// Output:
	// [0 0] -> [0.057274867]  :  [0]
	// [0 1] -> [0.9332078]  :  [1]
	// [1 0] -> [0.93215084]  :  [1]
	// [1 1] -> [0.08947442]  :  [0]
}

func ExampleRegressionNeural32() {
	rand.Seed(0)

	config := func(neural *Neural32) {
		neural.Init(WeightInitializer32FanIn, 2, 2, 1)
		neural.EnableRegression()
	}
	n := NewNeural32(config)

	n.Train(source, 1000, 0.6, 0.4)

	n.test(patterns)

	// Output:
	// [0 0] -> [0.00039592385]  :  [0]
	// [0 1] -> [1.0000901]  :  [1]
	// [1 0] -> [1.0000368]  :  [1]
	// [1 1] -> [6.206334e-05]  :  [0]
}

func ExampleDropoutNeural32() {
	rand.Seed(0)

	config := func(neural *Neural32) {
		neural.Init(WeightInitializer32FanIn, 2, 8, 1)
		neural.EnableDropout(.2)
	}
	n := NewNeural32(config)
	size := len(patterns)
	randomized := make([][][]float32, size)
	copy(randomized, patterns)
	src := func(iterations int) [][][]float32 {
		for i := 0; i < size; i++ {
			j := i + rand.Intn(size-i)
			randomized[i], randomized[j] = randomized[j], randomized[i]
		}
		return randomized
	}
	n.Train(src, 10000, 0.6, 0.4)

	n.test(patterns)

	// Output:
	// [0 0] -> [0.00061443914]  :  [0]
	// [0 1] -> [0.9990952]  :  [1]
	// [1 0] -> [0.9832545]  :  [1]
	// [1 1] -> [0.0011786821]  :  [0]
}

func ExampleDeepNeural32() {
	rand.Seed(0)

	config := func(neural *Neural32) {
		neural.Init(WeightInitializer32FanIn, 2, 2, 2, 1)
	}
	n := NewNeural32(config)

	n.Train(source, 10000, 0.6, 0.4)

	n.test(patterns)

	// Output:
	// [0 0] -> [0.01086853]  :  [0]
	// [0 1] -> [0.9948494]  :  [1]
	// [1 0] -> [0.98888844]  :  [1]
	// [1 1] -> [0.010121063]  :  [0]
}
