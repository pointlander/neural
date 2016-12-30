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

func ExampleSimpleNeural32() {
	rand.Seed(0)

	config := func(neural *Neural32) {
		neural.Init(WeightInitializer32FanIn, 2, 2, 1)
	}
	n := NewNeural32(config)

	n.Train(patterns, 1000, 0.6, 0.4)

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

	n.Train(patterns, 1000, 0.6, 0.4)

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

	n.Train(patterns, 10000, 0.6, 0.4)

	n.test(patterns)

	// Output:
	// [0 0] -> [0.025694605]  :  [0]
	// [0 1] -> [0.99999964]  :  [1]
	// [1 0] -> [0.9999012]  :  [1]
	// [1 1] -> [0.044942077]  :  [0]
}

func ExampleDeepNeural32() {
	rand.Seed(0)

	config := func(neural *Neural32) {
		neural.Init(WeightInitializer32FanIn, 2, 2, 2, 1)
	}
	n := NewNeural32(config)

	n.Train(patterns, 10000, 0.6, 0.4)

	n.test(patterns)

	// Output:
	// [0 0] -> [0.01086853]  :  [0]
	// [0 1] -> [0.9948494]  :  [1]
	// [1 0] -> [0.98888844]  :  [1]
	// [1 1] -> [0.010121063]  :  [0]
}
