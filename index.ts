import * as tf from '@tensorflow/tfjs';

// base template
const model = tf.sequential();
model.add(tf.layers.dense({units:1, inputShape:[2]}));

model.compile({loss:'meanSquaredError', optimizer:'adam'});

const xs = tf.tensor2d([[0,0], [0,1], [1,0], [1,1]], [4,2]);
const ys = tf.tensor2d([0,1,1,0], [4,1]);

model.fit(xs, ys).then(() => {
    model.predict(tf.tensor2d([[0,1]], [1,2])).print();
});


// tensor building block
const t1 = tf.tensor1d([1,2,3]);
t1.print();
// Tensor
//  [1, 2, 3]

const t2 = tf.tensor2d([1,2,3,4], [2,2]);
t2.print();
// Tensor
//  [[1, 2],
//   [3, 4]]

const t = tf.tensor1d([1,2,3]);

// Asynchronous API
t.data(d => {
    console.log(d); // Float32Array(3) [1, 2, 3]
});

// Synchronous API
console.log(t.dataSync()); // Float32Array(3) [1, 2, 3]


// Sequential Model
const model = tf.sequential({
    layers: [
        tf.layers.dense({inputShape:[784], units:16, activation:'relu'}),
        tf.layers.dense({units:10, activation:'softmax'})
    ]
});

// Layers API
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [784], units:16, activation:'relu'}))
model.add(tf.layers.dense({units:10, activation:'softmax'}))

// Functional Model API
const input = tf.input({shape:[784]})
const dense1 = tf.layers.dense({inputShape: [784], units:16, activation:'relu'}).apply(input);
const dense2 = tf.layers.dense({units:10, activation:'softmax'}).apply(dense2);