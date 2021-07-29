// 1.Import tensorflow
const tf=require('@tensorflow/tfjs-node')

//  2. Create a train dataset of sizeMB and its label which is the time it takes to download
//  Each with a shape of [20,1] a 2x1 dim
const trainData={
    sizeMB: [0.080, 9.000, 0.001, 0.100, 8.000,
        5.000, 0.100, 6.000, 0.050, 0.500,
        0.002, 2.000, 0.005, 10.00, 0.010,
        7.000, 6.000, 5.000, 1.000, 1.000],
         
    timeSec: [0.135, 0.739, 0.067, 0.126, 0.646,
        0.435, 0.069, 0.497, 0.068, 0.116,
        0.070, 0.289, 0.076, 0.744, 0.083,
        0.560, 0.480, 0.399, 0.153, 0.149]
}

// A test data to evaluate the model to be trained
const testData = {
    sizeMB: [5.000, 0.200, 0.001, 9.000, 0.002,
    0.020, 0.008, 4.000, 0.001, 1.000,
    0.005, 0.080, 0.800, 0.200, 0.050,
    7.000, 0.005, 0.002, 8.000, 0.008],
    timeSec: [0.425,0.078,0.052,0.558,
        0.098,0.070,0.063,0.066,0.052,
        0.375,0.183,0.068,0.686,0.058,
        0.087,0.610,0.066,0.136,0.066,0.057]
        };

// 3. Datasets are converted to tensors  
const trainTensors={
    sizeMB:tf.tensor2d(trainData.sizeMB,[20,1]),
    timeSec:tf.tensor2d(trainData.timeSec,[20,1])
}

const testTensors={
    sizeMB:tf.tensor2d(testData.sizeMB,[20,1]),
    timeSec:tf.tensor2d(testData.timeSec,[20,1])
}


async function app(){

//  4. Model is initialized
const model=tf.sequential()

// 5. We added a dense layer with an input shape of 1
model.add(tf.layers.dense({inputShape:[1],units:1}))
console.log(model.summary())

// 6. compile model with an optimizer to measure the loss distance
model.compile({optimizer:'sgd',loss:'meanAbsoluteError'})

// 7 Train model
await model.fit(trainTensors.sizeMB,trainTensors.timeSec,{epochs:200})

// 8 Evaluate model against the test data
model.evaluate(testTensors.sizeMB,testTensors.timeSec).print()

const smallFileMB=1
const bigFileMB=100
const hugeFileMB=10000

// 9. Make predictions with new dataset
model.predict(tf.tensor2d([[smallFileMB],[bigFileMB],[hugeFileMB]])).print()

}
app()

