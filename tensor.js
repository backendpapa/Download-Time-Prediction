const tf=require('@tensorflow/tfjs-node')

let tensorA=tf.tensor([[1,2,4,6]])
let tensorB=tf.tensor([2,3,5,3])


let addt=tensorA.add(tensorB)
console.log(addt.dataSync())
tf.dispose(tf)