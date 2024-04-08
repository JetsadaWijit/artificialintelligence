import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path'; // Import path module for handling file paths

// Hyperparameters (adjust as needed)
const learningRate = 0.01;
const epochs = 1;
const batchSize = 32;

// Function to define the model architecture (replace with your specific model)
function createModel(inputShape) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // Output layer for binary classification
  model.compile({ loss: 'binaryCrossentropy', optimizer: tf.train.adam(learningRate) });
  return model;
}

async function trainModel(model, xTrain, yTrain, xTest, yTest, saveCondition) {
  const history = []; // Track training progress

  for (let epoch = 0; epoch < epochs; epoch++) {
    const shuffledIndices = Array.from({ length: xTrain.shape[0] }, (_, i) => i); // Array from 0 to xTrain.shape[0] - 1
    shuffleArray(shuffledIndices); // Shuffle the indices

    for (let i = 0; i < xTrain.shape[0]; i += batchSize) {
      const end = Math.min(i + batchSize, xTrain.shape[0]);
      const batchIndices = shuffledIndices.slice(i, end); // Select batch indices

      const batchX = tf.gather(xTrain, batchIndices);
      const batchY = tf.gather(yTrain, batchIndices);
      
      const batchLogs = await model.fit(batchX, batchY, { epochs: 100, validationData: [xTest, yTest] });
      history.push(batchLogs.history);
    }

    // Conditional logic to save model based on your criteria
    if (saveCondition(history)) {
      const savePath = path.join(__dirname, 'data'); // Save model in the 'data' directory in the current directory
      await model.save(`file://${savePath}`);
    }
  }

  return history; // Return training history for analysis
}

// Fisher-Yates shuffle algorithm
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}


// Example usage (replace with your actual data)
const xTrain = tf.tensor2d([[1, 2], [3, 4], [5, 6]]); // Features (training data)
const yTrain = tf.tensor2d([[0], [1], [0]]);          // Labels (training data)
const xTest = tf.tensor2d([[7, 8], [9, 10]]);         // Features (testing data)
const yTest = tf.tensor2d([[1], [0]]);                // Labels (testing data)

const model = createModel([2]); // Input shape of 2 features

// Example condition for saving (replace with your own logic)
const saveCondition = (history) => {
  // Check if there's any history and if the last epoch's validation loss is less than 0.2
  if (history.length === 0) {
    return false; // No history available
  }

  const lastEpochLogs = history[history.length - 1];
  const lastEpochValidationLoss = lastEpochLogs.val?.loss; // Access validation loss

  return typeof lastEpochValidationLoss === 'number' && lastEpochValidationLoss < 0.2;
};

trainModel(model, xTrain, yTrain, xTest, yTest, saveCondition)
  .then((history) => {
    console.log('Training complete!');
    console.log(history); // Analyze training progress metrics
  })
  .catch((err) => console.error(err));
