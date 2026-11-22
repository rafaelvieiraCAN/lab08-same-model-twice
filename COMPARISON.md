# PyTorch vs Keras Comparison

## Overview

This lab implemented the same neural network architecture in both PyTorch and Keras and compared their behavior under identical conditions. Both models used the same dataset, preprocessing pipeline, network structure, optimizer, loss function, learning rate, batch size, and training epochs.

The primary objective was to determine whether both frameworks could represent the same function and whether they produced comparable prediction performance.

## Architectural Equivalence

The equivalence test demonstrated that weights could be copied directly from the PyTorch implementation into the Keras implementation. After the weight transfer, both models produced matching predictions for the same input data within numerical tolerance.

This confirms that both frameworks are capable of representing the same mathematical function.

## Experimental Results

Three random seeds were evaluated for each framework.

### PyTorch

| Seed | Validation MAE | Test MAE |
|------|------|------|
| 42 | 2.703652 | 2.823953 |
| 100 | 3.461853 | 3.315019 |
| 777 | 5.155214 | 5.388521 |

### Keras

| Seed | Validation MAE | Test MAE |
|------|------|------|
| 42 | 2.713721 | 2.909970 |
| 100 | 3.023979 | 2.882880 |
| 777 | 3.871483 | 3.889358 |

## Observed Differences

### 1. Both frameworks achieved comparable performance

For seed 42, the difference between validation MAE values was extremely small. This indicates that both implementations learned nearly identical solutions when initialized under similar conditions.

### 2. Random initialization still matters

The variation between different seeds was larger than the variation between frameworks. Seed choice had a stronger impact on performance than the framework itself.

### 3. Keras showed lower variation across seeds

In these experiments, the Keras implementation produced more consistent test errors across the evaluated seeds, while PyTorch exhibited larger fluctuations.

## Recommendation

I would recommend Keras for rapid prototyping and experimentation because it provides a simpler and more compact training workflow through features such as model.fit(), EarlyStopping, and ModelCheckpoint.

I would recommend PyTorch when greater flexibility, custom training loops, or research-oriented experimentation is required.

## What Is Lost When Choosing One Framework

Choosing Keras sacrifices some of the low-level control that PyTorch provides.

Choosing PyTorch requires more implementation effort because training loops, checkpoint handling, and monitoring are often implemented manually.

## What Would Change My Recommendation

If a project required highly customized architectures, advanced debugging, or research-driven development, I would favor PyTorch.

If the goal was rapid deployment, maintainability, and integration with TensorFlow production tools, I would favor Keras.

## Conclusion

The results show that both frameworks are capable of implementing the same neural network architecture and achieving very similar predictive performance. Framework choice should therefore be based primarily on development workflow, maintainability, deployment requirements, and team preferences rather than expected model accuracy.