// @flow

import * as bodyPix from '@tensorflow-models/body-pix';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import JitsiStreamBlurEffect from './JitsiStreamBlurEffect';

/**
 * Creates a new instance of JitsiStreamBlurEffect. This loads the bodyPix model that is used to
 * extract person segmentation.
 *
 * @returns {Promise<JitsiStreamBlurEffect>}
 */
export async function createBlurEffect() {
    if (!MediaStreamTrack.prototype.getSettings && !MediaStreamTrack.prototype.getConstraints) {
        throw new Error('JitsiStreamBlurEffect not supported!');
    }

    let result;

    try {
        console.log('TensorFlow backend is set to', tf.getBackend());
        result = await tf.setBackend('wasm');
        console.log('TensorFlow backend is set to', tf.getBackend());
    } catch (err) {
        throw new Error('JitsiStreamBlurEffect not supported!');
    }
    if (!result) {
        console.log('TensorFlow backend initialization failed');
        throw new Error('JitsiStreamBlurEffect not supported!');
    }

    // An output stride of 16 and a multiplier of 0.5 are used for improved
    // performance on a larger range of GPUs.
    const bpModel = await bodyPix.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        multiplier: 0.75,
        quantBytes: 2
    });

    return new JitsiStreamBlurEffect(bpModel);
}
