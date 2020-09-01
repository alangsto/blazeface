"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tf = require("@tensorflow/tfjs-core");
exports.disposeBox = function (box) {
    box.startEndTensor.dispose();
    box.startPoint.dispose();
    box.endPoint.dispose();
};
exports.createBox = function (startEndTensor) { return ({
    startEndTensor: startEndTensor,
    startPoint: tf.slice(startEndTensor, [0, 0], [-1, 2]),
    endPoint: tf.slice(startEndTensor, [0, 2], [-1, 2])
}); };
exports.scaleBox = function (box, factors) {
    var starts = tf.mul(box.startPoint, factors);
    var ends = tf.mul(box.endPoint, factors);
    var newCoordinates = tf.concat2d([starts, ends], 1);
    return exports.createBox(newCoordinates);
};
//# sourceMappingURL=box.js.map