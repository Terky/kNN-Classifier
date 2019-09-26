//
//  fScore.swift
//  NonparametricRegressionLabSwift
//
//  Created by Артём Бурмистров on 9/24/19.
//  Copyright © 2019 Артём Бурмистров. All rights reserved.
//

import Foundation

func calcC(index: Int, confusionMatrix: [[Double]]) -> Double {
    var res = 0.0
    for row in confusionMatrix {
        res += row[index]
    }
    return res
}

func calcP(index: Int, confusionMatrix: [[Double]]) -> Double {
    var res = 0.0
    for i in confusionMatrix.indices {
        res += confusionMatrix[index][i]
    }
    return res
}

func calcPrecisionW(confusionMatrix: [[Double]], all: Int) -> Double {
    var res = 0.0
    for i in confusionMatrix.indices {
        let C = calcC(index: i, confusionMatrix: confusionMatrix)
        let P = calcP(index: i, confusionMatrix: confusionMatrix)
        res += C == 0 ? 0 : confusionMatrix[i][i] *  P / C
    }
    return all == 0 ? 0 : res / Double(all)
}

func calcRecallW(confusionMatrix: [[Double]], all: Int) -> Double {
    var res = 0.0
    for i in confusionMatrix.indices {
        res += confusionMatrix[i][i]
    }
    return all == 0 ? 0 : res / Double(all)
}

func calcF1Score(precision: Double, recall: Double) -> Double {
    if precision + recall == 0 {
        return 0
    }
    return 2 * precision * recall / (precision + recall)
}

func calcMicroF1(confusionMatrix: [[Double]], all: Int) -> Double {
    var res = 0.0
    for i in confusionMatrix.indices {
        let C = calcC(index: i, confusionMatrix: confusionMatrix)
        let P = calcP(index: i, confusionMatrix: confusionMatrix)
        let precision = C == 0 ? 0 : confusionMatrix[i][i] / C
        let recall = P == 0 ? 0 : confusionMatrix[i][i] / P
        res += calcF1Score(precision: precision, recall: recall) * P;
    }
    return all == 0 ? 0 : res / Double(all);
}

func calcMacroF1(confusionMatrix: [[Double]], all: Int) -> Double {
    return calcF1Score(precision: calcPrecisionW(confusionMatrix: confusionMatrix, all: all),
                       recall: calcRecallW(confusionMatrix: confusionMatrix, all: all));
}
